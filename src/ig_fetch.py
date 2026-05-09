"""IG bar fetcher for the qqq_swing_backtest backtest basket.

Two-step flow, both gated behind CLI flags:

  python -m src.ig_fetch --search
      Searches IG's market list for plausible epics for each symbol in our
      basket, prints a table with epic / instrumentName / type / streamable
      so we can confirm we're hitting the right markets.

  python -m src.ig_fetch --pull
      Reads epic mappings from data/ig/epics.yml, fetches 15min and 1hour
      bars for the configured history window, writes parquet files to
      data/ig/{SYMBOL}_{tf}.parquet.

Credentials come from environment: IG_API_KEY, IG_USERNAME, IG_PASSWORD,
IG_ACCOUNT_TYPE (LIVE or DEMO).
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from .ig_client import IGClient, IGError


# Symbols we run in the backtest -> IG search terms. Multiple terms tried.
SEARCH_TERMS: Dict[str, List[str]] = {
    "QQQ": ["US Tech 100", "Nasdaq 100"],
    "SPY": ["US 500", "S&P 500"],
    "GLD": ["Spot Gold", "Gold"],
    "BNO": ["Brent Crude", "Brent"],
    "EWU": ["FTSE 100"],
    "EWJ": ["Japan 225", "Nikkei"],
}

# Backtest timeframe -> IG resolution code.
RESOLUTION = {
    "15min": "MINUTE_15",
    "1hour": "HOUR",
    "1day": "DAY",
}


def _client() -> IGClient:
    """Build an IGClient from env vars."""
    missing = [k for k in ("IG_API_KEY", "IG_USERNAME", "IG_PASSWORD", "IG_ACCOUNT_TYPE")
               if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"missing env vars: {', '.join(missing)}")
    account_type = os.environ["IG_ACCOUNT_TYPE"].strip().upper()
    if account_type not in ("LIVE", "DEMO"):
        raise SystemExit(f"IG_ACCOUNT_TYPE must be LIVE or DEMO, got {account_type!r}")
    return IGClient(
        api_key=os.environ["IG_API_KEY"],
        username=os.environ["IG_USERNAME"],
        password=os.environ["IG_PASSWORD"],
        account_type=account_type,
    )


def search() -> None:
    """Search IG for plausible epics matching each backtest symbol."""
    client = _client()
    client.login()
    print("=" * 110)
    print(f"{'Symbol':<6} {'Search term':<22} {'Epic':<30} {'Instrument':<35} {'Type':<14} {'Stream'}")
    print("=" * 110)
    for sym, terms in SEARCH_TERMS.items():
        for term in terms:
            try:
                results = client.search_markets(term)
            except IGError as e:
                print(f"{sym:<6} {term:<22} (search error: {e})")
                continue
            for r in results[:8]:
                epic = r.get("epic", "")
                inst = (r.get("instrumentName") or "")[:35]
                itype = (r.get("instrumentType") or "")[:14]
                streamable = bool(r.get("streamingPricesAvailable"))
                print(f"{sym:<6} {term:<22} {epic:<30} {inst:<35} {itype:<14} {streamable}")
            print("-" * 110)
    print()
    allowance = client.get_allowance()
    print(f"Historical-data allowance: {allowance}")


def _parse_ts(raw: str) -> pd.Timestamp | None:
    if not raw:
        return None
    try:
        if "T" in raw:
            return pd.Timestamp(raw, tz="UTC")
        return pd.Timestamp(raw.replace("/", "-"), tz="UTC")
    except Exception:
        return None


def _mid(d: dict | None) -> float | None:
    if not d:
        return None
    bid, ask = d.get("bid"), d.get("ask")
    if bid is not None and ask is not None:
        return (float(bid) + float(ask)) / 2.0
    lt = d.get("lastTraded")
    return float(lt) if lt is not None else None


def pull(epics_yml_path: Path, out_dir: Path, history_days: Dict[str, int]) -> None:
    if not epics_yml_path.exists():
        raise SystemExit(
            f"epics file not found: {epics_yml_path}\n"
            "Run --search first, then write your chosen epics to this YAML."
        )
    mapping: Dict[str, Dict[str, Any]] = yaml.safe_load(epics_yml_path.read_text()) or {}
    if not mapping:
        raise SystemExit(f"epics file empty: {epics_yml_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    client = _client()
    client.login()
    now_utc = datetime.now(timezone.utc).replace(microsecond=0)
    summary: List[Dict[str, Any]] = []

    for sym, info in mapping.items():
        epic = info.get("epic")
        if not epic:
            print(f"[{sym}] no epic configured; skipping")
            continue
        for tf, days in history_days.items():
            resolution = RESOLUTION.get(tf)
            if not resolution:
                print(f"[{sym}/{tf}] unsupported timeframe; skipping")
                continue
            start = now_utc - timedelta(days=days)
            start_str = start.strftime("%Y-%m-%dT%H:%M:%S")
            end_str = now_utc.strftime("%Y-%m-%dT%H:%M:%S")
            try:
                resp = client.get_prices(
                    epic=epic, resolution=resolution,
                    start=start_str, end=end_str, max_points=10000,
                )
            except IGError as e:
                print(f"[{sym}/{tf}] fetch error: {e}")
                summary.append({"symbol": sym, "tf": tf, "epic": epic, "ok": False,
                                "n_bars": 0, "error": str(e)[:200]})
                continue

            prices = resp.get("prices") or []
            allowance = resp.get("allowance") or {}
            print(f"[{sym}/{tf}] epic={epic} bars={len(prices)} "
                  f"remaining={allowance.get('remainingAllowance')}")

            rows = []
            for p in prices:
                ts = _parse_ts(p.get("snapshotTimeUTC") or p.get("snapshotTime"))
                if ts is None:
                    continue
                op, hi, lo, cl = (_mid(p.get(k)) for k in
                                  ("openPrice", "highPrice", "lowPrice", "closePrice"))
                if any(x is None for x in (op, hi, lo, cl)):
                    continue
                rows.append({"timestamp": ts, "open": op, "high": hi, "low": lo,
                             "close": cl, "volume": int(p.get("lastTradedVolume") or 0)})

            if not rows:
                summary.append({"symbol": sym, "tf": tf, "epic": epic, "ok": True,
                                "n_bars": 0, "error": "no parseable bars"})
                continue

            df = pd.DataFrame(rows).set_index("timestamp").sort_index()
            df = df[~df.index.duplicated(keep="last")]
            out_path = out_dir / f"{sym}_{tf}.parquet"
            df.to_parquet(out_path)
            summary.append({"symbol": sym, "tf": tf, "epic": epic, "ok": True,
                            "n_bars": len(df), "error": "",
                            "first": str(df.index.min()), "last": str(df.index.max())})

    md = [f"# IG bar pull summary", "",
          f"_Pull at {now_utc.isoformat()}_", "",
          "| Symbol | TF | Epic | OK | Bars | First | Last | Error |",
          "|---|---|---|---|---|---|---|---|"]
    for r in summary:
        md.append(
            f"| {r['symbol']} | {r['tf']} | `{r['epic']}` | "
            f"{'OK' if r['ok'] else 'FAIL'} | {r.get('n_bars', 0)} | "
            f"{r.get('first', '-')} | {r.get('last', '-')} | {r.get('error') or '-'} |"
        )
    (out_dir / "pull_summary.md").write_text("\n".join(md))
    print()
    print("=== Summary ===")
    for r in summary:
        print(f"  {r['symbol']:<5} {r['tf']:<7} epic={r['epic']:<30} "
              f"bars={r.get('n_bars', 0):<6} ok={r['ok']}")


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="IG bar fetcher")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--epics", type=Path, default=Path("data/ig/epics.yml"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/ig"))
    ap.add_argument("--history-15min-days", type=int, default=30)
    ap.add_argument("--history-1hour-days", type=int, default=90)
    args = ap.parse_args(argv)

    if not (args.search or args.pull):
        ap.error("must pass either --search or --pull")

    if args.search:
        search()
    if args.pull:
        pull(args.epics, args.out_dir,
             history_days={
                 "15min": args.history_15min_days,
                 "1hour": args.history_1hour_days,
             })


if __name__ == "__main__":
    main()
