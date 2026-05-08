"""FMP 1H data fetcher with 24h parquet cache.

The FMP `historical-chart/1hour` endpoint returns intraday bars in US/Eastern.
We convert to UTC tz-aware on load. Output schema:
    index: tz-aware UTC DatetimeIndex
    columns: open, high, low, close, volume (all float, volume int64)

Also exports `fetch_daily_close(symbol, start, end)` for the SPY buy-and-hold
baseline used in comparison plots.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# Map our timeframe slug -> FMP path segment + parquet filename.
# Adam confirmed FMP's "stable" hierarchy 2026-05-08; intraday endpoints truncate to ~3mo per request.
INTRADAY_TF = {
    "15min": ("15min", DATA_DIR / "qqq_15min.parquet"),
    "1hour": ("1hour", DATA_DIR / "qqq_1hour.parquet"),
    "4hour": ("4hour", DATA_DIR / "qqq_4hour.parquet"),
}
FMP_INTRADAY_BASE = "https://financialmodelingprep.com/stable/historical-chart"
FMP_BASE_DAILY = "https://financialmodelingprep.com/stable/historical-price-eod/full"

# Backwards-compatible default cache (legacy 1h-only callers).
CACHE_PATH = INTRADAY_TF["1hour"][1]


def _fmp_key() -> str:
    key = os.environ.get("FMP_KEY", "").strip()
    if not key:
        raise RuntimeError("FMP_KEY env var is not set; cannot fetch from FMP.")
    return key


def _normalise_intraday(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "date" not in df.columns:
        raise ValueError(f"Unexpected FMP payload columns: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"])
    df["date"] = (
        df["date"]
        .dt.tz_localize("America/New_York", ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )
    df = df.set_index("date").sort_index()
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep]
    cast = {c: "float64" for c in keep if c != "volume"}
    if "volume" in keep:
        cast["volume"] = "int64"
    df = df.astype(cast)
    df.index.name = "date"
    return df


def _cache_is_fresh(path: Path, max_age_hours: float) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < max_age_hours * 3600.0


def _cache_covers(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    if df.empty:
        return False
    return df.index.min() <= start and df.index.max() >= end - pd.Timedelta(days=2)


def _load_cache(path: "Path | str" = CACHE_PATH) -> Optional[pd.DataFrame]:
    p = Path(path) if not isinstance(path, Path) else path
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def _save_cache(df: pd.DataFrame, path: Path = CACHE_PATH) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _fetch_intraday_chunk(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    if timeframe not in INTRADAY_TF:
        raise ValueError(f"Unknown timeframe {timeframe!r}; expected one of {list(INTRADAY_TF)}")
    fmp_seg, _ = INTRADAY_TF[timeframe]
    url = f"{FMP_INTRADAY_BASE}/{fmp_seg}?symbol={symbol}&from={start}&to={end}&apikey={_fmp_key()}"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected FMP payload: {payload!r}")
    return _normalise_intraday(pd.DataFrame(payload))


def fetch_qqq_intraday(
    timeframe: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    cache_max_age_hours: float = 24.0,
) -> pd.DataFrame:
    """Return tz-aware UTC intraday QQQ bars at `timeframe` (15min, 1hour, 4hour)."""
    if timeframe not in INTRADAY_TF:
        raise ValueError(f"Unknown timeframe {timeframe!r}; expected one of {list(INTRADAY_TF)}")
    _, cache_path = INTRADAY_TF[timeframe]

    start_ts = pd.Timestamp(start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    end_ts = pd.Timestamp(end)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")

    cached = _load_cache(cache_path)
    if (
        cached is not None
        and _cache_is_fresh(cache_path, cache_max_age_hours)
        and _cache_covers(cached, start_ts, end_ts)
    ):
        sliced = cached.loc[(cached.index >= start_ts) & (cached.index <= end_ts)]
        print(f"[data:{timeframe}] loaded {len(sliced)} bars from cache")
        return sliced

    # 4h has fewer bars per call; can chunk wider. Intraday capped by FMP at ~3mo.
    chunk_days = pd.Timedelta(days=60) if timeframe in ("15min", "1hour") else pd.Timedelta(days=180)
    cursor = start_ts
    step = pd.Timedelta(minutes={"15min": 15, "1hour": 60, "4hour": 240}[timeframe])
    chunks: list[pd.DataFrame] = []
    while cursor < end_ts:
        chunk_end = min(cursor + chunk_days, end_ts)
        df = _fetch_intraday_chunk("QQQ", timeframe, cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"))
        if not df.empty:
            print(f"[data:{timeframe}] chunk {cursor.date()}->{chunk_end.date()}: {len(df)} bars, {df.index[0]} to {df.index[-1]}")
        else:
            print(f"[data:{timeframe}] chunk {cursor.date()}->{chunk_end.date()}: 0 bars (empty)")
        chunks.append(df)
        if chunk_end >= end_ts:
            break
        cursor = chunk_end + step

    if not chunks:
        raise RuntimeError("FMP returned no data")

    full = pd.concat(chunks).sort_index()
    full = full[~full.index.duplicated(keep="last")]
    print(f"[data:{timeframe}] total after dedupe: {len(full)} bars, {full.index[0] if len(full) else 'N/A'} to {full.index[-1] if len(full) else 'N/A'}")
    _save_cache(full, cache_path)
    sliced = full.loc[(full.index >= start_ts) & (full.index <= end_ts)]
    print(f"[data:{timeframe}] returning {len(sliced)} bars in requested window, cached to {cache_path}")
    return sliced


# Backwards-compat alias for any legacy 1h-only call site.
def fetch_qqq_1h(*args, **kwargs):
    return fetch_qqq_intraday("1hour", *args, **kwargs)


def fetch_daily_close(symbol: str, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.Series:
    """Return tz-aware UTC daily close series for the buy-and-hold baseline."""
    start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end).strftime("%Y-%m-%d")
    url = f"{FMP_BASE_DAILY}?symbol={symbol}&from={start_s}&to={end_s}&apikey={_fmp_key()}"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    # New /stable/ EOD endpoint returns a direct array of bar dicts.
    # Old v3 wrapped them in {"historical": [...]}; support both for safety.
    if isinstance(payload, dict):
        rows = payload.get("historical")
    else:
        rows = payload
    if not rows:
        return pd.Series(dtype="float64")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("UTC")
    df = df.set_index("date").sort_index()
    return df["close"].astype("float64")


def parse_end_date(end: str) -> pd.Timestamp:
    """Resolve config 'today' to a real timestamp."""
    if end == "today":
        return pd.Timestamp.utcnow().normalize().tz_localize(None).tz_localize("UTC") if pd.Timestamp.utcnow().tzinfo is None else pd.Timestamp.utcnow().normalize()
    ts = pd.Timestamp(end)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts
