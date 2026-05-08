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
CACHE_PATH = DATA_DIR / "qqq_1h.parquet"
FMP_BASE_INTRADAY = "https://financialmodelingprep.com/stable/historical-chart/1hour"
FMP_BASE_DAILY = "https://financialmodelingprep.com/stable/historical-price-eod/full"


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


def _load_cache() -> Optional[pd.DataFrame]:
    if not CACHE_PATH.exists():
        return None
    try:
        return pd.read_parquet(CACHE_PATH)
    except Exception:
        return None


def _save_cache(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH)


def _fetch_intraday_chunk(symbol: str, start: str, end: str) -> pd.DataFrame:
    url = f"{FMP_BASE_INTRADAY}?symbol={symbol}&from={start}&to={end}&apikey={_fmp_key()}"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected FMP payload: {payload!r}")
    return _normalise_intraday(pd.DataFrame(payload))


def fetch_qqq_1h(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    cache_max_age_hours: float = 24.0,
) -> pd.DataFrame:
    """Return tz-aware UTC 1H QQQ bars between [start, end] inclusive."""
    start_ts = pd.Timestamp(start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    end_ts = pd.Timestamp(end)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")

    cached = _load_cache()
    if (
        cached is not None
        and _cache_is_fresh(CACHE_PATH, cache_max_age_hours)
        and _cache_covers(cached, start_ts, end_ts)
    ):
        sliced = cached.loc[(cached.index >= start_ts) & (cached.index <= end_ts)]
        print(f"Loaded {len(sliced)} bars from cache")
        return sliced

    chunks: list[pd.DataFrame] = []
    cursor = start_ts
    one_year = pd.Timedelta(days=365)
    while cursor < end_ts:
        chunk_end = min(cursor + one_year, end_ts)
        df = _fetch_intraday_chunk("QQQ", cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"))
        if not df.empty:
            print(f"[data] chunk {cursor.date()}->{chunk_end.date()}: {len(df)} bars, {df.index[0]} to {df.index[-1]}")
        else:
            print(f"[data] chunk {cursor.date()}->{chunk_end.date()}: 0 bars (empty)")
        chunks.append(df)
        if chunk_end >= end_ts:
            break
        cursor = chunk_end + pd.Timedelta(hours=1)

    if not chunks:
        raise RuntimeError("FMP returned no data")

    full = pd.concat(chunks).sort_index()
    full = full[~full.index.duplicated(keep="last")]
    print(f"[data] total after dedupe: {len(full)} bars, {full.index[0] if len(full) else 'N/A'} to {full.index[-1] if len(full) else 'N/A'}")
    _save_cache(full)
    sliced = full.loc[(full.index >= start_ts) & (full.index <= end_ts)]
    print(f"Fetched {len(sliced)} bars from FMP, cached to {CACHE_PATH}")
    return sliced


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
