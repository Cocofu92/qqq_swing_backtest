"""Plot helpers (kept thin -- backtest.py inlines its own matplotlib).

This module exists so other entrypoints/notebooks can re-render charts from
saved trade logs / equity curves without re-running the backtest. Iteration
1 is intentionally minimal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def load_trade_log(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, parse_dates=["entry_date", "exit_date"])


def r_distribution_summary(trades: pd.DataFrame) -> Optional[dict]:
    if trades.empty or "R_multiple" not in trades.columns:
        return None
    series = pd.to_numeric(trades["R_multiple"], errors="coerce").dropna()
    if series.empty:
        return None
    return {
        "n": int(len(series)),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "win_rate": float((series > 0).mean()),
        "expectancy": float(series.mean()),
    }
