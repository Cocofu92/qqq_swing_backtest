"""No-lookahead invariant test.

Build a synthetic 1H series spanning 30 calendar days. After computing daily
bias and forward-filling to 1H, assert that EVERY 1H bar at timestamp T uses
daily values strictly older than T.normalize() -- i.e. yesterday's CLOSED
daily, never today's still-in-progress daily.

Edge case: a 1H bar at 09:30 ET on day N must NOT use day N's daily close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators import compute_daily_bias, forward_fill_daily_to_1h


def _make_series(days: int = 30, start: str = "2024-01-02") -> pd.DataFrame:
    """Deterministic 1H series, 6.5 hours per US trading day, 21 trading days."""
    rng = np.random.default_rng(42)
    bars = []
    cur = pd.Timestamp(start, tz="US/Eastern").replace(hour=9, minute=30)
    base_price = 400.0
    for d in range(days):
        # Skip weekends
        while cur.weekday() >= 5:
            cur = (cur + pd.Timedelta(days=1)).replace(hour=9, minute=30)
        for h in range(7):  # 09:30, 10:30, ..., 15:30 -> 7 bars
            ts = (cur + pd.Timedelta(hours=h)).tz_convert("UTC")
            drift = 0.5 * d + 0.05 * h
            price = base_price + drift + rng.normal(0, 0.1)
            bars.append({
                "date": ts,
                "open": price - 0.05,
                "high": price + 0.10,
                "low": price - 0.10,
                "close": price,
                "volume": 100000 + d * 1000,
            })
        cur = (cur + pd.Timedelta(days=1)).replace(hour=9, minute=30)
    df = pd.DataFrame(bars).set_index("date")
    df.index.name = "date"
    return df


def test_no_lookahead_respects_yesterday_close():
    df_1h = _make_series(days=30)
    daily = compute_daily_bias(df_1h, fast_ema=3, slow_ema=5, short_ema=2,
                               medium_ema=4, donchian_low=3, daily_atr_period=3)
    merged = forward_fill_daily_to_1h(daily, df_1h)

    # For every 1H bar T, daily_close_yesterday must come from a daily index
    # date <= T.normalize() - 1 day.
    daily_dates = pd.to_datetime(daily.index).normalize()
    daily_close_map = {d: daily.loc[d, "close"] for d in daily.index}

    violations = 0
    checked = 0
    for ts, row in merged.iterrows():
        if pd.isna(row["daily_close_yesterday"]):
            continue
        cutoff = ts.normalize() - pd.Timedelta(days=1)
        # The merged value must equal a daily close <= cutoff
        candidates = [d for d in daily_dates if d <= cutoff]
        if not candidates:
            continue
        latest = max(candidates)
        expected = daily.loc[latest, "close"]
        if not np.isclose(row["daily_close_yesterday"], expected, equal_nan=False):
            violations += 1
        checked += 1

    assert checked > 0, "no bars checked -- test setup error"
    assert violations == 0, f"{violations}/{checked} bars used today's not-yet-closed daily"


def test_market_open_bar_uses_prior_day():
    """A 09:30 ET bar on day N must use day (N-1) daily, not day N."""
    df_1h = _make_series(days=15)
    daily = compute_daily_bias(df_1h, fast_ema=3, slow_ema=5, short_ema=2,
                               medium_ema=4, donchian_low=3, daily_atr_period=3)
    merged = forward_fill_daily_to_1h(daily, df_1h)

    # Find all bars at 14:30 UTC (= 09:30 ET in EST, 13:30 UTC in EDT).
    # We'll just check: for the first bar of each calendar date in merged,
    # the daily_close_yesterday must equal the prior CALENDAR day's daily close.
    by_date = merged.groupby(merged.index.normalize())
    daily_idx_norm = pd.DatetimeIndex(daily.index).normalize()

    found = 0
    for date, group in by_date:
        first_bar = group.iloc[0]
        if pd.isna(first_bar["daily_close_yesterday"]):
            continue
        # The daily close of "yesterday" must be from a date < date
        prior_dates = daily_idx_norm[daily_idx_norm < date]
        if len(prior_dates) == 0:
            continue
        latest = prior_dates.max()
        expected = daily.loc[daily.index.normalize() == latest].iloc[0]["close"]
        # Today's close must NOT match (that would be lookahead)
        today_match = daily.loc[daily.index.normalize() == date]
        if len(today_match):
            assert not np.isclose(first_bar["daily_close_yesterday"], today_match.iloc[0]["close"]) \
                or np.isclose(today_match.iloc[0]["close"], expected), (
                f"09:30 bar on {date} appears to use today's daily close")
        assert np.isclose(first_bar["daily_close_yesterday"], expected), \
            f"first bar on {date} should use prior daily close"
        found += 1
    assert found > 0, "no market-open bars checked"
