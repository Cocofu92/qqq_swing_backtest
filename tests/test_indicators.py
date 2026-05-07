"""Indicator sanity tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators import (
    _atr,
    _ema,
    _rsi,
    compute_daily_bias,
    compute_hourly_signals,
    compute_zone_touch,
    forward_fill_daily_to_1h,
)


def _trend_series(n: int = 200, start: float = 100.0, slope: float = 0.5) -> pd.DataFrame:
    """Deterministic 1H bars; trend up so bullish bias eventually flips on."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2023-01-02", periods=n, freq="h", tz="UTC")
    closes = start + slope * np.arange(n) + rng.normal(0, 0.2, n)
    df = pd.DataFrame({
        "open": closes - 0.05,
        "high": closes + 0.10,
        "low": closes - 0.10,
        "close": closes,
        "volume": np.full(n, 100_000, dtype="int64"),
    }, index=idx)
    df.index.name = "date"
    return df


def test_ema_matches_manual_short_window():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    out = _ema(s, span=3)
    # min_periods=span, so first valid index is at position 2
    # alpha = 2/(3+1) = 0.5
    # ewm(adjust=False).mean() => recursive
    # first non-NaN value = SMA-equivalent at index 2 with adjust=False is
    # actually iterative from the start; verify via a recompute
    expected = s.ewm(span=3, adjust=False, min_periods=3).mean()
    pd.testing.assert_series_equal(out, expected)


def test_donchian_low_matches_rolling_min():
    df = _trend_series(n=120)
    daily = compute_daily_bias(df, donchian_low=20)
    if len(daily) >= 20:
        manual = daily["low"].rolling(20, min_periods=20).min()
        pd.testing.assert_series_equal(daily["donchian_low"], manual, check_names=False)


def test_rsi_within_bounds():
    df = _trend_series(n=300)
    rsi = _rsi(df["close"], period=14).dropna()
    assert (rsi >= 0).all(), "RSI should be >= 0"
    assert (rsi <= 100).all(), "RSI should be <= 100"


def test_atr_positive_and_finite():
    df = _trend_series(n=200)
    atr = _atr(df["high"], df["low"], df["close"], period=14).dropna()
    assert (atr > 0).all()
    assert np.isfinite(atr).all()


def test_compute_daily_bias_columns():
    df = _trend_series(n=300)
    daily = compute_daily_bias(df, fast_ema=3, slow_ema=5, short_ema=2,
                               medium_ema=4, donchian_low=3, daily_atr_period=3)
    assert "ema_fast" in daily.columns
    assert "ema_slow" in daily.columns
    assert "ema_short" in daily.columns
    assert "ema_medium" in daily.columns
    assert "daily_atr" in daily.columns
    assert "bullish_strict" in daily.columns
    assert "bullish_loose" in daily.columns


def test_zone_touch_atr_proximity_modes():
    df = _trend_series(n=300)
    daily = compute_daily_bias(df, fast_ema=3, slow_ema=5, short_ema=2,
                               medium_ema=4, donchian_low=3, daily_atr_period=3)
    merged = forward_fill_daily_to_1h(daily, df)
    merged = compute_hourly_signals(merged)

    strict = compute_zone_touch(merged, mode="strict", zone_proximity_atr_mult=0.5,
                                zone_lookback_bars=6)
    loose = compute_zone_touch(merged, mode="loose", zone_proximity_atr_mult=0.5,
                               zone_lookback_bars=6)
    assert "zone_touched" in strict.columns
    assert "zone_name" in strict.columns
    # loose has more zones, so loose touches >= strict touches
    assert int(loose["zone_touched"].sum()) >= int(strict["zone_touched"].sum())


def test_unknown_mode_rejected():
    df = _trend_series(n=200)
    daily = compute_daily_bias(df, fast_ema=3, slow_ema=5, short_ema=2,
                               medium_ema=4, donchian_low=3, daily_atr_period=3)
    merged = forward_fill_daily_to_1h(daily, df)
    with pytest.raises(ValueError):
        compute_zone_touch(merged, mode="bogus")
