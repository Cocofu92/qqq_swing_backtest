"""Indicator calculations for QQQ multi-timeframe trend-pullback strategy.

Closed-bar enforcement: when a daily indicator is forward-filled into 1H bars,
the value used for any 1H bar at time T comes from the most recent daily bar
that has CLOSED -- i.e. the daily bar whose calendar date is strictly before
T.normalize(). The 09:30 ET bar of day N must use day (N-1)'s close, never
day N's still-in-progress daily.

Zone proximity is ATR-scaled: a zone is touched if the bar's low is within
+- (atr_mult * daily_atr_yesterday) of the zone level. This adapts to
volatility regimes -- a static % does not.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(s: pd.Series, period: int) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def compute_daily_bias(
    df_1h: pd.DataFrame,
    fast_ema: int = 50,
    slow_ema: int = 200,
    short_ema: int = 21,
    medium_ema: int = 100,
    donchian_low: int = 20,
    daily_atr_period: int = 14,
) -> pd.DataFrame:
    """Resample 1H -> daily bars (UTC), compute trend EMAs, daily ATR and zones.

    Returns a daily DataFrame with columns:
        open, high, low, close, volume,
        ema_fast, ema_slow, ema_short, ema_medium,
        donchian_low, daily_atr,
        bullish_strict (close > ema_fast AND ema_fast > ema_slow),
        bullish_loose  (close > ema_slow)
    """
    if df_1h.empty:
        return pd.DataFrame()

    daily = df_1h.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()

    daily["ema_fast"] = _ema(daily["close"], fast_ema)
    daily["ema_slow"] = _ema(daily["close"], slow_ema)
    daily["ema_short"] = _ema(daily["close"], short_ema)
    daily["ema_medium"] = _ema(daily["close"], medium_ema)
    daily["donchian_low"] = daily["low"].rolling(donchian_low, min_periods=donchian_low).min()
    daily["daily_atr"] = _atr(daily["high"], daily["low"], daily["close"], daily_atr_period)
    daily["bullish_strict"] = (daily["close"] > daily["ema_fast"]) & (
        daily["ema_fast"] > daily["ema_slow"]
    )
    daily["bullish_loose"] = daily["close"] > daily["ema_slow"]
    return daily


def forward_fill_daily_to_1h(daily: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    """Map each 1H bar to YESTERDAY'S CLOSED daily bar values.

    For 1H bar at timestamp T, lookup is done on daily bars with index date
    <= T.normalize() - 1 day. Today's daily is in-progress so it is NEVER used.
    """
    if df_1h.empty or daily.empty:
        return df_1h.copy()

    out = df_1h.copy()
    daily_y = daily.copy()
    # Move the daily index forward by 1 day so a 1H bar at time T finds the
    # previous day's daily values via merge_asof on T (rather than T+1).
    daily_y.index = daily_y.index + pd.Timedelta(days=1)

    sub = daily_y[
        [
            "close", "ema_fast", "ema_slow", "ema_short", "ema_medium",
            "donchian_low", "daily_atr", "bullish_strict", "bullish_loose",
        ]
    ].rename(
        columns={
            "close": "daily_close_yesterday",
            "ema_fast": "daily_ema50_y",
            "ema_slow": "daily_ema200_y",
            "ema_short": "daily_ema21_y",
            "ema_medium": "daily_ema100_y",
            "donchian_low": "daily_donchian_low_y",
            "daily_atr": "daily_atr_y",
            "bullish_strict": "daily_bullish_strict_y",
            "bullish_loose": "daily_bullish_loose_y",
        }
    )

    out_reset = out.reset_index().rename(columns={out.index.name or "index": "ts"})
    if "date" in out_reset.columns and "ts" not in out_reset.columns:
        out_reset = out_reset.rename(columns={"date": "ts"})
    sub_reset = sub.reset_index().rename(columns={sub.index.name or "index": "ts"})
    if "date" in sub_reset.columns and "ts" not in sub_reset.columns:
        sub_reset = sub_reset.rename(columns={"date": "ts"})

    merged = pd.merge_asof(
        out_reset.sort_values("ts"),
        sub_reset.sort_values("ts"),
        on="ts",
        direction="backward",
    )
    merged = merged.set_index("ts")
    merged.index.name = out.index.name or "date"
    for col in ("daily_bullish_strict_y", "daily_bullish_loose_y"):
        merged[col] = merged[col].astype("object").where(merged[col].notna(), False).astype(bool)
    return merged


def compute_hourly_signals(
    df: pd.DataFrame,
    rsi_period: int = 14,
    rsi_threshold: float = 35.0,
    atr_period: int = 14,
    trail_ema: int = 21,
) -> pd.DataFrame:
    """Add 1H-level columns: rsi, atr, ema_trail, engulfing, rsi_cross_up."""
    out = df.copy()
    out["rsi"] = _rsi(out["close"], rsi_period)
    out["atr"] = _atr(out["high"], out["low"], out["close"], atr_period)
    out["ema_trail"] = _ema(out["close"], trail_ema)

    prev_open = out["open"].shift(1)
    prev_close = out["close"].shift(1)
    cur_body_low = out[["open", "close"]].min(axis=1)
    cur_body_high = out[["open", "close"]].max(axis=1)
    prev_body_low = pd.concat([prev_open, prev_close], axis=1).min(axis=1)
    prev_body_high = pd.concat([prev_open, prev_close], axis=1).max(axis=1)
    out["engulfing"] = (
        (out["close"] > out["open"])  # current bullish
        & (prev_close < prev_open)     # prior bearish
        & (cur_body_low <= prev_body_low)
        & (cur_body_high >= prev_body_high)
    )

    prev_rsi = out["rsi"].shift(1)
    out["rsi_cross_up"] = (prev_rsi < rsi_threshold) & (out["rsi"] >= rsi_threshold)
    return out


# ---------------------------------------------------------------------------
# ATR-scaled zone proximity
# ---------------------------------------------------------------------------

ZONE_COLUMNS = {
    "ema21": "daily_ema21_y",
    "ema50": "daily_ema50_y",
    "ema100": "daily_ema100_y",
    "ema200": "daily_ema200_y",
    "donchian_low": "daily_donchian_low_y",
}

# Zones consulted in each mode. Order = priority for `zone_name` reporting.
MODE_ZONES = {
    "strict": ("ema21", "ema50", "donchian_low"),
    "loose":  ("ema21", "ema50", "ema100", "ema200", "donchian_low"),
}


def compute_zone_touch(
    df: pd.DataFrame,
    mode: str,
    zone_proximity_atr_mult: float = 0.5,
    zone_lookback_bars: int = 6,
) -> pd.DataFrame:
    """Add `zone_touched` (bool) and `zone_name` (str | None) using ATR proximity.

    Touch for a given bar = bar.low <= zone + atr_mult*daily_atr AND
    bar.high >= zone - atr_mult*daily_atr. Then OR-rolled over the prior
    `zone_lookback_bars` bars (inclusive of current).

    `mode` selects which zones are consulted (see MODE_ZONES).
    """
    if mode not in MODE_ZONES:
        raise ValueError(f"unknown mode: {mode!r} -- expected one of {list(MODE_ZONES)}")
    out = df.copy()

    if "daily_atr_y" not in out.columns:
        raise ValueError("daily_atr_y column missing -- run forward_fill_daily_to_1h first")
    daily_atr = out["daily_atr_y"]

    touch_flags = pd.DataFrame(index=out.index)
    for name in MODE_ZONES[mode]:
        col = ZONE_COLUMNS[name]
        if col not in out.columns:
            touch_flags[name] = False
            continue
        zone_val = out[col]
        band = zone_proximity_atr_mult * daily_atr
        upper = zone_val + band
        lower = zone_val - band
        bar_touched = (out["low"] <= upper) & (out["high"] >= lower)
        touch_flags[name] = (
            bar_touched.fillna(False)
            .rolling(zone_lookback_bars, min_periods=1)
            .max()
            .astype(bool)
        )

    out["zone_touched"] = touch_flags.any(axis=1)
    # Report which zone (priority order = MODE_ZONES[mode] order)
    name_arr: list = [None] * len(out)
    flags = {n: touch_flags[n].values for n in MODE_ZONES[mode]}
    for idx in range(len(out)):
        for n in MODE_ZONES[mode]:
            if flags[n][idx]:
                name_arr[idx] = n
                break
    out["zone_name"] = name_arr
    return out
