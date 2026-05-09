"""backtesting.py Strategy implementing the trend-pullback rules.

The bias signal is mode-dependent: the prepared DataFrame supplies a column
`daily_bullish_y` whose values are filled by the entrypoint depending on the
strict/loose mode. The strategy itself is mode-agnostic.

Entry conditions (all true on the SIGNAL bar):
    - daily_bullish_y (mode-specific bias from yesterday's closed daily)
    - zone_touched in the last N 1H bars (mode-specific zones, ATR-scaled)
    - engulfing OR rsi_cross_up fires THIS 1H bar

Execution: market buy at the OPEN of the NEXT 1H bar (no same-bar fills).
Stop: entry - atr_multiplier_stop * 1H_ATR(14).
Targets:
    - 2R partial: sell partial_exit_pct%, move stop to breakeven
    - Trail: full close < 1H 21 EMA -> exit remainder
Costs: enter at open + slippage_ticks*tick_size + spread/2; exit at price -
slippage - spread/2.
"""

from __future__ import annotations

import numpy as np
from backtesting import Strategy


class TrendPullback(Strategy):
    # --- knobs (overridden at instantiation by backtest.py) ---
    atr_multiplier_stop = 1.0
    take_partial_at_R = 2.0
    partial_exit_pct = 50
    spread = 0.03
    slippage_ticks = 1
    tick_size = 0.01
    risk_pct = 0.01
    zone_lookback_bars = 6
    trail_type = "ema21"        # "ema21" | "ema50" | "atr"
    trail_atr_mult = 2.0        # used when trail_type == "atr"
    rsi_threshold = 35          # propagated from config; not used inside the strategy itself
                                #  — signals are already pre-computed using this value
    margin = 1.0                # required-cash fraction; passed to Backtest(). Strategy uses
                                #  it only to compute affordability-cap (allowed buying power
                                #  is equity / margin). 1.0 = cash, 0.5 = 2x leverage.
    eod_force_close_utc = ""    # "HH:MM" UTC; force close any position at/after this time. Empty = disabled.
    eod_block_entries_after_utc = ""  # "HH:MM" UTC; no new entries after this time. Empty = disabled.

    def init(self):
        df = self.data.df
        engulf = df["engulfing"].values.astype(bool) if "engulfing" in df.columns else np.zeros(len(df), dtype=bool)
        rsi_cross = df["rsi_cross_up"].values.astype(bool) if "rsi_cross_up" in df.columns else np.zeros(len(df), dtype=bool)
        bias = df["daily_bullish_y"].values.astype(bool) if "daily_bullish_y" in df.columns else np.zeros(len(df), dtype=bool)
        zone = df["zone_touched"].values.astype(bool) if "zone_touched" in df.columns else np.zeros(len(df), dtype=bool)
        atr = df["atr"].values if "atr" in df.columns else np.full(len(df), np.nan)
        # Trail series (one of these is consulted depending on self.trail_type)
        ema21_arr = df["hourly_ema21"].values if "hourly_ema21" in df.columns else np.full(len(df), np.nan)
        ema50_arr = df["hourly_ema50"].values if "hourly_ema50" in df.columns else np.full(len(df), np.nan)
        zone_name = df["zone_name"].values if "zone_name" in df.columns else np.full(len(df), None)

        self._signal = bias & zone & (engulf | rsi_cross)
        self._atr = atr
        self._ema21 = ema21_arr
        self._ema50 = ema50_arr
        self._zone_name = zone_name
        self._high_water = -np.inf  # for ATR trail — ratchets up with price

        self._pending_entry = False
        self._pending_zone = None
        self._entry_price = None
        self._initial_stop = None
        self._target_price = None
        self._risk_per_share = None
        self._partial_taken = False
        self._stop_price = None
        self._zone_at_entry = None
        self._entry_date = None
        self._trade_log = []

    def _slip_buy(self, price: float) -> float:
        return price + self.slippage_ticks * self.tick_size + self.spread / 2.0

    def _slip_sell(self, price: float) -> float:
        return price - self.slippage_ticks * self.tick_size - self.spread / 2.0

    def _record(self, exit_reason: str, exit_price: float):
        if self._entry_price is None:
            return
        r_mult = (exit_price - self._entry_price) / self._risk_per_share if self._risk_per_share else 0.0
        self._trade_log.append(
            dict(
                entry_date=self._entry_date,
                entry_price=self._entry_price,
                stop=self._initial_stop,
                target=self._target_price,
                exit_date=self.data.index[-1],
                exit_price=exit_price,
                R_multiple=r_mult,
                exit_reason=exit_reason,
                zone_triggered=self._zone_at_entry,
            )
        )

    def _reset_trade(self):
        self._entry_price = None
        self._initial_stop = None
        self._target_price = None
        self._risk_per_share = None
        self._partial_taken = False
        self._stop_price = None
        self._zone_at_entry = None
        self._entry_date = None
        self._high_water = -np.inf  # reset ratchet for next ATR-trail trade

    def next(self):
        i = len(self.data) - 1
        price = float(self.data.Close[-1])
        bar_open = float(self.data.Open[-1])
        bar_low = float(self.data.Low[-1])
        bar_high = float(self.data.High[-1])

        # ----- EOD logic for day-trading (forces no overnight financing on IG spread bet) -----
        bar_ts = self.data.index[-1]
        bar_time_str = bar_ts.strftime("%H:%M") if hasattr(bar_ts, "strftime") else ""
        is_force_close = bool(self.eod_force_close_utc) and bar_time_str >= self.eod_force_close_utc
        is_block_entries = bool(self.eod_block_entries_after_utc) and bar_time_str >= self.eod_block_entries_after_utc

        # If force-close window active and we're holding, exit at this bar's open.
        if is_force_close and self.position:
            exit_px = self._slip_sell(bar_open)
            self.position.close()
            self._record("eod_force_close", exit_px)
            self._reset_trade()
            self._pending_entry = False
            self._pending_zone = None
            return

        # If past block-entry window, kill any pending entry.
        if is_block_entries and self._pending_entry and not self.position:
            self._pending_entry = False
            self._pending_zone = None

        if self._pending_entry and not self.position:
            entry_fill = self._slip_buy(bar_open)
            atr_at_signal = self._atr[i - 1] if i > 0 else np.nan
            if np.isnan(atr_at_signal) or atr_at_signal <= 0:
                self._pending_entry = False
                self._pending_zone = None
                return
            stop = entry_fill - self.atr_multiplier_stop * atr_at_signal
            risk_per_share = entry_fill - stop
            target = entry_fill + self.take_partial_at_R * risk_per_share
            # Scale risk_pct by 1/margin: margin=1.0 -> 1×risk, margin=0.5 -> 2×risk, margin=0.33 -> 3×risk.
            # This is what "2× leverage" actually means in Adam-speak: double position, double DD potential.
            effective_risk_pct = self.risk_pct / max(self.margin, 0.001)
            risk_dollars = self.equity * effective_risk_pct
            risk_based = int(risk_dollars // risk_per_share)
            # Buying power expands with leverage too (so affordability cap doesn't strangle risk-based sizing)
            buying_power = self.equity / max(self.margin, 0.001)
            max_affordable = int((buying_power * 0.95) // entry_fill)  # 5% buffer for variance
            size_shares = max(1, min(risk_based, max_affordable))
            if size_shares < 1:
                # Can't afford even 1 share -- skip
                self._pending_entry = False
                self._pending_zone = None
                return
            self.buy(size=size_shares)
            self._entry_price = entry_fill
            self._initial_stop = stop
            self._stop_price = stop
            self._target_price = target
            self._risk_per_share = risk_per_share
            self._zone_at_entry = self._pending_zone
            self._entry_date = self.data.index[-1]
            self._pending_entry = False
            self._pending_zone = None
            return

        if self.position:
            if bar_low <= self._stop_price:
                exit_px = self._slip_sell(self._stop_price)
                self.position.close()
                self._record("stop_hit", exit_px)
                self._reset_trade()
                return

            if (not self._partial_taken) and bar_high >= self._target_price:
                frac = self.partial_exit_pct / 100.0
                for tr in list(self.trades):
                    tr.close(portion=frac)
                self._partial_taken = True
                self._stop_price = self._entry_price
                exit_px = self._slip_sell(self._target_price)
                self._trade_log.append(
                    dict(
                        entry_date=self._entry_date,
                        entry_price=self._entry_price,
                        stop=self._initial_stop,
                        target=self._target_price,
                        exit_date=self.data.index[-1],
                        exit_price=exit_px,
                        R_multiple=self.take_partial_at_R,
                        exit_reason="target_hit_partial",
                        zone_triggered=self._zone_at_entry,
                    )
                )

            # ----- trailing exit dispatch (configurable per run) -----
            if self.trail_type == "ema21":
                trail_level = self._ema21[i]
            elif self.trail_type == "ema50":
                trail_level = self._ema50[i]
            elif self.trail_type == "atr":
                # Chandelier-style: ratchet high-water and trail by atr_mult * ATR
                self._high_water = max(self._high_water, bar_high)
                cur_atr = self._atr[i]
                if np.isnan(cur_atr) or cur_atr <= 0:
                    trail_level = np.nan
                else:
                    trail_level = self._high_water - self.trail_atr_mult * cur_atr
            else:
                trail_level = np.nan

            if not np.isnan(trail_level) and price < trail_level:
                exit_px = self._slip_sell(price)
                self.position.close()
                self._record("trail_exit", exit_px)
                self._reset_trade()
                return

        if not self.position and not self._pending_entry and not is_block_entries:
            if self._signal[i]:
                self._pending_entry = True
                self._pending_zone = self._zone_name[i] if self._zone_name is not None else None
