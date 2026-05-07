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

    def init(self):
        df = self.data.df
        engulf = df["engulfing"].values.astype(bool) if "engulfing" in df.columns else np.zeros(len(df), dtype=bool)
        rsi_cross = df["rsi_cross_up"].values.astype(bool) if "rsi_cross_up" in df.columns else np.zeros(len(df), dtype=bool)
        bias = df["daily_bullish_y"].values.astype(bool) if "daily_bullish_y" in df.columns else np.zeros(len(df), dtype=bool)
        zone = df["zone_touched"].values.astype(bool) if "zone_touched" in df.columns else np.zeros(len(df), dtype=bool)
        atr = df["atr"].values if "atr" in df.columns else np.full(len(df), np.nan)
        ema_trail = df["ema_trail"].values if "ema_trail" in df.columns else np.full(len(df), np.nan)
        zone_name = df["zone_name"].values if "zone_name" in df.columns else np.full(len(df), None)

        self._signal = bias & zone & (engulf | rsi_cross)
        self._atr = atr
        self._ema_trail = ema_trail
        self._zone_name = zone_name

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

    def next(self):
        i = len(self.data) - 1
        price = float(self.data.Close[-1])
        bar_open = float(self.data.Open[-1])
        bar_low = float(self.data.Low[-1])
        bar_high = float(self.data.High[-1])

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
            risk_dollars = self.equity * self.risk_pct
            size_shares = max(int(risk_dollars // risk_per_share), 1)
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

            ema = self._ema_trail[i]
            if not np.isnan(ema) and price < ema:
                exit_px = self._slip_sell(price)
                self.position.close()
                self._record("trail_exit", exit_px)
                self._reset_trade()
                return

        if not self.position and not self._pending_entry:
            if self._signal[i]:
                self._pending_entry = True
                self._pending_zone = self._zone_name[i] if self._zone_name is not None else None
