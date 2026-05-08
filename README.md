# qqq_swing_backtest (v2: + 1H 21/50/200 EMA zones)

Multi-timeframe trend-pullback backtest on QQQ. Iteration 1: correctness +
configurability over completeness.

## Strategy in one paragraph

Daily timeframe gives us the bias and the candidate "pullback zones." Hourly
timeframe gives us the trigger. While the daily is bullish (mode-dependent --
see below), watch for the 1H price action to pull back into a daily zone, then
fire on a bullish-engulfing 1H candle OR a 1H RSI(14) cross back above 35.
Entry is at the OPEN of the next 1H bar -- never the same bar as the signal,
so there is no lookahead. Stop is 1 x 1H ATR below entry. Take 50% off at +2R
and move stop to breakeven; trail the remainder with the 1H 21 EMA (full close
below it = exit).

## Two modes

This project intentionally runs both filter regimes back-to-back so we can
compare:

| | Strict | Loose |
|---|---|---|
| Bias | close > 50EMA AND 50EMA > 200EMA | close > 200EMA only |
| Zones | 21EMA, 50EMA, 20-day Donchian low | 21EMA, 50EMA, 100EMA, 200EMA, 20-day Donchian low |

Switching modes is a single config flip (`strategy.daily_filter_mode`) or
a CLI flag (`--mode strict|loose`).

## Zone proximity is ATR-based

A zone is "touched" when the 1H bar's range crosses within
`zone_proximity_atr_mult` x daily ATR(14) of the zone level. Static % cannot
adapt to volatility regimes -- 0.5% on QQQ at $400 is $2, but daily ATR moves
between roughly $3 (calm) and $8 (volatile). The ATR-scaled band scales with
the regime.

## How to run

### Locally

```bash
pip install -e .
export FMP_KEY=...                 # FMP API key
pytest tests/ -v                   # unit tests + no-lookahead invariant
python -m src.backtest             # both modes
python -m src.backtest --mode strict
```

Outputs land in `outputs/<mode>/` plus `outputs/comparison.{png,md}`.

### Via GitHub Actions

1. Add `FMP_KEY` as a repo secret
   (`Settings > Secrets and variables > Actions > New repository secret`).
2. Trigger `Run QQQ backtest` from the Actions tab. The workflow runs both
   modes and commits `outputs/` back to the repo.

## Where the knobs live

`config.yml` is the single source of truth. Tweak EMAs, ATR multipliers, RSI
threshold, costs, train/holdout split, `daily_filter_mode`, etc. Nothing else
needs to change to re-run with new numbers.

## v1 status & deliberate non-goals

- No walk-forward optimisation, no parameter sweep, no out-of-sample beyond
  the train (2022 -> 2024-12-31) and holdout (2025-01-01 -> today) split.
- One position at a time, no pyramiding.
- 1H QQQ only. SPY is fetched only as the buy-and-hold baseline for the
  comparison overlay.
- Long-only.
- Costs are fixed (single spread + fixed slippage ticks). No regime-aware
  cost model.

These are intentional cuts for v1. Future iterations can layer them in
without rewriting the core.

## Outputs

```
outputs/
  strict/{equity_curve.png, stats.md, trade_log.csv, last_6mo_chart.png}
  loose/ {equity_curve.png, stats.md, trade_log.csv, last_6mo_chart.png}
  comparison.png    # strict vs. loose vs. SPY buy & hold
  comparison.md     # side-by-side stats
```


## v3 — multi-timeframe sweep

Now sweeps **3 timeframes × 2 modes = 6 backtests per workflow run**:
- 15min / 1hour / 4hour intraday execution
- strict (close>50EMA AND 50EMA>200EMA, daily 21/50/donchian zones)
- loose (close>200EMA, daily 21/50/100/200 + 1H 21/50/200 + donchian zones)

Outputs are written to `outputs/{15min,1hour,4hour}/{strict,loose}/` plus a unified `outputs/comparison.{md,png}` overlaying all 6 against SPY buy & hold.

