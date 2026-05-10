# Portfolio: qqq_gld_pullback @ 5x

_Pullback variant on the same 2-symbol basket, for comparison_

## Portfolio metrics

| Metric | Train+Holdout | Holdout-only |
|---|---|---|
| Starting equity | £10,000 | £10,000 (notional) |
| **Ending equity** | **£42,550** | £42,550 |
| Total return [%] | 325.50 | 122.18 |
| CAGR [%] | 40.03 | -- |
| Max drawdown [%] | -31.59 | -31.59 |
| Max drawdown (£) | £3,159 | £3,159 |

## Members

| Symbol | TF | Variant | Train Return [%] | Holdout Return [%] | Member MaxDD [%] | Trades | Status |
|---|---|---|---|---|---|---|---|
| GLD | 15min | `pullback_atr_2.5_rsi40_tiered` | 24.15 | 223.53 | -48.50 | 1135 | ok |
| QQQ | 1hour | `pullback_atr_2.5_rsi40_legacy50` | 158.86 | 73.58 | -33.53 | 275 | ok |