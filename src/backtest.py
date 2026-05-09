"""Backtest entry point.

Iterates over `modes` from config (default: strict, loose), running each on
TRAIN and HOLDOUT slices independently. Writes outputs/<mode>/{equity_curve.png,
stats.md, trade_log.csv, last_6mo_chart.png} and outputs/comparison.{png,md}
overlaying the equity curves with a SPY buy-and-hold baseline.

CLI:
    python -m src.backtest                      # both modes
    python -m src.backtest --mode strict        # only strict
    python -m src.backtest --mode loose         # only loose
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from backtesting import Backtest

from .data import fetch_daily_close, fetch_intraday, fetch_qqq_intraday, parse_end_date
from .indicators import (
    compute_daily_bias,
    compute_hourly_signals,
    compute_zone_touch,
    forward_fill_daily_to_1h,
)
from .strategy import TrendPullback

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"


def load_config() -> Dict[str, Any]:
    with open(ROOT / "config.yml") as f:
        return yaml.safe_load(f)


def prepare_base_data(cfg: Dict[str, Any], timeframe: str = "1hour", symbol: str = "QQQ") -> pd.DataFrame:
    """Fetch `symbol` at `timeframe` and attach all daily indicators (mode-agnostic)."""
    start = cfg["data"]["start_date"]
    end_raw = cfg["data"]["end_date"]
    end = parse_end_date(end_raw) if end_raw == "today" else pd.Timestamp(end_raw, tz="UTC")
    df_1h = fetch_intraday(symbol, timeframe, start, end, cache_max_age_hours=cfg["data"]["cache_max_age_hours"])

    daily = compute_daily_bias(
        df_1h,
        fast_ema=cfg["strategy"]["daily"]["fast_ema"],
        slow_ema=cfg["strategy"]["daily"]["slow_ema"],
        short_ema=cfg["strategy"]["daily"]["short_ema"],
        medium_ema=cfg["strategy"]["daily"]["medium_ema"],
        donchian_low=cfg["strategy"]["daily"]["donchian_low"],
        daily_atr_period=cfg["strategy"]["daily"]["daily_atr_period"],
    )
    df = forward_fill_daily_to_1h(daily, df_1h)
    df = compute_hourly_signals(
        df,
        rsi_period=cfg["strategy"]["hourly"]["rsi_period"],
        rsi_threshold=cfg["strategy"]["hourly"]["rsi_threshold"],
        atr_period=cfg["strategy"]["hourly"]["atr_period"],
        trail_ema=cfg["strategy"]["hourly"]["trail_ema"],
    )
    return df


def select_mode(df_base: pd.DataFrame, mode: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Return a copy with mode-specific bias and zones, ready for backtesting."""
    df = df_base.copy()
    bias_col = "daily_bullish_strict_y" if mode == "strict" else "daily_bullish_loose_y"
    df["daily_bullish_y"] = df[bias_col]
    df = compute_zone_touch(
        df,
        mode=mode,
        zone_proximity_atr_mult=cfg["strategy"]["daily"]["zone_proximity_atr_mult"],
        zone_lookback_bars=cfg["strategy"]["daily"]["zone_lookback_bars"],
    )
    df = df.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    return df


def run_slice(df: pd.DataFrame, cfg: Dict[str, Any], trail_type: str = "ema21", rsi_threshold: float = 35.0, atr_mult: Optional[float] = None, margin: float = 1.0) -> Dict[str, Any]:
    if df.empty:
        return {"stats": None, "equity_curve": pd.Series(dtype=float), "trade_log": []}

    bt = Backtest(
        df,
        TrendPullback,
        cash=cfg["execution"]["initial_capital"],
        commission=cfg["costs"]["commission"],
        exclusive_orders=cfg["execution"]["one_position_at_a_time"],
        trade_on_close=False,
        margin=margin,
    )
    stats = bt.run(
        atr_multiplier_stop=cfg["strategy"]["hourly"]["atr_multiplier_stop"],
        take_partial_at_R=cfg["strategy"]["hourly"]["take_partial_at_R"],
        partial_exit_pct=cfg["strategy"]["hourly"]["partial_exit_pct"],
        spread=cfg["costs"]["spread"],
        slippage_ticks=cfg["costs"]["slippage_ticks"],
        tick_size=cfg["costs"]["tick_size"],
        risk_pct=cfg["execution"]["risk_pct"],
        zone_lookback_bars=cfg["strategy"]["daily"]["zone_lookback_bars"],
        trail_type=trail_type,
        trail_atr_mult=atr_mult if atr_mult is not None else cfg["strategy"]["hourly"].get("trail_atr_mult", 2.0),
        rsi_threshold=rsi_threshold,
        margin=margin,
    )
    strat = stats._strategy
    return {
        "stats": stats,
        "equity_curve": stats["_equity_curve"]["Equity"],
        "trade_log": getattr(strat, "_trade_log", []),
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

STAT_KEYS = [
    ("Total Return [%]", "Return [%]"),
    ("CAGR [%]", "Return (Ann.) [%]"),
    ("Win Rate [%]", "Win Rate [%]"),
    ("Profit Factor", "Profit Factor"),
    ("Max Drawdown [%]", "Max. Drawdown [%]"),
    ("Sharpe Ratio", "Sharpe Ratio"),
    ("Total Trades", "# Trades"),
    ("Avg Trade Duration", "Avg. Trade Duration"),
]


def _stat(stats: Optional[Any], key: str) -> str:
    if stats is None or key not in stats:
        return "n/a"
    val = stats[key]
    if isinstance(val, float):
        return f"{val:.3f}"
    return str(val)


def write_stats_md(mode: str, train: Dict[str, Any], holdout: Dict[str, Any], outdir: Path) -> None:
    out = [f"# QQQ Trend-Pullback ({mode}) -- Stats", ""]
    out.append("| Metric | Train | Holdout |")
    out.append("|---|---|---|")
    for label, key in STAT_KEYS:
        out.append(f"| {label} | {_stat(train['stats'], key)} | {_stat(holdout['stats'], key)} |")
    out.append("")
    out.append(f"_Generated: {datetime.utcnow().isoformat()}Z_")
    (outdir / "stats.md").write_text("\n".join(out))


def write_trade_log(train: Dict[str, Any], holdout: Dict[str, Any], outdir: Path) -> None:
    rows = []
    for period, payload in (("train", train), ("holdout", holdout)):
        for tr in payload["trade_log"]:
            rows.append({**tr, "period": period})
    cols = [
        "entry_date", "entry_price", "stop", "target",
        "exit_date", "exit_price", "R_multiple",
        "exit_reason", "zone_triggered", "period",
    ]
    df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    df.to_csv(outdir / "trade_log.csv", index=False)


def write_equity_curve(mode: str, train: Dict[str, Any], holdout: Dict[str, Any], outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    if not train["equity_curve"].empty:
        ax.plot(train["equity_curve"].index, train["equity_curve"].values,
                color="steelblue", label="Train (2022 - 2024)", lw=1.2)
    if not holdout["equity_curve"].empty:
        ax.plot(holdout["equity_curve"].index, holdout["equity_curve"].values,
                color="darkorange", label="Holdout (2025+)", lw=1.2)
        cutoff = holdout["equity_curve"].index[0]
        ax.axvline(cutoff, ls="--", color="grey", lw=0.8)
        ax.annotate("Train | Holdout", xy=(cutoff, ax.get_ylim()[1]),
                    xytext=(5, -10), textcoords="offset points", fontsize=9, color="grey")
    ax.set_title(f"QQQ Trend-Pullback ({mode}) -- Equity Curve")
    ax.set_ylabel("Equity ($)")
    ax.set_xlabel("Date")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "equity_curve.png", dpi=120)
    plt.close(fig)


def write_last_6mo_chart(mode: str, df_full: pd.DataFrame, train: Dict[str, Any], holdout: Dict[str, Any], outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df_full.empty:
        return
    end = df_full.index.max()
    start = end - pd.Timedelta(days=180)
    sub = df_full.loc[df_full.index >= start]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sub.index, sub["Close"], color="black", lw=0.8)
    all_trades = list(train["trade_log"]) + list(holdout["trade_log"])
    for tr in all_trades:
        ed = pd.Timestamp(tr.get("entry_date"))
        xd = pd.Timestamp(tr.get("exit_date"))
        if pd.notna(ed) and ed >= start:
            ax.scatter(ed, tr["entry_price"], marker="^", color="green", s=40, zorder=5)
        if pd.notna(xd) and xd >= start:
            colour = "red" if tr.get("exit_reason") == "stop_hit" else "blue"
            ax.scatter(xd, tr["exit_price"], marker="v", color=colour, s=40, zorder=5)
    ax.set_title(f"QQQ 1H ({mode}) -- last 6 months with trade markers")
    ax.set_ylabel("Price ($)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "last_6mo_chart.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Comparison artefacts
# ---------------------------------------------------------------------------

def _stitch_equity(train_eq: pd.Series, holdout_eq: pd.Series, initial: float) -> pd.Series:
    """Return one continuous equity series. Holdout starts at the train final
    equity scaled by holdout returns (so visualisation is one curve)."""
    parts = []
    if not train_eq.empty:
        parts.append(train_eq)
    if not holdout_eq.empty:
        seed = train_eq.iloc[-1] if not train_eq.empty else initial
        ho = holdout_eq.copy()
        first = ho.iloc[0]
        if first and first != 0:
            ho = ho * (seed / first)
        parts.append(ho)
    if not parts:
        return pd.Series(dtype="float64")
    out = pd.concat(parts)
    return out[~out.index.duplicated(keep="last")].sort_index()


def _baseline_curve(symbol: str, span_index: pd.DatetimeIndex, initial: float) -> pd.Series:
    if span_index.empty:
        return pd.Series(dtype="float64")
    try:
        closes = fetch_daily_close(symbol, span_index.min(), span_index.max())
    except Exception as e:
        print(f"WARNING: baseline fetch failed for {symbol}: {e}")
        return pd.Series(dtype="float64")
    if closes.empty:
        return pd.Series(dtype="float64")
    closes = closes.loc[(closes.index >= span_index.min()) & (closes.index <= span_index.max())]
    if closes.empty:
        return pd.Series(dtype="float64")
    return closes / closes.iloc[0] * initial






def write_basket_summary(sweep: Dict[str, Dict[str, Any]], cfg: Dict[str, Any]) -> None:
    """Combine the per-symbol equity curves into one basket portfolio curve per leverage level.

    The basket = symbols listed in cfg['basket'] (or all symbols in sweep), equal-weighted.
    For each margin level we average the normalized equity curves across the basket symbols.
    Output: outputs/basket_summary.md + outputs/basket_curves.png
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    initial = cfg["execution"]["initial_capital"]
    basket = cfg.get("basket") or sorted({p["symbol"] for p in sweep.values()})

    # Group by margin_label
    by_margin: Dict[str, Dict[str, Any]] = {}
    for label, payload in sweep.items():
        ml = payload.get("margin_label", "1x")
        sym = payload.get("symbol")
        if sym not in basket:
            continue
        by_margin.setdefault(ml, {})[sym] = payload

    if not by_margin:
        print("[basket] no sweep results")
        return

    # ---- Compute basket curve per margin level ----
    fig, ax = plt.subplots(figsize=(12, 6))
    cutoff = pd.Timestamp(cfg["period"]["holdout_start"], tz="UTC")
    colours = ["#2ca02c", "#1f77b4", "#d62728", "#ff7f0e"]

    basket_stats: List[Dict[str, Any]] = []

    for j, (ml, sym_results) in enumerate(sorted(by_margin.items(), key=lambda kv: -float(kv[0].rstrip("x")))):
        # Collect each symbol's stitched equity curve
        eqs = []
        for sym in basket:
            if sym not in sym_results:
                continue
            eq = _stitch_equity(
                sym_results[sym]["train"]["equity_curve"],
                sym_results[sym]["holdout"]["equity_curve"],
                initial,
            )
            if eq.empty:
                continue
            # Normalize each curve to start at 1.0
            eqs.append(eq / eq.iloc[0])

        if not eqs:
            continue

        # Align on common index (intersection of all curves)
        common = eqs[0].index
        for eq in eqs[1:]:
            common = common.intersection(eq.index)
        eqs_aligned = [eq.reindex(common) for eq in eqs]

        # Mean across symbols at each timestamp
        basket_normed = sum(eqs_aligned) / len(eqs_aligned)
        basket_curve = basket_normed * initial

        # Plot
        ax.plot(basket_curve.index, basket_curve.values,
                label=f"Basket {ml}", lw=1.6, color=colours[j % len(colours)])

        # Compute basket stats: total return, CAGR, max DD
        total_ret = (basket_curve.iloc[-1] / basket_curve.iloc[0] - 1) * 100
        years = max((basket_curve.index[-1] - basket_curve.index[0]).days / 365.25, 1e-6)
        cagr = ((basket_curve.iloc[-1] / basket_curve.iloc[0]) ** (1 / years) - 1) * 100
        running_max = basket_curve.cummax()
        dd = (basket_curve / running_max - 1) * 100
        max_dd = dd.min()

        # Holdout-only stats
        holdout_curve = basket_curve.loc[basket_curve.index >= cutoff]
        if not holdout_curve.empty:
            h_ret = (holdout_curve.iloc[-1] / holdout_curve.iloc[0] - 1) * 100
            h_running = holdout_curve.cummax()
            h_dd_series = (holdout_curve / h_running - 1) * 100
            h_max_dd = h_dd_series.min()
        else:
            h_ret = 0.0
            h_max_dd = 0.0

        basket_stats.append({
            "margin": ml, "total_return": total_ret, "cagr": cagr, "max_dd": max_dd,
            "holdout_return": h_ret, "holdout_max_dd": h_max_dd,
        })

    # Equal-weight buy-and-hold of basket symbols
    try:
        bh_eqs = []
        s_min = None
        s_max = None
        for sym in basket:
            if not by_margin:
                continue
            # Use the first margin's curve to determine span
            first_m = next(iter(by_margin.values()))
            if sym not in first_m:
                continue
            eq = _stitch_equity(first_m[sym]["train"]["equity_curve"], first_m[sym]["holdout"]["equity_curve"], initial)
            if eq.empty:
                continue
            s_min = eq.index.min() if s_min is None else min(s_min, eq.index.min())
            s_max = eq.index.max() if s_max is None else max(s_max, eq.index.max())
        for sym in basket:
            if s_min is None:
                continue
            try:
                daily = fetch_daily_close(sym, s_min, s_max)
                if daily.empty:
                    continue
                normed = daily / daily.iloc[0]
                bh_eqs.append(normed)
            except Exception as e:
                print(f"[basket] {sym} B&H fetch failed: {e}")
        if bh_eqs:
            common_bh = bh_eqs[0].index
            for eq in bh_eqs[1:]:
                common_bh = common_bh.intersection(eq.index)
            bh_aligned = [eq.reindex(common_bh) for eq in bh_eqs]
            basket_bh = (sum(bh_aligned) / len(bh_aligned)) * initial
            ax.plot(basket_bh.index, basket_bh.values, label="Basket B&H (equal-weight)",
                    color="grey", lw=1.0, ls="--")
    except Exception as e:
        print(f"[basket] B&H computation failed: {e}")

    if cutoff is not None:
        ax.axvline(cutoff, ls=":", color="black", lw=0.7, alpha=0.6)
        ax.annotate("Holdout start", xy=(cutoff, ax.get_ylim()[1]),
                    xytext=(5, -10), textcoords="offset points", fontsize=8)

    ax.set_title(f"Basket portfolio ({', '.join(basket)}) -- equity curves at each leverage")
    ax.set_ylabel("Equity ($)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "basket_curves.png", dpi=120)
    plt.close(fig)

    # ---- Markdown summary ----
    md = ["# Basket portfolio summary", ""]
    md.append(f"Basket: **{', '.join(basket)}** -- equal-weighted, 4h, atr_2.5, RSI 40")
    md.append("")
    md.append("| Leverage | Total Return [%] | CAGR [%] | Max DD [%] | Holdout Return [%] | Holdout Max DD [%] |")
    md.append("|---|---|---|---|---|---|")
    for s_ in basket_stats:
        md.append(f"| {s_['margin']} | {s_['total_return']:.2f} | {s_['cagr']:.2f} | {s_['max_dd']:.2f} | {s_['holdout_return']:.2f} | {s_['holdout_max_dd']:.2f} |")
    md.append("")
    md.append(f"_Generated: {pd.Timestamp.utcnow().isoformat()}_")
    (OUTPUTS / "basket_summary.md").write_text("\n".join(md) + "\n")
    print(f"[basket] wrote basket_summary.md + basket_curves.png ({len(basket_stats)} margin levels)")


def write_sweep_comparison(sweep: Dict[str, Dict[str, Any]], cfg: Dict[str, Any]) -> None:
    """Cross-symbol summary: per-symbol equity-curve PNG + a cross-symbol stats table
    showing every (symbol, tf, variant) combination sorted by holdout PF.

    Benchmarks: each symbol's own buy-and-hold + above-200d-EMA filtered version."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    initial = cfg["execution"]["initial_capital"]

    # Group sweep results by symbol
    by_symbol: Dict[str, List[tuple]] = {}
    for label, payload in sweep.items():
        symbol = payload.get("symbol", "?")
        h_stats = payload["holdout"].get("stats")
        pf = float(h_stats.get("Profit Factor", 0)) if h_stats is not None and "Profit Factor" in h_stats else 0.0
        if pd.isna(pf):
            pf = 0.0
        by_symbol.setdefault(symbol, []).append((pf, label, payload))
    for sym in by_symbol:
        by_symbol[sym].sort(key=lambda x: x[0], reverse=True)

    # ---- Per-symbol equity curve plots ----
    cutoff = pd.Timestamp(cfg["period"]["holdout_start"], tz="UTC")
    for sym, rows in by_symbol.items():
        (OUTPUTS / sym).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 5.5))
        span_min: Optional[pd.Timestamp] = None
        span_max: Optional[pd.Timestamp] = None
        colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#17becf"]
        for j, (pf, label, payload) in enumerate(rows):
            eq = _stitch_equity(payload["train"]["equity_curve"], payload["holdout"]["equity_curve"], initial)
            if eq.empty:
                continue
            short_label = label.replace(f"{sym}/", "")
            ax.plot(eq.index, eq.values, label=f"{short_label} (PF={pf:.2f})", lw=1.4, color=colours[j % len(colours)])
            span_min = eq.index.min() if span_min is None else min(span_min, eq.index.min())
            span_max = eq.index.max() if span_max is None else max(span_max, eq.index.max())

        # Per-symbol benchmark: own buy-and-hold + own-above-200d-EMA
        if span_min is not None and span_max is not None:
            try:
                bh_daily = fetch_daily_close(sym, span_min, span_max)
                if not bh_daily.empty:
                    bh_curve = (bh_daily / bh_daily.iloc[0]) * initial
                    ax.plot(bh_curve.index, bh_curve.values, label=f"{sym} buy & hold",
                            color="grey", lw=1.0, ls="--")
                    ema200 = bh_daily.ewm(span=200, adjust=False).mean()
                    long_mask = (bh_daily > ema200).shift(1).fillna(False)
                    daily_ret = bh_daily.pct_change().fillna(0.0)
                    regime_ret = daily_ret.where(long_mask, 0.0)
                    regime_curve = (1.0 + regime_ret).cumprod() * initial
                    ax.plot(regime_curve.index, regime_curve.values, label=f"{sym} > 200d EMA",
                            color="black", lw=1.0, ls=":")
            except Exception as e:
                print(f"[plot] {sym} benchmark fetch failed: {e}")

        if span_min is not None and span_min <= cutoff <= span_max:
            ax.axvline(cutoff, ls=":", color="black", lw=0.7, alpha=0.6)
        ax.set_title(f"{sym} — strategy vs benchmarks")
        ax.set_ylabel("Equity ($)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUTS / sym / "comparison.png", dpi=120)
        plt.close(fig)

    # ---- Cross-symbol summary plot: best (highest holdout PF) variant per symbol ----
    fig, ax = plt.subplots(figsize=(13, 6))
    span_min = None
    span_max = None
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#17becf"]
    for j, (sym, rows) in enumerate(by_symbol.items()):
        if not rows:
            continue
        pf, label, payload = rows[0]
        eq = _stitch_equity(payload["train"]["equity_curve"], payload["holdout"]["equity_curve"], initial)
        if eq.empty:
            continue
        ax.plot(eq.index, eq.values, label=f"{sym} ({label.replace(f'{sym}/', '')}, PF={pf:.2f})",
                lw=1.5, color=colours[j % len(colours)])
        span_min = eq.index.min() if span_min is None else min(span_min, eq.index.min())
        span_max = eq.index.max() if span_max is None else max(span_max, eq.index.max())

    if span_min is not None and span_min <= cutoff <= span_max:
        ax.axvline(cutoff, ls=":", color="black", lw=0.7, alpha=0.6)
        ax.annotate("Holdout start", xy=(cutoff, ax.get_ylim()[1]),
                    xytext=(5, -10), textcoords="offset points", fontsize=8)
    ax.set_title("Best variant per symbol — equity overlay")
    ax.set_ylabel("Equity ($)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "comparison.png", dpi=120)
    plt.close(fig)

    # Flatten back to rows_by_pf for the markdown table
    rows_by_pf: List[tuple] = []
    for sym, rows in by_symbol.items():
        rows_by_pf.extend(rows)
    rows_by_pf.sort(key=lambda x: x[0], reverse=True)

    # ---- Markdown table ----
    rows = ["# Multi-symbol sweep -- v6 (atr_2.0/2.5 × 1h+4h × RSI40 across 6 symbols)", ""]
    headers = ["Symbol/TF/Variant", "Trail", "RSI", "Trades (T+H)",
               "Holdout Return [%]", "Holdout PF", "Holdout WR [%]", "Holdout MaxDD [%]",
               "Train Return [%]", "Train PF", "Train Trades"]
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("|" + "|".join(["---"] * len(headers)) + "|")
    for pf, label, payload in rows_by_pf:
        h_stats = payload["holdout"].get("stats")
        t_stats = payload["train"].get("stats")
        h_trades = int(h_stats.get("# Trades", 0)) if h_stats is not None else 0
        t_trades = int(t_stats.get("# Trades", 0)) if t_stats is not None else 0
        cells = [
            label,
            payload.get("trail", "?"),
            f'{payload.get("rsi", "?")}',
            f"{t_trades}+{h_trades}",
            _stat(h_stats, "Return [%]"),
            f"{pf:.2f}" if pf else "n/a",
            _stat(h_stats, "Win Rate [%]"),
            _stat(h_stats, "Max. Drawdown [%]"),
            _stat(t_stats, "Return [%]"),
            _stat(t_stats, "Profit Factor"),
            f"{t_trades}",
        ]
        rows.append("| " + " | ".join(cells) + " |")

    # Per-symbol benchmarks footer
    rows.append("")
    rows.append("## Buy-and-hold & regime-filtered benchmarks (per symbol)")
    rows.append("")
    rows.append("| Symbol | B&H Total [%] | B&H CAGR [%] | Above-200d-EMA Total [%] | Above-200d-EMA CAGR [%] |")
    rows.append("|---|---|---|---|---|")
    for sym in sorted(by_symbol.keys()):
        # Determine span for this symbol
        sym_eq = _stitch_equity(by_symbol[sym][0][2]["train"]["equity_curve"],
                                by_symbol[sym][0][2]["holdout"]["equity_curve"], initial) if by_symbol[sym] else pd.Series(dtype=float)
        if sym_eq.empty:
            continue
        s_min, s_max = sym_eq.index.min(), sym_eq.index.max()
        try:
            bh_daily = fetch_daily_close(sym, s_min, s_max)
            if bh_daily.empty:
                rows.append(f"| {sym} | n/a | n/a | n/a | n/a |")
                continue
            bh_total = (bh_daily.iloc[-1] / bh_daily.iloc[0] - 1) * 100
            yrs = max((s_max - s_min).days / 365.25, 1e-6)
            bh_cagr = ((bh_daily.iloc[-1] / bh_daily.iloc[0]) ** (1 / yrs) - 1) * 100
            ema200 = bh_daily.ewm(span=200, adjust=False).mean()
            long_mask = (bh_daily > ema200).shift(1).fillna(False)
            daily_ret = bh_daily.pct_change().fillna(0.0)
            regime_ret = daily_ret.where(long_mask, 0.0)
            regime_total = ((1.0 + regime_ret).prod() - 1) * 100
            regime_cagr = ((1.0 + regime_ret).prod() ** (1 / yrs) - 1) * 100
            rows.append(f"| {sym} | {bh_total:.2f} | {bh_cagr:.2f} | {regime_total:.2f} | {regime_cagr:.2f} |")
        except Exception as e:
            rows.append(f"| {sym} | error: {e} |  |  |  |")
    rows.append("")
    rows.append(f"_Generated: {pd.Timestamp.utcnow().isoformat()}_")
    (OUTPUTS / "comparison.md").write_text("\n".join(rows) + "\n")


def write_comparison_multi(
    tf_results: Dict[str, Dict[str, Dict[str, Any]]],
    cfg: Dict[str, Any],
) -> None:
    """Render combined equity-curve PNG and stats markdown across all
    (timeframe, mode) combinations vs. SPY buy-and-hold."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    initial = cfg["execution"]["initial_capital"]
    baseline_symbol = cfg["data"].get("baseline_symbol", "SPY")

    # Colour scheme: timeframe = hue, mode = strict (dashed) / loose (solid)
    tf_colour = {
        "15min": "#9b59b6",   # purple
        "1hour": "#2980b9",   # blue
        "4hour": "#27ae60",   # green
    }

    fig, ax = plt.subplots(figsize=(13, 6))
    span_min: Optional[pd.Timestamp] = None
    span_max: Optional[pd.Timestamp] = None

    for tf, mode_dict in tf_results.items():
        for mode, payload in mode_dict.items():
            eq = _stitch_equity(payload["train"]["equity_curve"], payload["holdout"]["equity_curve"], initial)
            if eq.empty:
                continue
            ls = "--" if mode == "strict" else "-"
            ax.plot(
                eq.index, eq.values,
                label=f"{tf} / {mode}",
                lw=1.4 if mode == "loose" else 1.0,
                ls=ls,
                color=tf_colour.get(tf, "#7f8c8d"),
            )
            span_min = eq.index.min() if span_min is None else min(span_min, eq.index.min())
            span_max = eq.index.max() if span_max is None else max(span_max, eq.index.max())

    if span_min is not None and span_max is not None:
        idx = pd.DatetimeIndex([span_min, span_max])
        baseline = _baseline_curve(baseline_symbol, idx, initial)
        if not baseline.empty:
            ax.plot(baseline.index, baseline.values, label=f"{baseline_symbol} buy & hold",
                    color="grey", lw=1.0, ls=":")
        cutoff = pd.Timestamp(cfg["period"]["holdout_start"], tz="UTC")
        if span_min <= cutoff <= span_max:
            ax.axvline(cutoff, ls=":", color="black", lw=0.7, alpha=0.6)
            ax.annotate("Holdout start", xy=(cutoff, ax.get_ylim()[1]),
                        xytext=(5, -10), textcoords="offset points", fontsize=8, color="black")

    ax.set_title("Timeframe × Mode comparison vs. SPY buy & hold")
    ax.set_ylabel("Equity ($)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "comparison.png", dpi=120)
    plt.close(fig)

    # Markdown table — rows = (timeframe, mode), columns = train + holdout per metric
    rows = ["# Timeframe × Mode -- Comparison", ""]
    headers = ["TF/Mode"]
    for label, _ in STAT_KEYS:
        headers.extend([f"{label} (Train)", f"{label} (Holdout)"])
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("|" + "|".join(["---"] * len(headers)) + "|")

    for tf, mode_dict in tf_results.items():
        for mode, payload in mode_dict.items():
            cells = [f"{tf}/{mode}"]
            for label, key in STAT_KEYS:
                cells.append(_stat(payload.get("train", {}).get("stats"), key))
                cells.append(_stat(payload.get("holdout", {}).get("stats"), key))
            rows.append("| " + " | ".join(cells) + " |")

    rows.append("")
    if span_min is not None and span_max is not None:
        try:
            bh = fetch_daily_close(baseline_symbol, span_min, span_max)
            if not bh.empty:
                total_return = (bh.iloc[-1] / bh.iloc[0] - 1) * 100
                years = max((span_max - span_min).days / 365.25, 1e-6)
                cagr = ((bh.iloc[-1] / bh.iloc[0]) ** (1 / years) - 1) * 100
                rows.append(
                    f"_{baseline_symbol} buy & hold over full span: total **{total_return:.2f}%**, "
                    f"CAGR **{cagr:.2f}%**_"
                )
        except Exception as e:
            rows.append(f"_(failed to fetch {baseline_symbol} baseline: {e})_")

    rows.append("")
    rows.append(f"_Generated: {pd.Timestamp.utcnow().isoformat()}_")
    (OUTPUTS / "comparison.md").write_text("\n".join(rows) + "\n")


def write_comparison(mode_results: Dict[str, Dict[str, Any]], cfg: Dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    initial = cfg["execution"]["initial_capital"]
    baseline_symbol = cfg["data"].get("baseline_symbol", "SPY")

    fig, ax = plt.subplots(figsize=(12, 5.5))
    span_min: Optional[pd.Timestamp] = None
    span_max: Optional[pd.Timestamp] = None
    colours = {"strict": "steelblue", "loose": "darkorange"}

    for mode, payload in mode_results.items():
        eq = _stitch_equity(payload["train"]["equity_curve"], payload["holdout"]["equity_curve"], initial)
        if eq.empty:
            continue
        ax.plot(eq.index, eq.values, label=f"{mode}", lw=1.4, color=colours.get(mode))
        span_min = eq.index.min() if span_min is None else min(span_min, eq.index.min())
        span_max = eq.index.max() if span_max is None else max(span_max, eq.index.max())

    if span_min is not None and span_max is not None:
        idx = pd.DatetimeIndex([span_min, span_max])
        baseline = _baseline_curve(baseline_symbol, idx, initial)
        if not baseline.empty:
            ax.plot(baseline.index, baseline.values, label=f"{baseline_symbol} buy & hold",
                    color="grey", lw=1.0, ls="--")
        # Holdout boundary
        cutoff = pd.Timestamp(cfg["period"]["holdout_start"], tz="UTC")
        if span_min <= cutoff <= span_max:
            ax.axvline(cutoff, ls=":", color="black", lw=0.7, alpha=0.6)
            ax.annotate("Holdout start", xy=(cutoff, ax.get_ylim()[1]),
                        xytext=(5, -10), textcoords="offset points", fontsize=8, color="black")

    ax.set_title("Strict vs. Loose vs. Buy & Hold")
    ax.set_ylabel("Equity ($)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "comparison.png", dpi=120)
    plt.close(fig)

    # Markdown comparison
    rows = ["# Strict vs. Loose -- Comparison", ""]
    rows.append("| Metric | Strict (Train) | Strict (Holdout) | Loose (Train) | Loose (Holdout) |")
    rows.append("|---|---|---|---|---|")
    for label, key in STAT_KEYS:
        s_t = _stat(mode_results.get("strict", {}).get("train", {}).get("stats"), key)
        s_h = _stat(mode_results.get("strict", {}).get("holdout", {}).get("stats"), key)
        l_t = _stat(mode_results.get("loose", {}).get("train", {}).get("stats"), key)
        l_h = _stat(mode_results.get("loose", {}).get("holdout", {}).get("stats"), key)
        rows.append(f"| {label} | {s_t} | {s_h} | {l_t} | {l_h} |")

    # Buy-and-hold row over the full span
    bh_row = "| Baseline (buy & hold) |"
    if span_min is not None and span_max is not None:
        try:
            bh = fetch_daily_close(baseline_symbol, span_min, span_max)
            if not bh.empty:
                ret_full = (bh.iloc[-1] / bh.iloc[0] - 1) * 100.0
                # Approximate annualised
                years = max((bh.index[-1] - bh.index[0]).days / 365.25, 1e-9)
                cagr = (bh.iloc[-1] / bh.iloc[0]) ** (1 / years) * 100 - 100
                bh_row = f"| {baseline_symbol} buy & hold (full span) | total {ret_full:.2f}% | CAGR {cagr:.2f}% |  |  |"
        except Exception as e:
            bh_row = f"| {baseline_symbol} buy & hold | n/a (fetch failed: {e}) |  |  |  |"
    rows.append("")
    rows.append(bh_row)
    rows.append("")
    rows.append(f"_Generated: {datetime.utcnow().isoformat()}Z_")
    (OUTPUTS / "comparison.md").write_text("\n".join(rows))


# ---------------------------------------------------------------------------

def main(
    modes_override: Optional[List[str]] = None,
    timeframes_override: Optional[List[str]] = None,
    trail_types_override: Optional[List[str]] = None,
    rsi_thresholds_override: Optional[List[float]] = None,
) -> None:
    """Sweep over (timeframe, trail_type, rsi_threshold). Mode is loose-only.

    Output structure: outputs/{tf}/{trail}_{rsi}/...
    """
    cfg = load_config()
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    train_end = pd.Timestamp(cfg["period"]["train_end"], tz="UTC") + pd.Timedelta(days=1)
    holdout_start = pd.Timestamp(cfg["period"]["holdout_start"], tz="UTC")
    timeframes = timeframes_override or cfg.get("timeframes", ["1hour"])
    trail_types = trail_types_override or cfg.get("trail_types", ["ema21", "ema50", "atr"])
    rsi_thresholds = rsi_thresholds_override or cfg.get("rsi_thresholds", [25, 30, 35, 40])
    modes = modes_override or cfg.get("modes", ["loose"])  # default loose-only
    margins = cfg.get("margins", [1.0])  # default no leverage
    use_4h_regime = bool(cfg.get("strategy", {}).get("regime_filter_4h", False))

    symbols = cfg.get("symbols", ["QQQ"])
    sweep_results: Dict[str, Dict[str, Any]] = {}  # key = "{symbol}/{tf}/{trail}_{rsi}"

    for symbol in symbols:
        print(f"\n========== Symbol: {symbol} ==========")

        for tf in timeframes:
            print(f"\n##### {symbol} / {tf} #####")
            try:
                df_base = prepare_base_data(cfg, timeframe=tf, symbol=symbol)
            except Exception as e:
                print(f"[bt] {symbol}/{tf}: data fetch failed: {e}; skipping.")
                continue
            if df_base.empty:
                print(f"No data for {symbol}/{tf}; skipping.")
                continue
            # Optional 4h regime overlay
            regime_4h_mask = None
            if use_4h_regime and tf != "4hour":
                regime_4h_mask = _build_4h_regime_mask(cfg, df_base)
                print(f"[regime] 4h trend filter active: {int(regime_4h_mask.sum())} / {len(regime_4h_mask)} bars qualify")

            for mode in modes:
                df = select_mode(df_base, mode, cfg)
                if regime_4h_mask is not None:
                    # Force-disable bias on bars where 4h trend is not bullish
                    df["daily_bullish_y"] = df["daily_bullish_y"] & regime_4h_mask.reindex(df.index, fill_value=False)
                if not len(df):
                    continue

                # Diagnostic on the base mode (not yet tied to specific trail/rsi)
                bias = df["daily_bullish_y"].fillna(False).astype(bool)
                zone = df["zone_touched"].fillna(False).astype(bool)
                print(f"[bt:{tf}] mode={mode} bias_true={int(bias.sum())} zone_true={int(zone.sum())} bias_and_zone={int((bias&zone).sum())}")

                for margin in margins:
                    margin_label = f"{int(round(1/max(margin,0.001))):d}x" if margin < 1.0 else "1x"
                    for trail in trail_types:
                        # Parse trail names: "atr_1.5" -> base="atr", mult=1.5; "ema21" -> base="ema21", mult=None
                        if trail.startswith("atr_"):
                            base_trail = "atr"
                            try:
                                atr_mult = float(trail.split("_", 1)[1])
                            except (ValueError, IndexError):
                                atr_mult = None
                        else:
                            base_trail = trail
                            atr_mult = None

                        for rsi_t in rsi_thresholds:
                            label = f"{trail}_rsi{rsi_t}"
                            print(f"  -> [{margin_label}] {symbol}/{tf}/{label}", flush=True)
                            df_v = df.copy()
                            if "rsi" in df_v.columns:
                                prev_rsi = df_v["rsi"].shift(1)
                                df_v["rsi_cross_up"] = (prev_rsi < rsi_t) & (df_v["rsi"] >= rsi_t)

                            train_df = df_v.loc[df_v.index < train_end]
                            holdout_df = df_v.loc[df_v.index >= holdout_start]
                            train = run_slice(train_df, cfg, trail_type=base_trail, rsi_threshold=rsi_t, atr_mult=atr_mult, margin=margin)
                            holdout = run_slice(holdout_df, cfg, trail_type=base_trail, rsi_threshold=rsi_t, atr_mult=atr_mult, margin=margin)

                            outdir = OUTPUTS / margin_label / symbol / tf / f"{label}"
                            outdir.mkdir(parents=True, exist_ok=True)
                            run_label = f"{margin_label}/{symbol}/{tf}/{label}"
                            write_stats_md(run_label, train, holdout, outdir)
                            write_trade_log(train, holdout, outdir)
                            write_equity_curve(run_label, train, holdout, outdir)
                            sweep_results[run_label] = {
                                "margin": margin, "margin_label": margin_label,
                                "symbol": symbol, "tf": tf, "mode": mode, "trail": trail, "rsi": rsi_t,
                                "atr_mult": atr_mult,
                                "train": train, "holdout": holdout,
                            }

    if sweep_results:
        write_sweep_comparison(sweep_results, cfg)
        write_basket_summary(sweep_results, cfg)
    print("\nDone. See outputs/.")


def _build_4h_regime_mask(cfg: Dict[str, Any], df_base: pd.DataFrame) -> pd.Series:
    """Fetch 4h QQQ, compute 4h close > 200 EMA, forward-fill onto df_base index."""
    from .data import fetch_qqq_intraday
    start = cfg["data"]["start_date"]
    end_raw = cfg["data"]["end_date"]
    end = parse_end_date(end_raw) if end_raw == "today" else pd.Timestamp(end_raw, tz="UTC")
    df_4h = fetch_qqq_intraday("4hour", start, end, cache_max_age_hours=cfg["data"]["cache_max_age_hours"])
    if df_4h.empty:
        return pd.Series(True, index=df_base.index)  # no filter if data missing
    # Compute 200 EMA on 4h closes
    ema_4h_200 = df_4h["close"].ewm(span=200, adjust=False).mean()
    bullish_4h = (df_4h["close"] > ema_4h_200).rename("regime_4h_bull")
    # Use the most-recent CLOSED 4h bar as the regime signal (no lookahead).
    # Shift by one 4h bar so each timestamp uses the *prior* 4h close vs prior EMA.
    bullish_4h = bullish_4h.shift(1).fillna(False)
    # Reindex onto df_base with backward-fill of last known 4h regime
    aligned = bullish_4h.reindex(df_base.index, method="ffill").fillna(False)
    return aligned


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Multi-symbol trend-pullback backtest sweep")
    parser.add_argument("--symbol", default=None, help="If set, run only this symbol (e.g. QQQ).")
    parser.add_argument("--timeframe", choices=["15min", "1hour", "4hour"], default=None)
    parser.add_argument("--trail", default=None)
    parser.add_argument("--rsi", type=float, default=None)
    args = parser.parse_args()
    main(
        timeframes_override=[args.timeframe] if args.timeframe else None,
        trail_types_override=[args.trail] if args.trail else None,
        rsi_thresholds_override=[args.rsi] if args.rsi is not None else None,
    )


if __name__ == "__main__":
    _cli()
