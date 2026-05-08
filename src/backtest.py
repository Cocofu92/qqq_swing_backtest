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

from .data import fetch_daily_close, fetch_qqq_1h, parse_end_date
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


def prepare_base_data(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Fetch QQQ 1H and attach all daily indicators (mode-agnostic)."""
    start = cfg["data"]["start_date"]
    end_raw = cfg["data"]["end_date"]
    end = parse_end_date(end_raw) if end_raw == "today" else pd.Timestamp(end_raw, tz="UTC")
    df_1h = fetch_qqq_1h(start, end, cache_max_age_hours=cfg["data"]["cache_max_age_hours"])

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


def run_slice(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    if df.empty:
        return {"stats": None, "equity_curve": pd.Series(dtype=float), "trade_log": []}

    bt = Backtest(
        df,
        TrendPullback,
        cash=cfg["execution"]["initial_capital"],
        commission=cfg["costs"]["commission"],
        exclusive_orders=cfg["execution"]["one_position_at_a_time"],
        trade_on_close=False,
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

def main(modes_override: Optional[List[str]] = None) -> None:
    cfg = load_config()
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    df_base = prepare_base_data(cfg)
    if df_base.empty:
        print("No data; nothing to backtest.")
        return

    train_end = pd.Timestamp(cfg["period"]["train_end"], tz="UTC") + pd.Timedelta(days=1)
    holdout_start = pd.Timestamp(cfg["period"]["holdout_start"], tz="UTC")
    modes = modes_override or cfg.get("modes", ["strict", "loose"])

    mode_results: Dict[str, Dict[str, Any]] = {}
    for mode in modes:
        print(f"\n=== Running mode: {mode} ===")
        df = select_mode(df_base, mode, cfg)
        train_df = df.loc[df.index < train_end]
        holdout_df = df.loc[df.index >= holdout_start]
        print(f"[bt] mode={mode}  train: {len(train_df)} bars  ({train_df.index[0] if len(train_df) else 'EMPTY'} -> {train_df.index[-1] if len(train_df) else ''})")
        print(f"[bt] mode={mode}  holdout: {len(holdout_df)} bars  ({holdout_df.index[0] if len(holdout_df) else 'EMPTY'} -> {holdout_df.index[-1] if len(holdout_df) else ''})")
        # Diagnostic: how many bars passed daily bias filter, how many had zone touch
        if len(df):
            bias = df.get('daily_bullish_y', pd.Series([], dtype=bool)).fillna(False).astype(bool)
            zone = df.get('zone_touched', pd.Series([], dtype=bool)).fillna(False).astype(bool)
            engulf = df.get('engulfing', pd.Series([], dtype=bool)).fillna(False).astype(bool)
            rsi_cross = df.get('rsi_cross_up', pd.Series([], dtype=bool)).fillna(False).astype(bool)
            trigger = engulf | rsi_cross
            all3 = bias & zone & trigger
            print(f"[bt] mode={mode} total={len(df)} bias_true={int(bias.sum())} zone_true={int(zone.sum())} bias_and_zone={int((bias&zone).sum())} trigger_only={int(trigger.sum())} ALL3_signals={int(all3.sum())}")

        train = run_slice(train_df, cfg)
        holdout = run_slice(holdout_df, cfg)

        outdir = OUTPUTS / mode
        outdir.mkdir(parents=True, exist_ok=True)
        write_stats_md(mode, train, holdout, outdir)
        write_trade_log(train, holdout, outdir)
        write_equity_curve(mode, train, holdout, outdir)
        write_last_6mo_chart(mode, df, train, holdout, outdir)
        mode_results[mode] = {"train": train, "holdout": holdout}

    if len(mode_results) >= 1:
        write_comparison(mode_results, cfg)
    print("\nDone. See outputs/.")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="QQQ trend-pullback backtest")
    parser.add_argument("--mode", choices=["strict", "loose"], default=None,
                        help="If set, run only this mode. Otherwise run all modes from config.")
    parser.add_argument("--timeframe", choices=["15min", "1hour", "4hour"], default=None,
                        help="If set, run only this timeframe. Otherwise run all timeframes from config.")
    args = parser.parse_args()
    main(
        modes_override=[args.mode] if args.mode else None,
        timeframes_override=[args.timeframe] if args.timeframe else None,
    )


if __name__ == "__main__":
    _cli()
