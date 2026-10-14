"""Robustness harness: score each variant on the full period and on two independent halves.

An idea is only worth keeping if it helps in BOTH halves; anything else is likely curve-fitting.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from . import backtest as bt
from . import metrics as mt

PERIODS = {
    "Full": ("1996-02-01", None),
    "1996-2010": ("1996-02-01", "2010-12-31"),
    "2011-now": ("2011-01-01", None),
}

# Recommended combination: every piece improved both halves on its own and is stable across
# parameter choices (see README "Improvements").
IMPROVED = dict(score="blend", regime_rule="sma_slope", max_weight=0.10)

IDEAS: dict[str, dict] = {
    "Spec as written": {},
    "Improved (blend + falling-SMA + 10% cap)": IMPROVED,
    # regime filter
    "Regime: bear only if SMA is falling": dict(regime_rule="sma_slope"),
    "Regime: re-enter above 50-day SMA": dict(regime_rule="fast_reentry"),
    "Regime: SMA + absolute momentum": dict(regime_rule="sma_absmom"),
    "Regime: 10-month SMA (Faber)": dict(regime_rule="faber"),
    "Regime: daily SMA check": dict(regime_daily=True),
    "Regime: daily SMA ±2% band": dict(regime_daily=True, sma_band=0.02),
    "Regime: keep 50% in bear": dict(bear_exposure=0.5),
    "Regime: off": dict(regime_filter=False),
    # stock selection
    "Score: 12-1 + 6-1 blend": dict(score="blend"),
    "Score: 52-week-high proximity": dict(score="high52"),
    "Score: smooth momentum (FIP)": dict(score="smooth"),
    "Score: return / volatility": dict(score="risk_adjusted"),
    "Sector cap: 5 per sector": dict(sector_cap=5),
    # sizing
    "Weights: inverse volatility": dict(weighting="inverse_vol", rebalance_mode="full"),
    "Weights: equal-weight reset": dict(rebalance_mode="full"),
    "Basket: 40 names / buffer 80": dict(top_n=40, buffer_rank=80),
    # portfolio
    "Core: 50% SPY + 50% momentum": dict(core_spy=0.5),
}


# One-knob changes relative to the Improved strategy (used by the research app's Compare tab).
IMPROVED_VARIANTS: dict[str, dict] = {
    "Improved (baseline)": {},
    "Regime: original rule (any close below SMA)": dict(regime_rule="sma"),
    "Regime: off (always invested)": dict(regime_filter=False),
    "Regime: re-enter above 50-day SMA": dict(regime_rule="fast_reentry"),
    "Regime: keep 50% in bear": dict(bear_exposure=0.5),
    "Regime: SMA 150 days": dict(sma_days=150),
    "Regime: SMA 250 days": dict(sma_days=250),
    "Regime: slope over 63 days": dict(slope_days=63),
    "Score: 12-1 only": dict(score="raw"),
    "Score: return / volatility": dict(score="risk_adjusted"),
    "Score: residual momentum (market)": dict(score="residual"),
    "Score: residual momentum (market + sector)": dict(score="residual_sector"),
    "No 10% position cap": dict(max_weight=0.0),
    "No rank-40 buffer": dict(buffer_rank=20),
    "Basket: 15 names / buffer 30": dict(top_n=15, buffer_rank=30),
    "Basket: 30 names / buffer 60": dict(top_n=30, buffer_rank=60),
    "Basket: 40 names / buffer 80": dict(top_n=40, buffer_rank=80),
    "Weights: equal-weight reset": dict(rebalance_mode="full"),
    "Weights: inverse volatility": dict(weighting="inverse_vol", rebalance_mode="full"),
    "Vol target 20%": dict(vol_target=0.20),
    "Sector cap: 5 per sector": dict(sector_cap=5),
    "Core: 30% SPY": dict(core_spy=0.3),
    "Today's members (biased)": dict(universe="current"),
}


def evaluate(variants: dict[str, dict], base: bt.Config | None = None, periods=PERIODS) -> pd.DataFrame:
    base = base or bt.Config()
    rows = []
    for name, kw in variants.items():
        row = {"Variant": name}
        for p, (s, e) in periods.items():
            r = bt.run(replace(base, **kw, start=s, end=e, label=name))
            S = mt.summary(r.nav, r.cash_rate)
            B = mt.summary(r.bench, r.cash_rate)
            row[f"{p} CAGR"] = S["CAGR"]
            row[f"{p} Sharpe"] = S["Sharpe"]
            row[f"{p} MaxDD"] = S["Max drawdown"]
            row[f"{p} vs SPY"] = S["CAGR"] - B["CAGR"]
            if p == "Full":
                row["Turnover"] = mt.turnover_per_year(r.rebalances)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Variant")


def benchmark(periods=PERIODS) -> dict:
    out = {}
    for p, (s, e) in periods.items():
        r = bt.run(bt.Config(start=s, end=e))
        B = mt.summary(r.bench, r.cash_rate)
        out[p] = (B["CAGR"], B["Sharpe"], B["Max drawdown"])
    return out


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 250)
    df = evaluate(IDEAS)
    cols = [c for c in df.columns if c.endswith(("CAGR", "Sharpe", "MaxDD"))] + ["Turnover"]
    print(df[cols].round(3).to_string())
    for p, (c, sh, dd) in benchmark().items():
        print(f"SPY {p}: CAGR {c:.3f} Sharpe {sh:.2f} MaxDD {dd:.3f}")
