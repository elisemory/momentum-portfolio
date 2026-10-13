"""After-tax comparison: momentum strategy in a taxable account vs. buy-and-hold SPY.

Run:  python -m momentum.taxes
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from . import backtest as bt
from . import data
from .experiments import IMPROVED


def spy_buy_and_hold(start: str, end: str | None = None, capital: float = 100_000.0,
                     div_tax: float = 0.15, ltcg: float = 0.15) -> dict:
    """Buy SPY once and never sell. Dividends are taxed each year at `div_tax` (qualified rate)
    and the rest is reinvested. Returns the value held and the value if sold at the end."""
    h = pd.read_parquet(data.DATA_DIR / "spy_unadjusted.parquet")
    h = h.loc[pd.Timestamp(start): pd.Timestamp(end) if end else None]
    px, div = h["Close"].to_numpy(), h["Dividends"].to_numpy()
    shares = capital / px[0]
    basis = capital
    tax_paid = 0.0
    for i in range(1, len(px)):
        if div[i] > 0:
            cash = shares * div[i]
            tax = cash * div_tax
            tax_paid += tax
            shares += (cash - tax) / px[i]
            basis += cash - tax
    value = shares * px[-1]
    years = (h.index[-1] - h.index[0]).days / 365.25
    sold = value - max(value - basis, 0) * ltcg
    return dict(value=value, liquidation_value=sold, tax_paid=tax_paid, years=years,
                cagr=(value / capital) ** (1 / years) - 1, cagr_sold=(sold / capital) ** (1 / years) - 1)


def strategy(cfg: bt.Config) -> dict:
    r = bt.run(cfg)
    years = (r.nav.index[-1] - r.nav.index[0]).days / 365.25
    cap = cfg.initial_capital
    if r.tax is None:
        v = r.nav.iloc[-1]
        return dict(value=v, liquidation_value=v, tax_paid=0.0, years=years,
                    cagr=(v / cap) ** (1 / years) - 1, cagr_sold=(v / cap) ** (1 / years) - 1, result=r)
    t = r.tax
    return dict(value=t["after_tax_value"], liquidation_value=t["liquidation_value"], tax_paid=t["paid"],
                years=years, cagr=(t["after_tax_value"] / cap) ** (1 / years) - 1,
                cagr_sold=(t["liquidation_value"] / cap) ** (1 / years) - 1, result=r)


def compare(start: str = "1996-02-01", end: str | None = None) -> pd.DataFrame:
    imp = bt.Config(**IMPROVED, start=start, end=end)
    spec = bt.Config(start=start, end=end)
    rows = {
        "SPY buy & hold, no taxes": spy_buy_and_hold(start, end, div_tax=0, ltcg=0),
        "SPY buy & hold, dividends taxed 15%": spy_buy_and_hold(start, end),
        "Improved, tax-free account (IRA/401k)": strategy(imp),
        "Improved, all gains taxed 30%": strategy(replace(imp, tax_short=0.30)),
        "Improved, 30% short / 15% long-term": strategy(replace(imp, tax_short=0.30, tax_long=0.15)),
        "Spec, tax-free account": strategy(spec),
        "Spec, all gains taxed 30%": strategy(replace(spec, tax_short=0.30)),
    }
    out = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != "result"} for k, v in rows.items()}).T
    return out[["cagr", "value", "cagr_sold", "liquidation_value", "tax_paid"]]


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 200)
    for s, e in [("1996-02-01", None), ("2011-01-01", None)]:
        print(f"\n=== {s} -> {e or 'today'} ($100,000 start) ===")
        df = compare(s, e)
        fmt = df.copy()
        for c in ["cagr", "cagr_sold"]:
            fmt[c] = (df[c] * 100).map("{:.1f}%".format)
        for c in ["value", "liquidation_value", "tax_paid"]:
            fmt[c] = df[c].map("${:,.0f}".format)
        print(fmt.to_string())
