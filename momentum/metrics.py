from __future__ import annotations

import numpy as np
import pandas as pd

TD = 252


def drawdown(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1.0


def summary(nav: pd.Series, rf_daily: pd.Series, bench: pd.Series | None = None) -> dict:
    nav = nav.dropna()
    r = nav.pct_change().dropna()
    rf = rf_daily.reindex(r.index).fillna(0.0)
    ex = r - rf
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    vol = r.std() * np.sqrt(TD)
    downside = ex[ex < 0].std() * np.sqrt(TD)
    dd = drawdown(nav)
    # longest time under water, in days
    under = dd < 0
    grp = (~under).cumsum()
    dd_len = under.groupby(grp).sum().max() if under.any() else 0
    yearly = nav.resample("YE").last().pct_change()
    yearly.iloc[0] = nav.resample("YE").last().iloc[0] / nav.iloc[0] - 1
    monthly = nav.resample("ME").last().pct_change().dropna()
    out = {
        "CAGR": cagr,
        "Total return": nav.iloc[-1] / nav.iloc[0] - 1,
        "Volatility": vol,
        "Sharpe": ex.mean() / r.std() * np.sqrt(TD) if r.std() > 0 else np.nan,
        "Sortino": ex.mean() * TD / downside if downside > 0 else np.nan,
        "Max drawdown": dd.min(),
        "Calmar": cagr / -dd.min() if dd.min() < 0 else np.nan,
        "Longest drawdown (days)": int(dd_len),
        "Best year": yearly.max(),
        "Worst year": yearly.min(),
        "% months positive": (monthly > 0).mean(),
        "Final value": nav.iloc[-1],
    }
    if bench is not None:
        br = bench.pct_change().reindex(r.index).dropna()
        rr = r.reindex(br.index)
        if br.var() > 0:
            beta = np.cov(rr, br)[0, 1] / br.var()
            out["Beta vs S&P"] = beta
            out["Correlation vs S&P"] = rr.corr(br)
    return out


def yearly_returns(series: dict[str, pd.Series]) -> pd.DataFrame:
    cols = {}
    for k, nav in series.items():
        y = nav.resample("YE").last()
        ret = y.pct_change()
        ret.iloc[0] = y.iloc[0] / nav.iloc[0] - 1
        cols[k] = ret
    df = pd.DataFrame(cols)
    df.index = df.index.year
    return df


def monthly_table(nav: pd.Series) -> pd.DataFrame:
    m = nav.resample("ME").last()
    ret = m.pct_change()
    ret.iloc[0] = m.iloc[0] / nav.iloc[0] - 1
    t = pd.DataFrame({"y": ret.index.year, "m": ret.index.month, "r": ret.values})
    tab = t.pivot(index="y", columns="m", values="r")
    tab.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][: len(tab.columns)]
    tab["Year"] = (1 + tab.fillna(0)).prod(axis=1) - 1
    return tab


def turnover_per_year(rebalances: pd.DataFrame) -> float:
    if rebalances.empty:
        return np.nan
    yrs = (rebalances.index[-1] - rebalances.index[0]).days / 365.25 or 1
    return rebalances["turnover"].sum() / yrs
