"""Trailing total returns using actual observations, never synthetic forward-filled closes."""
from __future__ import annotations

import pandas as pd


PERIODS = {"1D": None, "5D": None, "1M": 1, "3M": 3, "6M": 6, "YTD": None,
           "1Y": 12, "3Y": 36, "5Y": 60, "10Y": 120, "Max": None}


def period_start(idx: pd.DatetimeIndex, period: str) -> pd.Timestamp:
    """Previous close for 1D, five sessions back for 5D, calendar offsets otherwise."""
    end = idx[-1]
    if period in ("1D", "5D"):
        return idx[max(-len(idx), -1 - int(period[0]))]
    if period == "Max":
        return idx[0]
    target = (pd.Timestamp(end.year, 1, 1) - pd.Timedelta(days=1) if period == "YTD"
              else end - pd.DateOffset(months=PERIODS[period]))
    before = idx[idx <= target]
    return before[-1] if len(before) else idx[0]


def trailing_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """All series share the same endpoints; missing benchmark prices remain unavailable."""
    return pd.DataFrame({period: prices.iloc[-1] / prices.loc[period_start(prices.index, period)] - 1
                         for period in PERIODS})
