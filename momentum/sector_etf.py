"""Sector-ETF rotation: the Improved rules applied to the SPDR sector ETFs instead of stocks.

Tests how much of the stock strategy's edge is sector trend-following. ETFs don't get delisted,
so this sidesteps the survivorship gap in the stock price data. The engine is reused unchanged:
`universe()` temporarily swaps the stock market for an ETF one.

CLI:  python -m momentum.sector_etf --fetch
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pandas as pd

from . import backtest as bt
from . import data

ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",  # Dec 1998
        "XLRE", "XLC"]                                                   # Oct 2015, Jun 2018
OPEN_FILE = data.DATA_DIR / "sector_etf_open.parquet"
CLOSE_FILE = data.DATA_DIR / "sector_etf_close.parquet"
DEFAULTS = dict(score="blend", regime_rule="sma_slope", top_n=3, buffer_rank=3, rebalance_mode="full",
                max_weight=0.0, start="2000-02-01", label="Sector ETFs (top 3)")


def fetch(log=print) -> None:
    from .live import _download

    o, c = _download(ETFS, "1998-12-01", log)
    missing = set(ETFS) - set(c.columns)
    if missing:
        raise RuntimeError(f"no prices for {sorted(missing)}")
    o[ETFS].astype("float64").to_parquet(OPEN_FILE)
    c[ETFS].astype("float64").to_parquet(CLOSE_FILE)
    log(f"saved {len(ETFS)} sector ETFs through {c.index[-1].date()}")


def market(etfs: list[str] = ETFS) -> bt.Market:
    """Same construction as backtest.load_market, on the stock market's dates and benchmarks."""
    m = bt.load_market()
    dates = m.dates
    O = pd.read_parquet(OPEN_FILE).reindex(dates)[etfs]
    C = pd.read_parquet(CLOSE_FILE).reindex(dates)[etfs]
    Cff = C.ffill(limit=5)
    months = dates.to_period("M")
    me = Cff.groupby(months).last()
    last_day = pd.Series(dates, index=dates).groupby(months).last()
    me = me.where(C.reindex(last_day.values).ffill(limit=5).notna().values)
    return bt.Market(dates=dates, tickers=list(etfs), open=O.to_numpy(), close=C.to_numpy(), close_ff=Cff.to_numpy(),
                     month_end=me, index_close=m.index_close, bench_close=m.bench_close,
                     tbill_daily=m.tbill_daily, has_price=C.notna().to_numpy())


@contextmanager
def universe(etfs: list[str] = ETFS):
    """Run the backtest engine on the ETFs instead of S&P 500 stocks (not thread-safe). The cached
    regime inputs stay valid: the ETF market shares the stock market's dates, index and T-bills."""
    m = market(etfs)
    saved = bt.load_market, bt._membership
    bt.load_market, bt._membership = (lambda: m), (lambda universe: m.has_price)
    try:
        yield m
    finally:
        bt.load_market, bt._membership = saved


def run(etfs: list[str] = ETFS, **kw) -> bt.Result:
    with universe(etfs):
        return bt.run(bt.Config(**{**DEFAULTS, **kw}))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download the sector ETF prices")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    r = run()
    print(r.nav.iloc[-1], np.round(r.holdings.tail(3), 3))
