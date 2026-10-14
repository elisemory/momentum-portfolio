"""Supplement Yahoo with prices for former S&P 500 members it no longer carries.

Yahoo drops a stock's history once the symbol stops trading, so most companies that were bought
out or failed are missing (about half of the 1996-2000 members). The frozen Quandl WIKI archive
(Nasdaq Data Link, about 3,200 US stocks through March 2018, adjusted for splits and dividends)
fills part of that gap. It has few names that left before 2010 and none of the large failures
(Lehman, Enron, WorldCom), so results before 2010 still lean optimistic.

A member is filled only if Yahoo prices less than half of its membership and WIKI prices at least
90% of the trading days of that membership (up to the archive's end). US symbols are unique at any
one time, so full coverage of the membership window also guards against symbols that a different
company used before or after. Data stops at the end of February 2018: Nasdaq flagged the archive as
unreliable from April 2018, and a whole-month cut-off avoids a partial final month.

Build:  python -m momentum.supplement --wiki data/archive/wiki_prices.zip   (WIKI bulk export, zip or CSV)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data

END = "2018-02-28"
MIN_WIKI_COVERAGE = 0.9


def build(wiki_csv: str, log=print) -> pd.DataFrame:
    w = pd.read_csv(wiki_csv, usecols=["ticker", "date", "adj_open", "adj_close"], parse_dates=["date"])
    w["ticker"] = w["ticker"].str.replace("_", "-")  # WIKI writes share classes as BRK_B
    w = w[w["date"] <= END]
    W_o = w.pivot(index="date", columns="ticker", values="adj_open").sort_index()
    W_c = w.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    _, c = data.load_prices(supplement=False)
    mem = data.load_membership()

    spans: dict[str, list[pd.Timestamp]] = {}
    for d0, d1, ts in zip(mem.index, list(mem.index[1:]) + [mem.index[-1]], mem["tickers"]):
        for t in ts:
            spans.setdefault(t, [d0, d1])[1] = d1
    rows = []
    for t, (a, b) in spans.items():
        if t in c.columns and c[t].loc[a:b].notna().mean() >= 0.5:
            continue  # Yahoo has it
        days = c.index[(c.index >= a) & (c.index <= min(b, pd.Timestamp(END)))]
        cov = W_c[t].reindex(days).notna().mean() if t in W_c.columns and len(days) > 20 else 0.0
        rows.append(dict(ticker=t, member_from=a.date(), member_to=b.date(), wiki_coverage=round(cov, 3),
                         used=cov >= MIN_WIKI_COVERAGE))
    manifest = pd.DataFrame(rows).sort_values("ticker")
    use = manifest.loc[manifest["used"], "ticker"].tolist()
    bad = data.bad_tickers(W_c[use]) - data.BLOCKLIST
    if bad:
        log(f"dropped for extreme price jumps: {sorted(bad)}")
        manifest.loc[manifest["ticker"].isin(bad), "used"] = False
        use = [t for t in use if t not in bad]
    W_o[use].astype("float64").to_parquet(data.DATA_DIR / "supplement_open.parquet")
    W_c[use].astype("float64").to_parquet(data.DATA_DIR / "supplement_close.parquet")
    manifest.assign(source=np.where(manifest["used"], "Quandl WIKI", "")).to_csv(
        data.DATA_DIR / "supplement_manifest.csv", index=False)
    log(f"supplement: {len(use)} former members from WIKI, {len(manifest) - len(use)} still unpriced")
    return manifest


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", required=True, help="path to the WIKI_PRICES bulk export CSV")
    build(ap.parse_args().wiki)
