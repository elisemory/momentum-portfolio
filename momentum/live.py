"""Live portfolio support: incremental data refresh, monthly schedule, current holdings, next trades.

The live portfolio is the Improved strategy (experiments.IMPROVED) run from 1996 to the latest
close. Because the rank-40 buffer makes holdings path-dependent, current holdings are taken from
the end of that full simulation rather than from a fresh top-20 list.

CLI:  python -m momentum.live --refresh     (e.g. from cron on the 1st of each month)
"""
from __future__ import annotations

import io
import json
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from . import backtest as bt
from . import data
from .experiments import IMPROVED

STATE_FILE = data.DATA_DIR / "live_state.json"
REFRESH_HOUR = 6  # local time on the 1st of the month; before the US open, after month-end data is final
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

LIVE_CONFIG = bt.Config(**IMPROVED, label="Momentum (Improved)")


# ---------------------------------------------------------------- schedule
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def last_refresh() -> datetime | None:
    v = load_state().get("last_refresh")
    return datetime.fromisoformat(v) if v else None


def scheduled_time(now: datetime) -> datetime:
    """The most recent scheduled refresh: the 1st of this month at REFRESH_HOUR."""
    first = now.replace(day=1, hour=REFRESH_HOUR, minute=0, second=0, microsecond=0)
    if now < first:  # early on the 1st, before the refresh hour: last month's slot
        prev = (first - timedelta(days=1)).replace(day=1)
        return prev
    return first


def next_scheduled(now: datetime) -> datetime:
    first = now.replace(day=1, hour=REFRESH_HOUR, minute=0, second=0, microsecond=0)
    if now < first:
        return first
    return (first + timedelta(days=32)).replace(day=1)


def should_refresh(now: datetime | None = None) -> bool:
    """True if the 1st-of-month refresh hasn't run yet (also catches up after downtime)."""
    now = now or datetime.now()
    last = last_refresh()
    return last is None or last < scheduled_time(now)


# ---------------------------------------------------------------- data refresh
def refresh_membership(log=print) -> None:
    """Point-in-time history from fja05680/sp500, plus today's list from Wikipedia so additions
    made after that dataset's last update are picked up."""
    try:
        data.fetch_membership()
    except Exception as e:  # keep the existing file
        log(f"membership history download failed ({e}); keeping cached copy")
    path = data.DATA_DIR / "sp500_hist.csv"
    hist = pd.read_csv(path)
    try:
        html = requests.get(WIKI_URL, headers={"User-Agent": "Mozilla/5.0 momentum-backtest"}, timeout=30).text
        cur = pd.read_html(io.StringIO(html))[0]["Symbol"].astype(str).str.strip().tolist()
    except Exception as e:
        log(f"Wikipedia constituents download failed ({e})")
        return
    last = set(hist["tickers"].iloc[-1].split(","))
    if len(cur) > 450 and set(cur) != last:
        today = datetime.now().strftime("%Y-%m-%d")
        hist = hist[hist["date"] != today]
        hist.loc[len(hist)] = [today, ",".join(sorted(cur))]
        hist.to_csv(path, index=False)
        log(f"membership updated from Wikipedia: +{sorted(set(cur) - last)} −{sorted(last - set(cur))}")


def _download(tickers: list[str], start: str, log=print, chunk: int = 80) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf

    opens, closes = [], []
    for i in range(0, len(tickers), chunk):
        batch = tickers[i : i + chunk]
        for attempt in range(3):
            try:
                d = yf.download(batch, start=start, auto_adjust=True, progress=False, threads=True,
                                group_by="column")
                break
            except Exception as e:
                log(f"  retry ({e})")
                time.sleep(5 * (attempt + 1))
        else:
            continue
        o, c = d["Open"], d["Close"]
        if isinstance(o, pd.Series):  # single ticker
            o, c = o.to_frame(batch[0]), c.to_frame(batch[0])
        opens.append(o)
        closes.append(c)
        time.sleep(0.5)
    if not closes:
        return pd.DataFrame(), pd.DataFrame()
    o = pd.concat(opens, axis=1)
    c = pd.concat(closes, axis=1)
    o.index = pd.to_datetime(o.index).tz_localize(None)
    c.index = pd.to_datetime(c.index).tz_localize(None)
    return o.loc[:, ~o.columns.duplicated()], c.loc[:, ~c.columns.duplicated()]


def update_prices(log=print, overlap_days: int = 120) -> None:
    """Download only recent bars and merge them in. Yahoo's adjusted history is rescaled after
    every dividend or split, so each stock's older history is rescaled by the median ratio of
    new/old prices on the overlapping days before merging."""
    o_old, c_old = data.load_prices()
    recent_cut = c_old.index[-1] - pd.Timedelta(days=40)
    active = [t for t in c_old.columns if c_old[t].loc[recent_cut:].notna().any()]
    members = set().union(*data.load_membership()["tickers"].iloc[-3:])
    new_names = sorted(t for t in members if t not in c_old.columns)

    start = (c_old.index[-1] - pd.Timedelta(days=overlap_days)).strftime("%Y-%m-%d")
    log(f"updating {len(active)} active symbols since {start}")
    o_new, c_new = _download(sorted(set(active) | set(data.EXTRA)), start, log)
    if new_names:
        log(f"full history for new members: {', '.join(new_names)}")
        o_full, c_full = _download(new_names, data.PRICE_START, log)
        o_new, c_new = pd.concat([o_new, o_full], axis=1), pd.concat([c_new, c_full], axis=1)

    common = c_new.columns.intersection(c_old.columns)
    both = c_old.index.intersection(c_new.index)
    ratio = (c_new.loc[both, common] / c_old.loc[both, common]).median()
    ratio = ratio.where(np.isfinite(ratio) & (ratio > 0), 1.0)
    rescaled = int(((ratio - 1).abs() > 1e-4).sum())
    o_old = o_old.copy()
    c_old = c_old.copy()
    o_old[common] = o_old[common] * ratio
    c_old[common] = c_old[common] * ratio

    o = o_new.combine_first(o_old)
    c = c_new.combine_first(c_old)
    o, c = o.sort_index(), c.sort_index()
    o.astype("float64").to_parquet(data.DATA_DIR / "open.parquet")
    c.astype("float64").to_parquet(data.DATA_DIR / "close.parquet")
    log(f"prices now through {c.index[-1].date()} ({rescaled} histories rescaled for dividends/splits)")


def refresh(log=print) -> None:
    """Full monthly refresh: membership, prices, benchmarks, sectors for new names."""
    refresh_membership(log)
    update_prices(log)
    try:
        data.fetch_nasdaq()
        data.fetch_spy_dividends()
    except Exception as e:
        log(f"benchmark refresh failed ({e})")
    sectors = data.load_sectors()
    _, c = data.load_prices()
    missing = [t for t in c.columns if t not in sectors and t not in data.EXTRA]
    if missing:
        log(f"sectors for {len(missing)} new symbols")
        path = data.DATA_DIR / "sectors.csv"
        old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["ticker", "sector"])
        data.fetch_sectors(missing, log=lambda *_: None)
        new = pd.read_csv(path)
        pd.concat([old, new]).drop_duplicates("ticker", keep="last").to_csv(path, index=False)
    for f in (bt.load_market, bt._membership, bt._quality, bt._regime_inputs, bt._sector_codes):
        f.cache_clear()
    STATE_FILE.write_text(json.dumps({"last_refresh": datetime.now().isoformat(timespec="seconds")}))
    log("refresh complete")


# ---------------------------------------------------------------- portfolio
def portfolio(cfg: bt.Config = LIVE_CONFIG) -> dict:
    """Current holdings from the full simulation, plus the trades the next rebalance would make."""
    r = bt.run(cfg)
    m = bt.load_market()
    s = len(m.dates) - 1
    as_of = m.dates[-1]
    next_month = as_of.to_period("M") + 1
    last_regime = r.rebalances["regime"].iloc[-1] == "bull"
    bull, spx, sma = bt.regime_state(cfg, m, s, last_regime)
    x = bt._regime_inputs(cfg.sma_days, cfg.slope_days)
    sma_rising = bool(x["sma"][s] >= x["sma_prev"][s])

    ranks = bt.rank_universe(cfg, m, s, next_month)
    mom6 = bt.momentum_scores(m, next_month, 6, cfg.skip_months)
    ranks["momentum_6_1"] = mom6.reindex(ranks["ticker"]).to_numpy()
    rank_of = dict(zip(ranks["ticker"], ranks["rank"]))
    sectors = data.load_sectors()

    pos = r.positions.copy()
    pos["rank"] = pos["ticker"].map(rank_of)
    pos = pos.merge(ranks[["ticker", "momentum", "momentum_6_1", "score"]], on="ticker", how="left")
    pos["sector"] = pos["ticker"].map(sectors).fillna("Unknown")

    held = set(pos["ticker"])
    keep_rank = max(cfg.buffer_rank, cfg.top_n)
    if not bull:
        sells = [(t, "bear regime: move to T-bills") for t in pos["ticker"]]
        buys = []
    else:
        sells = [(t, "left the S&P 500" if rank_of.get(t) is None else f"rank {rank_of[t]} > {keep_rank}")
                 for t in pos["ticker"] if rank_of.get(t) is None or rank_of[t] > keep_rank]
        slots = cfg.top_n - (len(held) - len(sells))
        buys = []
        for t, rk in zip(ranks["ticker"], ranks["rank"]):
            if slots <= 0:
                break
            if t not in held:
                buys.append((t, f"rank {rk}"))
                slots -= 1
    trims = [(t, f"{w:.1%} > {cfg.max_weight:.0%} cap") for t, w in zip(pos["ticker"], pos["weight"])
             if bull and cfg.max_weight > 0 and w > cfg.max_weight and t not in dict(sells)]

    # first trading day of next month (US federal holidays approximate the NYSE calendar)
    bday = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    first = next_month.to_timestamp() - pd.Timedelta(days=1) + bday
    # signals are final once the month's last close is in the data
    final = (as_of + bday).to_period("M") != as_of.to_period("M") or datetime.now() >= next_month.to_timestamp()
    return dict(
        result=r, as_of=as_of, next_rebalance=first, final=final, bull=bull, spx=spx, sma=sma,
        sma_rising=sma_rising, positions=pos, ranks=ranks, sells=sells, buys=buys, trims=trims,
        cash_weight=max(0.0, 1 - pos["weight"].sum()) if len(pos) else 1.0,
        last_rebalance=r.rebalances.index[-1], last_trades=r.trades[r.trades["date"] == r.rebalances.index[-1]],
    )


# ---------------------------------------------------------------- intraday quotes
def quotes(tickers: list[str]) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Latest price and change vs the previous close (unadjusted, as a broker shows it).
    During market hours Yahoo's daily bar for today carries the latest trade."""
    import yfinance as yf

    d = yf.download(tickers, period="5d", interval="1d", auto_adjust=False, progress=False, threads=True)["Close"]
    if isinstance(d, pd.Series):
        d = d.to_frame(tickers[0])
    d = d.dropna(how="all").ffill()
    last, prev = d.iloc[-1], d.iloc[-2]
    return pd.DataFrame({"price": last, "prev_close": prev, "change": last / prev - 1}), d.index[-1]


def market_open(now: datetime | None = None) -> bool:
    """Regular US session, 09:30-16:00 New York time on business days (federal holidays approximate NYSE)."""
    from datetime import time as dtime
    from zoneinfo import ZoneInfo

    et = (now or datetime.now(ZoneInfo("America/New_York"))).astimezone(ZoneInfo("America/New_York"))
    if et.weekday() >= 5:
        return False
    hol = USFederalHolidayCalendar().holidays(start=f"{et.year}-01-01", end=f"{et.year}-12-31")
    if pd.Timestamp(et.date()) in hol:
        return False
    return dtime(9, 30) <= et.time() < dtime(16, 0)


# ---------------------------------------------------------------- data release (deployments)
# Hosted copies (e.g. Streamlit Cloud) start with an empty disk. A GitHub Action refreshes the
# data monthly and publishes it as a release asset; the app downloads it on a cold start.
import os  # noqa: E402
import tarfile  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

DATA_URL = os.environ.get(
    "MOMENTUM_DATA_URL",
    "https://github.com/elisemory/momentum-portfolio/releases/download/data-latest/momentum-data.tar.gz",
)
DATA_FILES = ["open.parquet", "close.parquet", "sp500_hist.csv", "sectors.csv", "nasdaq100_tr.parquet",
              "spy_unadjusted.parquet", "fundamentals.parquet", "live_state.json"]


def pack_data(out: str) -> None:
    with tarfile.open(out, "w:gz") as tar:
        for name in DATA_FILES:
            f = data.DATA_DIR / name
            if f.exists():
                tar.add(f, arcname=name)


def download_data(log=print, url: str = DATA_URL) -> bool:
    """Fetch the published data bundle into data/. Returns False if it isn't available."""
    try:
        r = requests.get(url, timeout=120, stream=True)
        if r.status_code != 200:
            log(f"data bundle not available ({r.status_code})")
            return False
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            for chunk in r.iter_content(1 << 20):
                tmp.write(chunk)
        data.DATA_DIR.mkdir(exist_ok=True)
        with tarfile.open(tmp.name, "r:gz") as tar:
            for member in tar.getmembers():  # only plain files with known names: no path tricks
                if member.isfile() and member.name in DATA_FILES:
                    src = tar.extractfile(member)
                    (data.DATA_DIR / member.name).write_bytes(src.read())
        Path(tmp.name).unlink(missing_ok=True)
        for f in (bt.load_market, bt._membership, bt._quality, bt._regime_inputs, bt._sector_codes):
            f.cache_clear()
        log("downloaded published data bundle")
        return True
    except Exception as e:
        log(f"data bundle download failed ({e})")
        return False


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="download new data now")
    ap.add_argument("--if-due", action="store_true", help="only refresh if the monthly refresh is due")
    ap.add_argument("--download", action="store_true", help="fetch the published data bundle first")
    ap.add_argument("--pack", metavar="PATH", help="write the data bundle to PATH and exit")
    a = ap.parse_args()
    if a.pack:
        pack_data(a.pack)
        raise SystemExit(0)
    if a.download and not download_data():
        raise SystemExit("could not download the data bundle")
    if a.refresh or (a.if_due and should_refresh()):
        refresh()
    p = portfolio()
    print(f"as of {p['as_of'].date()}  regime: {'BULL' if p['bull'] else 'BEAR'}")
    print(p["positions"][["ticker", "weight", "rank", "entry"]].to_string(index=False))
    print("next rebalance sells:", p["sells"], "\nbuys:", p["buys"])
