"""Data loading: point-in-time S&P 500 membership, daily prices, benchmarks, fundamentals.

Everything is cached under data/ so the app and backtests run offline after the first fetch.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
PRICE_START = "1994-01-01"
INDEX = "^GSPC"      # regime signal (price index, as specified)
BENCH = "SPY"        # benchmark (adjusted = total return)
TBILL = "^IRX"       # 13-week T-bill yield, % annualised
EXTRA = [INDEX, BENCH, TBILL]


# Symbols whose Yahoo history belongs to a different (later) company than the S&P 500 member
# that used the ticker. Found by scanning for implausible volatility / price jumps during the
# membership window and checked by hand (e.g. MCIC quoted in the millions, PARA ~$80k).
BLOCKLIST = {"MCIC", "CPWR", "EP", "SNT", "PARA", "COMS", "DIGI", "USBC", "MI", "POM", "STI", "UPC", "EQ", "DEC"}


def bad_tickers(close: pd.DataFrame, max_extreme_days: int = 3) -> set[str]:
    """Blocklist plus any symbol with >= max_extreme_days daily moves of > +100% or < -75%."""
    r = close.ffill(limit=5).pct_change(fill_method=None)
    ext = ((r > 1.0) | (r < -0.75)).sum()
    return BLOCKLIST | set(ext[ext >= max_extreme_days].index)


def yahoo_symbol(t: str) -> str:
    return t.replace(".", "-")


# ---------------------------------------------------------------- membership
def fetch_membership() -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / "sp500_hist.csv"
    r = requests.get(MEMBERSHIP_URL, timeout=60)
    r.raise_for_status()
    out.write_bytes(r.content)
    return out


def load_membership() -> pd.DataFrame:
    """Rows = change dates, value = list of member tickers effective from that date."""
    df = pd.read_csv(DATA_DIR / "sp500_hist.csv", parse_dates=["date"])
    df["tickers"] = df["tickers"].str.split(",").map(lambda xs: sorted({yahoo_symbol(x.strip()) for x in xs}))
    return df.set_index("date").sort_index()


def membership_matrix(dates: pd.DatetimeIndex, tickers: list[str], mode: str = "pit") -> pd.DataFrame:
    """Boolean (dates x tickers) frame: was ticker an index member on that date?

    mode="pit"     point-in-time membership (no survivorship bias from index composition)
    mode="current" today's constituents for all history (the spec as literally written; biased)
    """
    mem = load_membership()
    if mode == "current":
        cur = set(mem["tickers"].iloc[-1])
        row = np.array([t in cur for t in tickers])
        return pd.DataFrame(np.tile(row, (len(dates), 1)), index=dates, columns=tickers)
    col = {t: i for i, t in enumerate(tickers)}
    snap = np.zeros((len(mem), len(tickers)), dtype=bool)
    for i, members in enumerate(mem["tickers"]):
        idx = [col[t] for t in members if t in col]
        snap[i, idx] = True
    snap_df = pd.DataFrame(snap, index=mem.index, columns=tickers)
    return snap_df.reindex(dates, method="ffill").fillna(False).astype(bool)


def all_tickers() -> list[str]:
    mem = load_membership()
    return sorted(set().union(*mem["tickers"]))


# ---------------------------------------------------------------- prices
def fetch_prices(tickers: list[str] | None = None, chunk: int = 80, log=print) -> None:
    import yfinance as yf

    tickers = tickers or all_tickers()
    syms = list(dict.fromkeys(tickers + EXTRA))
    opens, closes = [], []
    for i in range(0, len(syms), chunk):
        batch = syms[i : i + chunk]
        log(f"downloading {i + 1}-{i + len(batch)} of {len(syms)}")
        for attempt in range(3):
            try:
                d = yf.download(batch, start=PRICE_START, auto_adjust=True, progress=False,
                                threads=True, group_by="column")
                break
            except Exception as e:  # rate limits etc.
                log(f"  retry ({e})")
                time.sleep(5 * (attempt + 1))
        else:
            continue
        opens.append(d["Open"])
        closes.append(d["Close"])
        time.sleep(1)
    o = pd.concat(opens, axis=1).dropna(how="all", axis=1)
    c = pd.concat(closes, axis=1).dropna(how="all", axis=1)
    o = o.loc[:, ~o.columns.duplicated()]
    c = c.loc[:, ~c.columns.duplicated()]
    o.index = pd.to_datetime(o.index).tz_localize(None)
    c.index = pd.to_datetime(c.index).tz_localize(None)
    o.astype("float64").to_parquet(DATA_DIR / "open.parquet")
    c.astype("float64").to_parquet(DATA_DIR / "close.parquet")
    log(f"saved prices for {c.shape[1]} symbols, {c.index[0].date()} .. {c.index[-1].date()}")


def load_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    o = pd.read_parquet(DATA_DIR / "open.parquet")
    c = pd.read_parquet(DATA_DIR / "close.parquet")
    return o, c


# ---------------------------------------------------------------- fundamentals (quality gate)
def fetch_fundamentals(tickers: list[str] | None = None, log=print) -> None:
    """Annual income statements for current constituents (Yahoo only serves ~4-5 fiscal years).

    Stored long-form: ticker, period_end, operating_income, ebit, interest_expense.
    """
    import yfinance as yf

    tickers = tickers or load_membership()["tickers"].iloc[-1]
    rows = []
    for n, t in enumerate(tickers, 1):
        if n % 50 == 0:
            log(f"fundamentals {n}/{len(tickers)}")
        try:
            inc = yf.Ticker(t).income_stmt
        except Exception:
            time.sleep(2)
            continue
        if inc is None or inc.empty:
            continue
        for period, col in inc.items():
            g = lambda k: float(col[k]) if k in col.index and pd.notna(col[k]) else np.nan  # noqa: E731
            rows.append(dict(ticker=t, period_end=pd.Timestamp(period),
                             operating_income=g("Operating Income"), ebit=g("EBIT"),
                             interest_expense=g("Interest Expense")))
        time.sleep(0.15)
    pd.DataFrame(rows).to_parquet(DATA_DIR / "fundamentals.parquet")
    log(f"saved fundamentals for {len({r['ticker'] for r in rows})} tickers")


def quality_matrix(dates: pd.DatetimeIndex, tickers: list[str], report_lag_days: int = 90,
                   min_interest_cover: float = 1.5, missing_passes: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Point-in-time quality gate.

    A fiscal year becomes 'known' report_lag_days after its period end. A stock passes if its
    latest known year has operating income > 0 and EBIT / interest expense >= min_interest_cover
    (distress proxy). Returns (passes, has_data) boolean frames.
    """
    f = DATA_DIR / "fundamentals.parquet"
    if not f.exists():
        ones = pd.DataFrame(True, index=dates, columns=tickers)
        return ones, ~ones
    fd = pd.read_parquet(f)
    fd["known"] = fd["period_end"] + pd.Timedelta(days=report_lag_days)
    ie = fd["interest_expense"].abs()
    cover = np.where(ie > 0, fd["ebit"] / ie, np.inf)
    ok = (fd["operating_income"] > 0) & (np.nan_to_num(cover, nan=np.inf) >= min_interest_cover)
    fd["ok"] = ok.astype(float).where(fd["operating_income"].notna())
    fd = fd.dropna(subset=["ok"]).sort_values("known")
    ok = fd.pivot_table(index="known", columns="ticker", values="ok", aggfunc="last")
    ok = ok.reindex(columns=tickers)
    full_idx = ok.index.union(dates)
    ok = ok.reindex(full_idx).ffill().reindex(dates)
    has = ok.notna()
    passes = ok.fillna(1.0 if missing_passes else 0.0).astype(bool)
    return passes, has


def fetch_all(log=print, fundamentals: bool = False) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    log("fetching point-in-time membership")
    fetch_membership()
    fetch_prices(log=log)
    log("fetching Nasdaq-100 and SPY dividends")
    fetch_nasdaq()
    fetch_spy_dividends()
    if fundamentals:
        fetch_fundamentals(log=log)


# ---------------------------------------------------------------- sectors (for sector caps)
def fetch_sectors(tickers: list[str] | None = None, log=print) -> None:
    """Current GICS-style sector per symbol from Yahoo. Sectors rarely change, so using today's
    label historically is a small approximation; symbols Yahoo no longer knows get 'Unknown'."""
    import yfinance as yf

    if tickers is None:
        _, c = load_prices()
        tickers = [t for t in c.columns if t not in EXTRA]
    rows = []
    for n, t in enumerate(tickers, 1):
        if n % 100 == 0:
            log(f"sectors {n}/{len(tickers)}")
        sector = None
        for attempt in range(2):
            try:
                sector = yf.Ticker(t).info.get("sector")
                break
            except Exception:
                time.sleep(2)
        rows.append((t, sector or "Unknown"))
        time.sleep(0.1)
    pd.DataFrame(rows, columns=["ticker", "sector"]).to_csv(DATA_DIR / "sectors.csv", index=False)
    log("saved sectors")


def load_sectors() -> dict[str, str]:
    f = DATA_DIR / "sectors.csv"
    if not f.exists():
        return {}
    return dict(pd.read_csv(f).itertuples(index=False, name=None))


def load_nasdaq() -> pd.Series:
    """Nasdaq-100 total return: QQQ (adjusted) from 1999-03, ^NDX price index before that."""
    return pd.read_parquet(DATA_DIR / "nasdaq100_tr.parquet")["NASDAQ100"]


def fetch_nasdaq() -> None:
    import yfinance as yf

    d = yf.download(["QQQ", "^NDX"], start=PRICE_START, auto_adjust=True, progress=False)["Close"]
    d.index = pd.to_datetime(d.index).tz_localize(None)
    q, n = d["QQQ"].dropna(), d["^NDX"].dropna()
    pre = n[: q.index[0]]
    spliced = pd.concat([pre[:-1] / pre.iloc[-1] * q.iloc[0], q])
    spliced.rename("NASDAQ100").to_frame().to_parquet(DATA_DIR / "nasdaq100_tr.parquet")


def fetch_spy_dividends() -> None:
    import yfinance as yf

    h = yf.Ticker(BENCH).history(start=PRICE_START, auto_adjust=False)
    h.index = h.index.tz_localize(None)
    h[["Close", "Dividends"]].to_parquet(DATA_DIR / "spy_unadjusted.parquet")
