"""12-1 momentum backtest engine.

Timeline for a rebalance on the first trading day `d` of month M:
  * signal uses information up to the close of the previous trading day `s` (last day of M-1)
  * momentum  = close(end of M-1-skip) / close(end of M-1-lookback) - 1   (12-1 by default)
  * regime    = S&P 500 close(s) > SMA_n(s)
  * trades fill at the open of `d`, costs charged per side in bps of traded notional
Idle cash accrues the 13-week T-bill yield (^IRX). Prices are split/dividend adjusted, so
returns are total returns.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from . import data


@dataclass(frozen=True)
class Config:
    start: str = "1996-02-01"
    end: str | None = None
    top_n: int = 20
    buffer_rank: int = 40           # hold while rank <= this; set == top_n for no buffer
    lookback_months: int = 12
    skip_months: int = 1
    regime_filter: bool = True
    sma_days: int = 200
    universe: str = "pit"           # "pit" (point-in-time) | "current" (today's members, biased)
    quality_gate: bool = False
    rebalance_mode: str = "proceeds"  # "proceeds": only new buys use sale proceeds | "full": reset to equal weight
    cost_bps: float = 10.0
    # ---- optional improvements (defaults = spec as written)
    score: str = "raw"              # "raw" 12-1 return | "risk_adjusted": 12-1 return / trailing 1y volatility
    sma_band: float = 0.0           # hysteresis around the SMA, e.g. 0.02 = exit < SMA-2%, re-enter > SMA+2%
    vol_target: float = 0.0         # e.g. 0.15 = scale equity exposure to ~15% ex-ante vol (0 = off)
    max_weight: float = 0.0         # e.g. 0.10 = trim any position above 10% of NAV (0 = off)
    regime_rule: str = "sma"        # "sma": index > SMA (spec) | "sma_absmom": bear only if index < SMA AND
                                    #   S&P 12-month total return < T-bills | "faber": index vs 10-month SMA
                                    #   | "fast_reentry": exit below SMA, re-enter above the 50-day SMA
                                    #   | "sma_slope": bear only if index < SMA AND the SMA is falling
    slope_days: int = 21            # "sma_slope": SMA compared with its value this many days earlier
    regime_daily: bool = False      # evaluate the regime every day (trade next open) instead of monthly
    bear_exposure: float = 0.0      # share of the basket kept in stocks during a bear regime (spec: 0)
    weighting: str = "equal"        # "equal" | "inverse_vol" (weights proportional to 1 / 6-month volatility)
    sector_cap: int = 0             # max holdings per sector when buying (0 = off)
    core_spy: float = 0.0           # share of capital held in SPY, rebalanced monthly (0 = off)
    # ---- taxes (taxable account). 0 = tax-free account.
    tax_short: float = 0.0          # rate on realised gains from lots held <= 1 year
    tax_long: float | None = None   # rate on lots held > 1 year (None = same as tax_short)
    tax_interest: float | None = None  # rate on T-bill interest (None = same as tax_short)
    initial_capital: float = 100_000.0
    label: str = "Strategy"


@dataclass
class Result:
    config: Config
    nav: pd.Series
    bench: pd.Series
    cash_rate: pd.Series                 # daily T-bill return
    rebalances: pd.DataFrame             # one row per rebalance date
    holdings: pd.DataFrame               # long: date, ticker, rank, momentum, weight
    trades: pd.DataFrame                 # long: date, ticker, side, shares, price, notional, reason
    weights_in_market: pd.Series = field(default=None)  # daily fraction of NAV in stocks
    tax: dict = field(default=None)      # tax summary when Config.tax_short > 0
    positions: pd.DataFrame = field(default=None)  # open positions at the end: ticker, shares, entry, cost


# ---------------------------------------------------------------- prepared market data
@dataclass
class Market:
    dates: pd.DatetimeIndex
    tickers: list[str]
    open: np.ndarray
    close: np.ndarray        # raw (NaN when not trading)
    close_ff: np.ndarray     # forward filled up to 5 days, for valuation
    month_end: pd.DataFrame  # month-end closes, index = Period[M]
    index_close: pd.Series
    bench_close: pd.Series
    tbill_daily: pd.Series
    has_price: np.ndarray    # bool, close available


@lru_cache(maxsize=2)
def load_market() -> Market:
    o, c = data.load_prices()
    idx = c[data.INDEX].dropna()
    dates = idx.index
    bad = data.bad_tickers(c.drop(columns=[x for x in data.EXTRA if x in c.columns]))
    tickers = [t for t in c.columns if t not in data.EXTRA and t not in bad]
    O = o.reindex(index=dates, columns=tickers)
    C = c.reindex(index=dates, columns=tickers)
    Cff = C.ffill(limit=5)
    me = Cff.groupby(dates.to_period("M")).last()
    # a month-end price only counts if the stock actually traded in the last week of the month
    last_day = pd.Series(dates, index=dates).groupby(dates.to_period("M")).last()
    me = me.where(C.reindex(last_day.values).ffill(limit=5).notna().values)
    irx = c[data.TBILL].reindex(dates).ffill().fillna(0.0)
    tb = (irx / 100.0 / 252.0).shift(1).fillna(0.0)
    return Market(
        dates=dates, tickers=tickers, open=O.to_numpy(), close=C.to_numpy(),
        close_ff=Cff.to_numpy(), month_end=me, index_close=idx,
        bench_close=c[data.BENCH].reindex(dates).ffill(), tbill_daily=tb,
        has_price=C.notna().to_numpy(),
    )


@lru_cache(maxsize=4)
def _membership(universe: str) -> np.ndarray:
    m = load_market()
    return data.membership_matrix(m.dates, m.tickers, universe).to_numpy()


@lru_cache(maxsize=2)
def _quality() -> tuple[np.ndarray, np.ndarray]:
    m = load_market()
    p, h = data.quality_matrix(m.dates, m.tickers)
    return p.to_numpy(), h.to_numpy()


def momentum_scores(m: Market, month: pd.Period, lookback: int, skip: int) -> pd.Series:
    """12-1 momentum known at the start of `month`."""
    recent, past = month - 1 - skip, month - 1 - lookback
    if past not in m.month_end.index or recent not in m.month_end.index:
        return pd.Series(np.nan, index=m.tickers)
    return m.month_end.loc[recent] / m.month_end.loc[past] - 1.0


@lru_cache(maxsize=16)
def _regime_inputs(sma_days: int, slope_days: int = 21) -> dict[str, np.ndarray]:
    m = load_market()
    px = m.index_close
    months = px.index.to_period("M")
    me = px.groupby(months).last()
    # 10-month SMA: the 9 previous month-end closes plus today's close
    prev9 = me.rolling(9).sum().shift(1).reindex(months).to_numpy()
    spy = m.bench_close
    tbill_growth = (1 + m.tbill_daily).cumprod()
    return dict(
        spx=px.to_numpy(),
        sma=px.rolling(sma_days).mean().to_numpy(),
        sma50=px.rolling(50).mean().to_numpy(),
        sma_prev=px.rolling(sma_days).mean().shift(slope_days).to_numpy(),
        faber=(prev9 + px.to_numpy()) / 10,
        spy_12m=(spy / spy.shift(252) - 1).to_numpy(),
        tbill_12m=(tbill_growth / tbill_growth.shift(252) - 1).to_numpy(),
    )


def regime_state(cfg: Config, m: Market, s: int, was_bull: bool | None = None) -> tuple[bool, float, float]:
    """(bull?, index close, trend line) using information up to the close of day index `s`."""
    x = _regime_inputs(cfg.sma_days, cfg.slope_days)
    spx = float(x["spx"][s])
    line = float(x["faber"][s] if cfg.regime_rule == "faber" else x["sma"][s])
    if not np.isfinite(line):
        return True, spx, np.nan
    band = cfg.sma_band
    if band <= 0 or was_bull is None:
        above = spx > line
    else:
        above = spx >= line * (1 - band) if was_bull else spx > line * (1 + band)
    if cfg.regime_rule == "fast_reentry" and was_bull is False and not above:
        # exit on the long SMA, but re-enter as soon as the index regains its 50-day SMA
        return bool(spx > x["sma50"][s]), spx, line
    if cfg.regime_rule == "sma_slope":
        # bear only when the index is below a *falling* SMA (SMA lower than one month ago)
        falling = np.isfinite(x["sma_prev"][s]) and x["sma"][s] < x["sma_prev"][s]
        return bool(above or not falling), spx, line
    if cfg.regime_rule == "sma_absmom":
        r12, t12 = x["spy_12m"][s], x["tbill_12m"][s]
        abs_ok = np.isfinite(r12) and np.isfinite(t12) and r12 > t12
        return bool(above or abs_ok), spx, line
    return bool(above), spx, line


def regime_is_bull(m: Market, s: int, sma_days: int, band: float = 0.0,
                   was_bull: bool | None = None) -> tuple[bool, float, float]:
    """Spec rule (kept for callers that only know the SMA parameters)."""
    return regime_state(Config(sma_days=sma_days, sma_band=band), m, s, was_bull)


@lru_cache(maxsize=1)
def _sector_codes() -> np.ndarray:
    m = load_market()
    sec = data.load_sectors()
    return np.array([sec.get(t, "Unknown") for t in m.tickers], dtype=object)


def clear_caches() -> None:
    """Drop everything derived from the files in data/; call after any data refresh."""
    for f in (load_market, _membership, _quality, _regime_inputs, _sector_codes, _monthly_returns):
        f.cache_clear()


@lru_cache(maxsize=1)
def _monthly_returns() -> tuple[np.ndarray, np.ndarray]:
    """(stock returns [month x ticker], SPY total return [month]) aligned with Market.month_end."""
    m = load_market()
    r = m.month_end.pct_change(fill_method=None).to_numpy()
    mkt = m.bench_close.groupby(m.dates.to_period("M")).last().pct_change()
    return r, mkt.reindex(m.month_end.index).to_numpy()


def residual_momentum(m: Market, month: pd.Period, cols: np.ndarray, lookback: int, skip: int,
                      sectors: np.ndarray | None = None, window: int = 36, min_obs: int = 24) -> np.ndarray:
    """Blitz, Huij & Martens (2011): regress each stock's monthly returns over the last `window`
    months on SPY (and, with `sectors`, on the equal-weighted return of the rest of its sector),
    then score = sum of residuals over the 12-1 window / their standard deviation. Uses month-end
    closes through month - 1 only. NaN where history is too short."""
    k = m.month_end.index.get_indexer([month - 1])[0]
    out = np.full(len(cols), np.nan)
    if k < window:
        return out
    r, mkt = _monthly_returns()
    Y = r[k - window + 1 : k + 1][:, cols]
    x = mkt[k - window + 1 : k + 1]
    form = slice(window - lookback, window - skip)  # rows of months M-lookback .. M-1-skip
    if sectors is not None:
        sec = sectors[cols]
        sec_ret = np.full_like(Y, np.nan)
        for name in set(sec) - {"Unknown"}:
            g = sec == name
            ok = np.isfinite(Y[:, g])
            tot, n = np.where(ok, Y[:, g], 0).sum(1, keepdims=True), ok.sum(1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):  # leave the stock itself out
                sec_ret[:, g] = np.where(n - ok > 0, (tot - np.where(ok, Y[:, g], 0)) / (n - ok), np.nan)
    for i in range(len(cols)):
        y = Y[:, i]
        cols_x = [np.ones(window), x] + ([sec_ret[:, i]] if sectors is not None and sec[i] != "Unknown" else [])
        X = np.column_stack(cols_x)
        ok = np.isfinite(y) & np.isfinite(X).all(1)
        if ok.sum() < min_obs or not ok[form].all():
            continue
        beta = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
        e = (y - X @ beta)[form]
        sd = e.std(ddof=1)
        out[i] = e.sum() / sd if sd > 0 else np.nan
    return out


def trailing_vol(m: Market, s: int, cols, days: int = 252) -> np.ndarray:
    px = m.close_ff[max(0, s - days) : s + 1][:, cols]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = px[1:] / px[:-1] - 1
    return np.nanstd(r, axis=0) * np.sqrt(252)


def trailing_vol_basket(m: Market, s: int, cols, days: int = 126) -> float:
    """Ex-ante annualised vol of an equal-weighted basket of `cols` over the last `days`."""
    px = m.close_ff[max(0, s - days) : s + 1][:, cols]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.nanmean(px[1:] / px[:-1] - 1, axis=1)
    r = r[np.isfinite(r)]
    return float(r.std() * np.sqrt(252)) if len(r) > 20 else 0.0


def rank_universe(cfg: Config, m: Market, s: int, month: pd.Period) -> pd.DataFrame:
    """Eligible stocks ranked by momentum as of signal day index `s`."""
    mom = momentum_scores(m, month, cfg.lookback_months, cfg.skip_months).to_numpy()
    elig = _membership(cfg.universe)[s] & m.has_price[s] & np.isfinite(mom)
    if cfg.quality_gate:
        elig &= _quality()[0][s]
    idx = np.flatnonzero(elig)
    score = mom.copy()
    if cfg.score == "risk_adjusted" and len(idx):
        vol = trailing_vol(m, s, idx)
        score[idx] = np.where(vol > 0, mom[idx] / vol, np.nan)
        idx = idx[np.isfinite(score[idx])]
    elif cfg.score == "blend" and len(idx):
        # average percentile rank of 12-1 and 6-1 momentum
        mom6 = momentum_scores(m, month, 6, cfg.skip_months).to_numpy()
        ok = idx[np.isfinite(mom6[idx])]
        pr = lambda v: pd.Series(v).rank(pct=True).to_numpy()  # noqa: E731
        score[:] = np.nan
        score[ok] = (pr(mom[ok]) + pr(mom6[ok])) / 2
        idx = ok
    elif cfg.score in ("residual", "residual_sector") and len(idx):
        # momentum net of market (and sector) moves: avoids buying a basket that is just one hot
        # sector or the highest-beta names
        score[idx] = residual_momentum(m, month, idx, cfg.lookback_months, cfg.skip_months,
                                       _sector_codes() if cfg.score == "residual_sector" else None)
        idx = idx[np.isfinite(score[idx])]
    elif cfg.score == "high52" and len(idx):
        # George & Hwang (2004): closeness to the 52-week high, ties broken by 12-1 momentum
        hi = np.nanmax(m.close_ff[max(0, s - 251) : s + 1][:, idx], axis=0)
        score[idx] = m.close_ff[s, idx] / hi + 1e-6 * pd.Series(mom[idx]).rank(pct=True).to_numpy()
        idx = idx[np.isfinite(score[idx])]
    elif cfg.score == "smooth" and len(idx):
        # Da, Gurun & Warachka (2014) "frog in the pan": prefer momentum built from many small
        # gains over a few jumps. Score = pct rank of 12-1 return + pct rank of share of up days.
        px = m.close_ff[max(0, s - 252) : max(1, s - 20)][:, idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            d = px[1:] / px[:-1] - 1
        up = (d > 0).sum(0) / np.maximum(np.isfinite(d).sum(0), 1)
        pr = lambda v: pd.Series(v).rank(pct=True).to_numpy()  # noqa: E731
        score[idx] = pr(mom[idx]) + pr(up)
    order = idx[np.argsort(-score[idx], kind="stable")] if len(idx) else idx
    return pd.DataFrame({"ticker": [m.tickers[i] for i in order], "col": order,
                         "momentum": mom[order], "score": score[order],
                         "rank": np.arange(1, len(order) + 1)})


# ---------------------------------------------------------------- engine
def run(cfg: Config) -> Result:
    m = load_market()
    dates = m.dates
    start = dates.searchsorted(pd.Timestamp(cfg.start))
    end = len(dates) if cfg.end is None else dates.searchsorted(pd.Timestamp(cfg.end), side="right")
    start = max(start, 1)
    months = dates.to_period("M")
    first_of_month = np.r_[True, months[1:] != months[:-1]]
    c = cfg.cost_bps / 1e4
    tb = m.tbill_daily.to_numpy()

    cash = cfg.initial_capital
    shares: dict[int, float] = {}
    nav = np.full(len(dates), np.nan)
    invested = np.full(len(dates), np.nan)
    rebal_rows, hold_rows, trade_rows = [], [], []
    last_px = {}
    entry_t: dict[int, int] = {}      # day index a position was opened
    cost_in: dict[int, float] = {}    # cash invested in the open position (incl. costs)

    # ---- tax lots (FIFO): j -> list of [day index, shares, cost per share]
    taxed = cfg.tax_short > 0
    r_st = cfg.tax_short
    r_lt = cfg.tax_short if cfg.tax_long is None else cfg.tax_long
    r_int = cfg.tax_short if cfg.tax_interest is None else cfg.tax_interest
    lots: dict[int, list[list[float]]] = {}
    yr = dict(st=0.0, lt=0.0, interest=0.0)
    carry = 0.0
    tax_paid = 0.0
    tax_rows = []

    def realise(t, j, q, proceeds):
        """Consume q shares of j's lots FIFO and book the gain as short- or long-term."""
        if not taxed:
            return
        per_share = proceeds / q if q > 0 else 0.0
        book = lots.get(j, [])
        while q > 1e-12 and book:
            lot = book[0]
            take = min(q, lot[1])
            gain = take * (per_share - lot[2])
            yr["lt" if (dates[t] - dates[int(lot[0])]).days > 365 else "st"] += gain
            lot[1] -= take
            q -= take
            if lot[1] <= 1e-12:
                book.pop(0)

    def tax_on(st, lt, carry_in):
        """US-style netting: cross-net short/long, then use loss carryforward. -> (tax, carry)."""
        if st < 0:
            lt, st = lt + st, 0.0
        if lt < 0:
            st, lt = st + lt, 0.0
        if st < 0:
            return 0.0, carry_in - st
        use = min(carry_in, st); st -= use; carry_in -= use
        use = min(carry_in, lt); lt -= use; carry_in -= use
        return st * r_st + lt * r_lt, carry_in

    def px_open(t, j):
        p = m.open[t, j]
        return p if np.isfinite(p) and p > 0 else m.close_ff[t - 1, j]

    def sell(t, j, price, reason):
        nonlocal cash
        q = shares.pop(j)
        entry_t.pop(j, None)
        cost_in.pop(j, None)
        cash += q * price * (1 - c)
        realise(t, j, q, q * price * (1 - c))
        lots.pop(j, None)
        trade_rows.append((dates[t], m.tickers[j], "SELL", q, price, q * price, reason))

    def buy(t, j, amount, price, reason):
        nonlocal cash
        if amount <= 0 or not np.isfinite(price) or price <= 0:
            return
        q = amount * (1 - c) / price
        if j not in shares:
            entry_t[j] = t
        cost_in[j] = cost_in.get(j, 0.0) + amount
        shares[j] = shares.get(j, 0.0) + q
        cash -= amount
        if taxed:
            lots.setdefault(j, []).append([t, q, amount / q])
        trade_rows.append((dates[t], m.tickers[j], "BUY", q, price, amount, reason))

    def trim(t, j, amount, price, reason):
        nonlocal cash
        q = min(shares[j], amount / price)
        cost_in[j] = cost_in.get(j, 0.0) * (1 - q / shares[j]) if shares[j] > 0 else 0.0
        shares[j] -= q
        cash += q * price * (1 - c)
        realise(t, j, q, q * price * (1 - c))
        trade_rows.append((dates[t], m.tickers[j], "TRIM", q, price, q * price, reason))
        return q * price

    def rebalance_to(t, targets, nav_now, rank_of, trims=True):
        """Move positions to target dollar values; skips trades smaller than 0.25% of NAV."""
        done = 0.0
        tol = 0.0025 * nav_now
        if trims:
            for j, tgt in targets.items():
                if j in shares:
                    p = px_open(t, j)
                    if shares[j] * p - tgt > tol:
                        done += trim(t, j, shares[j] * p - tgt, p, "resize")
        need = {}
        for j, tgt in targets.items():
            p = px_open(t, j) if j in shares else m.open[t, j]
            d = tgt - (shares[j] * p if j in shares else 0.0)
            if d > tol or (j not in shares and d > 0):
                need[j] = (d, p)
        tot = sum(d for d, _ in need.values())
        scale = min(1.0, cash / tot) if tot > 0 else 0.0
        for j, (d, p) in need.items():
            buy(t, j, d * scale, p, f"rank {rank_of.get(j)}" if j not in shares else "resize")
            done += d * scale
        return done

    sectors = _sector_codes() if cfg.sector_cap > 0 else None
    was_bull = None
    for t in range(start, end):
        if taxed and cash > 0:
            yr["interest"] += cash * tb[t]
        cash *= 1 + tb[t]

        if taxed and t > start and dates[t].year != dates[t - 1].year:
            # settle last year's tax bill on the first trading day of the new year
            due, carry = tax_on(yr["st"], yr["lt"], carry)
            due += max(yr["interest"], 0.0) * r_int
            tax_rows.append(dict(year=dates[t - 1].year, short_term=yr["st"], long_term=yr["lt"],
                                 interest=yr["interest"], tax=due, loss_carry=carry))
            yr = dict(st=0.0, lt=0.0, interest=0.0)
            cash -= due
            tax_paid += due
            if cash < 0 and shares:  # not enough cash: sell holdings pro rata to pay the bill
                vals = {j: q * px_open(t, j) for j, q in shares.items()}
                tot = sum(vals.values())
                need = -cash / (1 - c)
                for j, v in vals.items():
                    trim(t, j, need * v / tot, px_open(t, j), "pay taxes")

        trigger = first_of_month[t]
        if not trigger and cfg.regime_daily and cfg.regime_filter and was_bull is not None:
            trigger = regime_state(cfg, m, t - 1, was_bull)[0] != was_bull
        if trigger:
            s_ = t - 1
            bull, spx, sma = regime_state(cfg, m, s_, was_bull)
            if not cfg.regime_filter:
                bull = True
            was_bull = bull
            ranks = rank_universe(cfg, m, s_, months[t])
            rank_of = dict(zip(ranks["col"], ranks["rank"]))
            n_sold = n_bought = 0
            traded = 0.0
            nav_open = cash + sum(q * px_open(t, j) for j, q in shares.items())
            exposure = 1.0

            exposure_cap = 1.0 if bull else cfg.bear_exposure
            if exposure_cap <= 0:
                exposure = 0.0
                for j in list(shares):
                    p = px_open(t, j)
                    traded += shares[j] * p
                    sell(t, j, p, "regime: bear")
                    n_sold += 1
            else:
                keep_rank = max(cfg.buffer_rank, cfg.top_n)
                for j in list(shares):
                    r = rank_of.get(j)
                    if r is None or r > keep_rank:
                        p = px_open(t, j)
                        traded += shares[j] * p
                        sell(t, j, p, "left universe" if r is None else f"rank {r} > {keep_rank}")
                        n_sold += 1
                slots = cfg.top_n - len(shares)
                new = []
                sec_count: dict[str, int] = {}
                if sectors is not None:
                    for j in shares:
                        sec_count[sectors[j]] = sec_count.get(sectors[j], 0) + 1
                for j in ranks["col"]:
                    if slots <= 0:
                        break
                    if j in shares or not np.isfinite(m.open[t, j]):
                        continue
                    if sectors is not None and sectors[j] != "Unknown":
                        if sec_count.get(sectors[j], 0) >= cfg.sector_cap:
                            continue
                        sec_count[sectors[j]] = sec_count.get(sectors[j], 0) + 1
                    new.append(j)
                    slots -= 1
                final = list(shares) + new
                nav_now = cash + sum(q * px_open(t, j) for j, q in shares.items())
                cap = cfg.max_weight * nav_now if cfg.max_weight > 0 else np.inf
                exposure = exposure_cap
                if cfg.vol_target > 0 and final:
                    w = trailing_vol_basket(m, s_, final)
                    exposure = min(exposure, cfg.vol_target / w) if w > 0 else exposure

                if (cfg.rebalance_mode == "full" or cfg.vol_target > 0 or exposure < 1
                        or cfg.weighting != "equal"):
                    if cfg.weighting == "inverse_vol" and final:
                        iv = 1 / np.maximum(trailing_vol(m, s_, final, 126), 0.05)
                        w = iv / iv.sum() * len(final) / cfg.top_n
                    else:
                        w = np.full(len(final), 1 / cfg.top_n)
                    targets = {j: min(exposure * nav_now * wj, cap) for j, wj in zip(final, w)}
                    traded += rebalance_to(t, targets, nav_now, rank_of)
                    n_bought = len(new)
                else:
                    if cfg.max_weight > 0:
                        for j in list(shares):
                            p = px_open(t, j)
                            excess = shares[j] * p - cap
                            if excess > 0:
                                traded += trim(t, j, excess, p, f"above {cfg.max_weight:.0%} cap")
                    # new names get sale proceeds, up to an equal 1/N weight each (never above the cap)
                    per = min(nav_now / cfg.top_n, cap)
                    alloc = min(cash / len(new), per) if new else 0
                    for j in new:
                        buy(t, j, alloc, m.open[t, j], f"rank {rank_of[j]}")
                        traded += alloc
                        n_bought += 1
                    if cash > 0.005 * nav_now and shares:
                        # spare cash (e.g. after a partial-exposure period): top up holdings that are
                        # below 1/N; winners are never trimmed
                        traded += rebalance_to(t, {j: max(per, shares[j] * px_open(t, j)) for j in shares},
                                               nav_now, rank_of, trims=False)

            rebal_rows.append(dict(date=dates[t], regime="bull" if bull else "bear", spx=spx, sma=sma,
                                   eligible=len(ranks), members=int(_membership(cfg.universe)[s_].sum()),
                                   holdings=len(shares), sold=n_sold, bought=n_bought,
                                   exposure=exposure,
                                   turnover=traded / nav_open / 2 if nav_open > 0 else 0.0,
                                   nav=nav_open))
            shares = {j: q for j, q in shares.items() if q > 1e-12}
            pos_val = {j: q * px_open(t, j) for j, q in shares.items()}
            tot = cash + sum(pos_val.values())
            for j, v in pos_val.items():
                r = rank_of.get(j)
                mom = ranks.loc[ranks["col"] == j, "momentum"]
                hold_rows.append((dates[t], m.tickers[j], r, float(mom.iloc[0]) if len(mom) else np.nan, v / tot))

        # liquidate anything that stopped trading (delisting / acquisition) at its last price
        for j in list(shares):
            p = m.close_ff[t, j]
            if np.isfinite(p):
                last_px[j] = p
            else:
                sell(t, j, last_px.get(j, 0.0), "stopped trading")

        pos = sum(q * m.close_ff[t, j] for j, q in shares.items())
        nav[t] = cash + pos
        invested[t] = pos / nav[t] if nav[t] > 0 else 0.0

    tax_info = None
    if taxed:
        # bill for the (partial) final year, and what selling everything today would add
        t_end = end - 1
        accrued, _ = tax_on(yr["st"], yr["lt"], carry)
        accrued += max(yr["interest"], 0.0) * r_int
        st_u = lt_u = 0.0
        for j, book in lots.items():
            px = m.close_ff[t_end, j] * (1 - c)
            for lt_i, q, cost in book:
                g = q * (px - cost)
                if (dates[t_end] - dates[int(lt_i)]).days > 365:
                    lt_u += g
                else:
                    st_u += g
        if_sold, _ = tax_on(yr["st"] + st_u, yr["lt"] + lt_u, carry)
        if_sold += max(yr["interest"], 0.0) * r_int
        tax_info = dict(paid=tax_paid, accrued=accrued, liquidation_tax=if_sold,
                        after_tax_value=nav[t_end] - accrued, liquidation_value=nav[t_end] - if_sold,
                        by_year=pd.DataFrame(tax_rows))

    t_last = end - 1
    pos_rows = []
    for j, q in shares.items():
        px = m.close_ff[t_last, j]
        pos_rows.append(dict(ticker=m.tickers[j], shares=q, price=px, value=q * px,
                             weight=q * px / nav[t_last], entry=dates[entry_t.get(j, t_last)],
                             cost=cost_in.get(j, np.nan), gain=q * px / cost_in[j] - 1 if cost_in.get(j) else np.nan))
    positions = pd.DataFrame(pos_rows, columns=["ticker", "shares", "price", "value", "weight", "entry", "cost", "gain"])
    positions["entry"] = pd.to_datetime(positions["entry"])  # keeps a datetime dtype when all in cash

    sl = slice(start, end)
    nav_s = pd.Series(nav[sl], index=dates[sl], name=cfg.label)
    bench = m.bench_close.iloc[sl]
    bench = bench / bench.iloc[0] * cfg.initial_capital
    inv_s = pd.Series(invested[sl], index=dates[sl])
    if cfg.core_spy > 0:
        nav_s, inv_s = _with_core(nav_s, bench, inv_s, cfg.core_spy, c, first_of_month[sl])
    trades = pd.DataFrame(trade_rows, columns=["date", "ticker", "side", "shares", "price", "notional", "reason"])
    holdings = pd.DataFrame(hold_rows, columns=["date", "ticker", "rank", "momentum", "weight"])
    return Result(cfg, nav_s, bench.rename("S&P 500 (SPY TR)"), m.tbill_daily.iloc[sl],
                  pd.DataFrame(rebal_rows).set_index("date") if rebal_rows else pd.DataFrame(),
                  holdings, trades, inv_s, tax_info, positions.sort_values("weight", ascending=False))


def _with_core(sleeve: pd.Series, spy: pd.Series, invested: pd.Series, core: float, cost: float,
               rebalance_day: np.ndarray) -> tuple[pd.Series, pd.Series]:
    """Blend the momentum sleeve with a SPY core, both reset to target weights each month."""
    rs = sleeve.pct_change().fillna(0.0).to_numpy()
    rb = spy.pct_change().fillna(0.0).to_numpy()
    v_s, v_b = sleeve.iloc[0] * (1 - core), sleeve.iloc[0] * core
    out = np.empty(len(rs))
    w_core = np.empty(len(rs))
    for i in range(len(rs)):
        if i > 0:
            v_s *= 1 + rs[i]
            v_b *= 1 + rb[i]
            if rebalance_day[i]:
                tot = v_s + v_b
                moved = abs(v_b - core * tot)
                tot -= moved * cost
                v_s, v_b = tot * (1 - core), tot * core
        out[i] = v_s + v_b
        w_core[i] = v_b / out[i]
    nav = pd.Series(out, index=sleeve.index, name=sleeve.name)
    return nav, invested * (1 - w_core) + w_core


def live_signal(cfg: Config) -> dict:
    """What the model says today, for the next first-of-month rebalance."""
    m = load_market()
    s = len(m.dates) - 1
    next_month = m.dates[-1].to_period("M") + 1
    bull, spx, sma = regime_state(cfg, m, s)
    ranks = rank_universe(cfg, m, s, next_month)
    return dict(as_of=m.dates[-1], next_rebalance_month=str(next_month), bull=bull, spx=spx, sma=sma,
                ranks=ranks.drop(columns="col"))


def config_dict(cfg: Config) -> dict:
    return asdict(cfg)
