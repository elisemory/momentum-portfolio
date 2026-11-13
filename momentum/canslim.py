"""CAN SLIM backtest engine: William O'Neil's rules from "How to Make Money in Stocks" (4th ed.),
as far as daily closes and volume can express them.

Tested here
  N  buy only on a breakout: the first close above the highest close of the prior 52 weeks, out of a
     base at least 7 weeks long that corrected no more than 33%. Fill at the next open, skipped if
     that open is more than 5% above the pivot (don't chase).
  S  breakout-day volume at least 1.4x its 50-day average.
  L  RS rating >= 80: IBD-style 12-month return with the latest quarter double-weighted, ranked
     1-99 against point-in-time S&P 500 members.
  M  buy only while the market is in a confirmed uptrend. A cluster of distribution days (index down
     on higher volume) ends it; a follow-through day (a strong up day on higher volume, day 4 or
     later of a rally attempt) starts it again.
  Sell rules: 8% stop below cost; take profits at +20%, except that a stock up 20% within 3 weeks
     of the buy is held at least 8 weeks and then until it closes below its 50-day average.
Not tested: C, A and I need point-in-time earnings and ownership history, which free data sources
don't have. The screener checks C and A for today's candidates only (fetch_earnings).

Timeline: signals use closes up to day s; orders fill at the open of day s + 1. Stops and targets
are checked on closes, so a gap down can lose more than 8%. Idle cash earns the 13-week T-bill.

CLI:  python -m momentum.canslim      (baseline and one-knob variants, full period and both halves)
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd

from . import backtest as bt
from . import data

FAST_DAYS = 15       # "+20% within 3 weeks" ...
HOLD_DAYS = 40       # ... "hold at least 8 weeks"
BASE_LOOKBACK = 252  # pivot = highest close of the prior 52 weeks
PYRAMID = ((0.025, 0.3), (0.05, 0.2))  # add 30% of a slot at +2.5%, 20% at +5% (first buy = 50%)
PYRAMID_DAYS = 15    # adds only within 3 weeks of the first buy


@dataclass(frozen=True)
class Config:
    start: str = "1996-02-01"
    end: str | None = None
    max_positions: int = 6
    # ---- buy rules (N, S, L)
    min_rs: int = 80
    min_base_days: int = 35         # 7 weeks
    max_base_depth: float = 0.33
    max_chase: float = 0.05         # skip a buy if the open is more than this above the pivot
    min_volume_ratio: float = 1.4   # breakout volume / 50-day average (0 = off)
    # ---- market direction (M)
    market_rule: str = "ftd"        # "ftd": distribution / follow-through days (the book) | "sma200": index
                                    #   above its 200-day SMA | "sma_slope": bear only below a falling 200-day
                                    #   SMA (the momentum strategy's rule) | "off"
    dist_limit: int = 5             # distribution days within dist_window sessions that end an uptrend
    dist_window: int = 25
    dist_decline: float = 0.002     # index close down at least this much, on higher volume
    ftd_gain: float = 0.0125        # follow-through day: index up at least this much, on higher volume ...
    ftd_min_day: int = 4            # ... on this day of the rally attempt or later
    dist_expiry: float = 0.05       # a distribution day stops counting once the index closes this far above it
    on_correction: str = "sell_all"  # "sell_all" | "hold": keep positions, only stop buying
    # ---- sell rules
    stop_loss: float = 0.08
    profit_target: float = 0.20     # 0 = no profit target (exit only on the stop or the 50-day line)
    fast_gain_rule: bool = True     # +20% within 3 weeks: hold 8 weeks, then until a close below the 50-day
    # ---- sizing
    pyramid: bool = False           # buy half a slot, add 30% at +2.5% and 20% at +5%
    cost_bps: float = 10.0
    initial_capital: float = 100_000.0
    label: str = "CAN SLIM"


@dataclass
class Result:
    config: Config
    nav: pd.Series
    bench: pd.Series
    cash_rate: pd.Series
    invested: pd.Series             # daily fraction of NAV in stocks
    trades: pd.DataFrame            # every fill: date, ticker, side, shares, price, notional, reason
    round_trips: pd.DataFrame       # one row per position, closed or still open
    positions: pd.DataFrame         # open at the end, with stop and target prices
    market: pd.DataFrame            # daily market direction over the backtest window
    pending: dict                   # orders for the next open, from the last close


# ---------------------------------------------------------------- building blocks (pure functions)
def market_direction(close: np.ndarray, volume: np.ndarray, dist_limit: int = 5, dist_window: int = 25,
                     dist_decline: float = 0.002, ftd_gain: float = 0.0125, ftd_min_day: int = 4,
                     dist_expiry: float = 0.05) -> dict:
    """O'Neil's market-direction state machine on daily index closes and volume.

    Uptrend -> correction when distribution days (close down >= dist_decline on higher volume than
    the day before) since the last follow-through number dist_limit within dist_window sessions. A
    distribution day also stops counting once the index closes dist_expiry above it (IBD's rule;
    0 = off): without it, normal bull-market selling days add up to "corrections" every few weeks.
    In a correction, a rally attempt starts with the first up close after the lowest close; a new
    lower close resets it. A follow-through day is a close up >= ftd_gain on higher volume, on day
    ftd_min_day of the attempt or later, and starts a new uptrend. Starts in an uptrend.
    Returns boolean arrays uptrend / distribution / follow_through and the int array dist_count,
    each describing the state after that day's close.
    """
    n = len(close)
    prev = np.r_[np.nan, close[:-1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = close / prev - 1
    vol_up = np.r_[False, volume[1:] > volume[:-1]]  # NaN compares False
    is_dist = (ret <= -dist_decline) & vol_up
    up = np.zeros(n, bool)
    ftd = np.zeros(n, bool)
    count = np.zeros(n, int)
    state, low, day1 = True, np.inf, -1
    active: list[int] = []  # distribution days still counting
    for t in range(n):
        if is_dist[t]:
            active.append(t)
        active = [d for d in active if d > t - dist_window
                  and not (dist_expiry > 0 and close[t] >= close[d] * (1 + dist_expiry))]
        count[t] = len(active)
        if state:
            if count[t] >= dist_limit:
                state, low, day1 = False, close[t], -1
        elif close[t] < low:
            low, day1 = close[t], -1
        elif day1 < 0:
            if ret[t] > 0:
                day1 = t
        elif t - day1 + 1 >= ftd_min_day and ret[t] >= ftd_gain and vol_up[t]:
            state, active = True, []
            ftd[t] = True
            count[t] = 0
        up[t] = state
    return dict(uptrend=up, distribution=is_dist, follow_through=ftd, dist_count=count)


def rs_raw(close: pd.DataFrame) -> pd.DataFrame:
    """IBD-style relative-strength score: 0.4 r(3m) + 0.2 r(6m) + 0.2 r(9m) + 0.2 r(12m), so the
    latest quarter counts double."""
    r = lambda d: close / close.shift(d) - 1  # noqa: E731
    return 0.4 * r(63) + 0.2 * r(126) + 0.2 * r(189) + 0.2 * r(252)


def rs_rating(raw: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Percentile of raw among eligible stocks each day, scaled 1-99 (99 = strongest). NaN = not rated."""
    pct = pd.DataFrame(np.where(eligible, raw, np.nan)).rank(axis=1, pct=True).to_numpy()
    return np.clip(np.ceil(pct * 99), 1, 99).astype(np.float32)


def base_stats(window: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Base measurements for price windows (days x stocks, oldest first).

    pivot = the highest close; days = sessions from its latest occurrence to the end of the window
    (1 = it was the last row); depth = 1 - lowest close after the pivot / pivot (0 when none).
    """
    w = np.where(np.isfinite(window), window, -np.inf)
    n = len(w)
    last_max = n - 1 - np.argmax(w[::-1], axis=0)   # latest occurrence of the maximum
    pivot = w[last_max, np.arange(w.shape[1])]
    after = np.arange(n)[:, None] > last_max[None, :]
    low = np.where(after & np.isfinite(window), window, np.inf).min(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        depth = np.where(np.isfinite(low), 1 - low / pivot, 0.0)
    pivot = np.where(np.isfinite(pivot), pivot, np.nan)
    return pivot, n - last_max, depth


# ---------------------------------------------------------------- prepared data
@dataclass
class Prepared:
    m: bt.Market
    member: np.ndarray       # point-in-time S&P 500 membership
    rs: np.ndarray           # RS rating 1-99, NaN when not rated
    new_high: np.ndarray     # close above the highest close of the prior 52 weeks
    vol_ratio: np.ndarray    # volume / average volume of the prior 50 sessions (NaN = no volume data)
    index_volume: np.ndarray


@lru_cache(maxsize=1)
def prepare() -> Prepared:
    m = bt.load_market()
    vol = data.load_volume()
    if vol is None:
        raise FileNotFoundError("data/volume.parquet is missing: run `python -c 'from momentum import data; "
                                "data.fetch_volume()'` or use the app's update button")
    member = bt._membership("pit")
    C = pd.DataFrame(m.close_ff, index=m.dates, columns=m.tickers)
    raw = rs_raw(C).to_numpy()
    rs = rs_rating(raw, member & m.has_price & np.isfinite(raw))
    hi = C.rolling(BASE_LOOKBACK, min_periods=126).max().shift(1).to_numpy()
    with np.errstate(invalid="ignore"):
        new_high = m.has_price & (m.close_ff > hi)
    V = vol.reindex(index=m.dates, columns=m.tickers)
    ratio = (V / V.rolling(50, min_periods=20).mean().shift(1)).to_numpy(np.float32)
    iv = vol[data.INDEX].reindex(m.dates).to_numpy() if data.INDEX in vol.columns else np.full(len(m.dates), np.nan)
    return Prepared(m, member, rs, new_high, ratio, iv)


@lru_cache(maxsize=16)
def _market(dist_limit: int, dist_window: int, dist_decline: float, ftd_gain: float, ftd_min_day: int,
            dist_expiry: float) -> dict:
    p = prepare()
    return market_direction(p.m.index_close.to_numpy(), p.index_volume, dist_limit, dist_window,
                            dist_decline, ftd_gain, ftd_min_day, dist_expiry)


def market_state(cfg: Config) -> dict:
    """Daily market direction for cfg.market_rule; the distribution-day arrays are always included
    (from the book's rule) so they can be charted."""
    mk = dict(_market(cfg.dist_limit, cfg.dist_window, cfg.dist_decline, cfg.ftd_gain, cfg.ftd_min_day,
                      cfg.dist_expiry))
    if cfg.market_rule in ("sma200", "sma_slope"):
        px = prepare().m.index_close
        sma = px.rolling(200).mean()
        above = (px > sma) | sma.isna()
        if cfg.market_rule == "sma_slope":
            above |= sma >= sma.shift(21)
        mk["uptrend"] = above.to_numpy()
    elif cfg.market_rule == "off":
        mk["uptrend"] = np.ones(len(mk["uptrend"]), bool)
    return mk


def clear_caches() -> None:
    """Call after any data refresh (also clears the momentum engine's caches)."""
    bt.clear_caches()
    prepare.cache_clear()
    _market.cache_clear()


BREAKOUT_COLUMNS = ["col", "ticker", "rs", "close", "pivot", "base_days", "depth", "vol_ratio"]


def _breakout_arrays(cfg: Config, p: Prepared, s: int, exclude=()) -> dict[str, np.ndarray] | None:
    """Buyable breakouts at the close of day index s as arrays, strongest RS first (None = none)."""
    with np.errstate(invalid="ignore"):
        cand = np.flatnonzero(p.new_high[s] & p.member[s] & (p.rs[s] >= cfg.min_rs))
    if len(exclude):
        cand = cand[~np.isin(cand, list(exclude))]
    if cfg.min_volume_ratio > 0 and len(cand):
        cand = cand[~(p.vol_ratio[s, cand] < cfg.min_volume_ratio)]  # stocks without volume data pass
    if not len(cand):
        return None
    pivot, days, depth = base_stats(p.m.close_ff[max(0, s - BASE_LOOKBACK) : s, cand])
    ok = (days >= cfg.min_base_days) & (depth <= cfg.max_base_depth)
    if not ok.any():
        return None
    cand = cand[ok]
    rs, vr = p.rs[s, cand], p.vol_ratio[s, cand]
    order = np.lexsort((-np.nan_to_num(vr, nan=-1.0), -rs))  # RS first, then volume
    return dict(col=cand[order], rs=rs[order], pivot=pivot[ok][order], base_days=days[ok][order],
                depth=depth[ok][order], vol_ratio=vr[order], close=p.m.close_ff[s, cand[order]])


def breakouts(cfg: Config, p: Prepared, s: int, exclude=()) -> pd.DataFrame:
    """Stocks whose close on day index s is a buyable breakout, strongest RS first."""
    a = _breakout_arrays(cfg, p, s, exclude)
    if a is None:
        return pd.DataFrame(columns=BREAKOUT_COLUMNS)
    return pd.DataFrame({**a, "ticker": [p.m.tickers[j] for j in a["col"]]}, columns=BREAKOUT_COLUMNS)


# ---------------------------------------------------------------- engine
def run(cfg: Config) -> Result:
    p = prepare()
    m = p.m
    dates = m.dates
    start = max(dates.searchsorted(pd.Timestamp(cfg.start)), BASE_LOOKBACK + 1)
    end = len(dates) if cfg.end is None else dates.searchsorted(pd.Timestamp(cfg.end), side="right")
    mk = market_state(cfg)
    up = mk["uptrend"]
    c = cfg.cost_bps / 1e4
    tb = m.tbill_daily.to_numpy()
    slot = 1 / cfg.max_positions
    first_frac = 0.5 if cfg.pyramid else 1.0

    cash = cfg.initial_capital
    pos: dict[int, dict] = {}
    last_px: dict[int, float] = {}
    nav = np.full(len(dates), np.nan)
    invested = np.full(len(dates), np.nan)
    trade_rows, trips = [], []

    def px_open(t, j):
        o = m.open[t, j]
        return o if np.isfinite(o) and o > 0 else m.close_ff[t - 1, j]

    def sma50(s, j):
        return np.nanmean(m.close_ff[max(0, s - 49) : s + 1, j])

    def buy(t, j, amount, price, reason, info=None):
        nonlocal cash
        if amount <= 0 or not np.isfinite(price) or price <= 0:
            return
        q = amount * (1 - c) / price
        if j in pos:
            ps = pos[j]
            ps["basis"] = (ps["basis"] * ps["shares"] + price * q) / (ps["shares"] + q)
            ps["shares"] += q
            ps["cost"] += amount
            ps["adds"] += 1
        else:
            pos[j] = dict(shares=q, basis=price, entry_px=price, cost=amount, entry=t, hold_until=None,
                          adds=0, **(info or {}))
        cash -= amount
        trade_rows.append((dates[t], m.tickers[j], "BUY", q, price, amount, reason))

    def sell(t, j, price, reason):
        nonlocal cash
        ps = pos.pop(j)
        proceeds = ps["shares"] * price * (1 - c)
        cash += proceeds
        trade_rows.append((dates[t], m.tickers[j], "SELL", ps["shares"], price, ps["shares"] * price, reason))
        trips.append(_trip(m, j, ps, dates[t], price, proceeds / ps["cost"] - 1, t - ps["entry"], reason))

    def signals(s):
        """Orders for the open after day s: (sells, adds, buy candidates)."""
        sells, adds = [], []
        px = m.close_ff[s]
        for j, ps in pos.items():
            if not np.isfinite(px[j]):
                continue  # handled as "stopped trading"
            gain = px[j] / ps["basis"] - 1
            held = s - ps["entry"]
            if gain <= -cfg.stop_loss:
                sells.append((j, f"stop loss ({gain:+.1%})"))
                continue
            if cfg.on_correction == "sell_all" and not up[s]:
                sells.append((j, "market in correction"))
                continue
            if (cfg.fast_gain_rule and ps["hold_until"] is None and cfg.profit_target > 0
                    and gain >= cfg.profit_target and held < FAST_DAYS):
                ps["hold_until"] = ps["entry"] + HOLD_DAYS
            if ps["hold_until"] is not None:
                if s >= ps["hold_until"] and px[j] < sma50(s, j):
                    sells.append((j, f"below 50-day line after 8-week hold ({gain:+.0%})"))
                continue
            if cfg.profit_target > 0 and gain >= cfg.profit_target:
                sells.append((j, f"profit target ({gain:+.0%})"))
                continue
            if cfg.profit_target <= 0 and held >= HOLD_DAYS and px[j] < sma50(s, j):
                sells.append((j, f"below 50-day line ({gain:+.0%})"))
                continue
            if cfg.pyramid and ps["adds"] < len(PYRAMID) and held <= PYRAMID_DAYS:
                level, frac = PYRAMID[ps["adds"]]
                if px[j] >= ps["entry_px"] * (1 + level):
                    adds.append((j, frac, f"pyramid +{level:.1%}"))
        cands = _breakout_arrays(cfg, p, s, exclude=list(pos)) if up[s] else None
        return sells, adds, cands

    for t in range(start, end):
        cash *= 1 + tb[t]
        sells, adds, cands = signals(t - 1)
        for j, reason in sells:
            sell(t, j, px_open(t, j), reason)
        nav_open = cash + sum(ps["shares"] * px_open(t, j) for j, ps in pos.items())
        for j, frac, reason in adds:
            buy(t, j, min(frac * slot * nav_open, cash), px_open(t, j), reason)
        free = cfg.max_positions - len(pos)
        for i in range(len(cands["col"]) if cands is not None else 0):
            if free <= 0:
                break
            j = cands["col"][i]
            o = m.open[t, j]
            if not np.isfinite(o) or o <= 0 or o > cands["pivot"][i] * (1 + cfg.max_chase):
                continue  # gapped too far above the pivot: don't chase
            amount = min(first_frac * slot * nav_open, cash)
            if amount < 0.25 * first_frac * slot * nav_open:
                break  # out of cash
            buy(t, j, amount, o, f"breakout, RS {cands['rs'][i]:.0f}",
                {k: cands[k][i] for k in ("pivot", "rs", "base_days", "depth", "vol_ratio")})
            free -= 1
        # liquidate anything that stopped trading (delisting / acquisition) at its last price
        for j in list(pos):
            px = m.close_ff[t, j]
            if np.isfinite(px):
                last_px[j] = px
            else:
                sell(t, j, last_px.get(j, pos[j]["basis"]), "stopped trading")
        val = sum(ps["shares"] * m.close_ff[t, j] for j, ps in pos.items())
        nav[t] = cash + val
        invested[t] = val / nav[t] if nav[t] > 0 else 0.0

    t_last = end - 1
    sells, adds, _ = signals(t_last)
    cands = breakouts(cfg, p, t_last, exclude=list(pos)) if up[t_last] else pd.DataFrame(columns=BREAKOUT_COLUMNS)
    pending = dict(as_of=dates[t_last], uptrend=bool(up[t_last]),
                   sells=[(m.tickers[j], r) for j, r in sells],
                   adds=[(m.tickers[j], r) for j, _, r in adds],
                   buys=cands.drop(columns="col").head(max(cfg.max_positions - len(pos) + len(sells), 0)))
    open_trips = [_trip(m, j, ps, pd.NaT, m.close_ff[t_last, j],
                        ps["shares"] * m.close_ff[t_last, j] * (1 - c) / ps["cost"] - 1, t_last - ps["entry"], "open")
                  for j, ps in pos.items()]
    pos_rows = []
    for j, ps in pos.items():
        px = m.close_ff[t_last, j]
        hold = ps["hold_until"]
        pos_rows.append(dict(ticker=m.tickers[j], entry=dates[ps["entry"]], shares=ps["shares"], basis=ps["basis"],
                             price=px, value=ps["shares"] * px, weight=ps["shares"] * px / nav[t_last],
                             gain=px / ps["basis"] - 1, days=t_last - ps["entry"], stop=ps["basis"] * (1 - cfg.stop_loss),
                             target=np.nan if hold is not None or cfg.profit_target <= 0
                             else ps["basis"] * (1 + cfg.profit_target),
                             sma50=sma50(t_last, j), rs_now=p.rs[t_last, j],
                             rule="8-week hold" if hold is not None and t_last < hold
                             else "50-day line" if hold is not None or cfg.profit_target <= 0 else "stop / target"))
    positions = pd.DataFrame(pos_rows, columns=["ticker", "entry", "shares", "basis", "price", "value", "weight", "gain",
                                                "days", "stop", "target", "sma50", "rs_now", "rule"])

    sl = slice(start, end)
    nav_s = pd.Series(nav[sl], index=dates[sl], name=cfg.label)
    bench = m.bench_close.iloc[sl]
    bench = bench / bench.iloc[0] * cfg.initial_capital
    market = pd.DataFrame({"close": m.index_close.to_numpy()[sl], "uptrend": mk["uptrend"][sl],
                           "distribution": mk["distribution"][sl], "follow_through": mk["follow_through"][sl],
                           "dist_count": mk["dist_count"][sl]}, index=dates[sl])
    trades = pd.DataFrame(trade_rows, columns=["date", "ticker", "side", "shares", "price", "notional", "reason"])
    rt = pd.DataFrame(trips + open_trips, columns=TRIP_COLUMNS)
    return Result(cfg, nav_s, bench.rename("S&P 500 (SPY TR)"), m.tbill_daily.iloc[sl],
                  pd.Series(invested[sl], index=dates[sl]), trades, rt,
                  positions.sort_values("weight", ascending=False), market, pending)


TRIP_COLUMNS = ["ticker", "entry", "exit", "days", "entry_price", "exit_price", "return", "reason", "rs",
                "pivot", "base_days", "depth", "vol_ratio"]


def _trip(m, j, ps, exit_date, exit_px, ret, days, reason):
    return (m.tickers[j], m.dates[ps["entry"]], exit_date, days, ps["basis"], exit_px, ret, reason, ps.get("rs"),
            ps.get("pivot"), ps.get("base_days"), ps.get("depth"), ps.get("vol_ratio"))


def trade_stats(rt: pd.DataFrame) -> dict:
    """Win rate and payoff of closed positions."""
    closed = rt[rt["reason"] != "open"]
    if closed.empty:
        return {}
    r = closed["return"]
    wins, losses = r[r > 0], r[r <= 0]
    return {
        "Trades": len(closed),
        "Win rate": len(wins) / len(r),
        "Average gain": wins.mean() if len(wins) else np.nan,
        "Average loss": losses.mean() if len(losses) else np.nan,
        "Gain / loss ratio": wins.mean() / -losses.mean() if len(wins) and len(losses) and losses.mean() < 0 else np.nan,
        "Profit factor": wins.sum() / -losses.sum() if losses.sum() < 0 else np.nan,
        "Best trade": r.max(),
        "Worst trade": r.min(),
        "Average days held": closed["days"].mean(),
    }


def config_dict(cfg: Config) -> dict:
    return asdict(cfg)


# ---------------------------------------------------------------- live screen
def screen(cfg: Config, max_below_pivot: float = 0.15) -> pd.DataFrame:
    """Today's leaders (RS >= min_rs, point-in-time S&P 500 members) and where each stands:
    'breakout' = meets every buy rule at today's close; 'in buy zone' = above a proper base's pivot
    but not more than max_chase above; 'proper base' / 'base forming' = within max_below_pivot under
    the pivot of a base that has / hasn't lasted min_base_days yet; 'extended' = too far above."""
    p = prepare()
    m = p.m
    s = len(m.dates) - 1
    with np.errstate(invalid="ignore"):
        lead = np.flatnonzero(p.member[s] & (p.rs[s] >= cfg.min_rs))
    # base measured up to yesterday: today's close is compared against it
    pivot, days, depth = base_stats(m.close_ff[s - BASE_LOOKBACK : s, lead])
    close = m.close_ff[s, lead]
    dist = close / pivot - 1
    proper = (days >= cfg.min_base_days) & (depth <= cfg.max_base_depth)
    brk = set(breakouts(cfg, p, s)["col"])
    status = np.select(
        [np.isin(lead, list(brk)),
         proper & (dist > 0) & (dist <= cfg.max_chase),
         dist > 0,
         proper & (dist >= -max_below_pivot),
         (depth <= cfg.max_base_depth) & (dist >= -max_below_pivot)],
        ["breakout", "in buy zone", "extended", "proper base", "base forming"], "")
    sectors = data.load_sectors()
    out = pd.DataFrame({"ticker": [m.tickers[j] for j in lead], "sector": [sectors.get(m.tickers[j], "Unknown") for j in lead],
                        "rs": p.rs[s, lead], "close": close, "pivot": pivot, "vs_pivot": dist,
                        "base_weeks": days / 5, "depth": depth, "vol_ratio": p.vol_ratio[s, lead], "status": status})
    order = {"breakout": 0, "in buy zone": 1, "proper base": 2, "base forming": 3, "extended": 4, "": 5}
    out["_o"] = out["status"].map(order)
    return out.sort_values(["_o", "rs"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)


# ---------------------------------------------------------------- C and A (today only)
EARNINGS_FILE = data.DATA_DIR / "canslim_earnings.parquet"
MIN_EPS_GROWTH = 0.25    # C: latest quarter vs the same quarter a year earlier; A: each of the last 3 years
MIN_ROE = 0.17


def _yoy(row: pd.Series) -> float:
    """Latest quarter vs the same quarter a year earlier (NaN without one, or if that was a loss)."""
    row = row.dropna().sort_index(ascending=False)
    if row.empty:
        return np.nan
    d0, v0 = row.index[0], row.iloc[0]
    prior = row[(row.index <= d0 - pd.Timedelta(days=320)) & (row.index >= d0 - pd.Timedelta(days=410))]
    v1 = prior.iloc[0] if len(prior) else np.nan
    return (v0 - v1) / abs(v1) if np.isfinite(v1) and v1 > 0 else np.nan


def _row(df: pd.DataFrame, *names) -> pd.Series:
    for n in names:
        if df is not None and n in df.index:
            s = df.loc[n]
            s.index = pd.to_datetime(s.index)
            return pd.to_numeric(s, errors="coerce")
    return pd.Series(dtype=float)


def fetch_earnings(tickers: list[str], log=print) -> pd.DataFrame:
    """Latest quarterly and annual EPS growth, ROE and institutional ownership from Yahoo, merged
    into data/canslim_earnings.parquet. Yahoo serves ~5 quarters and ~4 years, enough for C and A today."""
    import yfinance as yf

    rows = []
    for n, t in enumerate(tickers, 1):
        log(f"earnings {n}/{len(tickers)}: {t}")
        try:
            tk = yf.Ticker(t)
            q, a = tk.quarterly_income_stmt, tk.income_stmt
            info = tk.info or {}
        except Exception as e:
            log(f"  {t} failed ({e})")
            continue
        eps_q = _yoy(_row(q, "Diluted EPS", "Basic EPS"))
        sales_q = _yoy(_row(q, "Total Revenue"))
        eps_a = _row(a, "Diluted EPS", "Basic EPS").dropna().sort_index()
        g = eps_a.pct_change().where(eps_a.shift() > 0).dropna().iloc[-3:]
        rows.append(dict(ticker=t, fetched=pd.Timestamp.now().normalize(),
                         latest_quarter=_row(q, "Diluted EPS", "Basic EPS").dropna().index.max(),
                         eps_q_growth=eps_q, sales_q_growth=sales_q,
                         eps_annual_min=g.min() if len(g) else np.nan, annual_years=len(g),
                         roe=info.get("returnOnEquity"), inst_own=info.get("heldPercentInstitutions")))
        time.sleep(0.2)
    new = pd.DataFrame(rows)
    old = load_earnings()
    out = pd.concat([old, new]).drop_duplicates("ticker", keep="last") if old is not None else new
    if not out.empty:
        out.to_parquet(EARNINGS_FILE)
    return out


def load_earnings() -> pd.DataFrame | None:
    return pd.read_parquet(EARNINGS_FILE) if EARNINGS_FILE.exists() else None


def earnings_checks(e: pd.DataFrame) -> pd.DataFrame:
    """Add pass / fail columns for C (quarterly EPS growth) and A (annual EPS growth and ROE)."""
    e = e.copy()
    e["C"] = np.where(e["eps_q_growth"].isna(), "?", np.where(e["eps_q_growth"] >= MIN_EPS_GROWTH, "✓", "✗"))
    a_ok = (e["eps_annual_min"] >= MIN_EPS_GROWTH) & (e["annual_years"] >= 3) & ~(e["roe"] < MIN_ROE)
    e["A"] = np.where(e["eps_annual_min"].isna(), "?", np.where(a_ok, "✓", "✗"))
    return e


# ---------------------------------------------------------------- data refresh
def refresh_prices(log=print) -> None:
    """Membership, recent prices (momentum.live) and volume. A partial bar for a session still in
    progress is dropped, so signals only ever use completed sessions."""
    from zoneinfo import ZoneInfo

    from . import live

    live.refresh_membership(log)
    live.update_prices(log)
    data.update_volume(log)
    today = pd.Timestamp(datetime.now(ZoneInfo("America/New_York")).date())
    if live.market_open():
        for name in ("open.parquet", "close.parquet", "volume.parquet"):
            f = data.DATA_DIR / name
            df = pd.read_parquet(f)
            if df.index[-1] >= today:
                df[df.index < today].to_parquet(f)
        log("market is open: dropped today's incomplete bar")
    clear_caches()


# ---------------------------------------------------------------- robustness CLI
PERIODS = {"Full": ("1996-02-01", None), "1996-2010": ("1996-02-01", "2010-12-31"), "2011-now": ("2011-01-01", None)}

VARIANTS: dict[str, dict] = {
    "CAN SLIM (baseline)": {},
    "M: hold positions in a correction": dict(on_correction="hold"),
    "M: off (ignore market direction)": dict(market_rule="off"),
    "M: index above 200-day SMA": dict(market_rule="sma200"),
    "M: bear only below a falling 200-day SMA": dict(market_rule="sma_slope"),
    "M: falling 200-day SMA, hold in a correction": dict(market_rule="sma_slope", on_correction="hold"),
    "M: distribution days without the 5% expiry": dict(dist_expiry=0.0),
    "M: 6 distribution days to end an uptrend": dict(dist_limit=6),
    "M: follow-through at +1.0%": dict(ftd_gain=0.01),
    "M: follow-through at +1.7%": dict(ftd_gain=0.017),
    "L: RS rating ≥ 90": dict(min_rs=90),
    "L: RS rating ≥ 70": dict(min_rs=70),
    "S: no volume confirmation": dict(min_volume_ratio=0.0),
    "N: base ≥ 5 weeks": dict(min_base_days=25),
    "Sell: stop loss 7%": dict(stop_loss=0.07),
    "Sell: no profit target (stop + 50-day line)": dict(profit_target=0.0),
    "Sell: profit target 25%": dict(profit_target=0.25),
    "Sell: no 8-week hold rule": dict(fast_gain_rule=False),
    "Size: 10 positions": dict(max_positions=10),
    "Size: 4 positions": dict(max_positions=4),
    "Size: pyramid into winners": dict(pyramid=True),
}


def split_summary(r: Result) -> dict[str, dict]:
    """Stats on the full window and on each half, cut from one run (positions carry across the cut)."""
    from . import metrics as mt

    out = {}
    for name, (s0, e0) in PERIODS.items():
        nav = r.nav.loc[s0:e0] if e0 else r.nav.loc[s0:]
        if len(nav) < 30:
            continue
        out[name] = mt.summary(nav, r.cash_rate, r.bench)
    return out


if __name__ == "__main__":
    from . import metrics as mt

    rows = []
    for name, kw in VARIANTS.items():
        t0 = time.time()
        r = run(Config(**kw))
        sp = split_summary(r)
        ts = trade_stats(r.round_trips)
        rows.append(dict(variant=name, cagr=sp["Full"]["CAGR"], sharpe=sp["Full"]["Sharpe"],
                         maxdd=sp["Full"]["Max drawdown"], cagr_96_10=sp["1996-2010"]["CAGR"],
                         cagr_11_now=sp["2011-now"]["CAGR"], trades=ts.get("Trades"), win=ts.get("Win rate"),
                         invested=r.invested.mean(), secs=time.time() - t0))
        print(rows[-1], flush=True)
    b = mt.summary(r.bench, r.cash_rate)
    print(pd.DataFrame(rows).to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"SPY  CAGR {b['CAGR']:.3f}  Sharpe {b['Sharpe']:.3f}  MaxDD {b['Max drawdown']:.3f}")
