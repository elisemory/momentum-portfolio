"""Momentum portfolio: current holdings and next trades for the Improved 12-1 strategy.

No strategy controls: the rules are fixed to the research winner (see FINDINGS.md).
Data refreshes automatically on the 1st of each month at 06:00 (and catches up on start-up if a
refresh was missed); the Update button refreshes on demand.

Run:  ./run_portfolio.sh      (http://localhost:8502)
"""
from __future__ import annotations

import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from momentum import data, live
from momentum import metrics as mt

st.set_page_config(page_title="Momentum Portfolio", layout="wide")

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GOOD, CRITICAL = "#006300", "#d03b3b"
LOG_TICKS = [v * 10 ** e for e in range(4, 10) for v in (1, 2, 5)]


def style(fig, height=380, yfmt=None, log=False):
    title = fig.layout.title.text
    if title:
        fig.update_layout(title=dict(x=0, xanchor="left", y=0.99, yanchor="top", font=dict(size=15, color=INK)))
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=78 if title else 30, b=8),
                      plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb", hovermode="x unified",
                      font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK2, size=13),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, title=None),
                      hoverlabel=dict(bgcolor="white", font_color=INK))
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont_color=MUTED)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont_color=MUTED, tickformat=yfmt)
    if log:
        fig.update_yaxes(type="log", tickvals=LOG_TICKS, tickformat="$~s")
    return fig


# ---------------------------------------------------------------- monthly auto-refresh
_lock = threading.Lock()


def run_refresh(log=print) -> None:
    with _lock:  # never run two refreshes at once (button + scheduler)
        live.refresh(log)


@st.cache_resource
def start_scheduler() -> threading.Thread:
    """One background thread per server: every 15 minutes, refresh if the 1st-of-month run is due."""
    def loop():
        while True:
            try:
                if live.should_refresh() and not _lock.locked():
                    try:
                        run_refresh(lambda *_: None)
                    except Exception as e:  # e.g. Yahoo blocking a cloud IP: use the published bundle
                        print(f"[scheduler] refresh failed ({e}); trying the published data bundle")
                        live.download_data(print)
            except Exception as e:  # keep the scheduler alive; next check retries
                print(f"[scheduler] refresh failed: {e}")
            time.sleep(15 * 60)
    th = threading.Thread(target=loop, name="monthly-refresh", daemon=True)
    th.start()
    return th



if not (data.DATA_DIR / "close.parquet").exists():
    with st.status("First run: loading price history…", expanded=True):
        if not live.download_data(st.write):  # published bundle; else build from scratch (slow)
            st.write("Downloading 30 years of prices from Yahoo (a few minutes)…")
            data.fetch_all(log=st.write)
            run_refresh(st.write)

start_scheduler()  # only once data exists, so it never races the first-run download


def data_version() -> tuple:
    files = [data.DATA_DIR / f for f in ("close.parquet", "sp500_hist.csv", "live_state.json")]
    return tuple(f.stat().st_mtime if f.exists() else 0 for f in files)


@st.cache_data(show_spinner="Computing portfolio…", max_entries=2)
def load_portfolio(version: tuple) -> dict:
    p = live.portfolio()
    r = p.pop("result")
    p["nav"], p["bench"], p["rf"] = r.nav, r.bench, r.cash_rate
    p["rebalances"] = r.rebalances
    return p


# ---------------------------------------------------------------- header
c1, c2 = st.columns([5, 1])
with c1:
    st.title("Momentum Portfolio")
    st.caption("Top 20 S&P 500 stocks by blended 12-1 / 6-1 momentum · hold while rank ≤ 40 · max 10% per stock · "
               "T-bills when the S&P 500 is below a falling 200-day average · rebalance on the first trading day "
               "of each month")
with c2:
    st.write("")
    if st.button("⟳ Update now", type="primary", width="stretch",
                 help="Download the latest prices and index changes, then recompute."):
        with st.status("Updating data…", expanded=True) as status:
            try:
                run_refresh(st.write)
                status.update(label="Updated", state="complete")
                ok = True
            except Exception as e:  # e.g. Yahoo returned no or stale prices: keep the cached data
                st.write(f"Update failed: {e}")
                status.update(label="Update failed; showing the cached data", state="error")
                ok = False
        if ok:
            st.cache_data.clear()
            st.rerun()


@st.fragment(run_every="10m")  # pick up scheduler refreshes without a manual reload
def main() -> None:
    p = load_portfolio(data_version())
    now = datetime.now()
    last = live.last_refresh()
    st.caption(f"Prices as of **{p['as_of']:%a %b %d, %Y}** · last update "
               f"{(last.strftime('%b %d %H:%M') if last else 'never')} · next automatic update "
               f"**{live.next_scheduled(now):%a %b %d, %H:%M}**")

    # ---- status tiles
    k = st.columns(5)
    k[0].metric("Market regime", "▲ Bull · invested" if p["bull"] else "▼ Bear · in T-bills")
    k[1].metric("S&P 500 vs 200-day avg", f"{(p['spx'] / p['sma'] - 1) * 100:+.1f}%",
                f"average is {'rising' if p['sma_rising'] else 'falling'}", delta_color="off", delta_arrow="off")
    k[2].metric("Next rebalance", f"{p['next_rebalance']:%a %b %d}",
                "trades are final" if p["final"] else "preview, final after month-end", delta_color="off", delta_arrow="off")
    k[3].metric("Holdings", f"{len(p['positions'])}", f"cash {p['cash_weight']:.1%}", delta_color="off", delta_arrow="off")
    y0 = period_start(p["nav"].index, "YTD")  # last close of the previous year
    ytd = lambda s: s.iloc[-1] / s[y0] - 1  # noqa: E731
    k[4].metric("Year to date", f"{ytd(p['nav']):+.1%}", f"S&P {ytd(p['bench']):+.1%}",
                delta_color="off", delta_arrow="off")

    # ---- next rebalance
    st.subheader(f"Next rebalance · {p['next_rebalance']:%A %B %d, %Y} at the open")
    if not p["final"]:
        st.caption("Preview based on the latest close. The final list is fixed by the last close of the month and "
                   "refreshes automatically on the 1st.")
    if not p["bull"]:
        st.error("**▼ Bear regime:** sell all stocks and hold T-bills (e.g. SGOV, BIL) until the regime turns bull.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**Sell ({len(p['sells'])})**")
        if p["sells"]:
            st.dataframe(pd.DataFrame(p["sells"], columns=["Ticker", "Why"]), hide_index=True, width="stretch")
        else:
            st.caption("Nothing to sell.")
    with c2:
        st.markdown(f"**Buy ({len(p['buys'])})**")
        if p["buys"]:
            b = pd.DataFrame(p["buys"], columns=["Ticker", "Why"])
            b["Sector"] = b["Ticker"].map(data.load_sectors()).fillna("—")
            st.dataframe(b, hide_index=True, width="stretch")
            st.caption("Each new name gets up to 5% of the portfolio, funded by the sales.")
        else:
            st.caption("Nothing to buy.")
    with c3:
        st.markdown(f"**Trim to 10% ({len(p['trims'])})**")
        if p["trims"]:
            st.dataframe(pd.DataFrame(p["trims"], columns=["Ticker", "Why"]), hide_index=True, width="stretch")
        else:
            st.caption("No position above 10%.")



@st.cache_data(ttl=55, show_spinner=False)
def load_quotes(tickers: tuple[str, ...]):
    return live.quotes(list(tickers))


def color_change(v):
    if not isinstance(v, (int, float)) or not np.isfinite(v) or v == 0:
        return ""
    return f"color: {GOOD}" if v > 0 else f"color: {CRITICAL}"


@st.fragment(run_every="1m")  # live prices: today's change refreshes every minute
def holdings_section() -> None:
    p = load_portfolio(data_version())
    pos = p["positions"].copy()
    st.subheader("Current holdings")
    if pos.empty:
        st.info(f"**All cash.** The portfolio holds no stocks (100% in T-bills) since the "
                f"{p['last_rebalance']:%b %d, %Y} rebalance.")
        return
    try:
        q, q_day = load_quotes(tuple(pos["ticker"]) + ("^GSPC", "^NDX"))
    except Exception as e:  # network hiccup: show the table without live prices
        q, q_day = None, None
        st.warning(f"Live quotes unavailable right now ({e}). Retrying every minute.")
    if q is not None:
        pos["price_now"] = pos["ticker"].map(q["price"])
        pos["today"] = pos["ticker"].map(q["change"])
        port_today = float((pos["weight"] * pos["today"].fillna(0)).sum())  # cash share earns ~0 in a day
        spx, ndx = float(q.loc["^GSPC", "change"]), float(q.loc["^NDX", "change"])
        is_open = live.market_open()
        session = "today" if is_open else f"last session ({q_day:%a %b %d})"
        k = st.columns(4)
        k[0].metric(f"Portfolio {'today' if is_open else 'last session'}", f"{port_today:+.2%}",
                    f"{(port_today - spx) * 100:+.2f} pts vs S&P")
        k[1].metric("S&P 500", f"{spx:+.2%}", session, delta_color="off", delta_arrow="off")
        k[2].metric("Nasdaq-100", f"{ndx:+.2%}", session, delta_color="off", delta_arrow="off")
        k[3].metric("Market", "● Open" if is_open else "○ Closed",
                    f"quotes updated {datetime.now():%H:%M:%S}", delta_color="off", delta_arrow="off")
    else:
        pos["price_now"], pos["today"] = np.nan, np.nan

    tab = pd.DataFrame({
        "Ticker": pos["ticker"], "Sector": pos["sector"], "Weight": pos["weight"],
        "Price": pos["price_now"], "Today": pos["today"],
        "Rank now": pos["rank"], "12-1 momentum": pos["momentum"], "6-1 momentum": pos["momentum_6_1"],
        "Held since": pos["entry"].dt.date, "Return since bought": pos["gain"],
        "Status": np.where(pos["ticker"].isin([t for t, _ in p["sells"]]), "sell next rebalance",
                           np.where(pos["rank"] <= 20, "top 20", "held by buffer")),
    })
    if q is None:
        tab = tab.drop(columns=["Price", "Today"])
    c1, c2 = st.columns([3, 1])
    with c1:
        styled = (tab.style.format({"Weight": "{:.1%}", "Price": "${:,.2f}", "Today": "{:+.2%}", "Rank now": "{:.0f}",
                                    "12-1 momentum": "{:+.0%}", "6-1 momentum": "{:+.0%}",
                                    "Return since bought": "{:+.1%}"}, na_rep="—")
                  .map(color_change, subset=[c for c in ["Today"] if c in tab.columns]))
        st.dataframe(styled, hide_index=True, width="stretch", height=36 * (len(tab) + 1) + 4)
        st.caption(f"Last rebalance {p['last_rebalance']:%b %d, %Y}. Today = change vs the previous close, "
                   "refreshed every minute (Yahoo quotes, may be delayed). Return since bought uses "
                   f"prices through {p['as_of']:%b %d}. The strategy does not trim winners except above 10%.")
    with c2:
        sec = pos.groupby("sector")["weight"].sum().sort_values()
        fig = go.Figure(go.Bar(x=sec.values, y=sec.index, orientation="h", marker_color=SERIES[0],
                               marker_line_width=0, hovertemplate="%{y}: %{x:.1%}<extra></extra>",
                               text=[f"{v:.0%}" for v in sec.values], textposition="outside",
                               textfont_color=INK2))
        fig.update_layout(title=dict(text="Sector weights"), showlegend=False, bargap=0.3, hovermode="closest")
        fig = style(fig, 90 + 56 * len(sec))
        fig.update_xaxes(showticklabels=False, showgrid=False, range=[0, sec.max() * 1.35])
        fig.update_yaxes(type="category", gridcolor="rgba(0,0,0,0)", tickfont_color=INK2)
        st.plotly_chart(fig, width="stretch", key="sector_chart")
        top = sec.idxmax()
        if sec.max() > 0.4:
            st.caption(f"⚠ {sec.max():.0%} in {top}: momentum baskets concentrate in leading sectors.")


def shade_bear(fig, rb: pd.DataFrame, x0: pd.Timestamp, x1: pd.Timestamp) -> None:
    """Gray bands where the strategy sat in T-bills, clipped to [x0, x1]."""
    bear = rb["regime"].eq("bear")
    for s0 in rb.index[bear & ~bear.shift(1, fill_value=False)]:
        after = rb.index[(rb.index > s0) & ~bear]
        s1 = after[0] if len(after) else x1
        if s1 > x0 and s0 < x1:
            fig.add_vrect(x0=max(s0, x0), x1=min(s1, x1), fillcolor="rgba(137,135,129,0.14)",
                          line_width=0, layer="below")


PERIODS = {"1D": None, "5D": None, "1M": 1, "3M": 3, "6M": 6, "YTD": None, "1Y": 12, "3Y": 36, "5Y": 60,
           "10Y": 120, "Max": None}


def period_start(idx: pd.DatetimeIndex, period: str) -> pd.Timestamp:
    """Base close for a trailing period ending at idx[-1] (Yahoo-style: 5D = five sessions back)."""
    end = idx[-1]
    if period in ("1D", "5D"):
        return idx[max(-len(idx), -1 - int(period[0]))]
    if period == "Max":
        return idx[0]
    target = pd.Timestamp(end.year, 1, 1) - pd.Timedelta(days=1) if period == "YTD" \
        else end - pd.DateOffset(months=PERIODS[period])
    before = idx[idx <= target]
    return before[-1] if len(before) else idx[0]


@st.fragment(run_every="10m")
def performance_section() -> None:
    p = load_portfolio(data_version())
    nav = p["nav"]
    ndx = data.load_nasdaq().reindex(nav.index, method="ffill")
    lines = {"Momentum portfolio": (nav, SERIES[0]), "S&P 500": (p["bench"], SERIES[1]),
             "Nasdaq-100": (ndx, SERIES[2])}
    idx, end = nav.index, nav.index[-1]
    st.subheader("Performance")

    # ---- every trailing period at a glance
    ret = lambda s, t0, t1: s[t1] / s[t0] - 1 if pd.notna(s[t0]) else np.nan  # noqa: E731
    grid = pd.DataFrame({per: {name: ret(s, period_start(idx, per), end) for name, (s, _) in lines.items()}
                         for per in PERIODS})
    st.dataframe(grid.style.format("{:+,.1%}", na_rep="—").map(color_change), width="stretch")

    # ---- chart for one period or a custom date range
    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    choice = c1.segmented_control("Period", list(PERIODS) + ["Custom"], default="1Y", key="perf_period",
                                  label_visibility="collapsed") or "1Y"
    if choice == "Custom":
        picked = c2.date_input("Date range", value=((end - pd.DateOffset(years=1)).date(), end.date()),
                               min_value=idx[0].date(), max_value=end.date(), key="perf_range",
                               format="YYYY-MM-DD", label_visibility="collapsed")
        if len(picked) < 2:
            st.caption("Pick an end date.")
            return
        on_or_before = lambda d: idx[idx <= pd.Timestamp(d)]  # noqa: E731
        t0 = on_or_before(picked[0])[-1] if len(on_or_before(picked[0])) else idx[0]
        t1 = on_or_before(picked[1])[-1]
        if t1 <= t0:
            st.caption("Pick a range that spans at least two trading days.")
            return
    else:
        t0, t1 = period_start(idx, choice), end

    fig = go.Figure()
    for name, (s, color) in lines.items():
        w = s[t0:t1]
        if w.isna().iloc[0]:
            continue
        r = w / w.iloc[0] - 1
        fig.add_scatter(x=r.index, y=r, name=f"{name} {r.iloc[-1]:+.1%}", line=dict(color=color, width=2),
                        mode="lines+markers" if len(r) <= 25 else "lines", hovertemplate=f"{name}: %{{y:+.2%}}<extra></extra>")
    shade_bear(fig, p["rebalances"], t0, t1)
    fig.add_hline(y=0, line_color=AXIS, line_width=1)
    fig.update_layout(title=dict(text=f"Return from {t0:%b %d, %Y} close to {t1:%b %d, %Y} · gray = in T-bills"))
    st.plotly_chart(style(fig, 400, yfmt="+.0%"), width="stretch", key="perf_chart")
    st.caption(f"Total returns through the {end:%b %d} close: the portfolio is the strategy's simulated "
               "daily value (it trades only at monthly rebalances), the S&P 500 is SPY and the Nasdaq-100 is "
               "QQQ, both with dividends reinvested. Periods are trailing from the last close, like Yahoo Finance.")


@st.fragment(run_every="10m")
def track_record() -> None:
    p = load_portfolio(data_version())
    # ---- track record
    st.subheader("Backtested track record")
    nav, bench = p["nav"], p["bench"]
    ndx = data.load_nasdaq().reindex(nav.index, method="ffill")
    ndx = ndx / ndx.iloc[0] * nav.iloc[0]
    S, B, N = (mt.summary(x, p["rf"]) for x in (nav, bench, ndx))
    k = st.columns(4)
    k[0].metric("CAGR since 1996", f"{S['CAGR']:.1%}", f"S&P {B['CAGR']:.1%} · Nasdaq {N['CAGR']:.1%}",
                delta_color="off", delta_arrow="off")
    k[1].metric("Sharpe", f"{S['Sharpe']:.2f}", f"S&P {B['Sharpe']:.2f} · Nasdaq {N['Sharpe']:.2f}", delta_color="off", delta_arrow="off")
    k[2].metric("Max drawdown", f"{S['Max drawdown']:.0%}",
                f"S&P {B['Max drawdown']:.0%} · Nasdaq {N['Max drawdown']:.0%}", delta_color="off", delta_arrow="off")
    last12 = nav.index >= nav.index[-1] - pd.Timedelta(days=365)
    r12 = lambda x: x[last12].iloc[-1] / x[last12].iloc[0] - 1  # noqa: E731
    k[3].metric("Last 12 months", f"{r12(nav):+.1%}", f"S&P {r12(bench):+.1%} · Nasdaq {r12(ndx):+.1%}",
                delta_color="off", delta_arrow="off")
    fig = go.Figure()
    fig.add_scatter(x=bench.index, y=bench, name="S&P 500", line=dict(color=SERIES[1], width=2),
                    hovertemplate="$%{y:,.0f}")
    fig.add_scatter(x=ndx.index, y=ndx, name="Nasdaq-100", line=dict(color=SERIES[2], width=2),
                    hovertemplate="$%{y:,.0f}")
    fig.add_scatter(x=nav.index, y=nav, name="Momentum portfolio", line=dict(color=SERIES[0], width=2),
                    hovertemplate="$%{y:,.0f}")
    shade_bear(fig, p["rebalances"], nav.index[0], nav.index[-1])
    fig.update_layout(title=dict(text="Growth of $100,000 since 1996 (log scale) · gray = in T-bills"))
    st.plotly_chart(style(fig, 420, log=True), width="stretch")

    with st.expander("Before you trade: what the research found"):
        st.markdown("""
- **Taxes matter a lot.** About two-thirds of gains are short-term. At a 30% rate the backtested CAGR falls from
  ~18% to ~14%, and since 2011 it roughly ties or trails buy-and-hold SPY after tax. Use a tax-advantaged account.
- **Since 2011 the Nasdaq-100 did better** (about 19% vs 16% a year) with less volatility. The strategy's edge over
  the full period comes mostly from avoiding the 2000–02 and 2008 crashes.
- **One-month crashes can't be avoided** by a monthly rule. COVID took the portfolio down 39% in 2020.
- **The basket can crash on its own.** It often concentrates in one theme. In June–July 2026 it fell 33% while the
  S&P 500 fell 2%, when the chip and memory stocks it held reversed together.
- **It's a backtest.** About 45 variants were tested before choosing these rules, so expect live results to be
  lower. Free price data also lacks most companies that failed before 2010, which flatters the early years. Full
  details are in `FINDINGS.md` and the research app (`./run.sh`).
- This is research output, not investment advice.
""")


main()
holdings_section()
performance_section()
track_record()
