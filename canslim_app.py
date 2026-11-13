"""Streamlit dashboard for a CAN SLIM backtest and daily screen (William O'Neil, "How to Make Money in
Stocks", 4th ed.).

Run:  ./run_canslim.sh      (http://localhost:8503)
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from momentum import backtest as bt
from momentum import canslim as cs
from momentum import data
from momentum import experiments as ex
from momentum import metrics as mt

st.set_page_config(page_title="CAN SLIM Backtest", layout="wide")

# ---------------------------------------------------------------- palette (same as the research app)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GOOD, CRITICAL = "#006300", "#d03b3b"
BEAR_SHADE = "rgba(137,135,129,0.14)"
LOG_TICKS = [v * 10 ** e for e in range(4, 10) for v in (1, 2, 5)]


def style(fig: go.Figure, height: int = 420, yfmt: str | None = None, log: bool = False) -> go.Figure:
    title = fig.layout.title.text
    if title:
        fig.update_layout(title=dict(x=0, xanchor="left", y=0.99, yanchor="top", pad=dict(t=4),
                                     font=dict(size=15, color=INK)))
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=78 if title else 36, b=8), plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK2, size=13),
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, title=None),
        hoverlabel=dict(bgcolor="white", font_color=INK),
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont_color=MUTED)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont_color=MUTED, tickformat=yfmt,
                     type="log" if log else "linear")
    if log:
        fig.update_yaxes(tickvals=LOG_TICKS, tickformat="$~s", minor=dict(showgrid=False))
    return fig


def pct(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def data_version() -> tuple:
    files = [data.DATA_DIR / f for f in ("close.parquet", "volume.parquet", "sp500_hist.csv")]
    return tuple(f.stat().st_mtime if f.exists() else 0 for f in files)


# ---------------------------------------------------------------- data presence
have_prices = (data.DATA_DIR / "close.parquet").exists() and (data.DATA_DIR / "sp500_hist.csv").exists()
have_volume = (data.DATA_DIR / "volume.parquet").exists()
with st.sidebar:
    st.header("Data")
    if have_prices and have_volume:
        m = bt.load_market()
        v_last = data.load_volume(supplement=False).index[-1]
        st.caption(f"Prices through **{m.dates[-1]:%b %d, %Y}** · volume through {v_last:%b %d} · "
                   f"{len(m.tickers)} stocks")
    if have_prices and st.button("⟳ Update prices & volume", type="primary", width="stretch",
                                 help="Index membership, recent prices and volume from Yahoo (1–3 minutes). "
                                      "Uses the same price update as the portfolio app."):
        with st.status("Updating data…", expanded=True) as status:
            try:
                cs.refresh_prices(st.write)
                status.update(label="Updated", state="complete")
                ok = True
            except Exception as e:  # e.g. Yahoo returned stale prices: keep the cached data
                st.write(f"Update failed: {e}")
                status.update(label="Update failed; showing the cached data", state="error")
                ok = False
        if ok:
            st.cache_data.clear()
            st.rerun()

if not have_prices:
    st.title("CAN SLIM Backtest")
    st.info("No price data yet. Start the research app (`./run.sh`) and click **Download / refresh data**, "
            "or run the portfolio app once.")
    st.stop()
if not have_volume:
    st.title("CAN SLIM Backtest")
    st.info("CAN SLIM needs daily volume (breakout volume, distribution and follow-through days), which the "
            "momentum apps don't download.")
    if st.button("Download volume history (about 2 minutes)", type="primary"):
        with st.status("Downloading volume…", expanded=True):
            data.fetch_volume(log=st.write)
        cs.clear_caches()
        st.rerun()
    st.stop()

# ---------------------------------------------------------------- sidebar: rules
MARKET_RULES = {
    "ftd": "Distribution & follow-through days (book)",
    "sma_slope": "Below a falling 200-day SMA",
    "sma200": "Below the 200-day SMA",
    "off": "Off: ignore the market",
}
PRESETS = {
    "As written in the book": {},
    "Book + falling-200-day market rule": dict(market_rule="sma_slope"),
}

with st.sidebar:
    st.header("Rules")
    preset = st.radio("Preset", list(PRESETS), key="preset",
                      help="The book's rules, or the same rules with the market filter from the momentum strategy "
                           "(the best single change in Compare variants). Change any knob below to test it.")
    d = cs.Config(**PRESETS[preset])
    P = lambda name: f"{preset}:{name}"  # noqa: E731  (each preset keeps its own widget state)

    c1, c2 = st.columns(2)
    last_full = m.dates[-1].to_period("M") - 1
    months = [str(x) for x in pd.period_range("1996-02", last_full, freq="M")]
    start_m = c1.selectbox("Start", months, index=months.index("2010-01"), key="start",
                           format_func=lambda x: pd.Period(x).strftime("%b %Y"),
                           help="Defaults to Jan 2010: before about 2010 the price data misses many companies "
                                "that left the index, especially ones that failed.")
    ends = ["latest"] + [str(x) for x in pd.period_range(start_m, last_full, freq="M")][::-1]
    end_m = c2.selectbox("End", ends, index=0, key="end",
                         format_func=lambda x: f"Latest ({m.dates[-1]:%b %d})" if x == "latest"
                         else "End of " + pd.Period(x).strftime("%b %Y"))
    start = pd.Period(start_m).start_time.date()
    end = None if end_m == "latest" else str(pd.Period(end_m).end_time.date())

    st.subheader("Buy: N · S · L")
    min_rs = st.slider("L: minimum RS rating", 50, 99, d.min_rs, key=P("rs"),
                       help="Relative strength 1–99 against S&P 500 members: 12-month return with the latest "
                            "quarter counted double. O'Neil: 80 or higher.")
    c1, c2 = st.columns(2)
    base_weeks = c1.number_input("N: base ≥ weeks", 3, 30, d.min_base_days // 5, key=P("base"),
                                 help="Weeks since the pivot (the prior 52-week closing high). Cup-with-handle: 7+.")
    depth = c2.number_input("N: max base depth", 0.10, 0.60, d.max_base_depth, 0.01, format="%.2f", key=P("depth"),
                            help="Deepest close in the base, below the pivot. O'Neil: about 33% at most.")
    c1, c2 = st.columns(2)
    chase = c1.number_input("Max % above pivot", 0.0, 0.20, d.max_chase, 0.01, format="%.2f", key=P("chase"),
                            help="Skip the buy if the next open is more than this above the pivot.")
    vol_ratio = c2.number_input("S: volume ≥ × avg", 0.0, 4.0, d.min_volume_ratio, 0.1, format="%.1f", key=P("vol"),
                                help="Breakout-day volume vs its 50-day average (0 = off). O'Neil: 40–50% above.")

    st.subheader("Market direction: M")
    market_rule = st.selectbox("Stop buying when…", list(MARKET_RULES), list(MARKET_RULES).index(d.market_rule),
                               format_func=MARKET_RULES.get, key=P("mrule"))
    ftd_rule = market_rule == "ftd"
    c1, c2 = st.columns(2)
    dist_limit = c1.number_input("Distribution days", 2, 10, d.dist_limit, key=P("dist"), disabled=not ftd_rule,
                                 help="This many distribution days within 25 sessions end an uptrend. A distribution "
                                      "day is an index close down ≥0.2% on higher volume than the day before.")
    dist_expiry = c2.number_input("Expire after index +", 0.0, 0.20, d.dist_expiry, 0.01, format="%.2f",
                                  key=P("exp"), disabled=not ftd_rule,
                                  help="A distribution day stops counting once the index closes this far above it "
                                       "(IBD's rule; 0 = off).")
    c1, c2 = st.columns(2)
    ftd_gain = c1.number_input("Follow-through ≥", 0.005, 0.03, d.ftd_gain, 0.0025, format="%.4f", key=P("ftd"),
                               disabled=not ftd_rule,
                               help="Index gain on higher volume, on day 4 or later of a rally attempt, that "
                                    "confirms a new uptrend.")
    ftd_day = c2.number_input("…on day ≥", 2, 10, d.ftd_min_day, key=P("ftdday"), disabled=not ftd_rule)
    on_corr = st.radio("In a correction", ["sell_all", "hold"], horizontal=True, key=P("oncorr"),
                       index=["sell_all", "hold"].index(d.on_correction), disabled=market_rule == "off",
                       format_func=lambda x: {"sell_all": "Sell everything", "hold": "Only stop buying"}[x])

    st.subheader("Sell rules")
    c1, c2 = st.columns(2)
    stop = c1.number_input("Stop loss", 0.02, 0.30, d.stop_loss, 0.01, format="%.2f", key=P("stop"),
                           help="Sell at the next open after a close this far below cost. O'Neil: 7–8%, no exceptions.")
    target = c2.number_input("Profit target (0 = off)", 0.0, 1.0, d.profit_target, 0.05, format="%.2f",
                             key=P("target"), help="O'Neil: take most profits at 20–25%. Off = hold until the stop "
                                                   "or, after 8 weeks, a close below the 50-day average.")
    fast = st.toggle("8-week hold for fast gainers", d.fast_gain_rule, key=P("fast"),
                     help="A stock up 20% within 3 weeks of the buy is held at least 8 weeks, then until it closes "
                          "below its 50-day average.")

    st.subheader("Sizing")
    max_pos = st.slider("Max positions", 1, 20, d.max_positions, key=P("npos"),
                        help="Each new buy gets an equal slot: 1/N of the portfolio. O'Neil: 4–6 for most accounts.")
    pyramid = st.toggle("Pyramid into winners", d.pyramid, key=P("pyr"),
                        help="Buy half a slot at the breakout; add 30% at +2.5% and 20% at +5% within 3 weeks.")
    cost_bps = st.number_input("Trading cost (bps per side)", 0.0, 100.0, d.cost_bps, step=5.0, key=P("cost"))

cfg = cs.Config(start=str(start), end=end, max_positions=max_pos, min_rs=min_rs, min_base_days=int(base_weeks) * 5,
                max_base_depth=depth, max_chase=chase, min_volume_ratio=vol_ratio, market_rule=market_rule,
                dist_limit=int(dist_limit), dist_expiry=dist_expiry, ftd_gain=ftd_gain, ftd_min_day=int(ftd_day),
                on_correction=on_corr, stop_loss=stop, profit_target=target, fast_gain_rule=fast, pyramid=pyramid,
                cost_bps=cost_bps)


@st.cache_data(show_spinner="Running backtest…", max_entries=64)
def run_cached(c: cs.Config, version: tuple) -> cs.Result:
    return cs.run(c)


@st.cache_data(show_spinner="Running the momentum strategy for comparison…", max_entries=16)
def momentum_cached(s: str, e: str | None, version: tuple) -> pd.Series:
    return bt.run(bt.Config(**ex.IMPROVED, start=s, end=e, label="Momentum (Improved)")).nav


@st.cache_data
def nasdaq_series(version: tuple) -> pd.Series:
    return data.load_nasdaq()


V = data_version()
r = run_cached(cfg, V)
nav, bench, rf = r.nav, r.bench, r.cash_rate
ndx = nasdaq_series(V).reindex(nav.index, method="ffill")
ndx = (ndx / ndx.iloc[0] * cfg.initial_capital).rename("Nasdaq-100")
mom = momentum_cached(str(start), end, V).reindex(nav.index).ffill()
mom = mom / mom.iloc[0] * cfg.initial_capital
S, B, N, MO = mt.summary(nav, rf, bench), mt.summary(bench, rf), mt.summary(ndx, rf), mt.summary(mom, rf)
T = cs.trade_stats(r.round_trips)
NAMES = {"canslim": "CAN SLIM", "spy": "S&P 500 (SPY, total return)", "ndx": "Nasdaq-100 (QQQ, total return)",
         "mom": "Momentum (Improved)"}

# ---------------------------------------------------------------- header + KPI tiles
st.title("CAN SLIM vs. S&P 500, Nasdaq & Momentum")
st.caption(f"{nav.index[0].date()} → {nav.index[-1].date()} · point-in-time S&P 500 members · signals on the close, "
           f"fills at the next open · {cost_bps:g} bps/side costs · idle cash earns the 13-week T-bill yield · "
           "C, A and I aren't in the backtest (no point-in-time earnings data); the Today tab checks them live")

years = (nav.index[-1] - nav.index[0]).days / 365.25
RET = "CAGR" if years >= 1 else "Total return"
RET_LABEL = "CAGR" if years >= 1 else "Return (not annualized)"
k = st.columns(6)
k[0].metric(RET_LABEL, pct(S[RET]), f"{(S[RET] - B[RET]) * 100:+.1f} pts vs S&P")
k[1].metric("Sharpe", f"{S['Sharpe']:.2f}", f"{S['Sharpe'] - B['Sharpe']:+.2f} vs S&P")
k[2].metric("Max drawdown", pct(S["Max drawdown"]), f"{(S['Max drawdown'] - B['Max drawdown']) * 100:+.1f} pts vs S&P")
k[3].metric("Win rate", pct(T.get("Win rate", np.nan), 0), f"{T.get('Trades', 0)} closed trades",
            delta_color="off", delta_arrow="off")
k[4].metric("Avg gain / avg loss", f"{pct(T.get('Average gain', np.nan))} / {pct(T.get('Average loss', np.nan))}",
            f"ratio {T.get('Gain / loss ratio', np.nan):.2f}", delta_color="off", delta_arrow="off")
k[5].metric("$100k became", f"${S['Final value'] / 1e3:,.0f}k", f"S&P: ${B['Final value'] / 1e3:,.0f}k",
            delta_color="off", delta_arrow="off")
st.caption(f"**vs Nasdaq-100:** {RET_LABEL} {(S[RET] - N[RET]) * 100:+.1f} pts ({pct(N[RET])}) · "
           f"**vs the Improved momentum strategy:** {(S[RET] - MO[RET]) * 100:+.1f} pts ({pct(MO[RET])}) · "
           f"time in stocks {pct(r.invested.mean(), 0)}")


def correction_spans(up: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(start, end) of each stretch where the market filter says 'don't buy'."""
    down = ~up.astype(bool)
    starts = up.index[down & ~down.shift(1, fill_value=False)]
    spans = []
    for s0 in starts:
        after = up.index[(up.index > s0) & ~down]
        spans.append((s0, after[0] if len(after) else up.index[-1]))
    return spans


def shade(fig, up: pd.Series):
    if cfg.market_rule == "off":
        return
    for s0, e0 in correction_spans(up):
        fig.add_vrect(x0=s0, x1=e0, fillcolor=BEAR_SHADE, line_width=0, layer="below")


full_mk = cs.market_state(cfg)
mk_all = pd.DataFrame(full_mk, index=bt.load_market().dates)
up_bt = mk_all["uptrend"].reindex(nav.index)

tabs = st.tabs(["Today", "Performance", "Trades", "Market direction", "Calendar returns", "Compare variants",
                "How it works", "Findings"])

with tabs[7]:
    st.markdown((Path(__file__).parent / "CANSLIM_FINDINGS.md").read_text())

# ---------------------------------------------------------------- today
with tabs[0]:
    live_r = r if end is None else run_cached(replace(cfg, end=None), V)
    pend = live_r.pending
    as_of = pend["as_of"]
    stale = (pd.Timestamp.now().normalize() - as_of).days
    st.markdown(f"Data as of the close on **{as_of:%a %b %d, %Y}**. Orders below are for the **next open**.")
    if stale > 4:
        st.warning(f"Prices are {stale} days old. Click **Update prices & volume** in the sidebar for today's signals.")
    st.caption("Yahoo often finalizes the S&P 500's volume the next morning, so a distribution or follow-through day "
               "on the latest session can appear only after the next update.")

    last = mk_all.iloc[-1]
    upd = mk_all["uptrend"]
    since = upd.index[(upd != upd.iloc[-1])]
    began = upd.index[upd.index > since[-1]][0] if len(since) else upd.index[0]
    ftds = mk_all.index[mk_all["follow_through"]]
    c = st.columns(4)
    c[0].metric("Market direction", "▲ Uptrend" if last["uptrend"] else "▼ Correction",
                f"{'buy breakouts' if last['uptrend'] else 'no new buys'} since {began:%b %d, %Y}",
                delta_color="off", delta_arrow="off")
    if ftd_rule:
        c[1].metric("Distribution days (25 sessions)", f"{int(last['dist_count'])}",
                    f"uptrend ends at {cfg.dist_limit}", delta_color="off", delta_arrow="off")
        c[2].metric("Last follow-through day", f"{ftds[-1]:%b %d, %Y}" if len(ftds) else "—")
    else:
        px_ = bt.load_market().index_close
        sma_ = px_.rolling(200).mean()
        c[1].metric("S&P 500 vs 200-day SMA", f"{(px_.iloc[-1] / sma_.iloc[-1] - 1) * 100:+.1f}%",
                    f"SMA {'rising' if sma_.iloc[-1] >= sma_.iloc[-22] else 'falling'} over a month",
                    delta_color="off", delta_arrow="off")
        c[2].metric("Distribution days (info only)", f"{int(last['dist_count'])}")
    c[3].metric("Simulated holdings", f"{len(live_r.positions)} of {cfg.max_positions}",
                f"cash {pct(1 - live_r.invested.iloc[-1], 0)}", delta_color="off", delta_arrow="off")

    st.subheader("Orders for the next open")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown(f"**Sell ({len(pend['sells'])})**")
        if pend["sells"]:
            st.dataframe(pd.DataFrame(pend["sells"], columns=["Ticker", "Why"]), hide_index=True, width="stretch")
        else:
            st.caption("No sell rule triggered at the last close.")
        if pend["adds"]:
            st.markdown("**Pyramid adds**")
            st.dataframe(pd.DataFrame(pend["adds"], columns=["Ticker", "Why"]), hide_index=True, width="stretch")
    with c2:
        b = pend["buys"]
        st.markdown(f"**Buy ({len(b)})**")
        if not pend["uptrend"]:
            st.caption("No buys: the market is in a correction.")
        elif b.empty:
            st.caption("No stock broke out of a proper base at the last close (or no free slots).")
        else:
            b = b.assign(buy_up_to=b["pivot"] * (1 + cfg.max_chase), base_weeks=b["base_days"] / 5)
            st.dataframe(b[["ticker", "rs", "close", "pivot", "buy_up_to", "base_weeks", "depth", "vol_ratio"]]
                         .style.format({"rs": "{:.0f}", "close": "${:,.2f}", "pivot": "${:,.2f}",
                                        "buy_up_to": "${:,.2f}", "base_weeks": "{:.0f}", "depth": "{:.0%}",
                                        "vol_ratio": "{:.1f}×"}, na_rep="—"),
                         hide_index=True, width="stretch")
            st.caption(f"Each gets 1/{cfg.max_positions} of the portfolio. Skip a stock if it opens above "
                       "'buy up to' (more than "
                       f"{cfg.max_chase:.0%} over the pivot). Check C and A in the screener before buying.")

    st.subheader("Simulated holdings")
    pos = live_r.positions
    if pos.empty:
        st.caption("No open positions.")
    else:
        st.dataframe(pos[["ticker", "entry", "days", "basis", "price", "gain", "weight", "stop", "target", "sma50",
                          "rs_now", "rule"]].style.format(
            {"entry": "{:%Y-%m-%d}", "basis": "${:,.2f}", "price": "${:,.2f}", "gain": "{:+.1%}", "weight": "{:.1%}",
             "stop": "${:,.2f}", "target": "${:,.2f}", "sma50": "${:,.2f}", "rs_now": "{:.0f}"}, na_rep="—"),
            hide_index=True, width="stretch")
    st.caption(f"Positions the rules would hold today if started on {start:%b %Y}. A different start date gives "
               "different holdings, because each buy depends on which slots were free.")

    st.subheader("Screener: leaders (RS ≥ " f"{cfg.min_rs}) and where they stand")
    scr = cs.screen(cfg)
    shown = st.multiselect("Show", ["breakout", "in buy zone", "proper base", "base forming", "extended"],
                           ["breakout", "in buy zone", "proper base"], key="scr_status",
                           help="breakout: meets every buy rule at the last close · in buy zone: up to "
                                f"{cfg.max_chase:.0%} above a proper base's pivot · proper base: within 15% below the "
                                "pivot of a base that's long enough · base forming: not long enough yet · "
                                "extended: too far above the pivot to buy")
    scr = scr[scr["status"].isin(shown)]
    e = cs.load_earnings()
    cols = ["ticker", "status", "sector", "rs", "close", "pivot", "vs_pivot", "base_weeks", "depth", "vol_ratio"]
    if e is not None:
        e = cs.earnings_checks(e)
        scr = scr.merge(e[["ticker", "C", "A", "eps_q_growth", "sales_q_growth", "eps_annual_min", "roe",
                           "inst_own", "fetched"]], on="ticker", how="left")
        cols += ["C", "A", "eps_q_growth", "sales_q_growth", "eps_annual_min", "roe", "inst_own"]
    st.dataframe(scr[cols].style.format(
        {"rs": "{:.0f}", "close": "${:,.2f}", "pivot": "${:,.2f}", "vs_pivot": "{:+.1%}", "base_weeks": "{:.0f}",
         "depth": "{:.0%}", "vol_ratio": "{:.1f}×", "eps_q_growth": "{:+.0%}", "sales_q_growth": "{:+.0%}",
         "eps_annual_min": "{:+.0%}", "roe": "{:.0%}", "inst_own": "{:.0%}"}, na_rep="—"),
        hide_index=True, width="stretch", height=min(36 * (len(scr) + 1) + 4, 640),
        column_config={"eps_q_growth": st.column_config.TextColumn("EPS q/q-4", help="C: latest quarter's EPS vs "
                                                                   "the same quarter a year earlier. Want ≥ 25%."),
                       "sales_q_growth": st.column_config.TextColumn("Sales q/q-4"),
                       "eps_annual_min": st.column_config.TextColumn("Min annual EPS growth",
                                                                     help="A: weakest of the last 3 years. Want ≥ 25%."),
                       "roe": st.column_config.TextColumn("ROE", help="A: want ≥ 17%."),
                       "inst_own": st.column_config.TextColumn("Inst. own", help="I: share held by institutions.")})
    c1, c2 = st.columns([1, 3])
    todo = scr["ticker"].tolist()[:60]
    if c1.button(f"Check earnings (C & A) for {len(todo)} stocks", disabled=not todo,
                 help="Quarterly and annual EPS from Yahoo, about 1 second per stock. Results are saved."):
        with st.status("Fetching earnings…", expanded=True) as status:
            cs.fetch_earnings(todo, log=st.write)
            status.update(label="Done", state="complete")
        st.rerun()
    c2.caption("C ✓ = latest quarter's EPS up ≥ 25% on a year earlier. A ✓ = EPS up ≥ 25% in each of the last 3 "
               "years and ROE ≥ 17%. ? = Yahoo doesn't have enough history. Earnings are today's figures, so they "
               "aren't part of the backtest." + (f" Last fetched {e['fetched'].max():%b %d}." if e is not None else ""))

# ---------------------------------------------------------------- performance
with tabs[1]:
    log = st.toggle("Log scale", True)
    fig = go.Figure()
    for s_, key, col in ((bench, "spy", SERIES[1]), (ndx, "ndx", SERIES[2]), (mom, "mom", SERIES[3]),
                         (nav, "canslim", SERIES[0])):
        fig.add_scatter(x=s_.index, y=s_, name=NAMES[key], line=dict(color=col, width=2), hovertemplate="$%{y:,.0f}")
    shade(fig, up_bt)
    fig.update_layout(title=dict(text="Growth of $100,000 · gray bands = market filter says don't buy"))
    st.plotly_chart(style(fig, 460, "$,.0f", log), width="stretch")

    fig = go.Figure()
    for s_, key, col in ((bench, "spy", SERIES[1]), (ndx, "ndx", SERIES[2]), (mom, "mom", SERIES[3]),
                         (nav, "canslim", SERIES[0])):
        fig.add_scatter(x=s_.index, y=mt.drawdown(s_), name=NAMES[key], line=dict(color=col, width=2),
                        hovertemplate="%{y:.1%}")
    fig.update_layout(title=dict(text="Drawdown from previous peak"))
    st.plotly_chart(style(fig, 300, ".0%"), width="stretch")

    c1, c2 = st.columns([3, 2])
    with c1:
        inv = r.invested.rolling(21, min_periods=1).mean()
        fig = go.Figure()
        fig.add_scatter(x=inv.index, y=inv, name="Share in stocks", fill="tozeroy", line=dict(color=SERIES[0], width=1.5),
                        fillcolor="rgba(42,120,214,0.15)", hovertemplate="%{y:.0%}")
        fig.update_layout(title=dict(text="Share of portfolio in stocks (1-month average; rest in T-bills)"),
                          showlegend=False)
        fig.update_yaxes(range=[0, 1.02])
        st.plotly_chart(style(fig, 300, ".0%"), width="stretch")
    with c2:
        st.markdown("**Full statistics**")
        rows = []
        for key in S:
            fmt = (lambda v: f"${v:,.0f}") if key == "Final value" else \
                  (lambda v: f"{v:,.0f}") if "days" in key else \
                  (lambda v: f"{v:.2f}") if key in ("Sharpe", "Sortino", "Calmar", "Beta vs S&P", "Correlation vs S&P") \
                  else pct
            rows.append((key, fmt(S[key]), fmt(B[key]) if key in B else "—", fmt(N[key]) if key in N else "—",
                         fmt(MO[key]) if key in MO else "—"))
        rows.append(("Time invested in stocks", pct(r.invested.mean(), 0), "100%", "100%", "—"))
        st.dataframe(pd.DataFrame(rows, columns=["Metric", "CAN SLIM", "S&P 500", "Nasdaq-100", "Momentum"])
                     .set_index("Metric"), width="stretch", height=520)

# ---------------------------------------------------------------- trades
with tabs[2]:
    rt = r.round_trips
    closed = rt[rt["reason"] != "open"]
    if closed.empty:
        st.info("No closed trades in this window.")
    else:
        k = st.columns(5)
        k[0].metric("Closed trades", f"{T['Trades']}", f"{T['Trades'] / years:.0f} per year", delta_color="off",
                    delta_arrow="off")
        k[1].metric("Win rate", pct(T["Win rate"], 0))
        k[2].metric("Profit factor", f"{T['Profit factor']:.2f}", help="Sum of gains ÷ sum of losses (above 1 = profitable)")
        k[3].metric("Best / worst", f"{pct(T['Best trade'], 0)} / {pct(T['Worst trade'], 0)}")
        k[4].metric("Average days held", f"{T['Average days held']:.0f}", "trading days", delta_color="off",
                    delta_arrow="off")

        c1, c2 = st.columns([3, 2])
        with c1:
            fig = go.Figure()
            bins = dict(start=-0.30, end=1.0, size=0.02)
            fig.add_histogram(x=closed.loc[closed["return"] <= 0, "return"], name="Losses", xbins=bins,
                              marker_color=SERIES[7], hovertemplate="%{x}: %{y} trades")
            fig.add_histogram(x=closed.loc[closed["return"] > 0, "return"], name="Gains", xbins=bins,
                              marker_color=SERIES[0], hovertemplate="%{x}: %{y} trades")
            fig.update_layout(barmode="overlay", bargap=0.05, title=dict(text="Return per trade, after costs"))
            fig.update_traces(marker_line_width=0)
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(style(fig, 340), width="stretch")
        with c2:
            reasons = closed.assign(exit=closed["reason"].str.replace(r" \(.*\)$", "", regex=True))
            g = reasons.groupby("exit")["return"].agg(trades="count", avg_return="mean")
            g["share"] = g["trades"] / g["trades"].sum()
            st.markdown("**Why positions were sold**")
            st.dataframe(g.sort_values("trades", ascending=False)[["trades", "share", "avg_return"]].style.format(
                {"share": "{:.0%}", "avg_return": "{:+.1%}"}), width="stretch")
            by_year = closed.groupby(closed["exit"].dt.year)["return"].agg(
                trades="count", win_rate=lambda x: (x > 0).mean(), avg_return="mean")
            st.markdown("**By year (exit date)**")
            st.dataframe(by_year.style.format({"win_rate": "{:.0%}", "avg_return": "{:+.1%}"}), width="stretch",
                         height=240)

    st.markdown("**All positions** (open ones at the bottom, marked 'open')")
    st.dataframe(rt.assign(base_weeks=rt["base_days"] / 5)[
        ["ticker", "entry", "exit", "days", "entry_price", "exit_price", "return", "reason", "rs", "pivot", "base_weeks",
         "depth", "vol_ratio"]].style.format(
        {"entry": "{:%Y-%m-%d}", "exit": lambda x: "" if pd.isna(x) else f"{x:%Y-%m-%d}", "entry_price": "${:,.2f}",
         "exit_price": "${:,.2f}", "return": "{:+.1%}", "rs": "{:.0f}", "pivot": "${:,.2f}", "base_weeks": "{:.0f}",
         "depth": "{:.0%}", "vol_ratio": "{:.1f}×"}, na_rep="—"), hide_index=True, width="stretch", height=480)
    st.download_button("Download all fills (CSV)", r.trades.to_csv(index=False), "canslim_trades.csv")

# ---------------------------------------------------------------- market direction
with tabs[3]:
    span = st.radio("Window", ["1 year", "3 years", "Backtest window"], horizontal=True, key="mk_span")
    lo = {"1 year": mk_all.index[-1] - pd.DateOffset(years=1), "3 years": mk_all.index[-1] - pd.DateOffset(years=3),
          "Backtest window": nav.index[0]}[span]
    w = mk_all.loc[lo:]
    w = w.assign(close=bt.load_market().index_close.reindex(w.index))
    fig = go.Figure()
    fig.add_scatter(x=w.index, y=w["close"], name="S&P 500", line=dict(color=INK2, width=1.5),
                    hovertemplate="%{y:,.0f}")
    dd_ = w[w["distribution"]]
    fig.add_scatter(x=dd_.index, y=dd_["close"], mode="markers", name="Distribution day",
                    marker=dict(color=SERIES[7], size=7, symbol="triangle-down"), hovertemplate="%{y:,.0f}")
    ft_ = w[w["follow_through"]]
    fig.add_scatter(x=ft_.index, y=ft_["close"], mode="markers", name="Follow-through day",
                    marker=dict(color=GOOD, size=11, symbol="triangle-up"), hovertemplate="%{y:,.0f}")
    shade(fig, w["uptrend"])
    fig.update_layout(title=dict(text=f"S&P 500 · gray bands = {MARKET_RULES[cfg.market_rule].lower()}: no new buys"))
    st.plotly_chart(style(fig, 460, ",.0f"), width="stretch")
    fig = go.Figure()
    fig.add_scatter(x=w.index, y=w["dist_count"], name="Distribution days", line=dict(color=SERIES[7], width=1.5, shape="hv"),
                    hovertemplate="%{y}")
    fig.add_hline(y=cfg.dist_limit, line_color=AXIS, line_dash="dot")
    fig.update_layout(title=dict(text=f"Distribution days still counting (uptrend ends at {cfg.dist_limit})"),
                      showlegend=False)
    st.plotly_chart(style(fig, 220), width="stretch")
    yrs = mk_all.loc[nav.index[0]:]
    st.caption(f"Over the backtest window the market filter allowed buying {yrs['uptrend'].mean():.0%} of the time, "
               f"with {len(correction_spans(yrs['uptrend']))} separate corrections. The book's distribution-day rule "
               "fires often: the S&P 500 has about 50 distribution days a year, so 5 in 25 sessions is common even in "
               "strong markets.")

# ---------------------------------------------------------------- calendar
with tabs[4]:
    yr = mt.yearly_returns({"CAN SLIM": nav, "S&P 500": bench, "Nasdaq-100": ndx, "Momentum": mom})
    fig = go.Figure()
    for name, col in (("CAN SLIM", SERIES[0]), ("S&P 500", SERIES[1]), ("Nasdaq-100", SERIES[2]), ("Momentum", SERIES[3])):
        fig.add_bar(x=yr.index, y=yr[name], name=name, marker_color=col, hovertemplate="%{y:.1%}")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, title=dict(text="Calendar-year returns"))
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(style(fig, 380, ".0%"), width="stretch")
    st.caption(f"CAN SLIM beat the S&P in {(yr['CAN SLIM'] > yr['S&P 500']).sum()} of {len(yr)} calendar years.")
    st.markdown("**Monthly returns**")
    st.dataframe(mt.monthly_table(nav).style.format("{:.1%}", na_rep="").background_gradient(
        cmap="RdBu", vmin=-0.10, vmax=0.10, axis=None), width="stretch", height=420)

# ---------------------------------------------------------------- how it works
with tabs[6]:
    st.markdown(f"""
### The rules, letter by letter

| | O'Neil's rule | In this backtest |
|---|---|---|
| **C** | Current quarterly EPS up 25%+ on a year earlier | **Not tested.** Free data has no point-in-time earnings history. The screener checks it for today's stocks. |
| **A** | Annual EPS up 25%+ in each of the last 3 years, ROE 17%+ | **Not tested**, same reason. Checked live in the screener. |
| **N** | New highs, bought as the stock breaks out of a sound base | Buy the first close above the highest close of the prior 52 weeks, after a base of at least {cfg.min_base_days // 5} weeks that fell no more than {cfg.max_base_depth:.0%}. Fill at the next open, skipped if that open is more than {cfg.max_chase:.0%} above the pivot. |
| **S** | Supply and demand: big volume on the breakout | Breakout-day volume at least {cfg.min_volume_ratio:.1f}× its 50-day average. |
| **L** | Leaders: RS rating 80+ | RS rating 1–99 vs point-in-time S&P 500 members: 0.4 × 3-month + 0.2 × 6-, 9- and 12-month returns. Minimum {cfg.min_rs}. When several stocks break out on the same day, the highest RS goes first. |
| **I** | Increasing ownership by good funds | **Not tested.** The screener shows today's institutional ownership. |
| **M** | Market direction: buy only in a confirmed uptrend | {MARKET_RULES[cfg.market_rule]}. Book rule: {cfg.dist_limit} distribution days (index down 0.2%+ on higher volume) within 25 sessions end the uptrend; a distribution day stops counting after the index gains {cfg.dist_expiry:.0%}. A follow-through day (index up {cfg.ftd_gain:.2%}+ on higher volume, day {cfg.ftd_min_day}+ of a rally attempt) starts a new one. |

**Selling:** sell at the next open after a close {cfg.stop_loss:.0%} below cost; take profits at +{cfg.profit_target:.0%}; a stock up 20% within
3 weeks is held at least 8 weeks and then until it closes below its 50-day average. In a correction the backtest
{"sells everything" if cfg.on_correction == "sell_all" else "keeps positions but stops buying"}.

**Sizing:** up to {cfg.max_positions} positions, each bought with 1/{cfg.max_positions} of the portfolio. Idle cash earns T-bill interest.

### Caveats

- **Only half of CAN SLIM is tested.** C, A and I, the earnings and ownership screens, are the heart of the book. O'Neil's
  big winners combined explosive earnings with price leadership, and this backtest only sees the price side.
- **Universe.** Only stocks in the S&P 500 at the time. O'Neil's winners were often smaller, younger growth companies
  that weren't in the index yet.
- **Closing prices only.** No intraday highs or lows: breakouts and stops trigger on closes and fill at the next open, so a
  gap down can lose more than the stop, and buys happen a day after O'Neil's intraday entry.
- **Market direction is a judgment call.** IBD's market calls involve discretion (stalling days, leading-stock behavior).
  The mechanical distribution-day count here whipsaws much more than IBD's published calls.
- **Survivorship.** Before about 2010 the price data misses many companies that left the index, especially failures.
- **Volume.** Yahoo daily volume (split-adjusted), plus the Quandl WIKI archive for 184 former members Yahoo dropped.
""")
    st.markdown("**Configuration used**")
    st.json(cs.config_dict(cfg))

# ---------------------------------------------------------------- variants (rendered last: slow on first load)
with tabs[5]:
    st.markdown(f"Each row changes **one rule** of the **{preset.lower()}** preset (other sidebar settings are ignored "
                "here, except trading cost), from Feb 1996 to today. Each row is scored on the full period and on two "
                "halves cut from the same run. A change is only credible if it helps in **both** halves. "
                "▲ / ▼ = Sharpe more than 0.05 above / below the baseline in that half.")
    base = cs.Config(**PRESETS[preset], cost_bps=cfg.cost_bps)
    rows, navs = [], {}
    prog = st.progress(0.0, "Running variants… (about 15 seconds on first load, then cached)")
    for i, (name, kw) in enumerate(cs.VARIANTS.items()):
        rv = run_cached(replace(base, **kw), V)
        rows.append((name, cs.split_summary(rv), cs.trade_stats(rv.round_trips), rv.invested.mean()))
        navs[name] = rv.nav
        prog.progress((i + 1) / len(cs.VARIANTS), f"Running variants… {name}")
    prog.empty()
    bench_full = rv.bench
    halves = list(cs.PERIODS)[1:]
    b_sum = {p_: mt.summary(bench_full.loc[s0:e0] if e0 else bench_full.loc[s0:], rv.cash_rate)
             for p_, (s0, e0) in cs.PERIODS.items()}
    mom_full = momentum_cached("1996-02-01", None, V)
    m_sum = {p_: mt.summary(mom_full.loc[s0:e0] if e0 else mom_full.loc[s0:], rv.cash_rate)
             for p_, (s0, e0) in cs.PERIODS.items()}

    def mark(h, ref):
        d_ = h["Sharpe"] - ref["Sharpe"]
        return "▲" if d_ > 0.05 else "▼" if d_ < -0.05 else "≈"

    out = []
    for name, sp, ts, inv in rows:
        verdict = "—" if name == rows[0][0] else (f"96–10 {mark(sp[halves[0]], rows[0][1][halves[0]])}  ·  "
                                                  f"11–now {mark(sp[halves[1]], rows[0][1][halves[1]])}")
        out.append({"Variant": name, "CAGR": sp["Full"]["CAGR"], "Sharpe": sp["Full"]["Sharpe"],
                    "Max DD": sp["Full"]["Max drawdown"], "CAGR 96–10": sp[halves[0]]["CAGR"],
                    "CAGR 11–now": sp[halves[1]]["CAGR"],
                    "vs S&P 11–now": sp[halves[1]]["CAGR"] - b_sum[halves[1]]["CAGR"],
                    "Trades": ts.get("Trades"), "Win rate": ts.get("Win rate"), "In stocks": inv,
                    "Sharpe vs baseline (by half)": verdict})
    for name, ss in (("S&P 500 (SPY)", b_sum), ("Momentum (Improved)", m_sum)):
        out.append({"Variant": name, "CAGR": ss["Full"]["CAGR"], "Sharpe": ss["Full"]["Sharpe"],
                    "Max DD": ss["Full"]["Max drawdown"], "CAGR 96–10": ss[halves[0]]["CAGR"],
                    "CAGR 11–now": ss[halves[1]]["CAGR"],
                    "vs S&P 11–now": ss[halves[1]]["CAGR"] - b_sum[halves[1]]["CAGR"], "Trades": None,
                    "Win rate": None, "In stocks": None, "Sharpe vs baseline (by half)": ""})
    tab = pd.DataFrame(out)
    st.dataframe(tab.style.format({"CAGR": "{:.1%}", "Sharpe": "{:.2f}", "Max DD": "{:.1%}", "CAGR 96–10": "{:.1%}",
                                   "CAGR 11–now": "{:.1%}", "vs S&P 11–now": "{:+.1%}", "Trades": "{:.0f}",
                                   "Win rate": "{:.0%}", "In stocks": "{:.0%}"}, na_rep="—"),
                 width="stretch", height=36 * (len(tab) + 1) + 4, hide_index=True,
                 column_config={"Variant": st.column_config.TextColumn(width="large")})
    names = list(cs.VARIANTS)
    pick = st.multiselect("Plot (max 6, colors stay fixed per variant)", names,
                          [names[0], "M: off (ignore market direction)", "M: bear only below a falling 200-day SMA"],
                          max_selections=6)
    fig = go.Figure()
    fig.add_scatter(x=bench_full.index, y=bench_full, name="S&P 500", line=dict(color=MUTED, width=2, dash="dot"),
                    hovertemplate="$%{y:,.0f}")
    mf = mom_full / mom_full.iloc[0] * 100_000
    fig.add_scatter(x=mf.index, y=mf, name="Momentum (Improved)", line=dict(color=MUTED, width=2, dash="dash"),
                    hovertemplate="$%{y:,.0f}")
    color_of = {n: SERIES[[0, 2, 3, 4, 5, 6, 7, 1][i % 8]] for i, n in enumerate(names)}
    for n in pick:
        fig.add_scatter(x=navs[n].index, y=navs[n], name=n, line=dict(color=color_of[n], width=2),
                        hovertemplate="$%{y:,.0f}")
    fig.update_layout(title=dict(text="Growth of $100,000, 1996 → today (log scale)"))
    st.plotly_chart(style(fig, 480, "$,.0f", True), width="stretch")
