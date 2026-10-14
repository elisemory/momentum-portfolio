"""Streamlit dashboard for the 12-1 momentum strategy backtest.

Run:  .venv/bin/streamlit run app.py
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from momentum import backtest as bt
from momentum import data
from momentum import experiments as ex
from momentum import metrics as mt

st.set_page_config(page_title="12-1 Momentum Backtest", layout="wide")

# ---------------------------------------------------------------- palette (dataviz reference instance, light)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BEAR_SHADE = "rgba(137,135,129,0.14)"


LOG_TICKS = [v * 10 ** e for e in range(4, 10) for v in (1, 2, 5)]


def style(fig: go.Figure, height: int = 420, yfmt: str | None = None, log: bool = False) -> go.Figure:
    title = fig.layout.title.text
    if title:  # title on its own line, legend underneath it
        fig.update_layout(title=dict(x=0, xanchor="left", y=0.99, yanchor="top", pad=dict(t=4)))
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


# ---------------------------------------------------------------- data presence
have_data = (data.DATA_DIR / "close.parquet").exists() and (data.DATA_DIR / "sp500_hist.csv").exists()
with st.sidebar:
    st.header("Data")
    if have_data:
        m = bt.load_market()
        st.caption(f"Prices through **{m.dates[-1].date()}** · {len(m.tickers)} stocks with Yahoo history")
    if st.button("Download / refresh data", help="Membership + adjusted prices for every stock that has been "
                 "in the S&P 500 since 1996. Takes a few minutes."):
        with st.status("Fetching data…", expanded=True) as status:
            data.fetch_all(log=st.write, fundamentals=st.session_state.get("fetch_fund", False))
            status.update(label="Done", state="complete")
        bt.clear_caches()
        st.cache_data.clear()
        st.rerun()
    st.checkbox("Also refresh fundamentals (slow, ~3 min)", key="fetch_fund")

if not have_data:
    st.title("12-1 Momentum Backtest")
    st.info("No data yet. Click **Download / refresh data** in the sidebar.")
    st.stop()

# ---------------------------------------------------------------- sidebar: strategy parameters
REGIME_RULES = {
    "sma": "Below SMA (original rule)",
    "sma_slope": "Below a falling SMA",
    "fast_reentry": "Below SMA, re-enter above 50-day",
    "sma_absmom": "Below SMA and 12-mo return < T-bills",
    "faber": "Below 10-month SMA",
}
SCORES = {"raw": "12-1 return only", "blend": "Blend of 12-1 and 6-1", "risk_adjusted": "12-1 ÷ volatility",
          "smooth": "Smooth momentum (FIP)", "high52": "Near 52-week high",
          "residual": "Residual momentum (net of market)",
          "residual_sector": "Residual momentum (net of market and sector)"}

with st.sidebar:
    st.caption("Defaults are the Improved strategy: 12-1/6-1 blend, bear regime only when the 200-day SMA is "
               "falling, 10% position cap. Change any knob to test it; reload the page to reset.")
    d = bt.Config(**ex.IMPROVED)
    K = lambda name: name  # noqa: E731

    st.header("Strategy")
    c1, c2 = st.columns(2)
    # month pickers instead of calendars: any month from Feb 1996 up to last month, type to search
    last_full = m.dates[-1].to_period("M") - 1
    months = [str(x) for x in pd.period_range("1996-02", last_full, freq="M")]
    start_m = c1.selectbox("Start", months, index=0, key=K("start"),
                           format_func=lambda x: pd.Period(x).strftime("%b %Y"),
                           help="The backtest starts on the first trading day of this month. Type to search, "
                                "e.g. '2025'.")
    ends = ["latest"] + [str(x) for x in pd.period_range(start_m, last_full, freq="M")][::-1]
    end_m = c2.selectbox("End", ends, index=0, key=K("end"),
                         format_func=lambda x: f"Latest ({m.dates[-1]:%b %d})" if x == "latest"
                         else "End of " + pd.Period(x).strftime("%b %Y"))
    start = pd.Period(start_m).start_time.date()
    end = m.dates[-1].date() if end_m == "latest" else pd.Period(end_m).end_time.date()
    universe = st.radio("Universe", ["pit", "current"], horizontal=True, key=K("universe"),
                        format_func=lambda x: {"pit": "Point-in-time members", "current": "Today's members"}[x],
                        help="'Today's members' suffers survivorship bias: "
                             "it only ever picks companies we already know made it into the index.")
    top_n = st.slider("Basket size (top N)", 1, 50, d.top_n, key=K("top_n"))
    buffer_rank = st.slider("Hold buffer: keep while rank ≤", top_n, 150, max(d.buffer_rank, top_n), key=K("buf"),
                            help="Set equal to N to disable the buffer (strict top-N each month).")
    score = st.selectbox("Ranking score", list(SCORES), list(SCORES).index(d.score), format_func=SCORES.get,
                         key=K("score"))
    c1, c2 = st.columns(2)
    lookback = c1.number_input("Lookback (months)", 2, 24, d.lookback_months, key=K("lb"))
    skip = c2.number_input("Skip (months)", 0, 3, d.skip_months, key=K("skip"))
    cost_bps = st.number_input("Trading cost (bps per side)", 0.0, 100.0, d.cost_bps, step=5.0, key=K("cost"))

    st.header("Regime filter")
    regime = st.toggle("Move to T-bills in bear markets", d.regime_filter, key=K("regime"))
    regime_rule = st.selectbox("Bear market when S&P 500 is…", list(REGIME_RULES),
                               list(REGIME_RULES).index(d.regime_rule), format_func=REGIME_RULES.get,
                               disabled=not regime, key=K("rule"))
    c1, c2 = st.columns(2)
    sma_days = c1.number_input("SMA length (days)", 50, 400, d.sma_days, step=10, disabled=not regime, key=K("sma"))
    slope_days = c2.number_input("Slope lookback (days)", 5, 126, d.slope_days, disabled=regime_rule != "sma_slope",
                                 key=K("slope"), help="'Falling' = SMA lower than this many trading days ago.")
    band = st.slider("SMA hysteresis band", 0.0, 0.05, d.sma_band, 0.005, format="%.3f", key=K("band"),
                     help="Exit below SMA×(1−band), re-enter above SMA×(1+band). Reduces whipsaw.")
    bear_exposure = st.slider("Stock exposure kept in bear regime", 0.0, 1.0, d.bear_exposure, 0.1, key=K("bear"))
    regime_daily = st.toggle("Check regime daily (not monthly)", d.regime_daily, key=K("daily"),
                             help="Tested: daily checks whipsaw badly and lower returns.")

    st.header("Sizing & risk")
    rebal_mode = st.radio("Sizing", ["proceeds", "full"], horizontal=True, key=K("mode"),
                          index=["proceeds", "full"].index(d.rebalance_mode),
                          format_func=lambda x: {"proceeds": "No trimming",
                                                 "full": "Equal-weight reset"}[x],
                          help="No trimming: holds are never trimmed (except by the position cap); sale proceeds buy new names at up to 1/N each. "
                               "Reset: every holding is brought back to 1/N each month.")
    weighting = st.radio("Weights", ["equal", "inverse_vol"], horizontal=True, key=K("w"),
                         index=["equal", "inverse_vol"].index(d.weighting),
                         format_func=lambda x: {"equal": "Equal", "inverse_vol": "Inverse volatility"}[x])
    max_weight = st.slider("Max position weight (0 = off)", 0.0, 0.30, d.max_weight, 0.01, format="%.2f",
                           key=K("maxw"), help="Trim any holding above this share of the portfolio at rebalance.")
    sector_cap = st.slider("Max names per sector (0 = off)", 0, 10, d.sector_cap, key=K("sector"),
                           help="Uses today's Yahoo sector labels. Tested: caps cut returns noticeably.")
    vol_target = st.slider("Volatility target (0 = off)", 0.0, 0.30, d.vol_target, 0.01, format="%.2f",
                           key=K("vt"), help="Scale stock exposure so the basket's trailing 6-month volatility ≈ "
                                             "target; the rest sits in T-bills.")
    core_spy = st.slider("Core SPY allocation", 0.0, 0.9, d.core_spy, 0.1, key=K("core"),
                         help="Hold this share in SPY, rebalanced monthly; the rest runs the strategy.")
    quality = st.toggle("Quality gate", d.quality_gate, key=K("quality"),
                        help="Operating income > 0 and EBIT/interest ≥ 1.5, using the latest fiscal year "
                             "reported ≥90 days earlier. Free data only covers ~2021 onward; earlier years pass "
                             "everything through.")

cfg = bt.Config(start=str(start), end=str(end), top_n=top_n, buffer_rank=buffer_rank, lookback_months=int(lookback),
                skip_months=int(skip), regime_filter=regime, sma_days=int(sma_days), universe=universe,
                quality_gate=quality, rebalance_mode=rebal_mode, cost_bps=cost_bps, score=score,
                sma_band=band, vol_target=vol_target, max_weight=max_weight, regime_rule=regime_rule,
                slope_days=int(slope_days), regime_daily=regime_daily, bear_exposure=bear_exposure,
                weighting=weighting, sector_cap=sector_cap, core_spy=core_spy, label="Strategy")


@st.cache_data(show_spinner="Running backtest…", max_entries=64)
def run_cached(c: bt.Config):
    r = bt.run(c)
    return r.nav, r.bench, r.cash_rate, r.rebalances, r.holdings, r.trades, r.weights_in_market


nav, bench, rf, rebals, holdings, trades, invested = run_cached(cfg)
S = mt.summary(nav, rf, bench)
B = mt.summary(bench, rf)


@st.cache_data
def nasdaq_series() -> pd.Series:
    return data.load_nasdaq()


def nasdaq_like(index: pd.DatetimeIndex, capital: float) -> pd.Series:
    """Nasdaq-100 total return on the given dates, scaled to start at `capital`."""
    n = nasdaq_series().reindex(index, method="ffill")
    return (n / n.iloc[0] * capital).rename("Nasdaq-100")


ndx = nasdaq_like(nav.index, cfg.initial_capital)
N = mt.summary(ndx, rf)
NDX_NAME = "Nasdaq-100 (QQQ, total return)"

# ---------------------------------------------------------------- header + KPI tiles
st.title("Improved 12-1 Momentum vs. S&P 500 & Nasdaq")
st.caption(f"{nav.index[0].date()} → {nav.index[-1].date()} · total-return prices · fills at the open on the "
           f"first trading day of each month · {cost_bps:g} bps/side costs · idle cash earns 13-week T-bill yield")

years = (nav.index[-1] - nav.index[0]).days / 365.25
RET = "CAGR" if years >= 1 else "Total return"  # don't annualise periods shorter than a year
RET_LABEL = "CAGR" if years >= 1 else "Return (not annualized)"
k = st.columns(6)
k[0].metric(RET_LABEL, pct(S[RET]), f"{(S[RET] - B[RET]) * 100:+.1f} pts vs S&P")
k[1].metric("Sharpe", f"{S['Sharpe']:.2f}", f"{S['Sharpe'] - B['Sharpe']:+.2f} vs S&P")
k[2].metric("Max drawdown", pct(S["Max drawdown"]), f"{(S['Max drawdown'] - B['Max drawdown']) * 100:+.1f} pts vs S&P")
k[3].metric("Volatility", pct(S["Volatility"]), f"{(S['Volatility'] - B['Volatility']) * 100:+.1f} pts vs S&P",
            delta_color="inverse")
k[4].metric("Turnover / yr", pct(mt.turnover_per_year(rebals), 0), help="One-sided: average of buys and sells, ÷ NAV")
k[5].metric("$100k became", f"${S['Final value'] / 1e6:,.2f}M", f"S&P: ${B['Final value'] / 1e6:,.2f}M",
            delta_color="off")
st.caption(f"**vs Nasdaq-100:** {RET_LABEL} {(S[RET] - N[RET]) * 100:+.1f} pts ({pct(N[RET])}) · "
           f"Sharpe {S['Sharpe'] - N['Sharpe']:+.2f} ({N['Sharpe']:.2f}) · "
           f"Max drawdown {(S['Max drawdown'] - N['Max drawdown']) * 100:+.1f} pts ({pct(N['Max drawdown'])}) · "
           f"$100k → ${N['Final value'] / 1e6:,.2f}M. Before QQQ launched (Mar 1999) the Nasdaq-100 price index is used.")


def bear_shapes(fig, rb: pd.DataFrame):
    if rb.empty or not regime:
        return
    is_bear = rb["regime"].eq("bear")
    starts = rb.index[is_bear & ~is_bear.shift(1, fill_value=False)]
    for s0 in starts:
        after = rb.index[(rb.index > s0) & ~is_bear]
        e0 = after[0] if len(after) else nav.index[-1]
        fig.add_vrect(x0=s0, x1=e0, fillcolor=BEAR_SHADE, line_width=0, layer="below")


tabs = st.tabs(["Performance", "Compare variants", "Calendar returns", "Holdings & trades", "Live signal",
                "Data & caveats", "Findings"])

with tabs[6]:
    st.markdown((Path(__file__).parent / "FINDINGS.md").read_text())

# ---------------------------------------------------------------- performance
with tabs[0]:
    log = st.toggle("Log scale", True)
    fig = go.Figure()
    fig.add_scatter(x=bench.index, y=bench, name="S&P 500 (SPY, total return)", line=dict(color=SERIES[1], width=2),
                    hovertemplate="$%{y:,.0f}")
    fig.add_scatter(x=ndx.index, y=ndx, name=NDX_NAME, line=dict(color=SERIES[2], width=2),
                    hovertemplate="$%{y:,.0f}")
    fig.add_scatter(x=nav.index, y=nav, name="Momentum strategy", line=dict(color=SERIES[0], width=2),
                    hovertemplate="$%{y:,.0f}")
    bear_shapes(fig, rebals)
    fig.update_layout(title=dict(text="Growth of $100,000 · gray bands = regime filter in T-bills",
                                 font=dict(size=15, color=INK)))
    st.plotly_chart(style(fig, 460, "$,.0f", log), width="stretch")

    fig = go.Figure()
    fig.add_scatter(x=bench.index, y=mt.drawdown(bench), name="S&P 500", line=dict(color=SERIES[1], width=2),
                    hovertemplate="%{y:.1%}")
    fig.add_scatter(x=ndx.index, y=mt.drawdown(ndx), name="Nasdaq-100", line=dict(color=SERIES[2], width=2),
                    hovertemplate="%{y:.1%}")
    fig.add_scatter(x=nav.index, y=mt.drawdown(nav), name="Momentum strategy", line=dict(color=SERIES[0], width=2),
                    hovertemplate="%{y:.1%}")
    fig.update_layout(title=dict(text="Drawdown from previous peak", font=dict(size=15, color=INK)))
    st.plotly_chart(style(fig, 300, ".0%"), width="stretch")

    c1, c2 = st.columns([3, 2])
    with c1:
        roll = 252 * 3
        rs = (nav / nav.shift(roll)) ** (1 / 3) - 1
        rb_ = (bench / bench.shift(roll)) ** (1 / 3) - 1
        fig = go.Figure()
        rn_ = (ndx / ndx.shift(roll)) ** (1 / 3) - 1
        fig.add_scatter(x=rb_.index, y=rb_, name="S&P 500", line=dict(color=SERIES[1], width=2), hovertemplate="%{y:.1%}")
        fig.add_scatter(x=rn_.index, y=rn_, name="Nasdaq-100", line=dict(color=SERIES[2], width=2),
                        hovertemplate="%{y:.1%}")
        fig.add_scatter(x=rs.index, y=rs, name="Momentum strategy", line=dict(color=SERIES[0], width=2),
                        hovertemplate="%{y:.1%}")
        fig.add_hline(y=0, line_color=AXIS, line_width=1)
        fig.update_layout(title=dict(text="Rolling 3-year annualised return", font=dict(size=15, color=INK)))
        st.plotly_chart(style(fig, 320, ".0%"), width="stretch")
    with c2:
        st.markdown("**Full statistics**")
        rows = []
        for key in S:
            fmt = (lambda v: f"${v:,.0f}") if key == "Final value" else \
                  (lambda v: f"{v:,.0f}") if "days" in key else \
                  (lambda v: f"{v:.2f}") if key in ("Sharpe", "Sortino", "Calmar", "Beta vs S&P", "Correlation vs S&P") \
                  else pct
            rows.append((key, fmt(S[key]), fmt(B[key]) if key in B else "—", fmt(N[key]) if key in N else "—"))
        rows.append(("Time invested in stocks", pct(invested.mean(), 0), "100%", "100%"))
        rows.append(("Avg holdings", f"{rebals['holdings'][rebals['regime'] == 'bull'].mean():.1f}", "~500", "100"))
        st.dataframe(pd.DataFrame(rows, columns=["Metric", "Strategy", "S&P 500", "Nasdaq-100"]).set_index("Metric"),
                     width="stretch", height=600)

# ---------------------------------------------------------------- calendar
with tabs[2]:
    yr = mt.yearly_returns({"Momentum strategy": nav, "S&P 500": bench, "Nasdaq-100": ndx})
    fig = go.Figure()
    fig.add_bar(x=yr.index, y=yr["Momentum strategy"], name="Momentum strategy", marker_color=SERIES[0],
                hovertemplate="%{y:.1%}")
    fig.add_bar(x=yr.index, y=yr["S&P 500"], name="S&P 500", marker_color=SERIES[1], hovertemplate="%{y:.1%}")
    fig.add_bar(x=yr.index, y=yr["Nasdaq-100"], name="Nasdaq-100", marker_color=SERIES[2], hovertemplate="%{y:.1%}")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08,
                      title=dict(text="Calendar-year returns", font=dict(size=15, color=INK)))
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(style(fig, 380, ".0%"), width="stretch")
    yr["Difference"] = yr["Momentum strategy"] - yr["S&P 500"]
    st.caption(f"Strategy beat the S&P in {(yr['Difference'] > 0).sum()} of {len(yr)} calendar years, and the "
               f"Nasdaq-100 in {(yr['Momentum strategy'] > yr['Nasdaq-100']).sum()} of {len(yr)}.")
    st.markdown("**Monthly returns**")
    mtab = mt.monthly_table(nav)
    st.dataframe(mtab.style.format("{:.1%}", na_rep="").background_gradient(
        cmap="RdBu", vmin=-0.15, vmax=0.15, axis=None), width="stretch", height=420)

# ---------------------------------------------------------------- holdings & trades
with tabs[3]:
    dates_ = sorted(holdings["date"].unique())
    if dates_:
        if len(dates_) > 1:
            sel = st.select_slider("Rebalance date", options=[pd.Timestamp(d).date() for d in dates_],
                                   value=pd.Timestamp(dates_[-1]).date())
        else:
            sel = pd.Timestamp(dates_[0]).date()
        h = holdings[holdings["date"] == pd.Timestamp(sel)].sort_values("weight", ascending=False)
        rrow = rebals.loc[pd.Timestamp(sel)]
        c1, c2 = st.columns([2, 3])
        with c1:
            st.markdown(f"**Portfolio after rebalance on {sel}** · regime: **{rrow['regime']}** · "
                        f"{int(rrow['eligible'])} eligible stocks")
            st.dataframe(h[["ticker", "rank", "momentum", "weight"]].set_index("ticker").style.format(
                {"rank": "{:.0f}", "momentum": "{:+.0%}", "weight": "{:.1%}"}, na_rep="—"),
                width="stretch", height=560)
        with c2:
            t_ = trades[trades["date"] == pd.Timestamp(sel)]
            st.markdown(f"**Trades that day** ({len(t_)})")
            st.dataframe(t_[["ticker", "side", "notional", "price", "reason"]].set_index("ticker").style.format(
                {"notional": "${:,.0f}", "price": "${:,.2f}"}), width="stretch", height=560)
        fig = go.Figure()
        fig.add_scatter(x=invested.index, y=invested, name="Share of portfolio in stocks", fill="tozeroy",
                        line=dict(color=SERIES[0], width=1.5), fillcolor="rgba(42,120,214,0.15)",
                        hovertemplate="%{y:.0%}")
        fig.update_layout(title=dict(text="Share of portfolio in stocks (rest in T-bills)",
                                     font=dict(size=15, color=INK)), showlegend=False)
        st.plotly_chart(style(fig, 260, ".0%"), width="stretch")
        top = holdings.groupby("ticker").size().sort_values(ascending=False).head(25)
        st.markdown("**Most frequently held** (months held after rebalance)")
        st.dataframe(top.rename("months").to_frame().T, width="stretch")
        st.download_button("Download all trades (CSV)", trades.to_csv(index=False), "trades.csv")

# ---------------------------------------------------------------- live signal
with tabs[4]:
    sig = bt.live_signal(cfg)
    nxt = sig["next_rebalance_month"]
    st.markdown(f"Data as of **{sig['as_of'].date()}**. Next rebalance: first trading day of **{nxt}**.")
    c1, c2, c3 = st.columns(3)
    c1.metric("S&P 500", f"{sig['spx']:,.0f}")
    c2.metric(f"{cfg.sma_days}-day SMA", f"{sig['sma']:,.0f}", f"{(sig['spx'] / sig['sma'] - 1) * 100:+.1f}% vs SMA",
              delta_color="off")
    c3.metric("Regime today", "▲ Bull: stay invested" if sig["bull"] else "▼ Bear: move to T-bills")
    last_hold = holdings[holdings["date"] == holdings["date"].max()]["ticker"].tolist()
    r = sig["ranks"].head(max(cfg.buffer_rank, 60)).copy()
    if cfg.score == "raw":
        r = r.drop(columns="score")
    r["status"] = np.where(r["ticker"].isin(last_hold),
                           np.where(r["rank"] <= cfg.buffer_rank, "hold", "sell"),
                           np.where(r["rank"] <= cfg.top_n, "buy candidate", ""))
    st.markdown("Ranking if the rebalance happened at today's close (the regime is re-checked on the rebalance "
                "date, so treat this as a preview, not an order).")
    dropped = [t for t in last_hold if t not in set(r.loc[r["rank"] <= cfg.buffer_rank, "ticker"])]
    if dropped:
        st.warning("Would be sold (fell below the buffer rank or left the index): " + ", ".join(dropped))
    st.dataframe(r.set_index("rank").style.format({"momentum": "{:+.0%}", "score": "{:.2f}"}),
                 width="stretch", height=600)

# ---------------------------------------------------------------- caveats
with tabs[5]:
    mem = bt._membership("pit")
    cov = pd.Series((mem & m.has_price).sum(1) / np.maximum(mem.sum(1), 1), index=m.dates)
    cov = cov[cov.index >= "1996-01-01"]
    fig = go.Figure()
    fig.add_scatter(x=cov.index, y=cov, line=dict(color=SERIES[0], width=2), name="coverage",
                    hovertemplate="%{y:.0%}")
    fig.update_layout(title=dict(text="Share of point-in-time S&P 500 members with Yahoo price history",
                                 font=dict(size=15, color=INK)), showlegend=False)
    fig.update_yaxes(range=[0, 1.02])
    st.plotly_chart(style(fig, 280, ".0%"), width="stretch")
    st.markdown("""
**Read these before trusting any number above.**

- **Survivorship bias (remaining).** Membership is point-in-time, but Yahoo has no prices for many companies that
  later went bankrupt, were acquired, or changed ticker (coverage chart above: ~70% in the late 1990s, ~99% today).
  Missing names are disproportionately failures, so early-period results are still somewhat flattering.
  Compare the *Today's members* variant to see how large the full bias is.
- **Ticker reuse.** A few symbols were reused by different companies. Yahoo returns the current owner's history,
  so a handful of early data points may belong to the wrong company.
- **Quality gate.** Yahoo serves only ~4–5 fiscal years of statements, so the gate is effectively live only from
  ~2021. Before that it passes every stock. A real test needs point-in-time fundamentals (Compustat, Sharadar, etc.).
- **Execution.** Fills at the adjusted open with a flat cost per side. No market impact, taxes, or borrow.
  Dividends are reinvested through adjusted prices.
- **Regime index.** The SMA filter uses the S&P 500 price index (^GSPC). The benchmark is SPY with dividends.
- **The original spec's "historical profile" figures** (14–15% CAGR, −24% to −28% max drawdown, Sharpe ~1.0) are claims
  in the spec, not something this app reproduces. Compare them with what you see here.
""")
    st.markdown("**Configuration used**")
    st.json(bt.config_dict(cfg))


# ---------------------------------------------------------------- variants (rendered last: slow on first load)
VARIANTS = ex.IMPROVED_VARIANTS

with tabs[1]:
    st.markdown("Each row changes **one knob** of the Improved strategy (sidebar settings are ignored here, "
                "except trading cost). Every row is scored on the full period and on two independent halves; "
                "a change is only credible if it helps in **both** halves. "
                "▲ / ▼ = Sharpe more than 0.03 above / below the baseline in that half; ≈ = within 0.03.")
    rows, navs, navs_p, rf_p = [], {}, {}, {}
    base = bt.Config(**ex.IMPROVED, cost_bps=cfg.cost_bps)
    prog = st.progress(0.0, "Running variants… (first load takes about a minute, then it's cached)")
    for i, (name, kw) in enumerate(VARIANTS.items()):
        row = {"Variant": name}
        for pname, (ps, pe) in ex.PERIODS.items():
            n_, b_, rf_, rb_, *_ = run_cached(replace(base, **kw, start=ps, end=pe, label=name))
            s_, sb_ = mt.summary(n_, rf_), mt.summary(b_, rf_)
            row[pname] = (s_, sb_, rb_)
            navs_p[pname], rf_p[pname] = n_.index, rf_
            if pname == "Full":
                navs[name] = n_
                bench_full = b_
        rows.append(row)
        prog.progress((i + 1) / len(VARIANTS), f"Running variants… {name}")
    prog.empty()

    baseline = rows[0]
    nq = {}
    for pname, (ps, pe) in ex.PERIODS.items():
        idx_ = navs_p[pname]
        nq[pname] = mt.summary(nasdaq_like(idx_, 100_000), rf_p[pname])
    out = []
    for row in rows:
        full, h1, h2 = (row[k][0] for k in ex.PERIODS)
        def mark(h, ref):  # Sharpe within ±0.03 of the baseline counts as a tie
            d_ = h["Sharpe"] - ref["Sharpe"]
            return "▲" if d_ > 0.03 else "▼" if d_ < -0.03 else "≈"
        verdict = "—" if row is baseline else (f"96–10 {mark(h1, baseline['1996-2010'][0])}  ·  "
                                            f"11–now {mark(h2, baseline['2011-now'][0])}")
        out.append({"Variant": row["Variant"], "CAGR": full["CAGR"], "Sharpe": full["Sharpe"],
                    "Max DD": full["Max drawdown"], "CAGR 96–10": h1["CAGR"], "CAGR 11–now": h2["CAGR"],
                    "vs S&P 11–now": h2["CAGR"] - row["2011-now"][1]["CAGR"],
                    "vs Nasdaq 11–now": h2["CAGR"] - nq["2011-now"]["CAGR"],
                    "Turnover/yr": mt.turnover_per_year(row["Full"][2]), "Sharpe vs baseline (by half)": verdict})
    sb = {k: rows[0][k][1] for k in ex.PERIODS}
    out.append({"Variant": "S&P 500 (SPY)", "CAGR": sb["Full"]["CAGR"], "Sharpe": sb["Full"]["Sharpe"],
                "Max DD": sb["Full"]["Max drawdown"], "CAGR 96–10": sb["1996-2010"]["CAGR"],
                "CAGR 11–now": sb["2011-now"]["CAGR"], "vs S&P 11–now": 0.0,
                "vs Nasdaq 11–now": sb["2011-now"]["CAGR"] - nq["2011-now"]["CAGR"], "Turnover/yr": np.nan,
                "Sharpe vs baseline (by half)": ""})
    out.append({"Variant": "Nasdaq-100 (QQQ)", "CAGR": nq["Full"]["CAGR"], "Sharpe": nq["Full"]["Sharpe"],
                "Max DD": nq["Full"]["Max drawdown"], "CAGR 96–10": nq["1996-2010"]["CAGR"],
                "CAGR 11–now": nq["2011-now"]["CAGR"],
                "vs S&P 11–now": nq["2011-now"]["CAGR"] - sb["2011-now"]["CAGR"], "vs Nasdaq 11–now": 0.0,
                "Turnover/yr": np.nan, "Sharpe vs baseline (by half)": ""})
    tab = pd.DataFrame(out)
    st.dataframe(tab.style.format({"CAGR": "{:.1%}", "Sharpe": "{:.2f}", "Max DD": "{:.1%}", "CAGR 96–10": "{:.1%}",
                                   "CAGR 11–now": "{:.1%}", "vs S&P 11–now": "{:+.1%}", "vs Nasdaq 11–now": "{:+.1%}",
                                   "Turnover/yr": "{:.0%}"},
                                  na_rep="—"),
                 width="stretch", height=36 * (len(tab) + 1) + 4, hide_index=True,
                 column_config={"Variant": st.column_config.TextColumn(width="large")})
    pick = st.multiselect("Plot (max 6, colors stay fixed per variant)", list(VARIANTS),
                          ["Improved (baseline)", "Regime: original rule (any close below SMA)",
                           "Regime: off (always invested)"],
                          max_selections=6)
    fig = go.Figure()
    fig.add_scatter(x=bench_full.index, y=bench_full, name="S&P 500", line=dict(color=MUTED, width=2, dash="dot"),
                    hovertemplate="$%{y:,.0f}")
    ndx_full = nasdaq_like(bench_full.index, 100_000)
    fig.add_scatter(x=ndx_full.index, y=ndx_full, name="Nasdaq-100", line=dict(color=MUTED, width=2, dash="dash"),
                    hovertemplate="$%{y:,.0f}")
    color_of = {n: SERIES[[0, 2, 3, 4, 5, 6, 7, 1][i % 8]] for i, n in enumerate(VARIANTS)}
    for n in pick:
        fig.add_scatter(x=navs[n].index, y=navs[n], name=n, line=dict(color=color_of[n], width=2),
                        hovertemplate="$%{y:,.0f}")
    fig.update_layout(title=dict(text="Growth of $100,000, 1996 → today (log scale)", font=dict(size=15, color=INK)))
    st.plotly_chart(style(fig, 480, "$,.0f", True), width="stretch")

