"""Reproducible, offline follow-up experiments; never changes the live configuration.

Run python -m momentum.followup. Output includes the predeclared experiment plan,
frozen inputs/source, daily NAVs, cost sensitivity, historical subperiods and a report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from . import backtest as bt, data
from .experiments import IMPROVED
from .verification_checks import statistics


PRIMARY = {
    "Baseline": {},
    "Covariance penalty 0.05": {"covariance_penalty": 0.05},
    "Soft sector penalty 0.10": {"sector_penalty": 0.10},
    "Trade efficiency": {"replacement_margin": 0.05, "cap_trim_band": 0.005},
}
SENSITIVITY = {
    "Covariance penalty 0.02": {"covariance_penalty": 0.02},
    "Covariance penalty 0.10": {"covariance_penalty": 0.10},
    "Soft sector penalty 0.05": {"sector_penalty": 0.05},
    "Replacement margin 0.02": {"replacement_margin": 0.02},
    "Replacement margin 0.05": {"replacement_margin": 0.05},
    "Replacement margin 0.10": {"replacement_margin": 0.10},
    "Trim band 0.25%": {"cap_trim_band": 0.0025},
    "Trim band 0.50%": {"cap_trim_band": 0.005},
    "Old equal-weight vol target 25%": {"vol_target": 0.25},
    "Add 3-1 momentum": {"score": "blend3"},
    "Exclude highest-volatility 10%": {"max_vol_pct": 0.90},
    "Exclude highest-volatility 20%": {"max_vol_pct": 0.80},
    "Buffer 60": {"buffer_rank": 60},
    "Missing-quote liquidation haircut 25%": {"delisting_haircut": 0.25},
    "Missing-quote liquidation haircut 100%": {"delisting_haircut": 1.0},
}
PERIODS = {
    "Full": (None, None),
    "1996-2010": (None, "2010-12-31"),
    "2011+": ("2011-01-01", None),
    "1996-2005": (None, "2005-12-31"),
    "2006-2010": ("2006-01-01", "2010-12-31"),
    "2011-2015": ("2011-01-01", "2015-12-31"),
    "2016-2020": ("2016-01-01", "2020-12-31"),
    "2021-2025": ("2021-01-01", "2025-12-31"),
    "2026 YTD": ("2026-01-01", None),
}
FAMILIES = {
    "Covariance selection": ["Baseline", "Covariance penalty 0.02", "Covariance penalty 0.05", "Covariance penalty 0.10"],
    "Trade efficiency": ["Baseline", "Replacement margin 0.02", "Replacement margin 0.05",
                         "Replacement margin 0.10", "Trim band 0.25%", "Trim band 0.50%", "Trade efficiency"],
    "Risk overlay": ["Baseline", "Risk ceiling 20% weekly", "Risk ceiling 25% weekly",
                     "Risk ceiling 30% weekly", "Risk ceiling 25% monthly"],
    "Regime ensemble": ["Baseline", "Regime ensemble 150/200/250", "Regime ensemble 175/200/225"],
}


@dataclass
class Study:
    name: str
    frame: pd.DataFrame
    rates: pd.Series
    cost_bps: float
    config: dict


def check_days(dates: pd.DatetimeIndex, previous_dates: pd.DatetimeIndex, weekly: bool) -> np.ndarray:
    months = dates.to_period("M") != previous_dates.to_period("M")
    weeks = dates.to_period("W-FRI") != previous_dates.to_period("W-FRI")
    return np.asarray(months | weeks if weekly else months)


def sleeve_frame(result: bt.Result) -> pd.DataFrame:
    """Reconstruct pre-trade opens and lagged actual-weight risk from emitted fills."""
    market = bt.load_market(result.config.point_in_time_filter)
    lookup = {ticker: offset for offset, ticker in enumerate(market.tickers)}
    groups = {day: list(group.itertuples(index=False)) for day, group in result.trades.groupby("date")}
    offsets = market.dates.get_indexer(result.nav.index)
    checks = check_days(result.nav.index, market.dates[offsets - 1], True)
    cash = result.config.initial_capital
    held = {}
    last_prices = {}
    cost = result.config.cost_bps / 10000
    rows = []
    for offset, day, check in zip(offsets, result.nav.index, checks):
        previous_nav = cash + sum(amount * market.close_ff[offset - 1, column]
                                  for column, amount in held.items())
        volatility = np.nan
        if check:
            columns = list(held)
            weights = np.array([held[column] * market.close_ff[offset - 1, column] / previous_nav
                                for column in columns])
            covariance = bt.trailing_covariance(market, offset - 1, columns) if columns else np.empty((0, 0))
            volatility = float(np.sqrt(max(0.0, weights @ covariance @ weights)))
        rate = result.cash_rate.loc[day]
        cash *= 1 + rate
        open_equity = 0.0
        for column, amount in held.items():
            price = market.open[offset, column]
            if not np.isfinite(price):
                price = market.close_ff[offset - 1, column]
            if not np.isfinite(price):
                price = last_prices[column]
            open_equity += amount * price
        open_nav = cash + open_equity
        notional = 0.0
        for trade in groups.get(day, []):
            column = lookup[trade.ticker]
            notional += trade.notional
            if trade.side == "BUY":
                cash -= trade.shares * trade.price / (1 - cost)
                held[column] = held.get(column, 0.0) + trade.shares
            else:
                cash += trade.shares * trade.price * (1 - cost)
                held[column] -= trade.shares
                if abs(held[column]) < 1e-8:
                    held.pop(column)
        equity = sum(amount * market.close_ff[offset, column] for column, amount in held.items())
        for column in held:
            if np.isfinite(market.close_ff[offset, column]):
                last_prices[column] = market.close_ff[offset, column]
        nav = cash + equity
        if not np.isclose(nav, result.nav.loc[day], rtol=1e-10, atol=1e-6):
            raise AssertionError(f"Ledger mismatch on {day}: {nav} != {result.nav.loc[day]}")
        rows.append((nav, open_nav, open_equity, equity / nav, notional, volatility))
    return pd.DataFrame(rows, index=result.nav.index,
                        columns=["nav", "open_nav", "open_equity", "invested", "notional", "lagged_vol"])


def risk_overlay(base: Study, target: float, weekly: bool = True, band: float = 0.05) -> Study:
    """Trade proportional units of the shadow portfolio at the NEXT open.

    Shadow holdings retain their original relative weights and rank-buffer history.
    All underlying trading costs remain, plus equity-weighted overlay trading costs.
    Simultaneous opposing trades are not netted: costs are conservative. Idle cash
    earns the same lagged T-bill rate. No leverage or short positions are permitted.
    """
    frame = base.frame
    calendar = bt.load_market(bool(base.config.get("point_in_time_filter"))).dates
    offsets = calendar.get_indexer(frame.index)
    checks = check_days(frame.index, calendar[offsets - 1], weekly)
    units, cash = 1.0, 0.0
    rows = []
    cost = base.cost_bps / 10000
    for day, row, check in zip(frame.index, frame.itertuples(index=False), checks):
        cash *= 1 + base.rates.loc[day]
        sleeve = units * row.open_nav
        total = sleeve + cash
        external_notional = 0.0
        desired = min(1.0, target / row.lagged_vol) if row.lagged_vol > 0 and target > 0 else 1.0
        if check and abs(desired - sleeve / total) >= band:
            change = desired * total - sleeve
            equity_fraction = row.open_equity / row.open_nav
            effective_cost = cost * equity_fraction
            if change < 0:
                units += change / row.open_nav
                cash -= change * (1 - effective_cost)
            else:
                spend = min(change, cash)
                units += spend * (1 - effective_cost) / row.open_nav
                cash -= spend
                change = spend
            external_notional = abs(change) * equity_fraction
        nav = units * row.nav + cash
        if min(units, cash) < -1e-7:
            raise AssertionError("Overlay used leverage")
        rows.append((nav, total, row.open_equity * units, units * row.nav * row.invested / nav,
                     units * row.notional + external_notional, row.lagged_vol))
    name = f"Risk ceiling {target:.0%} {'weekly' if weekly else 'monthly'}"
    config = {**base.config, "overlay_target": target, "weekly": weekly, "exposure_band": band}
    return Study(name, pd.DataFrame(rows, index=frame.index, columns=frame.columns), base.rates, base.cost_bps, config)


def ensemble(name: str, sleeves: list[Study]) -> Study:
    """Equal initial capital in independent sleeves; sleeve weights subsequently drift."""
    if any(not sleeve.frame.index.equals(sleeves[0].frame.index) for sleeve in sleeves):
        raise ValueError("Ensemble calendars differ")
    frame = sum(sleeve.frame.drop(columns="invested") for sleeve in sleeves) / len(sleeves)
    frame["invested"] = sum(sleeve.frame.nav * sleeve.frame.invested for sleeve in sleeves) / len(sleeves) / frame.nav
    frame = frame.reindex(columns=sleeves[0].frame.columns)
    return Study(name, frame, sleeves[0].rates, sleeves[0].cost_bps,
                 {"sleeves": [sleeve.config for sleeve in sleeves], "initial_equal_allocation": True})


def summarize(study: Study) -> list[dict]:
    rows = []
    for period, (start, end) in PERIODS.items():
        frame = study.frame.loc[start:end]
        if len(frame) < 2:
            continue
        first = study.frame.index.get_loc(frame.index[0])
        capital = study.frame.nav.iloc[first - 1] if first else 100_000.0
        stats = statistics(frame.nav, study.rates, capital)
        years = (frame.index[-1] - frame.index[0]).days / 365.25
        returns = frame.nav.pct_change()
        returns.iloc[0] = frame.nav.iloc[0] / capital - 1
        stats.update(turnover=(frame.notional / frame.open_nav).sum() / 2 / years,
                     invested=frame.invested.mean(), worst_day=returns.min())
        rows.append(dict(variant=study.name, cost_bps=study.cost_bps, period=period, **stats))
    return rows


def paired_bootstrap(study: Study, base: Study, start="2011-01-01", draws=500) -> dict:
    """Paired 63-session moving-block bootstrap, conditional on tested specifications."""
    returns = study.frame.nav.pct_change().loc[start:].dropna().to_numpy()
    reference = base.frame.nav.pct_change().loc[start:].dropna().to_numpy()
    rates = base.rates.loc[start:].iloc[-len(reference):].to_numpy()
    generator = np.random.default_rng(20261023)
    blocks = int(np.ceil(len(returns) / 63))
    starts = generator.integers(0, len(returns) - 63 + 1, size=(draws, blocks))
    indices = (starts[:, :, None] + np.arange(63)).reshape(draws, -1)[:, :len(returns)]
    sampled, control, cash = returns[indices], reference[indices], rates[indices]
    cagr_difference = np.expm1(np.log1p(sampled).mean(1) * 252) - np.expm1(np.log1p(control).mean(1) * 252)
    sharpe_difference = ((sampled - cash).mean(1) / sampled.std(1, ddof=1)
                         - (control - cash).mean(1) / control.std(1, ddof=1)) * np.sqrt(252)
    return dict(variant=study.name, start=start, draws=draws, block_sessions=63,
                cagr_diff_low=np.quantile(cagr_difference, 0.025),
                cagr_diff_high=np.quantile(cagr_difference, 0.975),
                sharpe_diff_low=np.quantile(sharpe_difference, 0.025),
                sharpe_diff_high=np.quantile(sharpe_difference, 0.975))


def walk_forward(studies: dict, output: Path) -> pd.DataFrame:
    """Select on trailing ten years, then start fresh cash in the next test fold.

    Each fold is independent, not a stitched investable NAV. Baseline and selected
    portfolios both pay initial execution costs. The last fold is a partial year.
    This is retrospective model-selection analysis, not an untouched holdout.
    """
    rows = []
    base = studies[("Baseline", 10)]
    def simulate(config, start, end, name):
        result = bt.run(replace(bt.Config(**config), start=start, end=end))
        return Study(name, sleeve_frame(result), result.cash_rate, 10, asdict(result.config))
    for year in (2006, 2011, 2016, 2021, 2026):
        start, end = f"{year}-01-01", min(f"{year + 4}-12-31", str(base.frame.index[-1].date()))
        training_start, training_end = f"{year - 10}-01-01", f"{year - 1}-12-31"
        control = simulate(base.config, start, end, "Baseline")
        control_stats = statistics(control.frame.nav, control.rates)
        for family, names in FAMILIES.items():
            def training_sharpe(name):
                study = studies[(name, 10)]
                nav = study.frame.nav.loc[training_start:training_end]
                first = study.frame.index.get_loc(nav.index[0])
                capital = study.frame.nav.iloc[first - 1] if first else 100_000.0
                return statistics(nav, study.rates, capital)["Sharpe"]
            selected = max(names, key=training_sharpe)
            template = studies[(selected, 10)]
            config = template.config
            if selected == "Baseline":
                tested = control
            elif "overlay_target" in config:
                tested = risk_overlay(control, config["overlay_target"], config["weekly"], config["exposure_band"])
            elif "sleeves" in config:
                tested = ensemble(selected, [simulate(sleeve, start, end, selected) for sleeve in config["sleeves"]])
            else:
                tested = simulate(config, start, end, selected)
            stats = statistics(tested.frame.nav, tested.rates)
            rows.append(dict(family=family, train_start=training_start, train_end=training_end,
                             test_start=str(tested.frame.index[0].date()), test_end=str(tested.frame.index[-1].date()),
                             selected=selected, train_sharpe=training_sharpe(selected),
                             CAGR=stats["CAGR"], Sharpe=stats["Sharpe"], MaxDD=stats["MaxDD"],
                             total_return=stats["total_return"], baseline_CAGR=control_stats["CAGR"],
                             baseline_Sharpe=control_stats["Sharpe"], baseline_MaxDD=control_stats["MaxDD"],
                             CAGR_difference=stats["CAGR"] - control_stats["CAGR"],
                             Sharpe_difference=stats["Sharpe"] - control_stats["Sharpe"]))
            print(f"Walk-forward {year} {family}: {selected}, CAGR difference {rows[-1]['CAGR_difference']:+.2%}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(output / "walk_forward.csv", index=False)
    return result


def audit_causality(config: bt.Config) -> dict:
    """Adversarial future-price and physical input-truncation checks on cached stock data."""
    opens, closes = data.load_prices()
    bt.clear_caches()
    reference = bt.run(config)
    prefixes = []
    for cutoff in ("2000-12-29", "2010-12-31", "2020-03-31", "2022-12-30"):
        bt.clear_caches()
        with patch.object(data, "load_prices", return_value=(opens.loc[:cutoff], closes.loc[:cutoff])):
            truncated = bt.run(replace(config, end=cutoff))
        expected = reference.nav.loc[:cutoff]
        trades = reference.trades[reference.trades.date <= cutoff].reset_index(drop=True)
        prefixes.append(dict(cutoff=cutoff, nav_matches=bool(np.allclose(expected, truncated.nav, rtol=1e-12, atol=1e-6)),
                             maximum_nav_error=float((expected - truncated.nav).abs().max()),
                             trades_match=trades.equals(truncated.trades)))
    cutoff = "2010-12-31"
    changed = closes.copy()
    future = changed.index[changed.index > cutoff][:4]
    changed.loc[future, "AAPL"] = [1e8, 0.01, 1e8, 0.01]
    bt.clear_caches()
    with patch.object(data, "load_prices", return_value=(opens, changed)):
        altered = bt.run(replace(config, end=cutoff))
    original = reference.nav.loc[:cutoff]
    future_matches = bool(np.allclose(original, altered.nav, rtol=1e-12, atol=1e-6))
    bt.clear_caches()
    result = dict(prefixes=prefixes, future_ticker="AAPL", future_cutoff=cutoff,
                  future_nav_matches=future_matches,
                  future_maximum_nav_error=float((original - altered.nav).abs().max()),
                  future_trades_match=reference.trades[reference.trades.date <= cutoff].reset_index(drop=True).equals(altered.trades))
    if not future_matches or not result["future_trades_match"] or not all(row["nav_matches"] and row["trades_match"] for row in prefixes):
        raise AssertionError(f"Causality audit failed: {result}")
    return result


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(output: Path) -> dict:
    root = Path(__file__).resolve().parent.parent
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("momentum", "tests"):
        for source in (root / folder).glob("*.py"):
            target = output / "source" / folder / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    snapshot = output / "data"
    snapshot.mkdir()
    for source in data.DATA_DIR.iterdir():
        if source.is_file() and source.suffix in (".parquet", ".csv"):
            shutil.copy2(source, snapshot / source.name)
    return {str(path.relative_to(output)): digest(path)
            for folder in ("source", "data") for path in (output / folder).rglob("*") if path.is_file()}


def render_report(summary: pd.DataFrame, output: Path, names: list[str]) -> None:
    lines = ["# Momentum follow-up experiments", "", "Offline cached-data research, not prospective performance.", "",
             "| Variant | Cost/side | Full CAGR | Sharpe | Max DD | 2011+ CAGR | 2011+ Sharpe | Turnover/year |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cost in (10, 25, 50):
        for name in names:
            full = summary[(summary.variant == name) & (summary.cost_bps == cost) & (summary.period == "Full")]
            recent = summary[(summary.variant == name) & (summary.cost_bps == cost) & (summary.period == "2011+")]
            if full.empty or recent.empty:
                continue
            full, recent = full.iloc[0], recent.iloc[0]
            lines.append(f"| {name} | {cost} bps | {full.CAGR:.2%} | {full.Sharpe:.3f} | {full.MaxDD:.2%} | "
                         f"{recent.CAGR:.2%} | {recent.Sharpe:.3f} | {full.turnover:.2f}x |")
    lines += ["", "## All 10 bps sensitivity results", "",
              "| Variant | Full CAGR | Sharpe | Max DD | 2011+ CAGR | 2011+ Sharpe |",
              "|---|---:|---:|---:|---:|---:|"]
    for full in summary[(summary.cost_bps == 10) & (summary.period == "Full")].itertuples(index=False):
        if full.variant in names:
            continue
        recent = summary[(summary.variant == full.variant) & (summary.cost_bps == 10) & (summary.period == "2011+")].iloc[0]
        lines.append(f"| {full.variant} | {full.CAGR:.2%} | {full.Sharpe:.3f} | {full.MaxDD:.2%} | "
                     f"{recent.CAGR:.2%} | {recent.Sharpe:.3f} |")
    lines += ["", "## Method and limitations", "",
              "- Baseline and all research variants use cumulative, signal-date extreme-jump exclusions and actual last-session month-end prices. Live defaults are unchanged.",
              "- The manual identity blocklist/rename map and vendor adjusted-price vintages are still retrospective; missing bankrupt companies remain missing.",
              "- Signals use prior closes; execution and overlay changes use next opens. No taxes or leverage.",
              "- Covariance uses 126 sessions, at least 60 complete returns, and 25% diagonal shrinkage. Selection considers near-ranked candidates within 0.05 score of the best available candidate.",
              "- Sector penalties use current sector labels, not historical classifications: exploratory only.",
              "- Trade efficiency requires a 0.05 percentile-score replacement advantage and trims only above 10.5% NAV, back to 10%.",
              "- Risk overlays trade proportional units of a shadow baseline portfolio, preserving its holdings history. Checks are monthly, or weekly plus monthly; exposure changes below 5 percentage points are skipped.",
              "- Overlays retain all underlying trading costs and add equity-weighted overlay costs; simultaneous opposing orders are not netted.",
              "- Regime ensembles start with equal-capital independent sleeves, then let sleeve allocations drift. This is not a single fixed-average-exposure basket.",
              "- Initial execution costs/returns and each subperiod's first day are included. Subperiods preserve the continuous portfolio path.",
              "- Historical subperiods and bootstrap intervals are NOT untouched out-of-sample tests and do not correct for all prior strategy searches.",
              "- Walk-forward selection uses the preceding ten years' Sharpe, with the baseline available in each family. Each next-five-year fold starts from fresh cash for both selected strategy and baseline; the 2026 fold is partial. These folds are not stitched into a portfolio return.",
              "- Missing-quote haircuts affect observed held-stock liquidation events only; they cannot repair absent securities.",
              "- See plan.json, validation.json, summary.csv, bootstrap.csv, walk_forward.csv, annual_returns.csv and daily_nav.csv for full results.", ""]
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("verification_runs") / f"followup-{stamp}"
    hashes = freeze(output)
    original_data = data.DATA_DIR
    data.DATA_DIR = (output / "data").resolve()
    bt.clear_caches()
    base_config = bt.Config(**IMPROVED, point_in_time_filter=True)
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), base=asdict(base_config),
                primary=PRIMARY, sensitivity=SENSITIVITY, primary_costs=[10, 25, 50],
                risk_targets=[0.20, 0.25, 0.30], risk_band=0.05, risk_schedules=["weekly+monthly", "monthly"],
                ensembles=[[150, 200, 250], [175, 200, 225]], stagger_days=[0, 5, 10, 15],
                walk_forward_families=FAMILIES,
                periods=PERIODS, hashes=hashes,
                selection="Fixed specifications; no parameter selection based on this run")
    (output / "plan.json").write_text(json.dumps(plan, indent=2))
    summaries, studies, audits = [], {}, []
    primary_names = list(PRIMARY) + ["Risk ceiling 25% weekly", "Regime ensemble 150/200/250"]
    def save(study):
        studies[(study.name, study.cost_bps)] = study
        rows = summarize(study)
        summaries.extend(rows)
        full = rows[0]
        print(f"{study.cost_bps:2.0f} bps {study.name:45s} CAGR {full['CAGR']:.2%} "
              f"Sharpe {full['Sharpe']:.3f} DD {full['MaxDD']:.2%}", flush=True)
    def simulate(name, cost, **overrides):
        config = replace(base_config, cost_bps=cost, **overrides)
        result = bt.run(config)
        frame = sleeve_frame(result)
        market = bt.load_market(config.point_in_time_filter)
        columns = {ticker: offset for offset, ticker in enumerate(market.tickers)}
        unmatched = sum(not np.isclose(trade.price, market.open[market.dates.get_loc(trade.date), columns[trade.ticker]],
                                      equal_nan=False) for trade in result.trades.itertuples(index=False))
        audits.append(dict(variant=name, cost_bps=cost, trades=len(result.trades),
                           missing_quote_exits=int((result.trades.reason == "stopped trading").sum()),
                           fills_not_matching_open=unmatched))
        return Study(name, frame, result.cash_rate, cost, asdict(config))
    try:
        save(simulate("Legacy baseline", 10, point_in_time_filter=False))
        for cost in (10, 25, 50):
            for name, changes in PRIMARY.items():
                save(simulate(name, cost, **changes))
            baseline = studies[("Baseline", cost)]
            save(risk_overlay(baseline, 0.25))
            sleeves = [simulate(f"SMA {length}", cost, sma_days=length) if length != 200 else baseline
                       for length in (150, 200, 250)]
            save(ensemble("Regime ensemble 150/200/250", sleeves))
            if cost == 10:
                for sleeve in sleeves:
                    if sleeve.name != "Baseline":
                        save(sleeve)
        baseline = studies[("Baseline", 10)]
        for name, changes in SENSITIVITY.items():
            save(simulate(name, 10, **changes))
        for target in (0.20, 0.30):
            save(risk_overlay(baseline, target))
        save(risk_overlay(baseline, 0.25, weekly=False))
        sleeves = [simulate(f"SMA {length}", 10, sma_days=length) if length != 200 else baseline
                   for length in (175, 200, 225)]
        save(ensemble("Regime ensemble 175/200/225", sleeves))
        sleeves = [simulate(f"Trading-day signal, day {day}", 10, daily_signal=True, rebalance_day=day)
                   for day in (0, 5, 10, 15)]
        for sleeve in sleeves:
            save(sleeve)
        save(ensemble("Four staggered sleeves", sleeves))
        market = bt.load_market(True)
        opens, closes = data.load_prices()
        from .verification_checks import spy_portfolio
        spy, _ = spy_portfolio(opens, closes, str(baseline.frame.index[0].date()),
                               str(baseline.frame.index[-1].date()), False, 10)
        spy_frame = baseline.frame.copy()
        spy_frame["nav"], spy_frame["invested"] = spy.nav, spy.invested
        spy_frame["notional"] = 0.0
        spy_frame.loc[spy_frame.index[0], "notional"] = 100_000.0
        spy_frame["open_nav"] = spy.nav.shift(1).fillna(100_000.0)
        save(Study("SPY buy and hold", spy_frame, baseline.rates, 10, {}))
        summary = pd.DataFrame(summaries)
        summary.to_csv(output / "summary.csv", index=False)
        navs = pd.DataFrame({f"{name} [{cost}bps]": study.frame.nav for (name, cost), study in studies.items()})
        navs.to_csv(output / "daily_nav.csv", index_label="date")
        annual = navs.resample("YE").last().pct_change()
        annual.iloc[0] = navs.resample("YE").last().iloc[0] / 100_000 - 1
        annual.to_csv(output / "annual_returns.csv", index_label="year")
        pd.DataFrame(audits).to_csv(output / "execution_audit.csv", index=False)
        intervals = [paired_bootstrap(studies[(name, 10)], baseline) for name in primary_names if name != "Baseline"]
        pd.DataFrame(intervals).to_csv(output / "bootstrap.csv", index=False)
        walk_forward(studies, output)
        causality = audit_causality(base_config)
        validation = dict(as_of=str(market.dates[-1].date()), ledger_reconstruction_passes=True,
                          causality=causality,
                          snapshot_unchanged=all(digest(output / name) == checksum for name, checksum in hashes.items()),
                          missing_stock_data="Not repaired: no independent delisting dataset",
                          missing_price_stress_is_not_missing_security_stress=True,
                          tax_results="Not tested: existing tax model omits individual stock dividend taxes and wash sales",
                          live_config_changed=False)
        (output / "validation.json").write_text(json.dumps(validation, indent=2))
        render_report(summary, output, primary_names)
        (output / "study_configs.json").write_text(json.dumps({f"{name} [{cost}bps]": study.config
                                                               for (name, cost), study in studies.items()}, indent=2))
        print(f"RESULTS: {output.resolve()}", flush=True)
    finally:
        data.DATA_DIR = original_data
        bt.clear_caches()


if __name__ == "__main__":
    main()
