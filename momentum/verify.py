"""Freeze and audit the backtest without modifying production data or strategy code.

Run `.venv/bin/python -m momentum.verify`. Results and a frozen source/data baseline
are written under verification_runs/. The HTML report distinguishes reproduced results,
independent calculations, failed checks, and checks requiring unavailable vendor data.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import io
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import backtest as bt, data, experiments as ex, metrics as mt, taxes
from . import verification_checks as checks

ROOT = Path(__file__).resolve().parent.parent
FACTOR_URL = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n')


def freeze(out):
    out.mkdir(parents=True, exist_ok=False)
    baseline = out / 'baseline'
    source = baseline / 'source'
    source.mkdir(parents=True)
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    names += ['momentum/verify.py', 'momentum/verification_checks.py', 'tests/test_verification.py']
    for name in sorted(set(filter(None, names))):
        src = ROOT / name
        if src.is_file():
            dest = source / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    shutil.copytree(data.DATA_DIR, baseline / 'data')
    (out / 'working-tree.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT))
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
                    python=sys.version, platform=platform.platform(), packages=packages,
                    config=asdict(bt.Config(**ex.IMPROVED)),
                    hashes={str(p.relative_to(out)): digest(p) for p in baseline.rglob('*') if p.is_file()},
                    limitations=['Snapshot freezes present-day adjusted history, not historical vendor vintages.',
                                 'Current sector labels and manually curated ticker blocklist are not point-in-time.'])
    write_json(out / 'manifest.json', manifest)
    (out / 'requirements-installed.txt').write_text('\n'.join(f'{k}=={v}' for k, v in sorted(packages.items())) + '\n')
    return source


def membership_file(path):
    frame = pd.read_csv(path, parse_dates=['date']).set_index('date').sort_index()
    frame['tickers'] = frame.tickers.map(lambda s: [t.strip().replace('.', '-') for t in s.split(',')])
    return frame


def coverage(closes, membership, allowed):
    rows = []
    dates = closes['^GSPC'].dropna().index
    for day in pd.Series(dates, index=dates).groupby(dates.to_period('Y')).last():
        if day < membership.index[0]:
            continue
        members = membership.loc[:day].tickers.iloc[-1]
        available = closes.reindex(columns=members).loc[day].notna()
        rows.append(dict(date=day, all_members=len(members), with_quote=int(available.sum()),
                         fraction_with_quote=float(available.mean()),
                         usable_by_engine=int(sum(t in allowed and available[t] for t in members)),
                         missing=','.join(available.index[~available])))
    return pd.DataFrame(rows)


def truncate_inputs(source, target, cutoff):
    """Actually remove future rows from disk, including membership and fundamentals."""
    target.mkdir()
    for path in source.iterdir():
        dest = target / path.name
        if path.suffix == '.parquet':
            frame = pd.read_parquet(path)
            if isinstance(frame.index, pd.DatetimeIndex):
                frame = frame.loc[:cutoff]
            elif 'period_end' in frame:
                frame = frame[frame.period_end <= pd.Timestamp(cutoff)]
            frame.to_parquet(dest)
        elif path.name == 'sp500_hist.csv':
            frame = pd.read_csv(path)
            frame[frame.date <= cutoff].to_csv(dest, index=False)
        else:
            shutil.copy2(path, dest)


def prefix_checks(out, baseline):
    rows = []
    saved_dir = data.DATA_DIR
    full_tickers = set(bt.load_market().tickers)
    for cutoff in ['2000-12-29', '2008-12-31', '2010-12-31', '2020-03-31', '2022-12-30']:
        print('Physical truncation:', cutoff, flush=True)
        with tempfile.TemporaryDirectory(prefix='momentum-verification-') as tmp:
            path = Path(tmp) / 'data'
            truncate_inputs(saved_dir, path, cutoff)
            try:
                data.DATA_DIR = path
                bt.clear_caches()
                result = bt.run(replace(baseline.config, end=cutoff))
                truncated_tickers = set(bt.load_market().tickers)
                expected = baseline.nav.loc[:cutoff]
                delta = result.nav - expected
                expected_trades = baseline.trades[baseline.trades.date <= cutoff].reset_index(drop=True)
                actual_trades = result.trades.reset_index(drop=True)
                try:
                    pd.testing.assert_frame_equal(expected_trades, actual_trades, check_exact=False,
                                                  rtol=1e-10, atol=1e-7)
                    same_trades = True
                except AssertionError:
                    same_trades = False
                different = delta.abs() > 1e-7 + expected.abs() * 1e-10
                rows.append(dict(cutoff=cutoff, nav_matches=not different.any(), trades_match=same_trades,
                                 maximum_nav_error=delta.abs().max(),
                                 ending_nav_difference=delta.iloc[-1],
                                 first_difference=str(delta.index[different][0].date()) if different.any() else '',
                                 added_tickers=','.join(sorted(truncated_tickers - full_tickers)),
                                 removed_tickers=','.join(sorted(full_tickers - truncated_tickers))))
                actual_trades.to_csv(out / f'prefix-trades-{cutoff}.csv', index=False)
            finally:
                data.DATA_DIR = saved_dir
                bt.clear_caches()
    return pd.DataFrame(rows)


def future_filter_probe(opens, closes, cutoff='2010-12-31'):
    """Adversarial control: only mutate future prices, then test the historical prefix."""
    from unittest.mock import patch
    base = bt.run(bt.Config(**ex.IMPROVED, end=cutoff))
    ticker = base.holdings.ticker.value_counts().index[0]
    changed = closes.copy()
    later = changed.index[(changed.index > cutoff) & changed[ticker].notna()]
    for day in later[:6:2]:
        changed.at[day, ticker] *= 10
    try:
        with patch.object(data, 'load_prices', return_value=(opens, changed)):
            bt.clear_caches()
            altered = bt.run(base.config)
        error = (altered.nav - base.nav).abs()
        return dict(ticker=ticker, cutoff=cutoff, changed_only_future=True,
                    historical_nav_unchanged=bool((error < 1e-7).all()),
                    maximum_nav_error=float(error.max()),
                    explanation='Automatic extreme-jump filtering scans the entire input history before the run.')
    finally:
        bt.clear_caches()


def ablations(opens, closes):
    rows = []
    # These are predeclared diagnostic comparisons, not a search for a new optimum.
    for period, (start, end) in ex.PERIODS.items():
        end = end or str(closes['^GSPC'].last_valid_index().date())
        for cost in (10, 25, 50):
            for filtered in (False, True):
                for basket in ('SPY', 'Momentum'):
                    if basket == 'SPY':
                        frame, turnover = checks.spy_portfolio(opens, closes, start, end, filtered, cost)
                        nav = frame.nav
                        rf = closes['^IRX'].reindex(closes['^GSPC'].dropna().index).ffill().shift(1).fillna(0) / 25200
                        invested = frame.invested.mean()
                    else:
                        result = bt.run(bt.Config(**ex.IMPROVED, start=start, end=end,
                                                  regime_filter=filtered, cost_bps=cost))
                        nav, rf = result.nav, result.cash_rate
                        years = (nav.index[-1] - nav.index[0]).days / 365.25
                        turnover = result.rebalances.turnover.sum() / years
                        invested = result.weights_in_market.mean()
                    rows.append(dict(period=period, basket=basket, filter=filtered, cost_bps=cost,
                                     **checks.statistics(nav, rf), turnover=turnover, invested=invested))
        print('Ablation finished:', period, flush=True)
    return pd.DataFrame(rows)


def independent_market_check(out, closes):
    """A public CRSP-based broad-market sanity check, NOT another stock-level backtest."""
    try:
        response = requests.get(FACTOR_URL, timeout=45)
        response.raise_for_status()
        (out / 'french-daily-factors.zip').write_bytes(response.content)
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            text = archive.read(archive.namelist()[0]).decode('utf-8-sig')
        lines = [s for s in text.splitlines() if re.match(r'^\s*\d{8},', s)]
        frame = pd.read_csv(io.StringIO('date,Mkt-RF,SMB,HML,RF\n' + '\n'.join(lines)))
        frame.index = pd.to_datetime(frame.pop('date').astype(str), format='%Y%m%d')
        frame = frame.astype(float) / 100
        returns = pd.DataFrame({'SPY': closes.SPY.pct_change(fill_method=None),
                                'French_US_market': frame['Mkt-RF'] + frame.RF}).dropna().loc['1996-02-01':]
        returns.to_csv(out / 'independent-market-returns.csv')
        return dict(status='completed_benchmark_only', source=FACTOR_URL,
                    sha256=digest(out / 'french-daily-factors.zip'), start=str(returns.index[0].date()),
                    end=str(returns.index[-1].date()), sessions=len(returns),
                    daily_correlation=returns.corr().iloc[0, 1],
                    annualized_mean_return_gap=(returns.SPY - returns.French_US_market).mean() * 252,
                    limitation='Different universe: broad US market, not SPY or the strategy. Does not validate individual stocks, membership or delistings.')
    except Exception as error:
        return dict(status='unavailable', source=FACTOR_URL, error=str(error))


def report(out, results, tables):
    parts = ['<!doctype html><meta charset="utf-8"><title>Momentum verification</title>',
             '<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#222}'
             'table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}'
             'td,th{padding:7px;border:1px solid #ddd}th{text-align:left}pre{white-space:pre-wrap}'
             'h2{margin-top:36px}</style><h1>Momentum verification</h1>',
             '<p>Frozen-source execution using frozen Yahoo data. Reproduction is not proof of an investment edge.</p>',
             '<p>All output CSVs are spreadsheet-readable. No production data or strategy rules were changed.</p>']
    for title, value in results.items():
        parts.append(f'<h2>{html.escape(title)}</h2><pre>{html.escape(json.dumps(value, indent=2, default=str))}</pre>')
    for title, table in tables.items():
        parts.append(f'<h2>{html.escape(title)}</h2>{table.to_html(index=False, float_format=lambda v: f"{v:.6f}")}')
    (out / 'report.html').write_text('\n'.join(parts))


def run(out, include_external=True):
    data.DATA_DIR = out / 'baseline' / 'data'
    bt.clear_caches()
    opens, closes = data.load_prices()
    membership = membership_file(data.DATA_DIR / 'sp500_hist.csv')
    cfg = bt.Config(**ex.IMPROVED)
    baseline = bt.run(cfg)
    baseline.nav.to_csv(out / 'improved-nav.csv')
    baseline.trades.to_csv(out / 'improved-trades.csv', index=False)
    baseline.holdings.to_csv(out / 'improved-holdings.csv', index=False)
    print('Reproducing published variants and tax scenarios…', flush=True)
    variant = ex.evaluate(ex.IDEAS).reset_index()
    variant.to_csv(out / 'variants.csv', index=False)
    print('Reproducing research-app Improved variants…', flush=True)
    ex.evaluate(ex.IMPROVED_VARIANTS, base=cfg).reset_index().to_csv(out / 'improved-variants.csv', index=False)
    tax_frames = []
    for start in ('1996-02-01', '2011-01-01'):
        frame = taxes.compare(start).rename_axis('scenario').reset_index()
        frame.insert(0, 'start', start)
        tax_frames.append(frame)
    tax_table = pd.concat(tax_frames, ignore_index=True)
    tax_table.to_csv(out / 'taxes.csv', index=False)
    print('Reconstructing ledger and monthly signals independently…', flush=True)
    ledger, fills = checks.replay_ledger(baseline, opens, closes)
    ledger.to_csv(out / 'daily-ledger.csv')
    fills.to_csv(out / 'fill-audit.csv', index=False)
    signals, rankings = checks.audit_signals(baseline, opens, closes, membership, bt.load_market().tickers)
    signals.to_csv(out / 'signal-audit.csv', index=False)
    rankings.to_csv(out / 'independent-rankings.csv', index=False)
    selected = {'ordinary': pd.Timestamp('2019-06-03')}
    regimes = baseline.rebalances.regime
    changes = regimes.ne(regimes.shift())
    for label, mask in [('exit', changes & regimes.eq('bear')), ('reentry', changes & regimes.eq('bull') & regimes.shift().eq('bear'))]:
        selected[label] = regimes.index[mask][0]
    missing_fills = fills[~fills.matches_open]
    if len(missing_fills):
        selected['missing_quote_or_delisting'] = missing_fills.date.iloc[0]
    universe_exits = baseline.trades[baseline.trades.reason == 'left universe']
    if len(universe_exits):
        selected['universe_exit'] = universe_exits.date.iloc[0]
    for label, day in selected.items():
        # Full month worksheet includes opening cash, interest, fills, valuations and reconciliation.
        ledger.loc[str(day.to_period('M'))].to_csv(out / f'worksheet-{label}-ledger.csv')
        fills[fills.date.dt.to_period('M') == day.to_period('M')].to_csv(out / f'worksheet-{label}-fills.csv', index=False)
        rankings[rankings.date.dt.to_period('M') == day.to_period('M')].to_csv(out / f'worksheet-{label}-ranks.csv', index=False)
    prefixes = prefix_checks(out, baseline)
    prefixes.to_csv(out / 'prefix-checks.csv', index=False)
    future = future_filter_probe(opens, closes)
    cover = coverage(closes, membership, set(bt.load_market().tickers))
    cover.to_csv(out / 'membership-coverage.csv', index=False)
    attribution = ablations(opens, closes)
    attribution.to_csv(out / 'ablations.csv', index=False)
    external = independent_market_check(out, closes) if include_external else {'status': 'skipped'}
    # Immutable initial research record: no claim that a preview is a finalized signal or a fill.
    as_of = bt.load_market().dates[-1]
    next_month = as_of.to_period('M') + 1
    prospective = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), as_of=str(as_of.date()),
                       status='PREVIEW_ONLY_NOT_EXECUTED', rebalance_month=str(next_month),
                       config=asdict(cfg), baseline_manifest_sha256=digest(out / 'manifest.json'),
                       rule='Freeze this configuration; finalize only with the completed month-end close. '
                            'Record actual timestamped fills and costs; never rewrite prior records.',
                       independent_stock_dataset='Not available; user confirmed no additional dataset.')
    write_json(out / 'prospective-record.json', prospective)
    rank = checks.independent_ranking(closes, membership, bt.load_market().tickers, as_of, next_month.to_timestamp())
    rank.to_csv(out / 'prospective-preview-ranks.csv')
    baseline.positions.to_csv(out / 'prospective-starting-positions.csv', index=False)
    pd.DataFrame(columns=['recorded_utc', 'execution_utc', 'ticker', 'side', 'shares', 'unadjusted_price',
                          'fees', 'signal_manifest_sha256', 'note']).to_csv(out / 'prospective-fills-template.csv', index=False)
    manifest = json.loads((out / 'manifest.json').read_text())
    changed = [name for name, sha in manifest['hashes'].items() if digest(out / name) != sha]
    results = {
        'Scope and limitations': dict(as_of=str(as_of.date()), source_data='Frozen Yahoo adjusted history',
            overall='PARTIAL VALIDATION: read individual checks; successful script execution is not an all-clear.',
            independent_stock_data='BLOCKED: no second stock-level dataset or subscription available.',
            prospective_performance='PENDING: frozen preview and fills template created; no future outcomes exist yet.',
            tax_validation='Reproduced only. Adjusted stock prices reinvest dividends without a separate annual '
                           'dividend tax; SPY comparison taxes dividends explicitly. No wash-sale model. '
                           'These tax scenarios are approximations, not validated after-tax forecasts.',
            timestamp_caveat='Latest daily bar may have been downloaded intraday; vendor publication timestamps are absent.'),
        'Accounting reconstruction': dict(sessions=len(ledger), trades=len(fills),
            maximum_absolute_nav_error=ledger.error.abs().max(),
            passes=bool(np.allclose(ledger.nav_rebuilt, ledger.nav_engine, rtol=1e-10, atol=1e-7)),
            minimum_cash=ledger.ending_cash.min(), minimum_shares=ledger.smallest_shares.min(),
            trades_without_matching_open=len(missing_fills),
            stopped_trading_events=int(baseline.trades.reason.eq('stopped trading').sum()),
            maximum_notional_error=fills.notional_error.abs().max(),
            note='Confirms adjusted-price accounting, not that missing-quote or delisting fills were executable.'),
        'Independent signal reconstruction': dict(rebalances=len(signals),
            regime_mismatches=int(signals.bull.ne(signals.engine_bull).sum()),
            selection_mismatches=int((~signals.selection_matches).sum()),
            rank_mismatches=int(signals.rank_errors.sum()),
            worksheet_months={k: str(v.date()) for k, v in selected.items()},
            limitation='Uses the same vendor history and engine ticker exclusion set; prefixes separately audit future dependence.'),
        'Adversarial future-data control': future,
        'Metric convention': dict(published=mt.summary(baseline.nav, baseline.cash_rate),
            including_initial_day=checks.statistics(baseline.nav, baseline.cash_rate),
            note='Published metrics divide by first closing NAV, omitting the initial execution day. '
                 'Ablations use initial capital for all four portfolios and charge all initial buys.'),
        'Independent public market comparison': external,
        'Snapshot integrity': dict(passes=not changed, changed_files=changed,
                                   manifest_sha256=digest(out / 'manifest.json')),
    }
    write_json(out / 'results.json', results)
    tables = {'Historical truncation checks': prefixes,
              'Attribution at 10 bps (returns are decimal fractions)': attribution[attribution.cost_bps == 10],
              'Historical constituent coverage': cover.drop(columns='missing'),
              'Tax scenarios — reproduced, not independently validated': tax_table}
    report(out, results, tables)
    # Output hashes also cover the raw external download and prospective record; report outputs are append-free.
    write_json(out / 'output-hashes.json', {p.name: digest(p) for p in out.iterdir()
                                          if p.is_file() and p.name not in ('output-hashes.json', 'run.log')})
    print(json.dumps(results, indent=2, default=str), flush=True)
    print('Report:', out / 'report.html', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New output directory (must not already exist)')
    parser.add_argument('--snapshot', type=Path, help='Internal: run a previously frozen baseline')
    parser.add_argument('--offline', action='store_true', help='Skip the public independent market-data download')
    args = parser.parse_args()
    if args.snapshot:
        run(args.snapshot.resolve(), not args.offline)
    else:
        out = (args.output or ROOT / 'verification_runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
        source = freeze(out)
        command = [sys.executable, '-m', 'momentum.verify', '--snapshot', str(out)]
        if args.offline:
            command.append('--offline')
        with (out / 'run.log').open('w') as log:
            proc = subprocess.Popen(command, cwd=source, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            status = proc.wait()
        print(f'Frozen baseline and artifacts: {out}', flush=True)
        raise SystemExit(status)


if __name__ == '__main__':
    main()
