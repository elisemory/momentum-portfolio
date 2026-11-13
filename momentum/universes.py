"""Dashboard universes, with a frozen market-cap screen at the test's start.

Historical market caps are USD company values, supplied as dated observations.
Never infer past capitalization from today's shares or today's market cap.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import os
import tempfile

import numpy as np
import pandas as pd
import requests

from . import backtest as bt, data

LABELS = {'sp500': 'S&P 500', 'nasdaq100': 'Nasdaq-100', 'market_cap': 'US stocks by starting market cap'}
REGIMES = {'sp500': 'S&P 500', 'nasdaq100': 'Nasdaq-100'}


@dataclass(frozen=True)
class Selection:
    kind: str = 'sp500'
    start: str = '2010-01-01'
    minimum_billions: float = 20.0
    cap_source: str = 'imported'     # app defaults to yfinance estimates; retain CSV API compatibility

    def __post_init__(self):
        if self.kind not in LABELS:
            raise ValueError(f'Unknown stock universe: {self.kind}')
        if not np.isfinite(self.minimum_billions) or self.minimum_billions < 0:
            raise ValueError('The market-cap threshold must be finite and nonnegative.')
        if self.cap_source not in ('imported', 'yfinance'):
            raise ValueError('Unknown market-cap source')


def path(name):
    return data.DATA_DIR / name


def write_frame(frame, target):
    """Readers see a complete cache file, even during a refresh."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=target.suffix)
    os.close(fd)
    try:
        if target.suffix == '.parquet':
            frame.to_parquet(tmp)
        else:
            frame.to_csv(tmp, index=False)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def fingerprint():
    names = ['open.parquet', 'close.parquet', 'sp500_hist.csv', 'fundamentals.parquet',
             'supplement_open.parquet', 'supplement_close.parquet', 'sectors.csv',
             'equity_open.parquet', 'equity_close.parquet', 'us_equities.csv',
             'nasdaq100_members.csv', 'historical_market_caps.parquet',
             'yahoo_cap_inputs/manifest.parquet',
             'nasdaq100_index.parquet', 'nasdaq100_tr.parquet']
    return tuple((n, path(n).stat().st_mtime_ns, path(n).stat().st_size)
                 for n in names if path(n).exists())


def validate_caps(frame):
    required = {'date', 'ticker', 'market_cap'}
    if not required.issubset(frame):
        raise ValueError('Market-cap CSV needs date, ticker, market_cap columns; values must be in USD.')
    df = frame[list(sorted(required))].copy()
    df['date'] = pd.to_datetime(df.date, errors='raise').dt.normalize()
    df['ticker'] = df.ticker.astype('string').str.strip().str.upper().str.replace('.', '-', regex=False)
    df['market_cap'] = pd.to_numeric(df.market_cap, errors='raise')
    if df.empty or df.isna().any().any() or (df.ticker.str.len() == 0).any():
        raise ValueError('Market-cap rows must contain a date, ticker and value.')
    if not np.isfinite(df.market_cap).all() or (df.market_cap <= 0).any():
        raise ValueError('Market caps must be positive, finite USD values, not billions.')
    if 'currency' in frame and not frame.currency.eq('USD').all():
        raise ValueError('Convert all market caps to USD before importing.')
    if df.duplicated(['date', 'ticker']).any():
        raise ValueError('Only one market-cap observation per ticker and date is allowed.')
    return df.sort_values(['date', 'ticker']).reset_index(drop=True)


def import_caps(contents: bytes):
    incoming = validate_caps(pd.read_csv(StringIO(contents.decode('utf-8-sig'))))
    target = path('historical_market_caps.parquet')
    if target.exists():
        old = pd.read_parquet(target)
        incoming = pd.concat([old, incoming]).drop_duplicates(['date', 'ticker'], keep='last')
    write_frame(incoming.sort_values(['date', 'ticker']), target)
    return len(incoming)


def start_caps(caps, cutoff, tickers):
    """Capitalization at the preceding session's close, known before the first fill."""
    cutoff = pd.Timestamp(cutoff).normalize()
    rows = caps[caps.date == cutoff]
    return (rows.sort_values('date').drop_duplicates('ticker', keep='last')
            .set_index('ticker').reindex(tickers))


def fetch_members(kind):
    if kind == 'nasdaq100':
        response = requests.get('https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies',
                                headers={'User-Agent': 'MomentumBacktest/1.0'}, timeout=60)
        response.raise_for_status()
        for table in pd.read_html(StringIO(response.text)):
            if 'Ticker' in table and 'Company' in table:
                members = table.rename(columns={'Ticker': 'ticker', 'Company': 'name'})[['ticker', 'name']]
                members['ticker'] = members.ticker.str.replace('.', '-', regex=False)
                if not 95 <= len(members) <= 110 or members.ticker.duplicated().any():
                    raise ValueError('Unexpected Nasdaq-100 constituent list; cache was not updated.')
                members['as_of'] = str(pd.Timestamp.now().date())
                write_frame(members, path('nasdaq100_members.csv'))
                return members
        raise ValueError('Nasdaq-100 constituent table was not found.')
    if kind != 'market_cap':
        raise ValueError(f'Unknown expanded universe: {kind}')
    frames = []
    for filename, symbol in [('nasdaqlisted', 'Symbol'), ('otherlisted', 'ACT Symbol')]:
        r = requests.get(f'https://www.nasdaqtrader.com/dynamic/SymDir/{filename}.txt', timeout=60)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text), sep='|')
        df = df[df['Test Issue'].eq('N') & df.ETF.eq('N')]
        frames.append(df.rename(columns={symbol: 'ticker', 'Security Name': 'name'})[['ticker', 'name']])
    df = pd.concat(frames, ignore_index=True)
    excluded = df.name.str.contains(
        r'preferred|preference|\bpfd\b|warrant|\brights?\b|\bnotes?\b|debenture|'
        r'\bfunds?\b|\bETN\b|\bSTRATS\b|\bCorTS\b|mortgage bonds|corporate units|'
        r' - Units?$|\bTerm Trust\b|(?:BlackRock|Eaton Vance|Gabelli|GAMCO|Royce|'
        r'John Hancock|DWS|Invesco|Western Asset|Nuveen|Franklin).*\b(?:Trust|Portfolio)\b',
        case=False, regex=True)
    excluded |= df.ticker.str.contains(r'\$|\.W|\.R|\.U', regex=True)
    excluded |= df.ticker.isin(['ASA', 'CET', 'SOR', 'ECC', 'EIC', 'FSCO', 'IVAI'])
    # Ordinary equities / ADRs / BDCs whose exchange descriptions resemble funds or preferreds.
    exceptions = ['AMX', 'BNS', 'CIB', 'ITUB', 'PBR.A', 'FBRT', 'ARCC', 'BCIC', 'CGBD',
                  'FDUS', 'GBDC', 'GECC', 'GLAD', 'MFIC', 'OCSL', 'OFS', 'OXSQ', 'PSEC',
                  'RAND', 'SLRC', 'TCPC', 'WHF']
    df = df[~excluded | df.ticker.isin(exceptions)].copy()
    df['ticker'] = df.ticker.str.replace('.', '-', regex=False)
    df['as_of'] = str(pd.Timestamp.now().date())
    if len(df) < 1000 or df.ticker.duplicated().any():
        raise ValueError('Unexpected US listing directory; cache was not updated.')
    write_frame(df, path('us_equities.csv'))
    return df


def members(kind):
    return pd.read_csv(path('nasdaq100_members.csv' if kind == 'nasdaq100' else 'us_equities.csv'))


def fetch_nasdaq_index(log=print):
    import yfinance as yf
    log('Downloading Nasdaq-100 market-regime history…')
    df = yf.download('^NDX', start=data.PRICE_START, auto_adjust=False, progress=False)['Close']
    if isinstance(df, pd.Series):
        df = df.to_frame('^NDX')
    if df.empty:
        raise ValueError('No Nasdaq-100 index prices returned; retained existing data.')
    df.index = pd.to_datetime(df.index).tz_localize(None)
    write_frame(df, path('nasdaq100_index.parquet'))


def load_nasdaq_index():
    target = path('nasdaq100_index.parquet')
    if not target.exists():
        raise ValueError('Download Nasdaq-100 regime data in the sidebar first.')
    return pd.read_parquet(target).iloc[:, 0]


def refresh(kind, log=print):
    """Fetch the requested stock list plus full adjusted history, in bounded batches."""
    import yfinance as yf
    listing = fetch_members(kind)
    tickers = sorted(set(listing.ticker) | set(data.EXTRA))
    opens, closes = [], []
    for i in range(0, len(tickers), 80):
        batch = tickers[i:i+80]
        log(f'Prices {i+1}–{i+len(batch)} of {len(tickers)}')
        df = yf.download(batch, start=data.PRICE_START, auto_adjust=True, progress=False, threads=12)
        if df.empty:
            raise ValueError('Price download failed; existing price cache retained. Try refreshing again.')
        opens.append(df['Open'])
        closes.append(df['Close'])
    for field, parts in [('open', opens), ('close', closes)]:
        frame = pd.concat(parts, axis=1).sort_index()
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        target = path(f'equity_{field}.parquet')
        if target.exists():
            frame = frame.combine_first(pd.read_parquet(target))
        write_frame(frame, target)
    log('Stocks without usable price history are excluded from ranking.')


def market(kind):
    if kind == 'sp500':
        return bt.load_market()
    listing = members(kind)
    o = pd.read_parquet(path('equity_open.parquet'))
    c = pd.read_parquet(path('equity_close.parquet'))
    tickers = sorted(set(listing.ticker) & set(c.columns) & set(o.columns))
    if not tickers:
        raise ValueError('No stock price histories found for this universe. Download its prices first.')
    # Invalid prices are missing observations, never zero-valued purchases.
    o[tickers] = o[tickers].where(o[tickers] > 0)
    c[tickers] = c[tickers].where(c[tickers] > 0)
    return bt.prepare_market(o[tickers + data.EXTRA], c[tickers + data.EXTRA], True)


def prepare(selection: Selection, cfg: bt.Config, m=None):
    m = market(selection.kind) if m is None else m
    if selection.kind == 'sp500':
        return m, data.membership_matrix(m.dates, m.tickers, cfg.universe).to_numpy(), {}
    eligible = np.ones(len(m.tickers), dtype=bool)
    details = {'listed': len(m.tickers), 'membership': 'Current listings; delisted stocks omitted'}
    if selection.kind == 'market_cap':
        first = m.dates.searchsorted(pd.Timestamp(selection.start))
        if first <= 0 or first >= len(m.dates):
            raise ValueError('The test start needs an earlier trading session in the price history.')
        cutoff = m.dates[first-1]
        eligible &= m.has_price[first-1]
        if selection.minimum_billions > 0:
            if selection.cap_source == 'yfinance':
                from . import market_caps
                snapshot = market_caps.snapshot(cutoff, m.tickers)
                details['cap_source'] = 'yfinance estimate'
            else:
                target = path('historical_market_caps.parquet')
                if not target.exists():
                    raise ValueError(f'Historical market caps as of {cutoff.date()} are needed. Import a market-cap CSV in the sidebar.')
                snapshot = start_caps(pd.read_parquet(target), cutoff, m.tickers)
                details['cap_source'] = 'Imported historical caps'
            eligible &= (snapshot.market_cap > selection.minimum_billions * 1e9).fillna(False).to_numpy()
            details['caps_available'] = int(snapshot.market_cap.notna().sum())
        details.update(cap_cutoff=str(cutoff.date()), eligible_at_start=int(eligible.sum()),
                       minimum_billions=selection.minimum_billions)
        if not eligible.any():
            if selection.cap_source == 'yfinance' and selection.minimum_billions > 0:
                raise ValueError(f'No stocks qualify above ${selection.minimum_billions:g}B using Yahoo estimates as of '
                                 f'{cutoff.date()}. {details["caps_available"]:,} usable estimates were found. '
                                 'Try a later start, a lower threshold, or refresh the histories. '
                                 'Yahoo share-count coverage often starts around 2015; missing or stale counts are excluded.')
            raise ValueError(f'No stocks qualify above ${selection.minimum_billions:g}B as of {cutoff.date()}. '
                             'Caps must be dated at the last market close before the start; missing values are excluded.')
    return m, np.broadcast_to(eligible, m.has_price.shape), details
