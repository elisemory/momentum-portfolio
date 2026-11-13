"""Estimated historical capitalization from Yahoo share observations and prices.

Yahoo Close is split-adjusted even with auto_adjust=False. Share observations are
in the units at their observation date. Restore the price's original units and
carry shares through intervening splits before multiplying. Dividend-adjusted
backtest prices must never be used here.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

from . import data

VERSION = 1
MAX_SHARE_AGE_DAYS = 366


class DownloadLimited(ValueError):
    pass


def directory():
    return data.DATA_DIR / 'yahoo_cap_inputs'


def manifest_path():
    return directory() / 'manifest.parquet'


def input_path(ticker):
    if not re.fullmatch(r'[A-Za-z0-9.^=-]+', ticker):
        raise ValueError('Invalid Yahoo ticker')
    return directory() / f'{ticker}.parquet'


def dates(index):
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert('America/New_York').tz_localize(None)
    return idx.normalize()


def normalize_shares(shares):
    s = pd.to_numeric(shares, errors='coerce').sort_index().copy()
    s.index = dates(s.index)
    s = s[np.isfinite(s) & (s > 0)]
    return s.groupby(level=0).last().sort_index().rename('shares')


def estimate(frame, cutoff):
    """One dated estimate, or a specific exclusion reason. No future share counts."""
    cutoff = pd.Timestamp(cutoff).normalize()
    shares = frame['shares'].dropna().sort_index()
    shares = shares[(shares.index <= cutoff) & np.isfinite(shares) & (shares > 0)]
    if shares.empty:
        return {'reason': 'no historical shares before start'}
    observed = shares.index[-1]
    age = (cutoff - observed).days
    if age > MAX_SHARE_AGE_DAYS:
        return {'reason': 'share count older than one year', 'shares_date': observed, 'share_age_days': age}
    if cutoff not in frame.index or not np.isfinite(frame.at[cutoff, 'close']) or frame.at[cutoff, 'close'] <= 0:
        return {'reason': 'missing starting price'}
    splits = frame['split'].dropna()
    splits = splits[splits != 0]
    if not np.isfinite(splits).all() or (splits <= 0).any():
        return {'reason': 'invalid split history'}
    # Splits after cutoff undo Yahoo's retrospective price adjustments. They do
    # not change the cap estimate if price and split history use the same vintage.
    future_splits = float(splits[splits.index > cutoff].prod())
    intervening = float(splits[(splits.index > observed) & (splits.index <= cutoff)].prod())
    original_price = float(frame.at[cutoff, 'close']) * future_splits
    shares_at_start = float(shares.iloc[-1]) * intervening
    cap = original_price * shares_at_start
    if not np.isfinite(cap) or cap <= 0:
        return {'reason': 'invalid estimated capitalization'}
    return dict(reason='', market_cap=cap, shares_date=observed, share_age_days=age,
                observed_shares=float(shares.iloc[-1]), shares_at_start=shares_at_start,
                price_at_start=original_price, shares_split_factor=intervening,
                price_split_factor=future_splits)


def fetch_one(ticker):
    import yfinance as yf
    from .universes import write_frame
    stock = yf.Ticker(ticker)
    shares = stock.get_shares_full(start=data.PRICE_START)
    if shares is None or shares.empty:
        return dict(ticker=ticker, status='unavailable', reason='no historical share counts')
    shares = normalize_shares(shares)
    if shares.empty:
        return dict(ticker=ticker, status='unavailable', reason='no valid historical share counts')
    # Fetch through the present, including every split used to adjust old Close.
    # Requesting only the backtest interval would miss subsequent stock splits.
    history = stock.history(start=str(shares.index.min().date()), auto_adjust=False, actions=True,
                            raise_errors=True, timeout=30)
    if history.empty or not {'Close', 'Stock Splits'} <= set(history.columns):
        return dict(ticker=ticker, status='unavailable', reason='missing price or split history')
    currency = stock.history_metadata.get('currency')
    if currency != 'USD':
        return dict(ticker=ticker, status='unavailable', reason='quote currency is not USD')
    frame = history[['Close', 'Stock Splits']].rename(columns={'Close': 'close', 'Stock Splits': 'split'})
    frame.index = dates(frame.index)
    frame = frame.groupby(level=0).last().join(shares, how='outer').sort_index()
    frame.attrs = dict(version=VERSION, ticker=ticker, currency=currency,
                       fetched_at=pd.Timestamp.now(tz='UTC').isoformat())
    write_frame(frame, input_path(ticker))
    return dict(ticker=ticker, status='ok', reason='', first_shares=str(shares.index.min().date()),
                last_price=str(history.index[-1].date()))


def manifest():
    if not manifest_path().exists():
        return pd.DataFrame(columns=['ticker', 'status', 'reason', 'fetched_on', 'version'])
    return pd.read_parquet(manifest_path())


def pending(tickers, force=False, cutoff=None):
    if force:
        return sorted(set(tickers))
    old = manifest().set_index('ticker')
    today = str(pd.Timestamp.now(tz='America/New_York').date())
    result = []
    for ticker in sorted(set(tickers)):
        if ticker not in old.index or old.at[ticker, 'version'] != VERSION:
            result.append(ticker)
        elif old.at[ticker, 'status'] == 'ok':
            if (not input_path(ticker).exists() or
                    (cutoff is not None and pd.Timestamp(old.at[ticker, 'last_price']) < pd.Timestamp(cutoff))):
                result.append(ticker)
        elif old.at[ticker, 'fetched_on'] != today:
            result.append(ticker)
    return result


def fetch(tickers, force=False, log=print, workers=3, cutoff=None):
    """Cache each ticker separately so interrupted downloads resume without starting over."""
    from .universes import write_frame
    todo = pending(tickers, force, cutoff)
    if not todo:
        log('Using cached Yahoo share counts and price histories.')
        return
    rows = manifest().set_index('ticker').to_dict('index')
    today = str(pd.Timestamp.now(tz='America/New_York').date())
    log(f'Downloading historical shares and prices for {len(todo):,} stocks…')
    # Bounded batches stop new requests promptly if Yahoo throttles us. Retrying
    # only failed tickers preserves successful downloads and avoids another flood.
    for offset in range(0, len(todo), 50):
        batch = todo[offset:offset+50]
        for attempt in range(3):
            limited = []
            with ThreadPoolExecutor(max_workers=workers if attempt == 0 else 1) as pool:
                futures = {pool.submit(fetch_one, ticker): ticker for ticker in batch}
                for future in as_completed(futures):
                    ticker = futures[future]
                    try:
                        row = future.result()
                    except Exception as exc:
                        # Record only the class, never secrets or full HTTP URLs.
                        reason = type(exc).__name__
                        row = dict(ticker=ticker, status='error', reason=reason)
                        if reason == 'YFRateLimitError':
                            limited.append(ticker)
                    row.pop('ticker', None)
                    rows[ticker] = dict(**row, fetched_on=today, version=VERSION)
                df = pd.DataFrame.from_dict(rows, orient='index').rename_axis('ticker').reset_index()
                write_frame(df, manifest_path())
            if not limited:
                break
            if attempt == 2:
                _snapshot.cache_clear()
                raise DownloadLimited('Yahoo is temporarily rate-limiting downloads. Progress is saved; '
                                      'rerun to resume the remaining histories later.')
            log(f'Yahoo rate limit: waiting 30 seconds, then retrying {len(limited)} stocks sequentially…')
            time.sleep(30)
            batch = limited
        number = min(offset+50, len(todo))
        log(f'{number:,}/{len(todo):,} checked · {(df.status == "ok").sum():,} histories cached')
    _snapshot.cache_clear()


def snapshot(cutoff, tickers):
    target = manifest_path()
    stamp = (target.stat().st_mtime_ns, target.stat().st_size) if target.exists() else None
    return _snapshot(str(directory()), stamp, str(pd.Timestamp(cutoff).date()), tuple(tickers)).copy()


@lru_cache(maxsize=4)
def _snapshot(folder, stamp, cutoff, tickers):
    rows = []
    manifest_file = Path(folder) / 'manifest.parquet'
    statuses = (pd.read_parquet(manifest_file).set_index('ticker').to_dict('index')
                if manifest_file.exists() else {})
    for ticker in tickers:
        target = Path(folder) / input_path(ticker).name
        row = dict(ticker=ticker, date=pd.Timestamp(cutoff), market_cap=np.nan,
                   source='yfinance estimate', reason='history not downloaded')
        status = statuses.get(ticker, {})
        row['download_status'] = status.get('status', 'pending')
        row['download_note'] = status.get('reason', '')
        if status.get('reason') == 'YFRateLimitError':
            row['reason'] = 'Yahoo download rate-limited'
        elif status.get('status') in ('error', 'unavailable'):
            row['reason'] = status.get('reason', 'history unavailable')
        if target.exists():
            try:
                frame = pd.read_parquet(target)
                if frame.attrs.get('version') == VERSION:
                    row.update(estimate(frame, cutoff))
            except (OSError, ValueError, KeyError, TypeError):
                row['reason'] = 'unusable cached history; refresh to retry'
        rows.append(row)
    return pd.DataFrame(rows).set_index('ticker')
