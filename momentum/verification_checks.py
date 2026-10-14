"""Independent checks for the default Improved strategy; no trading-engine helpers.

These checks deliberately read raw files / emitted trades rather than reusing the engine's
membership, momentum, regime, accounting or performance functions. They still share the
underlying vendor data, so passing is not independent validation of that data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def statistics(nav: pd.Series, rf: pd.Series, capital: float = 100_000) -> dict:
    """Use initial capital, including the initial execution day's gain/loss and costs."""
    returns = nav.div(nav.shift(1)).sub(1)
    returns.iloc[0] = nav.iloc[0] / capital - 1
    excess = returns - rf.reindex(nav.index).fillna(0)
    peak = nav.cummax().clip(lower=capital)
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    std = returns.std(ddof=1)
    return dict(CAGR=(nav.iloc[-1] / capital) ** (1 / years) - 1,
                Sharpe=excess.mean() / std * np.sqrt(252) if std else np.nan,
                Volatility=std * np.sqrt(252), MaxDD=(nav / peak - 1).min(),
                final_value=nav.iloc[-1], total_return=nav.iloc[-1] / capital - 1)


def replay_ledger(result, opens: pd.DataFrame, closes: pd.DataFrame):
    """Rebuild every day's NAV from fills, before tax; flag non-market executions.

    The current engine charges a buy fee on cash allocated, not on filled notional.
    That convention is checked explicitly rather than silently treating them as equal.
    """
    cfg = result.config
    if cfg.tax_short or cfg.core_spy:
        raise ValueError("Ledger replay is for a standalone, pre-tax sleeve")
    dates = closes['^GSPC'].dropna().index
    close = closes.reindex(dates).ffill(limit=5)
    rate = closes['^IRX'].reindex(dates).ffill().fillna(0).shift(1).fillna(0) / 25200
    groups = {d: g for d, g in result.trades.groupby('date', sort=False)}
    cash, held = cfg.initial_capital, {}
    daily, fills = [], []
    cost = cfg.cost_bps / 10000
    for day in result.nav.index:
        begin_cash = cash
        interest = cash * rate.loc[day]
        cash += interest
        fees = bought = sold = 0.0
        for trade in groups.get(day, pd.DataFrame()).itertuples(index=False):
            t, q, p = trade.ticker, trade.shares, trade.price
            raw_open = opens.at[day, t] if t in opens and day in opens.index else np.nan
            if trade.side == 'BUY':
                debit = q * p / (1 - cost)
                cash -= debit
                held[t] = held.get(t, 0) + q
                fees += debit - q * p
                bought += debit
                notional_error = debit - trade.notional
            else:
                cash += q * p * (1 - cost)
                held[t] = held.get(t, 0) - q
                if abs(held[t]) < 1e-8:
                    held.pop(t)
                fees += q * p * cost
                sold += q * p * (1 - cost)
                notional_error = q * p - trade.notional
            fills.append(dict(date=day, ticker=t, side=trade.side, shares=q,
                              fill=p, raw_adjusted_open=raw_open,
                              matches_open=bool(np.isfinite(raw_open) and np.isclose(p, raw_open)),
                              notional_error=notional_error, reason=trade.reason))
        stock_value = sum(q * close.at[day, t] for t, q in held.items())
        total = cash + stock_value
        daily.append(dict(date=day, beginning_cash=begin_cash, interest=interest,
                          buy_cash=bought, sale_cash=sold, fees=fees, ending_cash=cash,
                          stock_value=stock_value, nav_rebuilt=total,
                          nav_engine=result.nav.loc[day], error=total - result.nav.loc[day],
                          positions=len(held), smallest_shares=min(held.values(), default=0)))
    return pd.DataFrame(daily).set_index('date'), pd.DataFrame(fills)


def independent_ranking(closes, membership, tickers, signal_day, trade_day):
    """Default 12-1 / 6-1 blend using explicit month-end trading dates.

    A missing month-end quote can use at most five prior *daily* sessions.
    Membership includes all historical members in the coverage denominator.
    """
    calendar = closes['^GSPC'].dropna().index
    history = closes.reindex(index=calendar, columns=tickers)
    history = history.loc[:signal_day]
    filled = history.ffill(limit=5)
    month = pd.Timestamp(trade_day).to_period('M')
    prices = {}
    for offset in (2, 7, 13):
        days = history.index[history.index.to_period('M') == month - offset]
        prices[offset] = filled.loc[days[-1]] if len(days) else pd.Series(np.nan, index=tickers)
    r12 = prices[2] / prices[13] - 1
    r6 = prices[2] / prices[7] - 1
    members = membership.loc[:signal_day, 'tickers'].iloc[-1]
    eligible = (pd.Index(tickers).isin(members) & history.loc[signal_day].notna()
                & np.isfinite(r12) & np.isfinite(r6))
    rank = pd.DataFrame({'momentum': r12[eligible], 'momentum_6_1': r6[eligible]})
    rank['score'] = (rank.momentum.rank(pct=True) + rank.momentum_6_1.rank(pct=True)) / 2
    rank = rank.sort_values('score', ascending=False, kind='stable')
    rank['rank'] = np.arange(1, len(rank) + 1)
    return rank


def independent_regime(index_close: pd.Series, signal_day) -> tuple[bool, float, float]:
    history = index_close.dropna().loc[:signal_day]
    if len(history) < 200:
        return True, np.nan, np.nan
    sma = history.iloc[-200:].mean()
    previous = history.iloc[-221:-21].mean() if len(history) >= 221 else np.nan
    bull = not (history.iloc[-1] <= sma and np.isfinite(previous) and sma < previous)
    return bull, sma, previous


def audit_signals(result, opens, closes, membership, tickers):
    """Independently check every monthly regime and resulting ticker selection.

    Prior holdings come from prior emitted trades. Sizing is separately audited by the
    ledger; this function checks the ranks, buffer and available-open selection rules.
    """
    dates = closes['^GSPC'].dropna().index
    previous = dict(zip(dates[1:], dates[:-1]))
    held = {}
    groups = {d: g for d, g in result.trades.groupby('date')}
    rb = result.rebalances
    audit, ranking_rows = [], []
    for day in result.nav.index:
        if day in rb.index:
            signal = previous[day]
            rank = independent_ranking(closes, membership, tickers, signal, day)
            bull, sma, sma_previous = independent_regime(closes['^GSPC'], signal)
            retained = [t for t in held if t in rank.index and rank.at[t, 'rank'] <= 40] if bull else []
            expected = list(retained)
            if bull:
                for ticker in rank.index:
                    if len(expected) >= 20:
                        break
                    if ticker not in expected and np.isfinite(opens.at[day, ticker]):
                        expected.append(ticker)
            actual = result.holdings.loc[result.holdings.date == day].set_index('ticker')
            common = actual.index.intersection(rank.index)
            rank_errors = int((actual.loc[common, 'rank'] != rank.loc[common, 'rank']).sum())
            audit.append(dict(date=day, signal_date=signal, bull=bull,
                              engine_bull=rb.at[day, 'regime'] == 'bull', sma=sma,
                              sma_21_sessions_ago=sma_previous, eligible=len(rank),
                              engine_eligible=rb.at[day, 'eligible'], rank_errors=rank_errors,
                              expected=','.join(sorted(expected)), actual=','.join(sorted(actual.index)),
                              selection_matches=set(expected) == set(actual.index)))
            details = rank.reset_index(names='ticker')
            details.insert(0, 'date', day)
            details['expected_held'] = details.ticker.isin(expected)
            ranking_rows.append(details)
        for trade in groups.get(day, pd.DataFrame()).itertuples(index=False):
            held[trade.ticker] = held.get(trade.ticker, 0) + (trade.shares if trade.side == 'BUY' else -trade.shares)
            if abs(held[trade.ticker]) < 1e-8:
                held.pop(trade.ticker)
    return pd.DataFrame(audit), pd.concat(ranking_rows, ignore_index=True)


def spy_portfolio(opens, closes, start, end, filtered, cost_bps=10):
    """Independent SPY implementation, same first-session-open fills and cash interest.

    Also used as a restricted reference for engine comparison; it does not call bt.run.
    """
    calendar = closes['^GSPC'].dropna().index
    dates = calendar[(calendar >= start) & (calendar <= end)]
    rates = closes['^IRX'].reindex(calendar).ffill().fillna(0).shift(1).fillna(0) / 25200
    cash, shares, cost = 100_000., 0., cost_bps / 10000
    rows, traded = [], []
    for day in dates:
        cash *= 1 + rates.loc[day]
        i = calendar.get_loc(day)
        signal = calendar[i - 1]
        first = day.to_period('M') != signal.to_period('M')
        if first:
            bull = independent_regime(closes['^GSPC'], signal)[0] if filtered else True
            price = opens.at[day, 'SPY']
            if not np.isfinite(price):
                raise ValueError(f'Missing SPY open on {day}')
            before = cash + shares * price
            amount = 0.
            if not bull and shares:
                amount = shares * price
                cash += amount * (1 - cost)
                shares = 0.
            elif bull and cash > .005 * before:
                # Same cash top-up threshold and fee convention as the momentum sleeve.
                amount = cash
                shares += cash * (1 - cost) / price
                cash = 0.
            traded.append(amount / before / 2)
        total = cash + shares * closes.at[day, 'SPY']
        rows.append((day, total, shares * closes.at[day, 'SPY'] / total))
    frame = pd.DataFrame(rows, columns=['date', 'nav', 'invested']).set_index('date')
    return frame, sum(traded) / ((dates[-1] - dates[0]).days / 365.25)
