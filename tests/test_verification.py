"""Small hand-calculated fixtures for the independent audit machinery."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from momentum import verification_checks as checks
from momentum.verify import coverage, truncate_inputs


class VerificationTests(unittest.TestCase):
    def ledger_fixture(self):
        dates = pd.to_datetime(['2024-12-31', '2025-01-02', '2025-01-03'])
        close = pd.DataFrame({'AAA': [100., 110., 130.], '^GSPC': [100., 100., 100.],
                              '^IRX': [0., 0., 0.]}, index=dates)
        opens = close.copy()
        opens['AAA'] = [100., 100., 120.]
        trades = pd.DataFrame([
            [dates[1], 'AAA', 'BUY', 990., 100., 100_000., 'entry'],
            [dates[2], 'AAA', 'SELL', 495., 120., 59_400., 'exit'],
        ], columns=['date', 'ticker', 'side', 'shares', 'price', 'notional', 'reason'])
        cfg = SimpleNamespace(tax_short=0, core_spy=0, initial_capital=100_000., cost_bps=100)
        result = SimpleNamespace(config=cfg, trades=trades,
                                 nav=pd.Series([108_900., 123_156.], index=dates[1:]))
        return result, opens, close

    def test_ledger_matches_hand_calculated_buy_fee_sale_fee_and_value(self):
        result, opens, close = self.ledger_fixture()
        daily, fills = checks.replay_ledger(result, opens, close)
        np.testing.assert_allclose(daily.error, 0)
        self.assertEqual(daily.ending_cash.iloc[-1], 58_806.)
        self.assertEqual(daily.fees.sum(), 1594.)
        self.assertTrue(fills.matches_open.all())

    def test_ledger_detects_corrupted_nav_and_execution(self):
        result, opens, close = self.ledger_fixture()
        result.nav.iloc[-1] += 100
        opens.iloc[1, opens.columns.get_loc('AAA')] = 90
        daily, fills = checks.replay_ledger(result, opens, close)
        self.assertEqual(daily.error.iloc[-1], -100)
        self.assertFalse(fills.matches_open.iloc[0])

    def test_cash_interest_uses_previous_session_yield(self):
        result, opens, close = self.ledger_fixture()
        result.trades = result.trades.iloc[:0]
        close['^IRX'] = [25.2, 0., 100.]
        result.nav[:] = 100_100.
        daily, _ = checks.replay_ledger(result, opens, close)
        np.testing.assert_allclose(daily.interest, [100., 0.])
        np.testing.assert_allclose(daily.error, 0)

    def test_statistics_include_initial_day_loss(self):
        dates = pd.to_datetime(['2024-01-02', '2025-01-02'])
        nav = pd.Series([90_000., 90_000.], index=dates)
        stats = checks.statistics(nav, pd.Series(0., index=dates))
        self.assertAlmostEqual(stats['total_return'], -.1)
        self.assertAlmostEqual(stats['MaxDD'], -.1)

    def test_regime_uses_only_data_through_signal(self):
        days = pd.bdate_range('2020-01-01', periods=260)
        values = pd.Series(np.linspace(300., 100., 260), index=days)
        expected = checks.independent_regime(values, days[230])
        self.assertFalse(expected[0])
        values.iloc[231:] = 100_000
        self.assertEqual(expected, checks.independent_regime(values, days[230]))

    def ranking_fixture(self):
        days = pd.bdate_range('2009-01-01', '2010-02-01')
        closes = pd.DataFrame({'^GSPC': 100., 'AAA': 100., 'BBB': 100.}, index=days)
        closes.loc['2009-12-01':, ['AAA', 'BBB']] = [200., 120.]
        members = pd.DataFrame({'tickers': [['AAA', 'BBB']]}, index=pd.to_datetime(['2009-01-01']))
        return closes, members

    def test_ranking_excludes_skipped_month_returns(self):
        closes, members = self.ranking_fixture()
        rank = checks.independent_ranking(closes, members, ['AAA', 'BBB'], '2010-01-29', '2010-02-01')
        self.assertEqual(list(rank.index), ['AAA', 'BBB'])
        self.assertEqual(rank.loc['AAA', 'momentum'], 1.)
        closes.loc['2010-01-01':, 'BBB'] = 10_000
        same = checks.independent_ranking(closes, members, ['AAA', 'BBB'], '2010-01-29', '2010-02-01')
        pd.testing.assert_frame_equal(rank, same)

    def test_month_end_quote_cannot_forward_fill_across_months(self):
        closes, members = self.ranking_fixture()
        closes.loc['2009-12-01':'2009-12-31', 'AAA'] = np.nan
        rank = checks.independent_ranking(closes, members, ['AAA', 'BBB'], '2010-01-29', '2010-02-01')
        self.assertNotIn('AAA', rank.index)

    def test_spy_reference_includes_first_open_to_close_return_and_cost(self):
        dates = pd.to_datetime(['2024-12-31', '2025-01-02', '2025-01-03'])
        closes = pd.DataFrame({'SPY': [100., 110., 120.], '^GSPC': [100., 100., 100.],
                               '^IRX': [0., 0., 0.]}, index=dates)
        opens = closes.copy()
        opens.loc['2025-01-02', 'SPY'] = 100
        frame, _ = checks.spy_portfolio(opens, closes, '2025-01-01', '2025-01-03', False, 100)
        np.testing.assert_allclose(frame.nav, [108_900., 118_800.])

    def test_truncate_removes_future_rows_from_disk_without_changing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / 'source', Path(tmp) / 'truncated'
            source.mkdir()
            pd.DataFrame({'A': [1., 2.]}, index=pd.to_datetime(['2000-01-01', '2001-01-01'])).to_parquet(source / 'close.parquet')
            pd.DataFrame({'date': ['2000-01-01', '2001-01-01'], 'tickers': ['A', 'B']}).to_csv(source / 'sp500_hist.csv', index=False)
            truncate_inputs(source, target, '2000-12-31')
            self.assertEqual(len(pd.read_parquet(target / 'close.parquet')), 1)
            self.assertEqual(len(pd.read_csv(target / 'sp500_hist.csv')), 1)
            self.assertEqual(len(pd.read_parquet(source / 'close.parquet')), 2)

    def test_coverage_uses_all_members_and_skips_price_warmup_years(self):
        dates = pd.to_datetime(['1994-12-30', '1996-12-31'])
        closes = pd.DataFrame({'^GSPC': [100., 100.], 'AAA': [10., 10.]}, index=dates)
        members = pd.DataFrame({'tickers': [['AAA', 'MISSING']]}, index=pd.to_datetime(['1996-01-02']))
        result = coverage(closes, members, {'AAA'})
        self.assertEqual(len(result), 1)
        self.assertEqual(result.all_members.iloc[0], 2)
        self.assertEqual(result.fraction_with_quote.iloc[0], .5)


if __name__ == '__main__':
    unittest.main()
