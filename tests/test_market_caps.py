"""Historical capitalization estimates, split conventions, and fetch/cache behavior."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from momentum import data, market_caps as mc, universes as uv, backtest as bt


class EstimateTests(unittest.TestCase):
    def frame(self, prices=(25., 25., 25.), splits=(0., 0., 4.), shares=(100., np.nan, 400.)):
        return pd.DataFrame(dict(close=prices, split=splits, shares=shares),
                            index=pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01']))

    def test_future_split_is_removed_from_price_without_using_future_share_count(self):
        frame = self.frame(shares=(100., np.nan, 999999.))
        got = mc.estimate(frame, '2020-02-01')
        self.assertEqual(got['market_cap'], 10000)
        self.assertEqual(got['price_at_start'], 100)
        self.assertEqual(got['shares_at_start'], 100)
        self.assertEqual(got['shares_date'], pd.Timestamp('2020-01-01'))

    def test_split_between_share_observation_and_start_adjusts_shares(self):
        frame = self.frame(shares=(100., np.nan, np.nan))
        got = mc.estimate(frame, '2020-03-01')
        self.assertEqual(got['market_cap'], 10000)
        self.assertEqual(got['price_at_start'], 25)
        self.assertEqual(got['shares_at_start'], 400)

    def test_share_observation_on_split_day_is_not_adjusted_twice(self):
        self.assertEqual(mc.estimate(self.frame(), '2020-03-01')['market_cap'], 10000)

    def test_reverse_split_preserves_capitalization(self):
        frame = self.frame(prices=(100., 100., 100.), splits=(0., 0., .1), shares=(1000., np.nan, np.nan))
        self.assertEqual(mc.estimate(frame, '2020-02-01')['market_cap'], 10000)
        self.assertEqual(mc.estimate(frame, '2020-03-01')['market_cap'], 10000)

    def test_later_split_with_matching_price_restatement_does_not_change_old_cap(self):
        before = self.frame(prices=(100., 100., 100.), splits=(0., 0., 0.))
        after = self.frame()
        self.assertEqual(mc.estimate(before, '2020-02-01')['market_cap'],
                         mc.estimate(after, '2020-02-01')['market_cap'])

    def test_missing_price_and_future_only_shares_are_excluded(self):
        frame = self.frame(shares=(np.nan, np.nan, 400.))
        self.assertIn('no historical shares', mc.estimate(frame, '2020-02-01')['reason'])
        self.assertEqual(mc.estimate(self.frame(), '2020-02-02')['reason'], 'missing starting price')
        self.assertEqual(mc.estimate(self.frame(prices=(25., 0., 25.)), '2020-02-01')['reason'], 'missing starting price')

    def test_stale_shares_are_excluded(self):
        frame = self.frame(shares=(100., np.nan, np.nan))
        frame.loc[pd.Timestamp('2021-02-01')] = [25., 0., np.nan]
        self.assertEqual(mc.estimate(frame, '2021-02-01')['reason'], 'share count older than one year')

    def test_duplicate_day_keeps_latest_observation_and_local_calendar_date(self):
        idx = pd.to_datetime(['2020-01-03T01:00:00Z', '2020-01-02T15:00:00Z'])
        got = mc.normalize_shares(pd.Series([110, 100], index=idx))
        self.assertEqual(got.to_dict(), {pd.Timestamp('2020-01-02'): 110})


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.patch = patch.object(data, 'DATA_DIR', Path(self.temp.name))
        self.patch.start()
        mc._snapshot.cache_clear()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()
        mc._snapshot.cache_clear()

    def stock(self):
        obj = Mock()
        obj.get_shares_full.return_value = pd.Series([100.], index=pd.to_datetime(['2020-01-01']))
        obj.history.return_value = pd.DataFrame({'Close': [25., 25., 25.], 'Stock Splits': [0., 0., 4.]},
                                              index=pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01']))
        obj.history_metadata = {'currency': 'USD'}
        return obj

    def test_fetch_requests_prices_without_dividend_adjustment_and_full_split_history(self):
        obj = self.stock()
        with patch('yfinance.Ticker', return_value=obj):
            mc.fetch(['AAA'], log=lambda _: None, workers=1)
        args = obj.history.call_args.kwargs
        self.assertFalse(args['auto_adjust'])
        self.assertTrue(args['actions'])
        self.assertNotIn('end', args)
        got = mc.snapshot('2020-02-01', ['AAA', 'MISSING'])
        self.assertEqual(got.at['AAA', 'market_cap'], 10000)
        self.assertTrue(np.isnan(got.at['MISSING', 'market_cap']))
        self.assertEqual(mc.pending(['AAA']), [])
        self.assertEqual(mc.pending(['AAA'], cutoff='2020-04-01'), ['AAA'])

    def test_fetch_failure_is_recorded_and_does_not_erase_good_history(self):
        with patch('yfinance.Ticker', return_value=self.stock()):
            mc.fetch(['AAA'], log=lambda _: None, workers=1)
        with patch('yfinance.Ticker', side_effect=RuntimeError('network failure')):
            mc.fetch(['AAA'], force=True, log=lambda _: None, workers=1)
        self.assertEqual(mc.manifest().iloc[0].status, 'error')
        self.assertEqual(mc.snapshot('2020-02-01', ['AAA']).at['AAA', 'market_cap'], 10000)

    def test_estimates_select_the_universe_without_an_imported_csv(self):
        with patch('yfinance.Ticker', return_value=self.stock()):
            mc.fetch(['AAA'], log=lambda _: None, workers=1)
        days = pd.to_datetime(['2020-01-31', '2020-02-01', '2020-02-03'])
        market = SimpleNamespace(dates=days, tickers=['AAA', 'MISSING'], has_price=np.ones((3, 2), dtype=bool))
        selection = uv.Selection('market_cap', '2020-02-03', .000005, 'yfinance')
        _, membership, details = uv.prepare(selection, bt.Config(), market)
        self.assertEqual(details['caps_available'], 1)
        self.assertEqual(details['cap_source'], 'yfinance estimate')
        self.assertTrue(membership[:, 0].all())
        self.assertFalse(membership[:, 1].any())
        self.assertFalse(uv.path('historical_market_caps.parquet').exists())

    def test_non_usd_prices_are_excluded(self):
        stock = self.stock()
        stock.history_metadata = {'currency': 'EUR'}
        with patch('yfinance.Ticker', return_value=stock):
            row = mc.fetch_one('AAA')
        self.assertEqual(row['status'], 'unavailable')
        self.assertFalse(mc.input_path('AAA').exists())

    def test_rate_limit_backs_off_and_retries_only_failed_stock(self):
        from yfinance.exceptions import YFRateLimitError
        results = [YFRateLimitError(), {'ticker': 'AAA', 'status': 'unavailable', 'reason': 'no shares'}]
        with patch.object(mc, 'fetch_one', side_effect=results) as fetch, patch.object(mc.time, 'sleep') as sleep:
            mc.fetch(['AAA'], workers=1, log=lambda _: None)
        self.assertEqual(fetch.call_count, 2)
        sleep.assert_called_once_with(30)
        self.assertEqual(mc.manifest().iloc[0].status, 'unavailable')

    def test_persistent_rate_limit_saves_progress_and_stops(self):
        from yfinance.exceptions import YFRateLimitError
        with patch.object(mc, 'fetch_one', side_effect=YFRateLimitError()), patch.object(mc.time, 'sleep'):
            with self.assertRaises(mc.DownloadLimited):
                mc.fetch(['AAA'], workers=1, log=lambda _: None)
        self.assertEqual(mc.manifest().iloc[0].status, 'error')
        self.assertEqual(mc.pending(['AAA']), [])  # don't repeat throttled requests on every UI rerun
        self.assertEqual(mc.pending(['AAA'], force=True), ['AAA'])


if __name__ == '__main__':
    unittest.main()
