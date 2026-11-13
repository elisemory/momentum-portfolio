"""Historical-cap boundaries and isolated universe/regime decisions."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from momentum import backtest as bt, data, universes as uv


def fixture():
    days = pd.bdate_range('2021-01-01', '2024-02-29')
    n = len(days)
    close = pd.DataFrame({'AAA': np.linspace(10, 100, n), 'BBB': np.linspace(10, 200, n),
                          '^GSPC': np.linspace(100, 300, n), 'SPY': np.linspace(10, 30, n),
                          '^IRX': 4.0}, index=days)
    return bt.prepare_market(close.copy(), close, True), close


class UniverseTests(unittest.TestCase):
    def setUp(self):
        bt.clear_caches()
        self.temp = tempfile.TemporaryDirectory()
        self.directory = patch.object(data, 'DATA_DIR', Path(self.temp.name))
        self.directory.start()
        self.market, self.close = fixture()

    def tearDown(self):
        self.directory.stop()
        self.temp.cleanup()
        bt.clear_caches()

    def caps(self, rows):
        return uv.validate_caps(pd.DataFrame(rows, columns=['date', 'ticker', 'market_cap']))

    def test_snapshot_does_not_use_future_or_stale_caps(self):
        caps = self.caps([['2022-12-30', 'AAA', 30e9], ['2023-01-03', 'AAA', 1e9],
                          ['2022-12-29', 'BBB', 100e9], ['2023-01-03', 'CCC', 100e9]])
        got = uv.start_caps(caps, '2022-12-30', ['AAA', 'BBB', 'CCC'])
        self.assertEqual(got.at['AAA', 'market_cap'], 30e9)
        self.assertTrue(got.loc[['BBB', 'CCC'], 'market_cap'].isna().all())

    def test_cap_screen_is_frozen_at_start_not_rebalanced_later(self):
        caps = self.caps([['2022-12-30', 'AAA', 30e9], ['2022-12-30', 'BBB', 5e9],
                          ['2023-01-31', 'AAA', 1e9], ['2023-01-31', 'BBB', 100e9]])
        uv.write_frame(caps, uv.path('historical_market_caps.parquet'))
        sel = uv.Selection('market_cap', '2023-01-01', 20)
        m, mask, details = uv.prepare(sel, bt.Config(), self.market)
        self.assertEqual(details['cap_cutoff'], '2022-12-30')
        self.assertTrue(mask[:, m.tickers.index('AAA')].all())
        self.assertFalse(mask[:, m.tickers.index('BBB')].any())
        with bt.using_market(m, mask):
            result = bt.run(bt.Config(start='2023-01-01', top_n=1, buffer_rank=1, regime_filter=False))
        self.assertEqual(set(result.holdings.ticker), {'AAA'})
        _, later, _ = uv.prepare(replace(sel, start='2023-02-01'), bt.Config(), self.market)
        self.assertFalse(later[:, m.tickers.index('AAA')].any())
        self.assertTrue(later[:, m.tickers.index('BBB')].all())

    def test_threshold_is_strict_and_missing_caps_are_excluded(self):
        uv.write_frame(self.caps([['2022-12-30', 'AAA', 20e9]]), uv.path('historical_market_caps.parquet'))
        with self.assertRaisesRegex(ValueError, 'No stocks qualify'):
            uv.prepare(uv.Selection('market_cap', '2023-01-01', 20), bt.Config(), self.market)

    def test_missing_historical_caps_never_fall_back_to_current_values(self):
        with self.assertRaisesRegex(ValueError, 'Historical market caps'):
            uv.prepare(uv.Selection('market_cap', '2023-01-01', 20), bt.Config(), self.market)

    def test_zero_threshold_still_excludes_stocks_that_did_not_exist_at_start(self):
        m = replace(self.market, has_price=self.market.has_price.copy())
        m.has_price[m.dates < '2023-02-01', m.tickers.index('BBB')] = False
        _, mask, _ = uv.prepare(uv.Selection('market_cap', '2023-01-01', 0), bt.Config(), m)
        self.assertTrue(mask[:, 0].all())
        self.assertFalse(mask[:, 1].any())

    def test_bad_cap_import_does_not_replace_good_cache(self):
        uv.import_caps(b'date,ticker,market_cap\n2022-12-30,AAA,30000000000\n')
        for payload in [b'date,ticker,market_cap\n2022-12-30,AAA,-1\n',
                        b'date,ticker,market_cap\n2022-12-30,AAA,inf\n',
                        b'date,ticker,market_cap,currency\n2022-12-30,AAA,10,EUR\n']:
            with self.assertRaises(ValueError):
                uv.import_caps(payload)
        self.assertEqual(pd.read_parquet(uv.path('historical_market_caps.parquet')).market_cap.tolist(), [30e9])

    def test_nested_context_restores_market_and_cache_even_on_error(self):
        m = self.market
        other = replace(m, index_close=m.index_close.iloc[::-1].set_axis(m.dates))
        with bt.using_market(m, np.ones_like(m.has_price)):
            first = bt._regime_inputs(200)['spx'][-1]
            with self.assertRaises(RuntimeError):
                with bt.using_market(other, np.zeros_like(m.has_price)):
                    self.assertNotEqual(bt._regime_inputs(200)['spx'][-1], first)
                    raise RuntimeError('test')
            self.assertIs(bt.load_market(), m)
            self.assertEqual(bt._regime_inputs(200)['spx'][-1], first)
            self.assertTrue(bt._membership('current').all())
        self.assertIsNone(bt._market_context.get())

    def test_concurrent_sessions_do_not_share_markets_or_membership(self):
        barrier = threading.Barrier(2)
        def worker(flag):
            with bt.using_market(self.market, np.full_like(self.market.has_price, flag)):
                barrier.wait(timeout=5)
                return bool(bt._membership('current').all())
        with ThreadPoolExecutor(2) as pool:
            self.assertEqual(list(pool.map(worker, [True, False])), [True, False])

    def test_nasdaq_selector_changes_trades_and_status_not_just_label(self):
        m = self.market
        falling = pd.Series(np.linspace(500, 100, len(m.dates)), index=m.dates)
        with patch.object(uv, 'load_nasdaq_index', return_value=falling), \
             patch.object(data, 'load_nasdaq', return_value=falling), \
             bt.using_market(m, np.ones_like(m.has_price)):
            cfg = bt.Config(start='2023-01-01', top_n=1, buffer_rank=1, regime_rule='sma_slope')
            sp = bt.run(cfg)
            nd = bt.run(replace(cfg, regime_index='nasdaq100'))
            self.assertFalse(sp.holdings.empty)
            self.assertTrue(nd.holdings.empty)
            self.assertTrue(bt.live_signal(cfg)['bull'])
            self.assertFalse(bt.live_signal(replace(cfg, regime_index='nasdaq100'))['bull'])
            self.assertEqual(bt.live_signal(replace(cfg, regime_index='nasdaq100'))['spx'], 100)

    def test_missing_signal_index_quote_is_an_error_not_a_bull_signal(self):
        missing = self.market.index_close.copy()
        missing.iloc[-1] = np.nan
        with patch.object(uv, 'load_nasdaq_index', return_value=missing), \
             patch.object(data, 'load_nasdaq', return_value=missing), \
             bt.using_market(self.market, np.ones_like(self.market.has_price)):
            with self.assertRaisesRegex(ValueError, 'Missing nasdaq100'):
                bt.live_signal(bt.Config(regime_index='nasdaq100'))


if __name__ == '__main__':
    unittest.main()
