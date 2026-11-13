"""Hand-built fixtures for the CAN SLIM rules: market direction, bases, RS ratings and the sell rules."""
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from momentum import backtest as bt
from momentum import canslim as cs


def steps(start, moves):
    """Closes from a start price and a list of daily returns."""
    return start * np.cumprod(np.r_[1.0, 1 + np.asarray(moves, float)])


class MarketDirectionTests(unittest.TestCase):
    def correction_fixture(self):
        # 20 quiet up days, then 5 distribution days (down 0.3% on higher volume) alternating with up days
        moves = [0.002] * 20 + [-0.003, 0.001] * 5
        vols = [100.0] * 21 + [200.0, 100.0] * 5
        return list(moves), list(vols)

    def test_five_distribution_days_end_the_uptrend(self):
        moves, vols = self.correction_fixture()
        mk = cs.market_direction(steps(100, moves), np.array(vols))
        dist = np.flatnonzero(mk["distribution"])
        self.assertEqual(len(dist), 5)
        self.assertTrue(mk["uptrend"][: dist[-1]].all())
        self.assertFalse(mk["uptrend"][dist[-1]])
        self.assertEqual(mk["dist_count"][dist[-1]], 5)

    def test_follow_through_on_day_four_restarts_the_uptrend(self):
        moves, vols = self.correction_fixture()
        moves += [-0.01, -0.01, 0.002, 0.002, 0.002, 0.015]  # low, then day 1-3 small gains, day 4 +1.5%
        vols += [100.0, 100.0, 100.0, 100.0, 100.0, 300.0]
        mk = cs.market_direction(steps(100, moves), np.array(vols))
        n = len(moves) + 1
        self.assertEqual(list(np.flatnonzero(mk["follow_through"])), [n - 1])
        self.assertFalse(mk["uptrend"][n - 2])
        self.assertTrue(mk["uptrend"][n - 1])
        self.assertEqual(mk["dist_count"][n - 1], 0)

    def test_undercutting_the_low_resets_the_rally_attempt(self):
        moves, vols = self.correction_fixture()
        moves += [-0.01, 0.002, 0.002, -0.02, 0.002, 0.015]  # day 3 closes below the low: attempt restarts
        vols += [100.0, 100.0, 100.0, 100.0, 100.0, 300.0]
        mk = cs.market_direction(steps(100, moves), np.array(vols))
        self.assertFalse(mk["follow_through"].any())
        self.assertFalse(mk["uptrend"][-1])

    def test_distribution_day_expires_after_a_5pct_rally(self):
        moves = [0.0, -0.003] + [0.01] * 6
        vols = [100.0, 100.0, 200.0] + [100.0] * 6
        close = steps(100, moves)
        mk = cs.market_direction(close, np.array(vols))
        self.assertEqual(mk["dist_count"][2], 1)
        expired = np.flatnonzero(close >= close[2] * 1.05)[0]
        self.assertEqual(mk["dist_count"][expired - 1], 1)
        self.assertEqual(mk["dist_count"][expired], 0)
        self.assertEqual(cs.market_direction(close, np.array(vols), dist_expiry=0)["dist_count"][-1], 1)


class BuildingBlockTests(unittest.TestCase):
    def test_base_stats(self):
        w = np.array([[10, 5], [12, 6], [11, np.nan], [9, 7], [10.5, 8]], float)
        pivot, days, depth = cs.base_stats(w)
        np.testing.assert_allclose(pivot, [12, 8])
        np.testing.assert_array_equal(days, [4, 1])     # pivot 4 sessions before the end / on the last row
        np.testing.assert_allclose(depth, [0.25, 0.0])  # lowest close after the pivot: 9

    def test_base_stats_uses_latest_equal_high(self):
        pivot, days, depth = cs.base_stats(np.array([[10.0], [8.0], [10.0], [9.0]]))
        self.assertEqual((pivot[0], days[0]), (10.0, 2))
        self.assertAlmostEqual(depth[0], 0.1)

    def test_rs_rating(self):
        raw = np.array([[0.1, 0.5, 0.3, np.nan]])
        rs = cs.rs_rating(raw, np.isfinite(raw))
        np.testing.assert_array_equal(rs[0, :3], [33, 99, 66])
        self.assertTrue(np.isnan(rs[0, 3]))


class SellRuleTests(unittest.TestCase):
    """One stock: a 200-day advance to a 100 pivot, a 60-day base down to 85, a breakout close at 101 on day 260
    (bought at the day-261 open of 101.5), then a scripted path."""

    BREAKOUT, FILL = 260, 101.5

    def run_path(self, after, **kw):
        pre = np.r_[np.linspace(50, 100, 201), np.linspace(99, 85, 30), np.linspace(85, 99, 29), 101.0]
        close = np.r_[pre, after]
        n = len(close)
        opens = np.r_[close[0], close[:-1]]
        opens[self.BREAKOUT + 1] = self.FILL
        dates = pd.bdate_range("2000-01-03", periods=n)
        c2 = np.column_stack([close, np.full(n, 50.0)])
        o2 = np.column_stack([opens, np.full(n, 50.0)])
        flat = pd.Series(100.0, index=dates)
        m = bt.Market(dates=dates, tickers=["AAA", "BBB"], open=o2, close=c2, close_ff=c2, month_end=pd.DataFrame(),
                      index_close=flat, bench_close=flat, tbill_daily=flat * 0, has_price=np.isfinite(c2))
        hi = pd.DataFrame(c2).rolling(cs.BASE_LOOKBACK, min_periods=126).max().shift(1).to_numpy()
        with np.errstate(invalid="ignore"):
            new_high = c2 > hi
        rs = np.column_stack([np.full(n, 95.0), np.full(n, 10.0)]).astype(np.float32)
        p = cs.Prepared(m, np.ones((n, 2), bool), rs, new_high, np.full((n, 2), 2.0, np.float32), np.ones(n))
        cs._market.cache_clear()
        try:
            with mock.patch.object(cs, "prepare", return_value=p):
                return cs.run(cs.Config(start=str(dates[self.BREAKOUT - 5].date()), market_rule="off", cost_bps=0,
                                        max_positions=1, **kw))
        finally:
            cs._market.cache_clear()

    def test_buys_breakout_at_next_open(self):
        r = self.run_path(np.full(10, 102.0))
        buy = r.trades.iloc[0]
        self.assertEqual((buy.side, buy.price, buy.date), ("BUY", self.FILL, r.nav.index[6]))  # nav starts at day 255
        trip = r.round_trips.iloc[0]
        self.assertEqual((trip.base_days, round(trip.depth, 2), trip.pivot), (60, 0.15, 100.0))

    def test_skips_a_buy_more_than_5pct_above_pivot(self):
        with mock.patch.object(self, "FILL", 105.5):
            r = self.run_path(np.full(10, 106.0))
        self.assertTrue(r.trades.empty)

    def test_stop_loss_sells_next_open_after_close_8pct_below_cost(self):
        after = [100.0, 97.0, 94.0, 93.0, 92.0, 91.0, 91.0]  # 93.0 is -8.4% from 101.5
        r = self.run_path(np.array(after))
        sell = r.trades[r.trades.side == "SELL"].iloc[0]
        self.assertEqual(sell.price, 93.0)  # the fixture opens at the previous close
        self.assertTrue(sell.reason.startswith("stop loss"))
        self.assertEqual(sell.date, r.nav.index[5 + 1 + 4])

    def test_profit_target_after_three_weeks(self):
        after = np.r_[np.linspace(102, 118, 20), 122.5, 123.0, 123.0]  # +20% on day 21 after the buy
        r = self.run_path(after)
        sell = r.trades[r.trades.side == "SELL"].iloc[0]
        self.assertTrue(sell.reason.startswith("profit target"))
        self.assertEqual(sell.price, 122.5)

    def test_fast_gainer_is_held_eight_weeks_then_until_50_day_line_breaks(self):
        after = np.r_[np.linspace(102, 125, 10), np.linspace(125, 140, 50), [120.0, 119.0, 119.0]]
        r = self.run_path(after)
        sells = r.trades[r.trades.side == "SELL"]
        self.assertEqual(len(sells), 1)
        self.assertIn("50-day line after 8-week hold", sells.iloc[0].reason)
        self.assertEqual(sells.iloc[0].price, 120.0)
        # the same path without the 8-week rule sells at the first close 20% above cost
        r2 = self.run_path(after, fast_gain_rule=False)
        self.assertTrue(r2.trades[r2.trades.side == "SELL"].iloc[0].reason.startswith("profit target"))


if __name__ == "__main__":
    unittest.main()
