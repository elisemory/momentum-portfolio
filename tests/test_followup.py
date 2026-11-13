"""Causality, selection and self-financing checks for opt-in research variants."""
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from momentum import backtest as bt, data, sector_etf
from momentum.followup import Study, check_days, ensemble, risk_overlay, sleeve_frame


class FollowupTests(unittest.TestCase):
    def tearDown(self):
        bt.clear_caches()

    def prices(self):
        dates = pd.bdate_range("1999-01-01", periods=900)
        generator = np.random.default_rng(42)
        returns = generator.normal([0.001, 0.0006, 0.0002], 0.008, size=(len(dates), 3))
        closes = pd.DataFrame(100 * np.exp(returns.cumsum(axis=0)), index=dates, columns=["AAA", "BBB", "CCC"])
        closes[data.INDEX] = np.linspace(100, 200, len(dates))
        closes[data.BENCH] = closes[data.INDEX]
        closes[data.TBILL] = 4.0
        return closes.shift(1).fillna(100), closes

    def membership(self, dates, tickers, mode):
        return pd.DataFrame(True, index=dates, columns=tickers)

    def test_point_in_time_filter_does_not_change_history_after_future_jumps(self):
        opens, closes = self.prices()
        cutoff = closes.index[650]
        config = bt.Config(start="2000-03-01", end=str(cutoff.date()), top_n=2, buffer_rank=3,
                           score="blend", regime_filter=False, point_in_time_filter=True)
        with patch.object(data, "load_prices", return_value=(opens, closes)), \
                patch.object(data, "membership_matrix", side_effect=self.membership):
            original = bt.run(config)
        changed = closes.copy()
        changed.loc[changed.index[700:704], "AAA"] = [1e8, 1.0, 1e8, 1.0]
        bt.clear_caches()
        with patch.object(data, "load_prices", return_value=(opens, changed)), \
                patch.object(data, "membership_matrix", side_effect=self.membership):
            future_changed = bt.run(config)
            market = bt.load_market(True)
            column = market.tickers.index("AAA")
            self.assertTrue(market.extreme_ok[650, column])
            self.assertFalse(market.extreme_ok[703, column])
        pd.testing.assert_series_equal(original.nav, future_changed.nav)
        pd.testing.assert_frame_equal(original.trades, future_changed.trades)

    def test_physical_truncation_preserves_nav_and_trades(self):
        opens, closes = self.prices()
        cutoff = closes.index[650]
        config = bt.Config(start="2000-03-01", end=str(cutoff.date()), top_n=2,
                           score="blend", point_in_time_filter=True, covariance_penalty=0.05)
        with patch.object(data, "load_prices", return_value=(opens, closes)), \
                patch.object(data, "membership_matrix", side_effect=self.membership):
            original = bt.run(config)
        bt.clear_caches()
        with patch.object(data, "load_prices", return_value=(opens.loc[:cutoff], closes.loc[:cutoff])), \
                patch.object(data, "membership_matrix", side_effect=self.membership):
            truncated = bt.run(config)
        pd.testing.assert_series_equal(original.nav, truncated.nav)
        pd.testing.assert_frame_equal(original.trades, truncated.trades)

    def test_month_end_uses_last_session_not_last_nonmissing_monthly_quote(self):
        opens, closes = self.prices()
        days = closes.loc["2000-12"].index
        closes.loc[days[-8:], "AAA"] = np.nan
        with patch.object(data, "load_prices", return_value=(opens, closes)):
            market = bt.load_market(True)
        self.assertTrue(np.isnan(market.month_end.loc[pd.Period("2000-12"), "AAA"]))

    def test_covariance_does_not_read_future(self):
        opens, closes = self.prices()
        with patch.object(data, "load_prices", return_value=(opens, closes)):
            market = bt.load_market(True)
        before = bt.trailing_covariance(market, 500, [0, 1])
        market.close_ff = market.close_ff.copy()
        market.close_ff[501:] *= 1000
        np.testing.assert_array_equal(before, bt.trailing_covariance(market, 500, [0, 1]))

    def test_covariance_penalty_prefers_less_correlated_near_ranked_stock(self):
        config = bt.Config(covariance_penalty=0.05)
        ranks = pd.DataFrame({"col": [1, 2], "score": [0.99, 0.97]})
        covariance = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]])
        with patch.object(bt, "trailing_covariance", return_value=covariance):
            order = bt.diversified_order(config, None, 0, ranks, {0: 100.0}, [1, 2], 1, None)
        self.assertEqual(order[0], 2)

    def test_diversification_does_not_reach_below_near_score_window(self):
        config = bt.Config(sector_penalty=1.0)
        ranks = pd.DataFrame({"col": [1, 2], "score": [0.99, 0.80]})
        sectors = np.array(["Tech", "Tech", "Health"])
        order = bt.diversified_order(config, None, 0, ranks, {0: 100.0}, [1, 2], 1, sectors)
        self.assertEqual(order[0], 1)

    def test_soft_sector_penalty_prefers_different_sector(self):
        config = bt.Config(sector_penalty=0.1)
        ranks = pd.DataFrame({"col": [1, 2], "score": [0.99, 0.97]})
        sectors = np.array(["Tech", "Tech", "Health"])
        order = bt.diversified_order(config, None, 0, ranks, {0: 100.0}, [1, 2], 1, sectors)
        self.assertEqual(order[0], 2)

    def overlay_fixture(self):
        dates = pd.to_datetime(["2024-12-31", "2025-01-02", "2025-01-03", "2025-01-06"])
        frame = pd.DataFrame({"nav": [100_000.0] * 3, "open_nav": [100_000.0] * 3,
                              "open_equity": [100_000.0] * 3, "invested": [1.0] * 3,
                              "notional": [0.0] * 3, "lagged_vol": [0.5] * 3}, index=dates[1:])
        study = Study("test", frame, pd.Series(0.0, index=dates[1:]), 10, {})
        return dates, study

    def test_disabled_overlay_exactly_preserves_baseline(self):
        dates, study = self.overlay_fixture()
        study.frame["nav"] = [101_000.0, 102_000.0, 103_000.0]
        study.frame["open_nav"] = [100_000.0, 101_000.0, 102_000.0]
        with patch.object(bt, "load_market", return_value=SimpleNamespace(dates=dates)):
            result = risk_overlay(study, 0)
        pd.testing.assert_series_equal(study.frame.nav, result.frame.nav)

    def test_overlay_fee_and_exposure_are_self_financing(self):
        dates, study = self.overlay_fixture()
        with patch.object(bt, "load_market", return_value=SimpleNamespace(dates=dates)):
            result = risk_overlay(study, 0.25)
        np.testing.assert_allclose(result.frame.nav, 99_950.0)
        np.testing.assert_allclose(result.frame.invested, 50_000.0 / 99_950.0)
        self.assertEqual(result.frame.notional.iloc[0], 50_000.0)

    def test_overlay_band_skips_small_exposure_change(self):
        dates, study = self.overlay_fixture()
        with patch.object(bt, "load_market", return_value=SimpleNamespace(dates=dates)):
            result = risk_overlay(study, 0.48)
        np.testing.assert_allclose(result.frame.nav, 100_000.0)
        np.testing.assert_allclose(result.frame.invested, 1.0)

    def test_overlay_does_not_apply_new_exposure_to_overnight_return(self):
        dates, study = self.overlay_fixture()
        study.frame.loc[dates[1], ["nav", "open_nav", "open_equity"]] = 90_000.0
        study.cost_bps = 0
        with patch.object(bt, "load_market", return_value=SimpleNamespace(dates=dates)):
            result = risk_overlay(study, 0.25)
        self.assertEqual(result.frame.nav.iloc[0], 90_000.0)

    def test_check_schedule_uses_first_session_of_week_not_only_mondays(self):
        dates = pd.to_datetime(["2025-01-21", "2025-01-22"])
        previous = pd.to_datetime(["2025-01-17", "2025-01-21"])
        np.testing.assert_array_equal(check_days(dates, previous, True), [True, False])

    def test_ensemble_starts_equal_and_allows_drift(self):
        _, first = self.overlay_fixture()
        second = Study("second", first.frame.copy(), first.rates, 10, {})
        second.frame["nav"] *= 2
        second.frame["invested"] = 0.5
        result = ensemble("combined", [first, second])
        np.testing.assert_allclose(result.frame.nav, 150_000.0)
        np.testing.assert_allclose(result.frame.invested, 2 / 3)

    def test_replay_matches_engine_with_research_options(self):
        opens, closes = self.prices()
        config = bt.Config(start="2000-03-01", top_n=2, buffer_rank=2, score="blend",
                           point_in_time_filter=True, replacement_margin=0.05,
                           max_weight=0.60, cap_trim_band=0.005)
        with patch.object(data, "load_prices", return_value=(opens, closes)), \
                patch.object(data, "membership_matrix", side_effect=self.membership):
            result = bt.run(config)
            frame = sleeve_frame(result)
            overlay = risk_overlay(Study("test", frame, result.cash_rate, 10,
                                        {"point_in_time_filter": config.point_in_time_filter}), 0)
        np.testing.assert_allclose(frame.nav, result.nav, rtol=1e-10)
        np.testing.assert_allclose(overlay.frame.nav, result.nav, rtol=1e-10)

    def test_replacement_requires_score_advantage(self):
        opens, closes = self.prices()
        config = bt.Config(start="2000-03-01", end="2000-04-03", top_n=1, buffer_rank=1,
                           regime_filter=False, point_in_time_filter=True)
        def ranks(config, market, signal, month):
            names = ["AAA", "BBB"] if str(month) == "2000-03" else ["BBB", "AAA"]
            return pd.DataFrame({"ticker": names, "col": [market.tickers.index(name) for name in names],
                                 "momentum": [0.5, 0.4], "score": [0.99, 0.97], "rank": [1, 2]})
        with patch.object(data, "load_prices", return_value=(opens, closes)), \
                patch.object(data, "membership_matrix", side_effect=self.membership), \
                patch.object(bt, "rank_universe", side_effect=ranks):
            control = bt.run(config)
            buffered = bt.run(replace(config, replacement_margin=0.05))
        self.assertEqual(control.positions.ticker.tolist(), ["BBB"])
        self.assertEqual(buffered.positions.ticker.tolist(), ["AAA"])
        self.assertLess(len(buffered.trades), len(control.trades))

    def test_cap_trim_band_skips_small_excess(self):
        opens, closes = self.prices()
        closes[["AAA", "BBB"]] = 100.0
        closes.loc["2000-03-15":, "AAA"] = 151.0
        opens[["AAA", "BBB"]] = closes[["AAA", "BBB"]].shift(1).fillna(100.0)
        config = bt.Config(start="2000-03-01", end="2000-04-03", top_n=2, buffer_rank=2,
                           regime_filter=False, point_in_time_filter=True, max_weight=0.60, cost_bps=0)
        def ranks(config, market, signal, month):
            return pd.DataFrame({"ticker": ["AAA", "BBB"], "col": [0, 1],
                                 "momentum": [0.5, 0.4], "score": [0.99, 0.97], "rank": [1, 2]})
        with patch.object(data, "load_prices", return_value=(opens, closes)), \
                patch.object(data, "membership_matrix", side_effect=self.membership), \
                patch.object(bt, "rank_universe", side_effect=ranks):
            control = bt.run(config)
            buffered = bt.run(replace(config, cap_trim_band=0.005))
        self.assertTrue((control.trades.side == "TRIM").any())
        self.assertFalse((buffered.trades.side == "TRIM").any())

    def test_delisting_stress_applies_to_unquoted_held_stock(self):
        opens, closes = self.prices()
        closes[data.TBILL] = 0.0
        closes["AAA"] = 100.0
        closes.loc["2000-03-07":, "AAA"] = np.nan
        opens["AAA"] = closes["AAA"]
        config = bt.Config(start="2000-03-01", end="2000-03-31", top_n=1, buffer_rank=1,
                           regime_filter=False, point_in_time_filter=True, cost_bps=0)
        def ranks(config, market, signal, month):
            return pd.DataFrame({"ticker": ["AAA"], "col": [0], "momentum": [0.5],
                                 "score": [0.99], "rank": [1]})
        with patch.object(data, "load_prices", return_value=(opens, closes)), \
                patch.object(data, "membership_matrix", side_effect=self.membership), \
                patch.object(bt, "rank_universe", side_effect=ranks):
            control = bt.run(config)
            stressed = bt.run(replace(config, delisting_haircut=1.0))
        self.assertEqual(control.nav.iloc[-1], 100_000.0)
        self.assertEqual(stressed.nav.iloc[-1], 0.0)

    def test_sector_etf_adapter_accepts_optional_research_arguments(self):
        opens, closes = self.prices()
        with patch.object(data, "load_prices", return_value=(opens, closes)):
            market = bt.load_market()
        with patch.object(sector_etf, "market", return_value=market):
            result = sector_etf.run(start="2000-03-01", end="2000-04-03")
        self.assertGreater(result.nav.iloc[-1], 0)
        self.assertEqual(len(result.positions), 3)


if __name__ == "__main__":
    unittest.main()
