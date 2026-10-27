"""Regression checks for stale benchmark returns and automatic price refreshes."""
import unittest
import io
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from momentum import data, live
from momentum.performance import period_start, trailing_returns


class PerformanceTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2025-12-30", "2026-10-27")

    def test_trailing_periods_use_sessions_and_calendar_cutoffs(self):
        expected = {"1D": "2026-10-26", "5D": "2026-10-20", "1M": "2026-09-25",
                    "3M": "2026-07-27", "6M": "2026-04-27", "YTD": "2025-12-31"}
        for period, date in expected.items():
            with self.subTest(period=period):
                self.assertEqual(period_start(self.dates, period), pd.Timestamp(date))

    def test_returns_include_today_against_the_correct_base(self):
        prices = pd.DataFrame({"SPY": 100.0, "QQQ": 200.0}, index=self.dates)
        prices.loc["2026-10-26"] = [110, 220]
        prices.loc["2026-10-27"] = [121, 242]
        result = trailing_returns(prices)
        np.testing.assert_allclose(result["1D"], [0.1, 0.1])
        np.testing.assert_allclose(result["5D"], [0.21, 0.21])
        np.testing.assert_allclose(result["1M"], [0.21, 0.21])

    def test_stale_nasdaq_does_not_become_zero_return(self):
        nav = pd.Series(100.0, index=self.dates)
        nasdaq = pd.Series(200.0, index=self.dates[self.dates <= "2026-10-13"])
        result = trailing_returns(pd.DataFrame({"Portfolio": nav, "Nasdaq": nasdaq}))
        self.assertTrue(result.loc["Nasdaq"].isna().all())
        self.assertEqual(result.loc["Portfolio", "1D"], 0)

    def test_missing_base_is_unavailable_even_with_current_quote(self):
        prices = pd.DataFrame({"SPY": 100.0}, index=self.dates)
        prices.loc["2026-09-25", "SPY"] = np.nan
        result = trailing_returns(prices)
        self.assertTrue(np.isnan(result.loc["SPY", "1M"]))
        self.assertEqual(result.loc["SPY", "1D"], 0)

    def test_short_history_clamps_to_first_observation(self):
        dates = pd.to_datetime(["2026-10-26", "2026-10-27"])
        for period in ("5D", "1M", "YTD", "10Y", "Max"):
            self.assertEqual(period_start(dates, period), dates[0])


class RefreshTests(unittest.TestCase):
    def et(self, stamp):
        return datetime.fromisoformat(stamp).replace(tzinfo=live.MARKET_TZ)

    def test_in_month_prices_refresh_without_waiting_for_rebalance(self):
        now = self.et("2026-10-27T10:46:00")
        with patch.object(live, "last_refresh", return_value=self.et("2026-10-27T10:31:00")):
            self.assertTrue(live.should_refresh(now))
        with patch.object(live, "last_refresh", return_value=self.et("2026-10-27T10:45:30")):
            self.assertFalse(live.should_refresh(now))
        self.assertEqual(live.next_scheduled(now), self.et("2026-10-27T11:00:00"))

    def test_final_close_and_weekend_slots(self):
        self.assertEqual(live.next_scheduled(self.et("2026-10-23T16:01:00")),
                         self.et("2026-10-23T16:15:00"))
        saturday = self.et("2026-10-24T12:00:00")
        self.assertEqual(live.scheduled_time(saturday), self.et("2026-10-23T16:15:00"))
        self.assertEqual(live.next_scheduled(saturday), self.et("2026-10-26T06:00:00"))

    def test_schedule_uses_new_york_time_across_dst(self):
        now = datetime.fromisoformat("2026-11-02T15:46:00+00:00")
        self.assertEqual(live.scheduled_time(now), self.et("2026-11-02T10:45:00"))
        self.assertEqual(live.next_scheduled(self.et("2026-10-30T17:00:00")),
                         self.et("2026-11-02T06:00:00"))

    def test_missing_latest_spy_rejects_download(self):
        prices = pd.DataFrame({data.INDEX: [100., 101.], data.BENCH: [100., np.nan],
                               "AAA": [50., 51.]}, index=pd.to_datetime(["2026-10-26", "2026-10-27"]))
        with self.assertRaisesRegex(RuntimeError, "no SPY price"):
            live._check_download(prices, ["AAA"], now=datetime(2026, 10, 27))

    def test_benchmark_failure_does_not_mark_refresh_successful(self):
        with patch.object(live, "refresh_membership"), patch.object(live, "update_prices"), \
                patch.object(data, "fetch_nasdaq", side_effect=RuntimeError("download failed")), \
                patch.object(live, "STATE_FILE") as state:
            with self.assertRaisesRegex(RuntimeError, "download failed"):
                live.refresh(log=lambda *_: None)
            state.write_text.assert_not_called()

    def test_stale_benchmark_download_does_not_mark_refresh_successful(self):
        prices = pd.DataFrame({data.INDEX: [100.]}, index=pd.to_datetime(["2026-10-27"]))
        nasdaq = pd.Series([200.], index=pd.to_datetime(["2026-10-13"]))
        with patch.object(live, "refresh_membership"), patch.object(live, "update_prices"), \
                patch.object(data, "fetch_nasdaq"), patch.object(data, "fetch_spy_dividends"), \
                patch.object(data, "load_prices", return_value=(prices, prices)), \
                patch.object(data, "load_nasdaq", return_value=nasdaq), \
                patch.object(live, "STATE_FILE") as state:
            with self.assertRaisesRegex(RuntimeError, "Nasdaq data is stale"):
                live.refresh(log=lambda *_: None)
            state.write_text.assert_not_called()

    def test_fallback_bundle_cannot_roll_prices_back(self):
        old = pd.DataFrame({data.BENCH: [100.]}, index=pd.to_datetime(["2026-10-23"]))
        current = pd.DataFrame({data.BENCH: [110.]}, index=pd.to_datetime(["2026-10-27"]))
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w:gz") as archive:
            contents = old.to_parquet()
            info = tarfile.TarInfo("close.parquet")
            info.size = len(contents)
            archive.addfile(info, io.BytesIO(contents))
        with tempfile.TemporaryDirectory() as directory, patch.object(data, "DATA_DIR", Path(directory)), \
                patch.object(live.requests, "get") as get:
            path = Path(directory) / "close.parquet"
            current.to_parquet(path)
            get.return_value.status_code = 200
            get.return_value.iter_content.return_value = [payload.getvalue()]
            self.assertTrue(live.download_data(log=lambda *_: None))
            pd.testing.assert_frame_equal(pd.read_parquet(path), current)


if __name__ == "__main__":
    unittest.main()
