# 12-1 Momentum Backtest

A backtester and Streamlit dashboard for a monthly 12-1 momentum strategy on the S&P 500. The strategy holds the top 20 stocks equal-weighted, with a rank-40 hold buffer and a 200-day SMA crash filter.

## Run

```bash
./run_portfolio.sh  # live portfolio app:  http://localhost:8502
./run.sh            # research app:        http://localhost:8501
./run_canslim.sh    # CAN SLIM app:        http://localhost:8503
```

### Research app (`app.py`)

**Pick stocks from** selects S&P 500 (historical or current membership), today's Nasdaq-100
constituents, or current US-listed stocks screened by market cap at the test's start.
Use **Download / refresh selected universe** to populate an expanded universe's prices.
Delisted stocks and stocks without the necessary price observations are omitted.

For **US stocks by starting market cap**, the threshold is in USD billions and the universe
is frozen for the entire test. The default **Yahoo Finance estimates (free)** source downloads
historical share counts and prices automatically, then caches them for later tests. The first
download can take several minutes and resumes after interruption. **Refresh estimated market-cap
data** updates the cache; **Retry failed histories** retries only failed downloads. Rate limits
trigger bounded retries with a pause; incomplete coverage is shown explicitly and unavailable
histories are excluded. The app defaults this mode to January 2016 unless you have selected a
start yourself. Choose a start around 2016 or later: share-count coverage often begins
in 2015, varies by ticker, and may restart when a ticker changes.

Estimated starting cap uses the close before the first test session and the most recent Yahoo
share observation on or before that date, at most 366 days old. Historical prices are fetched
without dividend adjustment. Later split adjustments embedded in Yahoo Close are reversed,
and splits between the share observation and the start are applied to the share count.
Missing prices, missing share counts, and stale counts exclude that stock. Today's shares are
never backfilled into the past. Estimates are explicitly labeled, since Yahoo observations are
not a complete daily or filing-time share-count record. Download the starting-cap audit CSV
from the results to inspect dates, split factors, estimates and exclusion reasons.

The **Imported historical market caps** source remains available. Import a CSV with columns
`date,ticker,market_cap`, values in USD (not billions), and observations at the last trading
close before the first test session. Multiple dates are supported. Imported values are kept
separate from Yahoo estimates. A threshold of zero uses every current listing with a price at
the start and needs no cap data. ETFs, funds, debt and warrants are excluded; ordinary equities,
ADRs, REITs and partnerships are included.

**Bear-market signal index** independently selects S&P 500 or Nasdaq-100 for the regime
rule and displayed Bull/Bear status. Download the Nasdaq-100 signal history from its sidebar
button if needed. Benchmarks remain S&P 500 and Nasdaq-100. The optional variant comparison
uses the selected universe, period and signal index, preserving the original starting-cap
screen in both halves. Expanded prices and imported caps are cached locally under `data/`.

### Portfolio app (`portfolio_app.py`)

This app has no strategy controls. It always runs the Improved strategy (`momentum.live.LIVE_CONFIG`). It shows:

- the current holdings, simulated from 1996 to today, because the rank-40 buffer makes holdings depend on past trades
- the next rebalance: what to sell, buy, and trim
- the market regime
- the track record against the S&P 500 and Nasdaq-100

**Data refresh:**

- **Automatic:** weekday catch-up at 06:00 New York time, then every 15 minutes from 09:30 through 16:15. A background thread checks every 15 minutes and catches up after downtime. Trading rules still rebalance monthly. The performance table and chart use the latest available daily bar (which may be intraday and delayed), with SPY/QQQ total returns explicitly labeled; missing benchmark observations remain unavailable rather than appearing flat.
- **Manual:** the **Update now** button.
- **What a refresh downloads:** S&P 500 membership changes from Wikipedia, about 4 months of recent prices (older history is rescaled for dividends and splits), full history for new index members, and benchmark data.

The background thread only runs while the app is running. To refresh without the app, use cron:

```cron
*/15 * * * *  cd ~/projects/momentum-backtest && .venv/bin/python -m momentum.live --if-due
```

The `data/` folder is already filled. To re-download data, click **Download / refresh data** in the sidebar.

### CAN SLIM app (`canslim_app.py`)

A backtest and daily screen for William O'Neil's CAN SLIM rules (*How to Make Money in Stocks*, 4th ed.). It buys
breakouts from 7-week-plus bases by stocks with an RS rating of 80 or higher, on above-average volume, only
while the market is in a confirmed uptrend. It sells on an 8% stop or a 20% gain, holding fast gainers for 8 weeks.
The backtest can't test C, A and I, because free data has no point-in-time earnings. The **Today** tab checks C and A
live from Yahoo for the screened stocks.

- **Tabs:** today's orders, simulated holdings, and screener; performance against SPY, the Nasdaq-100 and the Improved
  momentum strategy; trade statistics; market direction (distribution and follow-through days); one-rule variants.
- **Data:** it also needs daily volume (`data/volume.parquet`), which the momentum apps don't download. The app offers
  to fetch it on first run. **Update prices & volume** runs the portfolio app's price update plus a volume update,
  and drops the bar for a session still in progress. Former members from the WIKI supplement get their volume
  from `data/supplement_volume.parquet` (`python -m momentum.supplement --wiki data/archive/wiki_prices.zip --volume`).
- **Results:** the book's rules returned 3.0% a year from 1996 against 10.4% for SPY. See
  [CANSLIM_FINDINGS.md](CANSLIM_FINDINGS.md), or run `python -m momentum.canslim`.

## Layout

| File | Purpose |
|---|---|
| `momentum/data.py` | Point-in-time S&P 500 membership ([fja05680/sp500](https://github.com/fja05680/sp500)), Yahoo adjusted prices (plus the WIKI supplement), ticker renames, and annual fundamentals for the quality gate |
| `momentum/supplement.py` | Builds the price supplement for former members Yahoo dropped, from the frozen Quandl WIKI archive. Run `python -m momentum.supplement --wiki data/archive/wiki_prices.zip` |
| `momentum/sector_etf.py` | Runs the same rules on the SPDR sector ETFs, for comparison. Run `python -m momentum.sector_etf --fetch` |
| `momentum/backtest.py` | Engine. `Config` holds every parameter. `run(cfg)` returns NAV, holdings, and trades. `live_signal(cfg)` returns today's ranking |
| `momentum/metrics.py` | CAGR, Sharpe, Sortino, drawdowns, calendar tables, and turnover |
| `momentum/live.py` | Data refresh, market-day schedule, current portfolio, and next-rebalance trades |
| `momentum/taxes.py` | After-tax comparison against buy-and-hold SPY. Run `python -m momentum.taxes` |
| `FINDINGS.md` | All research findings. Also shown in the research app's Findings tab |
| `momentum/experiments.py` | Robustness harness. Scores every idea on 1996–2010 and 2011–now separately. Run `python -m momentum.experiments` |
| `app.py` | Dashboard |
| `momentum/canslim.py` | CAN SLIM engine: market direction, RS rating, bases and breakouts, sell rules, screener, live C and A checks. `python -m momentum.canslim` runs the variant table |
| `canslim_app.py` | CAN SLIM dashboard |

## Using the engine from Python

```python
from momentum import backtest as bt, metrics as mt
r = bt.run(bt.Config(vol_target=0.15, score="risk_adjusted"))
print(mt.summary(r.nav, r.cash_rate, r.bench))
```

## Verifying the backtest

Run `.venv/bin/python -m unittest discover -s tests -v` to check the independent audit
calculations against small hand-calculated examples. Then run
`.venv/bin/python -m momentum.verify` (several minutes). This creates a new ignored
`verification_runs/<UTC timestamp>/` directory containing:

- A frozen copy of the working-tree source and data, a SHA-256 manifest, and installed package versions.
- `report.html` and `results.json`, separating reproduced results from independently checked calculations.
- All published variants and tax scenarios; daily cash/holdings reconciliation; fill and monthly signal audits.
- Spreadsheet-readable worksheets for ordinary, exit, re-entry and missing-price months.
- Physical input-truncation checks and a control that alters only future prices.
- SPY versus momentum, with and without the same regime filter, at 10/25/50 bps per side.
- An optional public Kenneth French market-return comparison (use `--offline` to skip downloading).
- A timestamped frozen strategy/preview record and a blank prospective fills template. These do not
  execute orders, schedule future work, or establish prospective investment performance.

The audit executes inside the frozen source copy and never refreshes production data. The checks
use the same stock-price source: a second stock-level dataset with delisting returns is still needed
to validate vendor history. The public market comparison is only a benchmark sanity check.
Published tax scenarios are reproduced but not independently validated; the stock sleeve does not
separately tax dividends embedded in adjusted prices. Output returns are decimal fractions.

## Follow-up strategy experiments

Run `.venv/bin/python -m momentum.followup` for the offline follow-up study. It freezes the
source, cached data, and experiment plan under `verification_runs/followup-<timestamp>/`, then
tests proportional portfolio-risk overlays, covariance-aware stock selection, soft sector
penalties, score-margin/trim-band trading rules, and independent trend-filter ensembles.
Primary specifications run at 10, 25, and 50 bps per side. Additional sensitivity tests,
historical subperiods, paired block-bootstrap intervals, and retrospective walk-forward
selection are saved alongside daily NAVs and a Markdown report.

The research configuration opts into signal-date extreme-jump exclusions and strict daily
month-end quote validity. Live defaults and the live strategy configuration are unchanged.
The harness checks accounting, physical input truncation, and adversarial future-price changes.
These checks do not repair missing historical securities or make historical sector labels,
security mappings, taxes, or execution assumptions independently validated.

See [FOLLOWUP_RESULTS.md](FOLLOWUP_RESULTS.md) for the recorded results and limitations.

## Timing

- **Signal:** computed on the close of the last trading day of the month. Momentum is P(end of M-2) / P(end of M-13) - 1, and the regime check is S&P 500 close > SMA200.
- **Execution:** fills at the open of the first trading day. Costs are charged per side in bps.
- **Idle cash:** earns the 13-week T-bill yield (^IRX).

See the **Data & caveats** tab for known biases.

## Improvements

Run `python -m momentum.experiments` to reproduce these numbers. The **Compare variants** tab shows the same table.

**Diagnosis.** The spec beat the S&P 500 by 12 points a year in 1996–2010 but trailed it by 4 points a year from 2011 on. The cause is the 200-day regime filter. Since 2011 it has fired on short corrections, and the S&P rose an average of 2.3% in the months it kept the strategy in T-bills.

**Improved preset:** `score="blend", regime_rule="sma_slope", max_weight=0.10`

| | CAGR | Sharpe | Max drawdown | CAGR 1996–2010 | CAGR 2011–now |
|---|---|---|---|---|---|
| Spec as written | 14.3% | 0.61 | −33% | 18.7% | 10.3% |
| Improved | 18.1% | 0.72 | −39% | 20.1% | 16.4% |
| S&P 500 (SPY) | 10.4% | 0.49 | −55% | 6.4% | 14.1% |

- **Blend score:** ranks stocks on the average percentile rank of 12-1 and 6-1 momentum.
- **Falling-SMA regime rule:** a bear market is declared only when the S&P 500 is below its 200-day SMA *and* that SMA is lower than it was 21 trading days ago. It still caught 2000–02, 2008, and 2022, and it ignored most short corrections.
- **10% position cap:** trims runaway winners. It's a risk limit; on the current data it no longer adds return.

**Robustness.** The gain holds for slope lookbacks of 5–63 days, SMA lengths of 150–250, trading costs of 25 bps, and buffers of 30–60. Across all of those, full-period CAGR stays between 16.5% and 18%. The worst drawdown is the COVID crash in February–March 2020, which happened within a single month, so no monthly rule could have avoided it.

**Tested and rejected:**
- Daily regime checks: heavy whipsaw.
- 52-week-high and smooth-momentum scores.
- Volatility targeting and inverse-volatility weights: lower risk, but also lower return.
- Sector caps: cut returns by 3–4 points a year.
- Residual momentum (net of market or sector moves): less concentrated, but 5–6 points a year lower.
- Rotating sector ETFs instead of stocks: about half the return since 2011.

**Caveat.** Roughly 45 variants were tried, and the falling-SMA rule was chosen after seeing where the spec failed. The 2011–now half is therefore not truly out-of-sample for that rule. Expect live results to fall short of the backtest.

## Data cleaning

Some delisted S&P 500 tickers were later reused by other companies, such as MCIC, CPWR, COMS, PARA, and EP. Yahoo returns the new owner's history for those symbols. They're excluded through `data.BLOCKLIST` and an automatic check for extreme price jumps.

Yahoo also deletes the history of stocks that stop trading, so many former members have no prices (survivorship bias). Two partial fixes:

- **Renamed tickers:** `data.RENAMES` maps members listed under an old symbol (for example `FB`) to the symbol Yahoo keeps their history under (`META`).
- **Price supplement:** `data/supplement_*.parquet` holds prices from the frozen Quandl WIKI archive (through February 2018) for 184 former members. `data.load_prices()` merges it in. It's built locally from the WIKI bulk export, never written into the Yahoo files, and not part of the published data bundle, so the hosted app runs on Yahoo data only. The archive itself is kept in `data/archive/` (ignored by git).

Pre-2010 failures (Lehman, Enron and others) are still missing; see FINDINGS.md section 8.

## Deployment (Streamlit Community Cloud)

The live app runs `portfolio_app.py` from this repo on [Streamlit Community Cloud](https://share.streamlit.io).

- **Data isn't committed to git.** The `.github/workflows/refresh-data.yml` Action refreshes it every weekday at 21:30 UTC, after the US close. It publishes the result as `momentum-data.tar.gz` on the `data-latest` release, replacing the previous file, so the repo doesn't grow. To refresh by hand, go to **Actions → Refresh data → Run workflow**.
- **The hosted app downloads that file** whenever it starts with an empty disk. While it's awake, it also refreshes itself throughout the US session. If Yahoo blocks the request, it falls back to the published file.
- **To deploy:** at share.streamlit.io, choose **Create app**, then:
  - Repository: `elisemory/momentum-portfolio`
  - Branch: `main`
  - Main file: `portfolio_app.py`
  - Python version (under Advanced settings): **3.11**
- **To run locally:** `uv venv .venv && uv pip install -r requirements.txt`, then `./run_portfolio.sh`. On first start, the app downloads the published data.
