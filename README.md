# 12-1 Momentum Backtest

A backtester and Streamlit dashboard for a monthly 12-1 momentum strategy on the S&P 500. The strategy holds the top 20 stocks equal-weighted, with a rank-40 hold buffer and a 200-day SMA crash filter.

## Run

```bash
./run_portfolio.sh  # live portfolio app:  http://localhost:8502
./run.sh            # research app:        http://localhost:8501
```

### Portfolio app (`portfolio_app.py`)

This app has no strategy controls. It always runs the Improved strategy (`momentum.live.LIVE_CONFIG`). It shows:

- the current holdings, simulated from 1996 to today, because the rank-40 buffer makes holdings depend on past trades
- the next rebalance: what to sell, buy, and trim
- the market regime
- the track record against the S&P 500 and Nasdaq-100

**Data refresh:**

- **Automatic:** on the 1st of each month at 06:00 local time, run by a background thread in the app. The thread checks every 15 minutes. If the app wasn't running on the 1st, it refreshes when it next starts.
- **Manual:** the **Update now** button.
- **What a refresh downloads:** S&P 500 membership changes from Wikipedia, about 4 months of recent prices (older history is rescaled for dividends and splits), full history for new index members, and benchmark data.

The background thread only runs while the app is running. To refresh without the app, use cron:

```cron
0 6 1 * *  cd ~/projects/momentum-backtest && .venv/bin/python -m momentum.live --if-due
```

The `data/` folder is already filled. To re-download data, click **Download / refresh data** in the sidebar.

## Layout

| File | Purpose |
|---|---|
| `momentum/data.py` | Point-in-time S&P 500 membership ([fja05680/sp500](https://github.com/fja05680/sp500)), Yahoo adjusted prices (plus the WIKI supplement), ticker renames, and annual fundamentals for the quality gate |
| `momentum/supplement.py` | Builds the price supplement for former members Yahoo dropped, from the frozen Quandl WIKI archive. Run `python -m momentum.supplement --wiki data/archive/wiki_prices.zip` |
| `momentum/sector_etf.py` | Runs the same rules on the SPDR sector ETFs, for comparison. Run `python -m momentum.sector_etf --fetch` |
| `momentum/backtest.py` | Engine. `Config` holds every parameter. `run(cfg)` returns NAV, holdings, and trades. `live_signal(cfg)` returns today's ranking |
| `momentum/metrics.py` | CAGR, Sharpe, Sortino, drawdowns, calendar tables, and turnover |
| `momentum/live.py` | Data refresh, monthly schedule, current portfolio, and next-rebalance trades |
| `momentum/taxes.py` | After-tax comparison against buy-and-hold SPY. Run `python -m momentum.taxes` |
| `FINDINGS.md` | All research findings. Also shown in the research app's Findings tab |
| `momentum/experiments.py` | Robustness harness. Scores every idea on 1996–2010 and 2011–now separately. Run `python -m momentum.experiments` |
| `app.py` | Dashboard |

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

- **Data isn't committed to git.** The `.github/workflows/refresh-data.yml` Action refreshes it on the 1st of each month at 10:00 UTC. It publishes the result as `momentum-data.tar.gz` on the `data-latest` release, replacing the previous file, so the repo doesn't grow. To refresh by hand, go to **Actions → Refresh data → Run workflow**.
- **The hosted app downloads that file** whenever it starts with an empty disk. While it's awake, it also refreshes itself on the 1st. If Yahoo blocks the request, it falls back to the published file.
- **To deploy:** at share.streamlit.io, choose **Create app**, then:
  - Repository: `elisemory/momentum-portfolio`
  - Branch: `main`
  - Main file: `portfolio_app.py`
  - Python version (under Advanced settings): **3.11**
- **To run locally:** `uv venv .venv && uv pip install -r requirements.txt`, then `./run_portfolio.sh`. On first start, the app downloads the published data.
