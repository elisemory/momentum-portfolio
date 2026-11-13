# Follow-up strategy tests — October 23, 2026

## Conclusion

None of the four primary ideas delivers a convincing improvement over the current strategy.
The original configuration remains the best return/complexity tradeoff among these tests.
That is not evidence that its historical returns will persist: data coverage, execution
assumptions, and extensive prior strategy selection remain material limitations.

The study contains 46 strategy/cost combinations (34 distinct labels, including controls
and component sleeves), plus 20 retrospective walk-forward family/fold evaluations.
All use cached data from February 1, 1996 through October 13, 2026. No data was refreshed,
no orders were placed, and no live strategy settings were changed.

## Primary results

Trading costs: 10 bps per side. Returns are before tax. The full-period and subperiod
calculations include their first day's return and initial execution costs where applicable.
Subperiods retain the continuous portfolio history. These conventions explain small
differences from the previous quick review and the fresh-start experiments harness.

| Strategy | Full CAGR | Full Sharpe | Max drawdown | 1996–2010 CAGR | 2011+ CAGR | 2011+ Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| **Baseline** | **18.14%** | **0.717** | **−39.09%** | **20.11%** | **16.32%** | **0.677** |
| Covariance-aware additions, penalty 0.05 | 16.95% | 0.698 | −41.01% | 19.00% | 15.06% | 0.659 |
| Soft sector penalty, 0.10 | 16.20% | 0.663 | −39.41% | 18.87% | 13.73% | 0.601 |
| Score-margin and trim-band trading | 18.10% | 0.716 | −39.09% | 20.11% | 16.25% | 0.674 |
| Proportional 25% volatility overlay, weekly | 16.33% | 0.704 | −38.33% | 18.75% | 14.10% | 0.640 |
| Independent 150/200/250-day regime sleeves | 17.51% | 0.698 | −40.21% | 19.57% | 15.60% | 0.653 |
| SPY buy and hold | 10.38% | 0.492 | −55.19% | — | 14.21% | 0.777 |

### What each experiment actually tests

1. **Proportional risk overlay.** Estimate risk from the shadow baseline's actual stock
   weights and a trailing 126-session covariance matrix, with 25% diagonal shrinkage and
   at least 60 complete returns. Trade proportional portfolio units at the next open,
   checking weekly plus monthly, with a five-percentage-point exposure no-trade band.
   Selection, relative weights, and rank-buffer history stay unchanged. No leverage.
   This is an estimated-risk target, not a guaranteed realized-volatility or drawdown cap.
   Underlying trading costs remain; overlay costs are added without netting opposing orders.
2. **Covariance-aware selection.** Sequentially select new holdings from near-ranked
   candidates within 0.05 blended-score units of the best available candidate, penalizing
   covariance with retained holdings and already selected additions. Existing positions
   are not forcibly diversified. A separate soft-sector variant penalizes sector weight.
3. **Trading efficiency.** Keep an otherwise sellable stock unless a replacement improves
   its blended score by at least 0.05. Allow positions to reach 10.5% before trimming them
   to 10%. Forced universe exits and market-regime exits still occur.
4. **Regime ensemble.** Allocate equal initial capital to independent 150/200/250-day
   falling-SMA strategies, then allow allocations to drift. This is an ensemble of
   separately managed sleeves, not one stock basket at a fixed average regime exposure.

### Interpretation

- The risk overlay lowers full-period volatility from 24.15% to 21.56%, but costs 1.80
  percentage points of CAGR and reduces the worst drawdown by only 0.76 points.
- The new proportional overlay is better than the old 25% volatility-targeting setting
  (16.33% vs 14.78% CAGR; 0.704 vs 0.658 Sharpe), but not better than the baseline.
- Covariance and sector penalties lower ordinary volatility, but do not improve the worst
  drawdown in the primary tests. Less concentration is not automatically better performance.
- Trading efficiency changes very little: 3,565 trades versus 3,571 for the baseline,
  and approximately 257.4% versus 257.5% annual one-way turnover. The existing rank buffer
  already prevents many marginal trades.
- The regime ensembles do not solve the timing problem on this history.

## Cost sensitivity

Full-period CAGR, all other settings held fixed:

| Strategy | 10 bps/side | 25 bps/side | 50 bps/side |
|---|---:|---:|---:|
| Baseline | 18.14% | 17.23% | 15.73% |
| Covariance penalty 0.05 | 16.95% | 16.03% | 14.52% |
| Soft sector penalty 0.10 | 16.20% | 15.29% | 13.79% |
| Trading efficiency | 18.10% | 17.21% | 15.71% |
| Weekly 25% volatility overlay | 16.33% | 15.43% | 13.94% |
| Regime ensemble 150/200/250 | 17.51% | 16.59% | 15.08% |

None of the primary alternatives overtakes the baseline at higher costs.

## Sensitivity checks

All values below use 10 bps/side. The full generated report lists every specification.

| Variant | Full CAGR | Sharpe | Max drawdown | 2011+ CAGR |
|---|---:|---:|---:|---:|
| Light covariance penalty 0.02 | 18.03% | 0.723 | −40.34% | 16.42% |
| Strong covariance penalty 0.10 | 16.39% | 0.686 | −41.44% | 14.21% |
| Soft sector penalty 0.05 | 17.88% | 0.714 | −39.39% | 15.87% |
| Replacement margin 0.10 | 17.82% | 0.704 | −39.14% | 15.95% |
| Weekly 20% volatility overlay | 14.87% | 0.684 | −36.94% | 12.94% |
| Weekly 30% volatility overlay | 17.19% | 0.718 | −38.64% | 14.89% |
| Monthly 25% volatility overlay | 16.83% | 0.717 | −39.09% | 14.68% |
| Regime ensemble 175/200/225 | 17.44% | 0.695 | −39.67% | 15.40% |
| Add 3–1 momentum | 16.70% | 0.685 | −38.80% | 13.54% |
| Exclude highest-volatility 10% | 14.01% | 0.654 | −40.95% | 9.54% |
| Exclude highest-volatility 20% | 10.20% | 0.511 | −41.42% | 7.53% |
| Hold buffer 60 | 17.07% | 0.680 | −37.01% | 14.86% |
| Four staggered sleeves | 17.00% | 0.680 | −38.33% | 14.26% |

The light covariance penalty is the closest candidate: slightly higher Sharpe, slightly
lower full-period return, and a worse maximum drawdown. A post-run paired moving-block
bootstrap for its 2011+ advantage gives an approximate 95% CAGR-difference interval of
**−1.01 to +1.11 percentage points/year**, and a Sharpe-difference interval of
**−0.028 to +0.055**. Both include zero. This is not convincing evidence of an improvement.
The bootstrap uses 500 draws of 63-session blocks and does not correct for multiple testing.

Timing sensitivity is large: individual trading-day-signal sleeves at offsets 0/5/10/15
return 17.36%/15.44%/18.02%/16.73%. Picking the best offset after observing these results
would be another optimization, not independent validation. The equal-initial-capital
staggered ensemble returns 17.00%.

## Retrospective walk-forward check

For each family, select the highest-Sharpe specification over the preceding ten years,
then evaluate it over the next five years. The unchanged baseline is an available choice.
Both selected strategy and control start each test fold from fresh cash and pay entry
costs. Folds are independent and are not concatenated into a purported investable return.
Current-sector-label variants are excluded from model selection.

CAGR difference versus the fresh-start baseline, in percentage points/year:

| Test fold | Covariance selection | Trading efficiency | Risk overlay | Regime ensemble |
|---|---:|---:|---:|---:|
| 2006–2010 | −0.62 | 0.00 | −0.93 | −1.51 |
| 2011–2015 | +0.04 | −0.01 | 0.00 | 0.00 |
| 2016–2020 | −0.86 | +0.32 | 0.00 | −1.71 |
| 2021–2025 | −1.86 | −0.77 | 0.00 | 0.00 |

For risk overlays and regime ensembles, zero generally means the training procedure chose
the baseline, not that a modified strategy matched it. The partial 2026 fold is recorded
separately in the CSV; three families choose the baseline and the trim-band selection
does not improve it. There is no repeatable improvement across the completed test folds.

These are retrospective checks on already examined history, not untouched holdouts.
Freezing the final experiment plan makes the run reproducible; it does not undo prior
research or make the parameter search statistically independent.

## Data and validation findings

- **Point-in-time research filtering:** extreme-jump exclusions now use only information
  available by the signal date. Research month-end prices require a valid quote within
  the preceding five daily sessions. The opt-in corrections leave the actual baseline
  NAV and trades unchanged on the current cached dataset.
- **Real-data causality checks:** truncating input prices at 2000-12-29, 2010-12-31,
  2020-03-31, and 2022-12-30 leaves historical NAV and trades exactly unchanged. Injecting
  extreme AAPL prices after 2010 also leaves earlier results exactly unchanged.
- **Accounting:** research sleeve NAVs reconcile to the emitted fills. Frozen input/source
  hashes remain unchanged. The current test suite passes all 28 tests, including the
  sector-ETF adapter compatibility check added after the frozen experiment run.
- **Execution remains partly hypothetical:** the baseline has eight sales without matching
  adjusted opening quotes. Four are automatic `stopped trading` exits; four are other
  sells using stale-price fallback. Matching the accounting is not evidence these fills
  were executable.
- **Stress on the four automatic missing-quote liquidations:** reducing proceeds by 25%
  lowers full CAGR from 18.14% to 17.96%; reducing proceeds to zero lowers it to 17.42%.
  The 2011+ CAGR falls from 16.32% to 14.94% in the zero-proceeds scenario. These are severe
  sensitivity assumptions, not estimates of actual corporate-action proceeds. The other
  four stale-price fallback sales are not haircut by this test.
- **Still unresolved:** missing historical securities, independent stock-level prices and
  corporate-action proceeds, retrospective ticker mappings, current sector classifications,
  and historical vendor revisions. Stressing observed holdings cannot repair absent names.
- **Taxes:** no new after-tax claim is made. The existing tax model omits separate stock
  dividend taxation and wash sales, so after-tax optimization needs a different model.

## Files and reproduction

Run:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m momentum.followup
```

The recorded final run is `verification_runs/followup-20261023-final/`:

- `report.md`: primary cost tables and every 10-bps sensitivity result.
- `summary.csv`: full-period, split-period, and additional historical-window metrics.
- `daily_nav.csv`, `annual_returns.csv`: complete return paths.
- `walk_forward.csv`: training choices and test-fold comparisons.
- `bootstrap.csv`: primary paired uncertainty checks.
- `plan.json`, `study_configs.json`: frozen parameters and input/source hashes.
- `validation.json`, `execution_audit.csv`: causality and accounting checks.
- `source/`, `data/`: frozen source and cached inputs used for the experiment.

Additional post-run diagnostics in the same directory: `comparison.png`,
`non_open_fills.csv`, and `bootstrap_light_covariance.csv`.

The live configuration remains unchanged. No tested addition is promoted to production.
