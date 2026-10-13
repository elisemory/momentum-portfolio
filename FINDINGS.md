# Research findings: 12-1 momentum on the S&P 500

These results come from the backtests in this repo: February 1996 to October 2026, point-in-time S&P 500 members, Yahoo adjusted prices, 10 bps cost per side, fills at the open on the first trading day of each month, and idle cash earning the 13-week T-bill rate. They were recorded on 2026-10-13. To reproduce them:

- Strategy variants: `python -m momentum.experiments`
- Tax comparison: `python -m momentum.taxes`

## 1. The spec as written

| | Spec | S&P 500 (SPY, dividends reinvested) |
|---|---|---|
| CAGR | 13.8% | 10.4% |
| Sharpe | 0.59 | 0.49 |
| Max drawdown | −34% | −55% |
| Turnover per year | 187% | — |
| CAGR 1996–2010 | 18.5% | 6.4% |
| CAGR 2011–now | **9.1%** | **14.1%** |

- **The spec's own claims don't hold.** It claims a Sharpe of about 1.0 and a −24% to −28% drawdown. The backtest gives a Sharpe of 0.59 and a −34% drawdown. With today's members as the universe (survivorship bias), the result is 27.9% CAGR and a Sharpe of 1.01, so the spec's figures probably come from a biased backtest.
- **The single-stock claim doesn't hold either.** The spec says one stock returns −10.2% a year. The backtest gives +22.5% CAGR with an 82% drawdown.
- **All of the outperformance came before 2011.** The strategy beat the S&P in only 16 of 31 calendar years.

## 2. Why it lagged after 2011: the regime filter

| | 1996–2010 | 2011–now |
|---|---|---|
| With the 200-day SMA filter | 18.5% | 9.1% |
| Without the filter | 11.2% | 16.0% |

- **The filter helped before 2011 and hurt after.** It sat in T-bills through 2000–02 and 2008. Since 2011 it fired on 31 months of corrections, and the S&P averaged **+2.5% in the month after each exit**, which is classic whipsaw on V-shaped recoveries.
- **It adds the drawdown protection, not the return.** Without it, the full-period CAGR is about the same (13.9%), but the max drawdown is −63%.
- **While invested, the strategy is high-beta:** about 1.3× the S&P's moves, with a 0.76 correlation.

## 3. What worked

A variant counts as robust when it improves the Sharpe in **both** halves, 1996–2010 and 2011–now, or ties in one and improves the other.

| Idea | CAGR | Sharpe | Max DD | 96–10 | 11–now | Verdict |
|---|---|---|---|---|---|---|
| Spec | 13.8% | 0.59 | −34% | 18.5% | 9.1% | baseline |
| **Bear only if the SMA is falling** | 15.9% | 0.63 | −37% | 18.5% | 13.2% | ties, then improves |
| **Blend of 12-1 and 6-1 ranks** | 15.0% | 0.65 | −31% | 17.9% | 12.4% | ties, then improves |
| **10% max position** | ≈ | +0.01 | ≈ | ≈ | ≈ | small free gain |
| **Improved (all three)** | **17.7%** | **0.71** | −36% | 18.7% | **16.9%** | ties, then improves |

- **Falling-SMA rule.** A bear market is declared only when the S&P 500 is below its 200-day SMA *and* that SMA is lower than it was 21 trading days ago. It still caught 2000–02, 2008, and 2022, and it ignored most short corrections.
- **Robustness of the Improved version.** Full-period CAGR stays at 16–18% across slope lookbacks of 5–63 days, SMA lengths of 150–250, buffers of 30–60, and costs of 25 bps.
- **Worst drawdown.** The COVID crash in February–March 2020 cost 36%. It happened within one month, and no monthly rule could have avoided it.

## 4. What didn't work

| Idea | Result |
|---|---|
| Checking the regime daily | Whipsaw: 470% turnover and 10.4% CAGR. Daily checks plus fast re-entry gave 1,320% turnover. |
| Hysteresis band (±2%) | Lower return and a deeper drawdown |
| Re-entering above the 50-day SMA | Helps 2011+, hurts 1996–2010 |
| SMA plus absolute momentum | Helps 2011+, hurts 1996–2010, −46% max drawdown |
| 10-month SMA (Faber) | Worse in both halves |
| Keeping 50% invested in bear markets | Helps 2011+, hurts 1996–2010 |
| Proximity to the 52-week high | 8.0% CAGR, 647% turnover |
| Smooth momentum ("frog in the pan") | Worse in both halves |
| Return ÷ volatility | Lower CAGR and lower drawdown, about the same Sharpe |
| Volatility targeting (15–25%) | Lower risk, but proportionally lower return |
| Inverse-volatility weights | Same pattern as volatility targeting |
| Sector cap (4–6 names per sector) | −3 to −4 points of CAGR. Concentration in leading sectors is part of momentum's return. |
| 50% SPY core | A smoother ride with a lower return |
| Equal-weight reset every month | Slightly worse than the spec's no-trim rule |

## 5. Basket size

| N | CAGR | Sharpe | Max DD | Turnover |
|---|---|---|---|---|
| 10 | 14.7% | 0.57 | −38% | 200% |
| 15 | 14.8% | 0.60 | −36% | 193% |
| 20 | 13.5% | 0.58 | −34% | 187% |
| 40 | 12.6% | 0.62 | −26% | 159% |
| 50 | 12.4% | 0.63 | −24% | 148% |
| 100 | 11.2% | 0.66 | −26% | 107% |

The buffer was always 2×N. Return peaks at 10–15 names and the Sharpe ratio rises with N. The differences, about 3.5 points of CAGR across all sizes, fall within the ±3.5-point statistical uncertainty of a 30-year backtest, so 20 names is a reasonable middle.

## 6. Taxes (taxable account, $100k start)

| Scenario | CAGR 1996–now | CAGR 2011–now |
|---|---|---|
| SPY, buy and hold, dividends taxed at 15%, never sold | 10.1% | 13.8% |
| SPY, as above, sold at the end (15% long-term rate) | 9.6% | 12.9% |
| Improved, tax-free account (IRA/401k) | **17.7%** | **16.9%** |
| Improved, all gains taxed at 30% | 13.6% | 13.3% |
| Improved, 30% short-term / 15% long-term | 14.3% | 13.9% |
| Spec, all gains taxed at 30% | 10.3% | 6.9% |

- **Tax drag is about 3.5–4 points of CAGR a year.** About two-thirds of realized gains are short-term. T-bill interest is taxed as ordinary income.
- **Since 2011, the after-tax result roughly ties buy-and-hold SPY.** The pre-tax edge disappears in a taxable account.
- **Conclusion:** run the strategy in a tax-advantaged account. In a taxable account, SPY buy-and-hold is about as good since 2011, and simpler.

## 7. Comparison with the Nasdaq-100

The benchmark is QQQ with dividends reinvested, from March 1999. Before that it uses the Nasdaq-100 price index.

| | CAGR (Sharpe) 1996–now | 1996–2010 | 2011–now | Max DD |
|---|---|---|---|---|
| Improved | **17.7% (0.71)** | **18.7% (0.73)** | 16.9% (0.70) | −36% |
| Spec | 13.8% (0.59) | 18.5% (0.76) | 9.1% (0.43) | −34% |
| Nasdaq-100 | 14.2% (0.54) | 9.3% (0.34) | **18.9% (0.87)** | **−83%** |
| S&P 500 | 10.4% (0.49) | 6.4% (0.25) | 14.1% (0.77) | −55% |

- **Since 2011, the Nasdaq-100 beat every momentum variant on both return and risk-adjusted return.** The 2011–2026 bull market was led by a handful of mega-cap tech stocks. Equal-weighting 20 names can't fully capture that, and the regime filter still costs some time out of the market.
- **Over the full period, the strategy wins** because the Nasdaq lost 83% in 2000–02, while the regime filter kept the strategy in T-bills.
- **Takeaway:** the strategy's value is mainly crash protection plus a diversified momentum tilt. If you're confident the mega-cap tech era continues, QQQ was the better hold since 2011, and it's also far more tax-efficient.

## 8. Data issues found

- **Survivorship bias.** Yahoo has no prices for about 30% of the late-1990s index members, mostly failed or acquired companies. Results before about 2005 are flattered.
- **Ticker reuse.** MCIC, CPWR, COMS, DIGI, PARA, EP, SNT, and USBC were reused by OTC penny stocks; MCIC was quoted in the millions. These are excluded through `data.BLOCKLIST` and an extreme-jump check.
- **Quality gate.** Only about 4–5 fiscal years of fundamentals are available for free, so the gate can't be tested historically.
- **Sizing bug (fixed).** When the regime flipped back to bull, idle cash was poured into the few new names, which reached about 25% each. New buys are now capped at 1/N.

## 9. Caveats

- **Multiple testing.** About 40 variants were tried, and the falling-SMA rule was found *after* seeing where the spec failed. The 2011–now half is not truly out-of-sample for that rule. Expect live results to come in below the backtest.
- **Lower Sharpe than the index since 2011:** 0.70 vs SPY's 0.77, even pre-tax.
- **Concentration.** The basket is often concentrated in one theme; in October 2026 it is mostly semiconductors and memory stocks.
