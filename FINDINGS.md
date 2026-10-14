# Research findings: 12-1 momentum on the S&P 500

These results come from the backtests in this repo: February 1996 to October 2026, point-in-time S&P 500 members, adjusted prices, 10 bps cost per side, fills at the open on the first trading day of each month, and idle cash earning the 13-week T-bill rate. Prices come from Yahoo, plus a supplement from the frozen Quandl WIKI archive for 184 former members Yahoo no longer carries (see section 8). They were recorded on 2026-10-13. To reproduce them:

- Strategy variants: `python -m momentum.experiments`
- Tax comparison: `python -m momentum.taxes`
- Sector-ETF comparison: `python -m momentum.sector_etf --fetch`
- Price supplement: `python -m momentum.supplement --wiki data/archive/wiki_prices.zip` (Nasdaq Data Link WIKI bulk export, free API key)

## 1. The spec as written

| | Spec | S&P 500 (SPY, dividends reinvested) |
|---|---|---|
| CAGR | 14.3% | 10.4% |
| Sharpe | 0.61 | 0.49 |
| Max drawdown | −33% | −55% |
| Turnover per year | 189% | — |
| CAGR 1996–2010 | 18.7% | 6.4% |
| CAGR 2011–now | **10.3%** | **14.1%** |

- **The spec's own claims don't hold.** It claims a Sharpe of about 1.0 and a −24% to −28% drawdown. The backtest gives a Sharpe of 0.61 and a −33% drawdown. With today's members as the universe (survivorship bias), the result is 28.1% CAGR and a Sharpe of 1.01, so the spec's figures probably come from a biased backtest.
- **The single-stock claim doesn't hold either.** The spec says one stock returns −10.2% a year. The backtest gives +26.3% CAGR with a 73% drawdown.
- **All of the outperformance came before 2011.** The strategy beat the S&P in 19 of 31 calendar years, but since 2011 it trails by about 4 points a year.

## 2. Why it lagged after 2011: the regime filter

| | 1996–2010 | 2011–now |
|---|---|---|
| With the 200-day SMA filter | 18.7% | 10.3% |
| Without the filter | 9.4% | 16.7% |

- **The filter helped before 2011 and hurt after.** It sat in T-bills through 2000–02 and 2008. Since 2011 it has fired 12 times, mostly on short corrections, and kept the strategy out of the market for 31 months in which the S&P averaged **+2.3%**, which is classic whipsaw on V-shaped recoveries.
- **It adds the drawdown protection, not the return.** Without it, the full-period CAGR is a little lower (13.1%) and the max drawdown is −69%.
- **While invested, the strategy is high-beta:** about 1.3× the S&P's moves, with a 0.76 correlation.

## 3. What worked

A variant counts as robust when it improves the Sharpe in **both** halves, 1996–2010 and 2011–now, or ties in one and improves the other.

| Idea | CAGR | Sharpe | Max DD | CAGR 96–10 | CAGR 11–now | Sharpe 96–10 / 11–now | Verdict |
|---|---|---|---|---|---|---|---|
| Spec | 14.3% | 0.61 | −33% | 18.7% | 10.3% | 0.74 / 0.48 | baseline |
| **Bear only if the SMA is falling** | 16.7% | 0.65 | −39% | 19.0% | 14.5% | 0.70 / 0.60 | return ties, then improves |
| **Blend of 12-1 and 6-1 ranks** | 15.4% | 0.66 | −33% | 18.7% | 12.4% | 0.74 / 0.58 | ties, then improves |
| **Improved (both, plus a 10% cap)** | **18.1%** | **0.72** | −39% | 20.1% | **16.4%** | **0.76 / 0.68** | improves both |

- **Falling-SMA rule.** A bear market is declared only when the S&P 500 is below its 200-day SMA *and* that SMA is lower than it was 21 trading days ago. It still caught 2000–02, 2008, and 2022, and it ignored most short corrections. On its own it slightly lowers the 1996–2010 Sharpe (0.70 vs 0.74) while matching that period's return; combined with the blend score, both halves improve.
- **10% position cap.** It no longer earns anything: within the Improved strategy it costs 0.3 points of CAGR (18.1% vs 18.5% without it) and leaves the Sharpe unchanged. It's kept as a risk limit on single names, not for return. (On the earlier Yahoo-only data it added about 0.01 of Sharpe.)
- **Robustness of the Improved version.** Full-period CAGR stays at 16.5–18% across slope lookbacks of 5–63 days, SMA lengths of 150–250, and buffers of 30–60; at 25 bps per side it's 17.2%.

### Crash behaviour

| Episode (S&P peak to trough) | Improved | S&P 500 |
|---|---|---|
| Dot-com, 2000–02 | −10% | −48% |
| Financial crisis, 2007–09 | −12% | −55% |
| Q4 2018 | −9% | −19% |
| COVID, Feb–Mar 2020 | **−39%** | −34% |
| 2022 bear market | +11% | −24% |
| Tariff sell-off, Feb–Apr 2025 | **−27%** | −19% |

- **Slow bear markets are where the filter pays.** It had moved to T-bills before most of the 2000–02, 2008 and 2022 losses.
- **Fast crashes get through.** COVID and April 2025 fell and recovered within the month, before a monthly rule could react, and the high-momentum basket fell harder than the index. The COVID crash is the strategy's worst drawdown (−39%).
- **It also has crashes of its own.** From June 22 to July 29, 2026 it fell 33% while the S&P fell 2%: the basket was about 80% technology, mostly memory and chip stocks, and they reversed together. The market filter can't catch this.

## 4. What didn't work

| Idea | Result |
|---|---|
| Checking the regime daily | Whipsaw: 476% turnover and 10.6% CAGR. Daily checks plus fast re-entry gave 1,324% turnover. |
| Hysteresis band (±2%) | Lower return and a slightly deeper drawdown |
| Re-entering above the 50-day SMA | Helps 2011+, hurts 1996–2010 |
| SMA plus absolute momentum | Helps 2011+, hurts 1996–2010, −50% max drawdown |
| 10-month SMA (Faber) | Worse in both halves |
| Keeping 50% invested in bear markets | Helps 2011+, hurts 1996–2010 |
| Proximity to the 52-week high | 7.8% CAGR, 683% turnover |
| Smooth momentum ("frog in the pan") | Worse in both halves |
| Return ÷ volatility | Lower CAGR and lower drawdown, about the same Sharpe |
| Residual momentum (net of market, or market and sector) | 12.5–13.0% CAGR; Sharpe since 2011 falls from 0.68 to about 0.52. Cuts sector concentration. See section 10. |
| Volatility targeting (15–25%) | Lower risk, but proportionally lower return |
| Inverse-volatility weights | Same pattern as volatility targeting |
| Sector cap (4–6 names per sector) | −3 to −4 points of CAGR on the Improved strategy. Concentration in leading sectors is part of momentum's return. |
| Rotating sector ETFs instead of stocks | About half the return since 2011, and no better than holding all sectors. See section 10. |
| 50% SPY core | A smoother ride with a lower return |
| Equal-weight reset every month | Slightly worse than the spec's no-trim rule |

## 5. Basket size

| N | CAGR | Sharpe | Max DD | Turnover |
|---|---|---|---|---|
| 10 | 14.9% | 0.57 | −37% | 206% |
| 15 | 15.3% | 0.62 | −35% | 195% |
| 20 | 14.3% | 0.61 | −33% | 189% |
| 40 | 12.6% | 0.61 | −26% | 167% |
| 50 | 12.3% | 0.62 | −24% | 156% |
| 100 | 11.8% | 0.68 | −25% | 118% |

These use the spec's rules, with the buffer always 2×N. Return peaks at 10–15 names and the Sharpe ratio tends to rise with N. The differences, about 3.5 points of CAGR across all sizes, fall within the ±3.5-point statistical uncertainty of a 30-year backtest, so 20 names is a reasonable middle.

## 6. Taxes (taxable account, $100k start)

| Scenario | CAGR 1996–now | CAGR 2011–now |
|---|---|---|
| SPY, buy and hold, dividends taxed at 15%, never sold | 10.1% | 13.8% |
| SPY, as above, sold at the end (15% long-term rate) | 9.6% | 12.9% |
| Improved, tax-free account (IRA/401k) | **18.1%** | **16.4%** |
| Improved, all gains taxed at 30% | 13.9% | 12.9% |
| Improved, 30% short-term / 15% long-term | 14.6% | 13.5% |
| Spec, all gains taxed at 30% | 10.7% | 7.8% |

- **Tax drag is about 3.5–4 points of CAGR a year.** About two-thirds of realized gains are short-term. T-bill interest is taxed as ordinary income.
- **Since 2011, the after-tax result roughly ties or trails buy-and-hold SPY.** The pre-tax edge disappears in a taxable account.
- **Conclusion:** run the strategy in a tax-advantaged account. In a taxable account, SPY buy-and-hold is about as good since 2011, and simpler.

## 7. Comparison with the Nasdaq-100

The benchmark is QQQ with dividends reinvested, from March 1999. Before that it uses the Nasdaq-100 price index.

| | CAGR (Sharpe) 1996–now | 1996–2010 | 2011–now | Max DD |
|---|---|---|---|---|
| Improved | **18.1% (0.72)** | **20.1% (0.76)** | 16.4% (0.68) | −39% |
| Spec | 14.3% (0.61) | 18.7% (0.74) | 10.3% (0.48) | −33% |
| Nasdaq-100 | 14.2% (0.54) | 9.3% (0.34) | **18.9% (0.87)** | **−83%** |
| S&P 500 | 10.4% (0.49) | 6.4% (0.25) | 14.1% (0.77) | −55% |

- **Since 2011, the Nasdaq-100 beat every momentum variant on both return and risk-adjusted return.** The 2011–2026 bull market was led by a handful of mega-cap tech stocks. Equal-weighting 20 names can't fully capture that, and the regime filter still costs some time out of the market.
- **Over the full period, the strategy wins** because the Nasdaq lost 83% in 2000–02, while the regime filter kept the strategy in T-bills.
- **Takeaway:** the strategy's value is mainly crash protection plus a diversified momentum tilt. If you're confident the mega-cap tech era continues, QQQ was the better hold since 2011, and it's also far more tax-efficient.

## 8. Data issues found

- **Survivorship bias.** Yahoo deletes a stock's history once its symbol stops trading, so failed and acquired companies disappear. On Yahoo alone, only 40% of the index's members had prices in 1996, 49% in 2000, 64% in 2008 and 72% in 2012.
- **Renamed tickers (fixed).** The membership file lists some companies under old symbols, such as Facebook as `FB` for 2013–22, while Yahoo keeps the history only under the current one. `data.RENAMES` maps 31 verified renames (FB→META, UTX→RTX, PX→LIN, BK→BNY and others). Ambiguous cases are left out, such as the old Ingersoll-Rand (`IR`, now `TT`) while today's `IR` is a different member.
- **Price supplement (partial fix).** `momentum/supplement.py` adds adjusted prices from the frozen Quandl WIKI archive (through February 2018) for 184 former members that Yahoo lacks, each covering at least 90% of its membership. WIKI agrees with Yahoo on 97.7% of the stocks both carry. Coverage is now 49% in 1996, 60% in 2000, 81% in 2008, 91% in 2012 and 95% in 2016. The supplement lives in its own files: refreshes never write it into the Yahoo files, and the published data bundle (and so the hosted app) doesn't include it.
- **Still missing: nearly all pre-2010 failures** (Lehman, Bear Stearns, Washington Mutual, Enron, WorldCom). No free source has them. Leaving out failures makes results look better than reality, so 1996–2010 still leans optimistic. A survivorship-free dataset (Norgate Data at the Platinum level, or CRSP) would fix this.
- **Ticker reuse.** MCIC, CPWR, COMS, DIGI, PARA, EP, SNT, and USBC were reused by OTC penny stocks; MCIC was quoted in the millions. These are excluded through `data.BLOCKLIST` and an extreme-jump check.
- **Quality gate.** Only about 4–5 fiscal years of fundamentals are available for free, so the gate can't be tested historically.
- **Sizing bugs (fixed).** When the regime flipped back to bull, idle cash was poured into the few new names, which reached about 25% each; new buys are now capped at 1/N. The `max_weight` cap also wasn't applied to full-rebalance, volatility-sized or top-up purchases; it now applies to every target and buy. Neither change affects the Improved strategy's results.

## 9. Caveats

- **Multiple testing.** About 45 variants were tried, and the falling-SMA rule was found *after* seeing where the spec failed. The 2011–now half is not truly out-of-sample for that rule. Expect live results to come in below the backtest.
- **Lower Sharpe than the index since 2011:** 0.68 vs SPY's 0.77, even pre-tax. Since 2011 the strategy earned more than SPY (16.4% vs 14.1%) by taking more risk, not by being more efficient.
- **The best evidence rests on the weakest data.** The risk-adjusted edge over SPY comes from 1996–2010, which is also the period with the most missing stocks (section 8).
- **Concentration.** The basket is often concentrated in one theme. In October 2026 it is about 80% technology, mostly semiconductors and memory stocks.

## 10. Later tests: residual momentum and sector ETFs

Both tests kept the Improved strategy's other rules and were judged on the robustness rule in section 3.

**Residual momentum** (Blitz, Huij & Martens, 2011): rank stocks on the part of their 12-1 return that the market (and optionally their sector) doesn't explain, estimated over 36 months and divided by the residuals' volatility. Available as `score="residual"` and `score="residual_sector"`.

| Ranking | CAGR | Sharpe | Max DD | Sharpe 96–10 / 11–now | CAGR 11–now |
|---|---|---|---|---|---|
| Improved (12-1/6-1 blend) | **18.1%** | **0.72** | −39% | 0.76 / **0.68** | **16.4%** |
| Raw 12-1 | 16.5% | 0.65 | −39% | 0.71 / 0.59 | 14.1% |
| Residual, market | 13.0% | 0.63 | −39% | 0.75 / 0.52 | 9.9% |
| Residual, market + sector | 12.5% | 0.65 | −41% | 0.79 / 0.53 | 9.7% |

- It does what it's meant to on concentration: the market + sector version cuts the average largest-sector weight from about 35% to 26%, and the June–July 2026 drop from −33% to −10%.
- But it costs 5–6 points of CAGR and fails the robustness rule: better only in 1996–2010, clearly worse since 2011. It doesn't help in COVID either.

**Sector-ETF rotation** (`momentum/sector_etf.py`): the same rules applied to the 11 SPDR sector ETFs, holding the top 3. February 2000 to now, so every line covers the same period:

| | CAGR | Sharpe 2000–10 | Sharpe 2011–now | Max DD |
|---|---|---|---|---|
| Stocks: Improved | **15.4%** | 0.59 | **0.67** | −39% |
| ETFs: top 3 sectors | 8.9% | 0.67 | 0.41 | −36% |
| ETFs: all 9 sectors + filter (control) | 9.5% | 0.69 | 0.57 | −37% |
| 50/50 stocks + ETF top 3 | 12.4% | 0.65 | 0.60 | −38% |
| SPY buy and hold | 8.5% | 0.03 | 0.77 | −55% |

- Picking sectors added nothing over holding all of them with the same filter, and the ETF version earned about half as much since 2011. Top 2, top 4 and raw 12-1 ranking gave the same picture.
- A 50/50 mix diversifies (monthly correlation 0.64 since 2011) but lowers the Sharpe since 2011.
- **Conclusion:** the edge comes from picking individual stocks, often the leaders within a hot sector, not from sector rotation. That same concentration is what produced the June–July 2026 drawdown.
