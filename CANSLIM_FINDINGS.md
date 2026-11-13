# CAN SLIM findings

Run `python -m momentum.canslim` to reproduce these numbers. The **Compare variants** tab shows the same table.
Feb 1996 to Oct 2026, point-in-time S&P 500 members, 10 bps per side. The halves are cut from the same run.

## Summary

**The book's rules, as far as prices and volume can test them, lost to buy-and-hold.** They returned 3.0% a year
against 10.4% for SPY, and 0.7% a year since 2011. No single change or small combination beat SPY after 2011.

| | CAGR | Sharpe | Max drawdown | CAGR 1996–2010 | CAGR 2011–now | Time in stocks |
|---|---|---|---|---|---|---|
| CAN SLIM, book rules | 3.0% | 0.13 | −28% | 5.5% | 0.7% | 24% |
| Book + falling-200-day market rule | 8.1% | 0.49 | −25% | 9.9% | 6.4% | 63% |
| S&P 500 (SPY) | 10.4% | 0.49 | −55% | 6.4% | 14.1% | 100% |
| Momentum (Improved) | 18.1% | 0.72 | −39% | 20.1% | 16.2% | — |

## Why

1. **The distribution-day market filter whipsaws.** The S&P 500 closes down 0.2% or more on higher volume
   about 50 times a year, so "5 distribution days in 25 sessions" is roughly the normal rate, not a warning. The
   mechanical rule declared 109 corrections in 31 years. It kept the system out of the market in most of 2017, 2019
   and 2021, and still called 2008 an uptrend two-thirds of the time. Selling everything on each call caused 78% of
   all exits. IBD's rule that a distribution day expires after a 5% index gain barely changes this.
2. **The 8% stop and 20% target don't suit large caps.** S&P 500 stocks with 25–35% annual volatility hit an 8%
   stop from ordinary noise. With the market filter off, 318 trades hit the stop and 199 hit the target, a 40% win
   rate. The profit target also caps the winners that drive momentum returns. Removing it didn't help, though:
   without the target, more positions rode back down to the stop.
3. **The earnings half is missing.** C and A screen for companies with accelerating earnings. O'Neil's case is that
   price leadership plus explosive earnings marks the big winners. Price leadership alone, among large caps, is
   what the momentum strategy already captures better with monthly ranking and no stops.

## One-knob variants of the book rules

| Variant | CAGR | Sharpe | Max DD | CAGR 96–10 | CAGR 11–now |
|---|---|---|---|---|---|
| Book rules (baseline) | 3.0% | 0.13 | −28% | 5.5% | 0.7% |
| M: hold positions in a correction | 5.7% | 0.33 | −47% | 5.4% | 5.9% |
| M: off (ignore market direction) | 6.8% | 0.37 | −49% | 6.5% | 7.1% |
| M: index above 200-day SMA | 7.3% | 0.45 | −20% | 9.0% | 5.8% |
| M: bear only below a falling 200-day SMA | 8.1% | 0.49 | −25% | 9.9% | 6.4% |
| M: falling 200-day SMA, hold in a correction | 8.1% | 0.48 | −30% | 11.0% | 5.5% |
| M: distribution days without the 5% expiry | 3.6% | 0.20 | −26% | 5.6% | 1.7% |
| M: 6 distribution days to end an uptrend | 3.0% | 0.12 | −38% | 3.8% | 2.3% |
| M: follow-through at +1.0% | 4.7% | 0.33 | −24% | 7.6% | 2.1% |
| M: follow-through at +1.7% | 3.5% | 0.21 | −20% | 4.3% | 2.7% |
| L: RS rating ≥ 90 | 3.0% | 0.15 | −17% | 5.2% | 1.0% |
| L: RS rating ≥ 70 | 3.2% | 0.14 | −30% | 6.2% | 0.4% |
| S: no volume confirmation | 4.4% | 0.26 | −33% | 4.7% | 4.2% |
| N: base ≥ 5 weeks | 2.8% | 0.10 | −29% | 5.3% | 0.5% |
| Sell: stop loss 7% | 2.8% | 0.10 | −25% | 5.8% | 0.0% |
| Sell: no profit target (stop + 50-day line) | 3.2% | 0.15 | −26% | 6.4% | 0.2% |
| Sell: profit target 25% | 3.3% | 0.17 | −25% | 6.2% | 0.7% |
| Sell: no 8-week hold rule | 2.9% | 0.12 | −28% | 5.4% | 0.7% |
| Size: 10 positions | 3.0% | 0.15 | −15% | 5.3% | 0.8% |
| Size: 4 positions | 2.2% | 0.03 | −39% | 4.9% | −0.3% |
| Size: pyramid into winners | 2.9% | 0.12 | −18% | 5.0% | 0.9% |

Only the market rule moves results much. Replacing the distribution-day count with the momentum strategy's
falling-200-day rule (chosen in that research, not fitted here) helps in both halves. It still trails SPY by 7–8
points a year since 2011. A few combinations on top of it (no profit target, 10 positions, wider stop, no volume
rule) ranged from 5.9% to 8.6% CAGR, all with 2.0–6.4% since 2011.

## What this doesn't show

- Whether CAN SLIM works with the earnings screens, on smaller growth stocks, or with an experienced trader's
  judgment on chart patterns and market calls. Testing that needs point-in-time fundamentals (Compustat,
  Sharadar) and a broader universe.
- Intraday entries and stops. Closes and next-day opens make entries a day late and stops looser.
