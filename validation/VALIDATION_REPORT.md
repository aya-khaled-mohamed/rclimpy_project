# Validation against RClimDex (sampledata, 1915-2009, base 1915-2009, lat 0)

Input: `sampledataindcal.csv` (RClimDex's own QC'd data, 34,699 days, no missing). Reference: trend statistics printed on the
RClimDex plots (R2 %, p-value, slope, slope error), transcribed in `reference_trend_stats.json`. A line counts as
**validated** only if all four printed numbers are reproduced exactly at printed precision.
Limitation: the annual values themselves were only supplied as plots, so this is a statistical fingerprint test,
not a value-by-value comparison.

## Validated (17): all four statistics identical
CDD, CWD, FD0, GSL, ID0, PRCPTOT, R10, R20, Rnn(25), RX1day, RX5day, SDII, SU25, TXx, TXn, TMAXmean, TMINmean

## Numerical differences (9)
| Index | RClimDex (R2, p, slope, SE) | This code | Notes / what was tested |
|---|---|---|---|
| DTR | 23.8, 0, -0.010, 0.002 | 24.3, 0, -0.010, 0.002 | slope/SE/p match. Mean of 12 monthly means gives 23.9 (closer, not exact) |
| TN10p | 21.9, 0, -0.061, 0.012 | 21.9, 0, -0.060, 0.012 | only slope differs in 3rd decimal |
| TN90p | 3.7, 0.063, 0.024, 0.013 | 3.4, 0.073, 0.023, 0.013 | |
| TX10p | 0.1, 0.783, -0.004, 0.013 | 0.0, 0.960, +0.001, 0.013 | |
| TX90p | 3.0, 0.095, -0.028, 0.017 | 2.1, 0.162, -0.022, 0.015 | |
| WSDI | 11.6, 0.001, -0.127, 0.036 | 3.7, 0.062, -0.055, 0.029 | same peak years, RClimDex counts more days in several years |
| CSDI | 1.2, 0.294, -0.017, 0.016 | 0.4, 0.544, -0.008, 0.013 | e.g. 1960: RClimDex ~26 vs 12 here |
| R95p | 10.8, 0.001, 1.001, 0.298 | 11.5, 0.001, 1.050, 0.302 | |
| R99p | 6.3, 0.014, 0.434, 0.173 | 4.2, 0.045, 0.371, 0.183 | |

## Not testable: TR20, TNx, TNn (plots not supplied)

## Investigation so far (nothing was changed in the calculation to force agreement)
* WSDI/CSDI: quantile type 6/7/8, 365-day vs 366-day vs day-of-year binning, Feb 29 treatment, circular vs continuous window,
  window half-width 0-7 and >= vs > were tried. None reproduces the reference, and none was adopted. The reference has
  larger spell totals in several years (peaks coincide) - consistent with slightly different daily thresholds or a different
  spell rule in the original program; cannot be resolved without RClimDex's daily thresholds or annual values.
* R95p/R99p: the manual's formula text says the thresholds come from **1961-1990** wet days. With a 1961-1990 threshold base,
  R99p reproduces R2, p and SE exactly (slope 0.432 vs 0.434) and R95p moves much closer (10.1 / 0.959 vs 10.8 / 1.001), but
  neither matches exactly. Evidence supports a 1961-1990 base but the exact threshold rule is still unknown. Default in code
  remains the user base period until this is settled.
* Manual (section on known bug): RClimDex needs data beyond the first base-period day (window), so its +-2-day window may use
  days outside the base period; this code uses in-base days only. This affects only the first/last days of the base period.
