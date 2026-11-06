# Assumptions to verify against the RClimDex manual / reference output

The RClimDex 1.0 User Manual (Sept 2004) is now available and was read in full. Items marked CONFIRMED are stated
in it; the rest remain assumptions. NOTHING is claimed numerically identical to RClimDex until compared with real
output (raw data + RClimDex indices still needed).

| # | Item | Implemented as | Status |
|---|---|---|---|
| 1 | Monthly / annual missing-day limits | 3 / 15 days | CONFIRMED (App. C) |
| 1b | "No annual value if any one month's data are missing" | every month must meet the <=3 rule (`annual_month_rule`) | INTERPRETATION - verify with output |
| 2 | 70 % availability for thresholds | n_valid >= 0.7 * n_expected | CONFIRMED (App. C); exact denominator assumed |
| 3 | Quantile algorithm | Hyndman-Fan type 8, clamped to min/max | CONFIRMED (App. D) |
| 4 | Leap day in percentile windows | 366 bins, +-2 actual dates | NOT in manual - verify |
| 5 | Percentile index denominator | days with a defined exceedance | verify |
| 6 | TX90p etc. annual value | % of defined days in the year | verify |
| 7 | GSL end/boundary convention, SH labelling | see INDICES.md | verify |
| 8 | CDD/CWD/WSDI/CSDI spells across year boundary | whole spell credited to the year it ends | CONFIRMED (App. C) |
| 9 | RX5day window crediting | by last day of window | CONFIRMED ("interval ending i") |
| 10 | SDII with no wet day | NaN | verify |
| 11 | WSDI/CSDI thresholds | non-bootstrapped daily thresholds | NOT in manual - verify |
| 13 | Bootstrap scheme | out-of-base year vs thresholds from other years + each repeated year in turn, averaged | CONFIRMED (App. D) |
| 14 | R95p/R99p percentile base | user base period (manual text says 1961-1990) | verify |
| 12 | Monthly vs annual output per index | as in INDICES.md | verify |

Needed to close these: the raw station file without header (e.g. sampledata.csv), the RClimDex index output
CSVs for the same run (same base period, latitude, thresholds), and the manual's appendix pages.

## QC-stage items (manual section 3.2)
| # | Item | Implemented as | Status |
|---|---|---|---|
| Q1 | -99.9 -> NA; PRCP<0 -> NA; TMAX<TMIN -> NA | exactly so; both logged per value | CONFIRMED |
| Q2 | Which of TMAX/TMIN becomes NA when TMAX<TMIN | both (`tmax_lt_tmin_action`) | manual silent |
| Q3 | Outlier bounds | mean +- n*std of the calendar day, n default 3 | CONFIRMED; same bounds for all years of a day and day listed if TMAX, TMIN or DTR is outside - inferred from the supplied reference file |
| Q4 | std definition / pooling of neighbouring days / min sample | ddof=1, halfwidth 0, n>=10 | manual silent - to be tested against sampledata_tepstdQC.csv once raw data is available |
| Q5 | Outliers are NOT changed | flagged only; changes only via explicit decisions | CONFIRMED |
| Q6 | PRCP upper limit (200 mm in the GUI screenshot) | flag only | NOT in 1.0 manual (newer RClimDex) |
| Q7 | Missing dates | allowed (manual); become NA in working copy and are reported | CONFIRMED |

## Result of the first numerical validation (see validation/VALIDATION_REPORT.md)
17 of 26 comparable indices reproduce RClimDex's printed trend statistics exactly; 9 differ (DTR, TN10p, TN90p, TX10p, TX90p, WSDI, CSDI, R95p, R99p).
Open questions: exact daily-threshold rule for percentile temperature indices / WSDI / CSDI; threshold base for R95p/R99p (evidence points to 1961-1990);
DTR annual aggregation. The equivalence claim is therefore NOT made.
