# Index definitions as implemented (stage 3-4)

TX/TN = daily max/min temperature (°C); PRCP = daily precipitation (mm). Wet day = PRCP >= 1 mm.
"Complete" = missing days <= 3 (month) / <= 15 (year) unless configured otherwise; incomplete -> NaN.

| Index | Freq | Unit | Definition (comparison as implemented) | Function |
|---|---|---|---|---|
| FD0 | annual | days | count TN < 0 | `temperature_indices.frost_days` |
| SU25 | annual | days | count TX > 25 | `summer_days` |
| ID0 | annual | days | count TX < 0 | `ice_days` |
| TR20 | annual | days | count TN > 20 | `tropical_nights` |
| FD/SU/TR/ID (user) | annual | days | same functions, thresholds from parameters (screenshot: lower TN, upper TX, upper TN, lower TX) | same |
| GSL | annual | days | TG=(TX+TN)/2; start = first run >=6 d TG>5 starting before 1 Jul; end = first run >=6 d TG<5 starting on/after 1 Jul (else 31 Dec); GSL = end-start; 0 if no start. SH: season 1 Jul..30 Jun, split 1 Jan | `growing_season_length` |
| TXx, TNx | monthly+annual | °C | max of TX / TN | `txx`, `tnx` |
| TXn, TNn | monthly+annual | °C | min of TX / TN | `txn`, `tnn` |
| TX10p, TN10p | monthly+annual | % days | % days TX/TN < daily 10th percentile (5-day window, base period, bootstrap in base years) | `percentile_index` |
| TX90p, TN90p | monthly+annual | % days | % days TX/TN > daily 90th percentile (same method) | `percentile_index` |
| WSDI | annual | days | days in runs >=6 d with TX > daily 90th pct (no bootstrap) | `spell_duration_indicator` |
| CSDI | annual | days | days in runs >=6 d with TN < daily 10th pct (no bootstrap) | `spell_duration_indicator` |
| DTR | monthly+annual | °C | mean(TX-TN) | `diurnal_temperature_range` |
| RX1day | monthly+annual | mm | max daily PRCP | `rx1day` |
| RX5day | monthly+annual | mm | max 5-day running sum, credited to period of the window's last day | `rx5day` |
| SDII | annual | mm/day | sum PRCP(wet) / n wet days; NaN if no wet day | `sdii` |
| R10, R20, Rnn | annual | days | count PRCP >= 10 / 20 / user threshold | `r10mm`, `r20mm`, `rnn` |
| CDD, CWD | annual | days | longest run PRCP < 1 / >= 1; missing day breaks run; cross-year runs credited to year of the run's end (`spell_span="split"` cuts at 31 Dec) | `cdd`, `cwd` |
| R95p, R99p | annual | mm | sum PRCP on days PRCP > base-period 95th/99th percentile of wet-day amounts | `very_wet_precip` |
| PRCPTOT | annual | mm | sum PRCP on wet days | `prcptot` |

Percentile method: Hyndman-Fan type 8 (`numpy method="median_unbiased"`), threshold needs >= 70 % of
expected pooled values, 366 calendar bins (Feb 29 has its own bin fed by leap years only).
Bootstrap: for base-year Y, thresholds are recomputed from {all base years except Y} + a copy of Z for
each Z != Y; the exceedance rate of Y is averaged over Z. Years outside the base period use plain thresholds.
