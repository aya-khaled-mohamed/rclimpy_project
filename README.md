# rclimpy — Python alternative to RClimDex (work in progress)

Not yet validated as equivalent to RClimDex. See `docs/ASSUMPTIONS_AND_PENDING.md`.

## Install & test
    pip install -r requirements.txt
    python -m pytest            # runs from project root (pytest.ini sets pythonpath=src)

## Clean data (QC)
    from rclimpy.data_import import read_station_file
    from rclimpy.quality_control import QCConfig, run_qc, apply_decisions, Decision, to_indcal_frame
    imp = read_station_file("station.txt")                    # header optional; original file never touched
    res = run_qc(imp, QCConfig(n_std=3, units_confirmed=True))
    res.temp_qc, res.prcp_qc, res.outliers.table, res.na_statistic, res.audit   # RClimDex-style QC tables + audit
    final, log = apply_decisions(res, [Decision("1995-07-15", "tmax", "exclude", "instrument fault")], analysis="trends")
    daily = daily_frame_from_ymd(to_indcal_frame(final))      # -> compute_indices(...)

## Use the calculation engine
    import pandas as pd, json
    from rclimpy.common import daily_frame_from_ymd
    from rclimpy.parameters import IndexParameters
    from rclimpy.pipeline import compute_indices
    raw = pd.read_csv("station.csv")                      # columns year,month,day,prcp,tmax,tmin (header!)
    daily = daily_frame_from_ymd(raw)                      # -99.9 -> NaN in a copy; original untouched
    params = IndexParameters.from_dict(json.load(open("config/example_config.json")))
    run = compute_indices(daily, params, selected=None)    # or ["TX90p", "GSL", ...]
    run.results[("TX90p", "annual")]; run.status_frame()   # status shows ok / unavailable / error + reason

## Status
| Stage | Component | State |
|---|---|---|
| 3 | 27 core indices + user-defined FD/SU/TR/ID | implemented, hand-case unit tests pass |
| 4 | percentile thresholds + bootstrap (TX/TN 10p/90p, WSDI, CSDI, R95p, R99p) | implemented; bootstrap checked against an independent naive implementation |
| – | completeness rules, parameters, pipeline with per-index status | implemented |
| 2 | data_import, schema_validation, quality_control, outlier_detection (manual 3.2 rules, audit log, decisions) | implemented, 70 tests pass; outlier numbers not yet compared with reference (raw data missing) |
| 5 | statistics_analysis, trend_analysis, visualization | PENDING (stubs) |
| 6 | Streamlit app, export_results, report_generation | PENDING (stubs) |
| 7 | validation vs RClimDex output | PARTIAL: 17 of 26 validated on printed trend stats, 9 differ (validation/VALIDATION_REPORT.md); value-by-value pending |

Layout: `src/rclimpy/` (engine) · `src/rclimpy/indices/` · `tests/` · `docs/` · `config/` · `app/` · `sample_data/` · `output/`.
