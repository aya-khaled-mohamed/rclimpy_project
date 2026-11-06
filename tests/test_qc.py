import numpy as np, pandas as pd, pytest
from rclimpy.common import ClimateIndexError
from rclimpy.data_import import read_station_file
from rclimpy.schema_validation import validate_schema
from rclimpy.quality_control import QCConfig, Decision, run_qc, apply_decisions, to_indcal_frame
from rclimpy.outlier_detection import daily_climatology, detect_temperature_outliers, to_rclimdex_tepstd, REFERENCE_COLUMNS

def make_df(y0=1990, y1=1999, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{y0}-01-01", f"{y1}-12-31")
    tx = 20 + 8 * np.sin((idx.dayofyear.to_numpy() - 110) / 365.25 * 2 * np.pi) + rng.normal(0, 1, len(idx))
    return pd.DataFrame({"year": idx.year, "month": idx.month, "day": idx.day, "prcp": np.round(np.where(rng.random(len(idx)) < .2, rng.gamma(.7, 8, len(idx)), 0.), 1),
                         "tmax": np.round(tx, 1), "tmin": np.round(tx - 9 + rng.normal(0, .5, len(idx)), 1)})

def write(df, path, header=False, sep=" "):
    df.to_csv(path, sep=sep, header=header, index=False)
    return path

def test_import_rclimdex_txt_no_header(tmp_path):
    p = write(make_df(), tmp_path / "s.txt")
    imp = read_station_file(p)
    assert not imp.had_header and imp.units_source == "rclimdex_convention" and imp.raw.shape == (3652, 6)

def test_import_csv_with_header_like_the_users_file(tmp_path):
    p = write(make_df(), tmp_path / "s.csv", header=True, sep=",")
    imp = read_station_file(p)
    assert imp.had_header and imp.units_source == "unspecified" and any("Header" in n for n in imp.notes)
    with pytest.raises(ClimateIndexError, match="units"):
        run_qc(imp)                                   # units never assumed for header files
    assert run_qc(imp, QCConfig(units_confirmed=True)).summary()["rows_original"] == 3652

def test_import_single_date_column_and_mapping(tmp_path):
    df = make_df(); out = pd.DataFrame({"Date": pd.to_datetime(df[["year", "month", "day"]]).dt.strftime("%Y-%m-%d"), "RR": df.prcp, "TX": df.tmax, "TN": df.tmin})
    out.to_csv(tmp_path / "d.csv", index=False)
    imp = read_station_file(tmp_path / "d.csv", column_map={"date": "Date", "prcp": "RR", "tmax": "TX", "tmin": "TN"})
    assert run_qc(imp, QCConfig(units_confirmed=True)).qc_data.shape[0] == 3652
    with pytest.raises(ClimateIndexError, match="column_map"):
        read_station_file(tmp_path / "d.csv", column_map={"date": "Nope", "prcp": "RR", "tmax": "TX", "tmin": "TN"})

def test_import_xlsx(tmp_path):
    make_df().to_excel(tmp_path / "s.xlsx", index=False)
    imp = read_station_file(tmp_path / "s.xlsx")
    assert imp.had_header and set(["year", "prcp"]) <= set(imp.column_map)

def test_schema_invalid_date_duplicates_nonnumeric(tmp_path):
    df = make_df(1990, 1990).iloc[:10].copy()
    df.loc[1, ["month", "day"]] = [2, 30]
    df.loc[3, ["month", "day"]] = [1, 1]                  # duplicate of row 0
    df["tmax"] = df["tmax"].astype(object); df.loc[5, "tmax"] = "abc"
    p = write(df, tmp_path / "bad.txt"); imp = read_station_file(p)
    rep = validate_schema(imp)
    assert not rep.ok and len(rep.errors) == 3
    with pytest.raises(ClimateIndexError, match="invalid"):
        run_qc(imp)

def test_missing_calendar_dates_and_unsorted_logged(tmp_path):
    df = make_df(1990, 1990).drop(index=range(10, 20)).sample(frac=1, random_state=1)
    imp = read_station_file(write(df, tmp_path / "u.txt"))
    res = run_qc(imp)
    assert len(res.gaps) == 1 and res.gaps.missing_days.iloc[0] == 10
    assert (res.audit.rule == "sorted").any()
    assert res.qc_data.tmax.isna().sum() == 10 and len(res.qc_data) == 365

def test_missing_code_negative_prcp_tmax_lt_tmin_and_audit(tmp_path):
    df = make_df(1990, 1992)
    df.loc[5, "prcp"] = -99.9; df.loc[6, "prcp"] = -2.5
    df.loc[10, ["tmax", "tmin"]] = [5.0, 8.0]
    df.loc[11, ["tmax", "tmin"]] = [7.0, 7.0]              # equal is allowed
    imp = read_station_file(write(df, tmp_path / "q.txt")); before = imp.raw.copy()
    res = run_qc(imp)
    q, o = res.qc_data, res.original
    d = lambda i: pd.Timestamp(f"{df.year[i]}-{df.month[i]}-{df.day[i]}")
    assert np.isnan(q.at[d(5), "prcp"]) and np.isnan(q.at[d(6), "prcp"])
    assert np.isnan(q.at[d(10), "tmax"]) and np.isnan(q.at[d(10), "tmin"]) and q.at[d(11), "tmax"] == 7.0
    assert o.at[d(5), "prcp"] == -99.9 and o.at[d(6), "prcp"] == -2.5 and o.at[d(10), "tmax"] == 5.0   # original intact
    assert len(res.prcp_qc) == 1 and res.prcp_qc.prcp.iloc[0] == -2.5            # -99.9 is NOT counted as negative
    assert len(res.temp_qc) == 1 and res.temp_qc["tmax-tmin"].iloc[0] == -3.0
    assert list(res.temp_qc.columns) == ["year", "month", "day", "prcp", "tmax", "tmin", "tmax-tmin"]
    a = res.audit
    assert ((a.rule == "missing_code") & (a.variable == "prcp")).sum() == 1
    assert ((a.rule == "prcp_negative").sum() == 1) and ((a.rule == "tmax_lt_tmin").sum() == 2)
    pd.testing.assert_frame_equal(imp.raw, before)
    o.iloc[0, 0] = 12345.0                                  # tampering with the returned copy ...
    assert res.original.iloc[0, 0] != 12345.0               # ... never reaches the stored original

def test_climatology_and_bounds_per_calendar_day_match_manual_definition():
    idx = pd.date_range("2000-01-01", "2009-12-31")
    x = pd.Series(np.tile([0.0, 10.0], len(idx) // 2 + 1)[:len(idx)], index=idx)
    c = daily_climatology(x)
    jan1 = c.iloc[0]
    vals = x[(x.index.month == 1) & (x.index.day == 1)].to_numpy()
    assert jan1["mean"] == pytest.approx(vals.mean()) and jan1["std"] == pytest.approx(vals.std(ddof=1)) and jan1["n"] == 10
    assert c.iloc[59]["n"] == 3                              # Feb 29: leap years 2000, 2004, 2008 only

def test_outlier_detection_flags_and_never_changes_values(tmp_path):
    df = make_df(1980, 2009)
    i = df.index[(df.year == 1995) & (df.month == 7) & (df.day == 15)][0]
    df.loc[i, "tmax"] = df.loc[i, "tmax"] + 25
    imp = read_station_file(write(df, tmp_path / "o.txt"))
    res = run_qc(imp, QCConfig(n_std=4))
    t = res.outliers.table
    hit = t[(t.year == 1995) & (t.month == 7) & (t.day == 15)]
    assert len(hit) == 1 and bool(hit.flag_tmax.iloc[0]) and bool(hit.flag_dtr.iloc[0])
    assert res.qc_data.at[pd.Timestamp("1995-07-15"), "tmax"] == df.loc[i, "tmax"]       # flagged, NOT changed
    assert ((res.audit.rule == "temp_outlier_tmax") & (res.audit.action == "flag_only")).sum() >= 1
    ref = to_rclimdex_tepstd(t)
    assert list(ref.columns) == REFERENCE_COLUMNS == list(pd.read_csv("/mnt/user-data/uploads/sampledata_tepstdQC.csv").columns)
    low, up = hit.tmaxlow.iloc[0], hit.tmaxup.iloc[0]
    assert low < hit.tmax.iloc[0] and hit.tmax.iloc[0] > up

def test_larger_n_flags_fewer():
    df = make_df(1980, 2009).assign(tmax=lambda d: d.tmax + np.random.default_rng(1).normal(0, 2, len(d)))
    d = pd.DataFrame({"tmax": df.tmax.to_numpy(), "tmin": df.tmin.to_numpy()}, index=pd.date_range("1980-01-01", periods=len(df)))
    assert len(detect_temperature_outliers(d, 3).table) >= len(detect_temperature_outliers(d, 4).table)

def test_zero_std_and_small_sample_are_explicit_not_silent():
    idx = pd.date_range("2000-01-01", "2004-12-31")                 # 5 years: n<10 per day
    d = pd.DataFrame({"tmax": 20.0, "tmin": 10.0}, index=idx)
    r = detect_temperature_outliers(d, 3, min_samples=10)
    assert len(r.table) == 0 and any("fewer than" in n for n in r.notes)
    idx = pd.date_range("1990-01-01", "2004-12-31")
    d = pd.DataFrame({"tmax": 20.0, "tmin": 10.0}, index=idx)
    r = detect_temperature_outliers(d, 3, min_samples=10)
    assert len(r.table) == 0 and any("zero standard deviation" in n for n in r.notes)

def test_prcp_upper_limit_flag_only(tmp_path):
    df = make_df(1990, 1992); df.loc[100, "prcp"] = 250.0
    res = run_qc(read_station_file(write(df, tmp_path / "p.txt")))
    assert len(res.prcp_flags) == 1 and res.qc_data.prcp.max() == 250.0

def test_na_statistic_layout_and_counts(tmp_path):
    df = make_df(1990, 1991); df.loc[0:2, "tmax"] = -99.9
    res = run_qc(read_station_file(write(df, tmp_path / "n.txt")))
    ref_cols = list(pd.read_csv("/mnt/user-data/uploads/sampledata_nastatistic.csv").columns)
    assert list(res.na_statistic.columns) == ref_cols
    r = res.na_statistic.query("TITLE=='TMAX' and YEAR==1990").iloc[0]
    assert r.JAN == 3 and r.ANN == 3 and res.na_statistic.query("TITLE=='PRCP'").ANN.sum() == 0

def test_decisions_exclude_replace_accept_logged_and_original_safe(tmp_path):
    df = make_df(1980, 1989)
    res = run_qc(read_station_file(write(df, tmp_path / "d.txt")))
    d0 = "1985-03-10"; orig = res.qc_data.at[pd.Timestamp(d0), "tmax"]
    decs = [Decision(d0, "tmax", "replace", "verified against paper log", 12.3, analysis="trend"),
            Decision("1985-03-11", "tmax", "exclude", "instrument fault suspected"),
            Decision("1985-03-12", "tmin", "accept", "genuine cold snap"),
            Decision("1985-03-13", "prcp", "investigate", "ask observer")]
    final, rows = apply_decisions(res, decs, analysis="trend")
    assert final.at[pd.Timestamp(d0), "tmax"] == 12.3 and np.isnan(final.at[pd.Timestamp("1985-03-11"), "tmax"])
    assert res.qc_data.at[pd.Timestamp(d0), "tmax"] == orig                  # qc_data untouched
    assert len(rows) == 4 and rows.loc[rows.action == "replace", "original_value"].iloc[0] == orig
    assert rows.reason.notna().all() and rows.timestamp.notna().all()
    f2, r2 = apply_decisions(res, decs, analysis="other")                    # replace was scoped to 'trend'
    assert f2.at[pd.Timestamp(d0), "tmax"] == orig and len(r2) == 3

def test_decision_validation():
    df = make_df(1980, 1985)
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as t:
        res = run_qc(read_station_file(write(df, pathlib.Path(t) / "x.txt")))
    for bad in (Decision("1981-01-01", "tmax", "replace", "r", None), Decision("1981-01-01", "tmax", "accept", " "),
                Decision("1981-01-01", "tmax", "delete", "r"), Decision("1999-01-01", "tmax", "accept", "r")):
        with pytest.raises(ClimateIndexError):
            apply_decisions(res, [bad])

def test_indcal_layout_roundtrip_into_indices(tmp_path):
    df = make_df(1990, 1999)
    res = run_qc(read_station_file(write(df, tmp_path / "i.txt")))
    ind = to_indcal_frame(res.qc_data)
    assert list(ind.columns) == ["year", "month", "day", "prcp", "tmax", "tmin"] and len(ind) == 3652
    from rclimpy.common import daily_frame_from_ymd
    from rclimpy.parameters import IndexParameters
    from rclimpy.pipeline import compute_indices
    run = compute_indices(daily_frame_from_ymd(ind), IndexParameters(base_start=1990, base_end=1999), selected=["FD0", "TXx", "PRCPTOT"])
    assert (run.status_frame().status == "ok").all()
