import numpy as np, pandas as pd, pytest
from rclimpy.common import ClimateIndexError, daily_frame_from_ymd
from rclimpy.parameters import IndexParameters
from rclimpy.pipeline import build_specs, compute_indices

def synth(y0=1961, y1=1980, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{y0}-01-01", f"{y1}-12-31")
    tx = 20 + 8 * np.sin((idx.dayofyear.to_numpy() - 110) / 365.25 * 2 * np.pi) + rng.normal(0, 2, len(idx))
    return pd.DataFrame({"prcp": np.where(rng.random(len(idx)) < .2, rng.gamma(.7, 8, len(idx)), 0.),
                         "tmax": tx, "tmin": tx - 9 + rng.normal(0, 1, len(idx))}, index=idx)

def test_27_core_indices_registered():
    core = {"FD0","SU25","ID0","TR20","GSL","TXx","TNx","TXn","TNn","TN10p","TX10p","TN90p","TX90p","WSDI","CSDI","DTR",
            "RX1day","RX5day","SDII","R10","R20","Rnn","CDD","CWD","R95p","R99p","PRCPTOT"}
    names = {s.name for s in build_specs(IndexParameters())}
    assert core <= names and len(core) == 27

def test_compute_all_runs_and_reports_status():
    d = synth()
    run = compute_indices(d, IndexParameters(base_start=1961, base_end=1975, latitude=30))
    st = run.status_frame()
    assert (st.status == "ok").all()
    assert ("TX90p", "monthly") in run.results and ("TX90p", "annual") in run.results and ("FD0", "monthly") not in run.results

def test_missing_variable_reported_unavailable_not_silent():
    d = synth(); d["prcp"] = np.nan
    run = compute_indices(d, IndexParameters(base_start=1961, base_end=1975), selected=["PRCPTOT", "FD0"])
    st = run.status_frame().set_index("index")
    assert st.loc["PRCPTOT", "status"] == "unavailable" and st.loc["FD0", "status"] == "ok"

def test_invalid_base_period_is_actionable_error():
    with pytest.raises(ClimateIndexError, match="outside the data range"):
        compute_indices(synth(), IndexParameters(base_start=1900, base_end=1990))

def test_input_not_modified():
    d = synth(); before = d.copy(deep=True)
    compute_indices(d, IndexParameters(base_start=1961, base_end=1975), selected=["TX90p", "GSL", "CDD"])
    pd.testing.assert_frame_equal(d, before)

def test_ymd_frame_missing_code_and_calendar():
    raw = pd.DataFrame({"year": [2001, 2001, 2001], "month": [1, 1, 1], "day": [1, 2, 4],
                        "prcp": [0.0, -99.9, 3.0], "tmax": [10, 11, -99.9], "tmin": [1, 2, 3]})
    raw0 = raw.copy()
    d = daily_frame_from_ymd(raw)
    assert len(d) == 365 and np.isnan(d.loc["2001-01-02", "prcp"]) and np.isnan(d.loc["2001-01-03", "tmin"])
    assert np.isnan(d.loc["2001-01-04", "tmax"]) and d.loc["2001-01-04", "prcp"] == 3.0
    pd.testing.assert_frame_equal(raw, raw0)

def test_ymd_frame_rejects_duplicates_and_invalid_dates():
    dup = pd.DataFrame({"year": [2001, 2001], "month": [1, 1], "day": [1, 1], "prcp": [0, 0], "tmax": [1, 1], "tmin": [0, 0]})
    with pytest.raises(ClimateIndexError):
        daily_frame_from_ymd(dup)
    bad = dup.assign(month=[2, 2], day=[30, 30])
    with pytest.raises(Exception):
        daily_frame_from_ymd(bad)
