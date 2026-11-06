import numpy as np, pandas as pd
from rclimpy.common import ANNUAL, MONTHLY
from rclimpy.indices import temperature_indices as ti
from conftest import daily

def test_strict_thresholds():
    tn = daily(5.0, years=(2001, 2001)); tn.iloc[0:4] = [-0.1, 0.0, 0.1, -5]       # FD: only <0
    assert ti.frost_days(tn).iloc[0] == 2
    tx = daily(10.0, years=(2001, 2001)); tx.iloc[0:3] = [25.0, 25.1, 30]          # SU: only >25
    assert ti.summer_days(tx).iloc[0] == 2
    tx = daily(10.0, years=(2001, 2001)); tx.iloc[0:3] = [0.0, -0.1, -3]           # ID: only <0
    assert ti.ice_days(tx).iloc[0] == 2
    tn = daily(10.0, years=(2001, 2001)); tn.iloc[0:3] = [20.0, 20.1, 25]          # TR: only >20
    assert ti.tropical_nights(tn).iloc[0] == 2

def test_user_defined_names_and_thresholds():
    tx = daily(30.0, years=(2001, 2001))
    r = ti.summer_days(tx, 28.0)
    assert r.name == "SU28" and r.iloc[0] == 365

def test_extremes_monthly_and_annual():
    tx = daily(np.arange(365.0), years=(2001, 2001)) if False else daily(list(range(365)), start="2001-01-01")
    m = ti.txx(tx, MONTHLY); assert m.iloc[0] == 30 and m.iloc[1] == 58 and len(m) == 12
    assert ti.txn(tx, MONTHLY).iloc[1] == 31
    assert ti.txx(tx, ANNUAL).iloc[0] == 364 and ti.txn(tx, ANNUAL).iloc[0] == 0

def test_dtr_mean_and_missing_pairing():
    tx = daily(15.0, years=(2001, 2001)); tn = daily(5.0, years=(2001, 2001))
    tx.iloc[0] = np.nan
    r = ti.diurnal_temperature_range(tx, tn, MONTHLY)
    assert r.iloc[0] == 10.0 and r.name == "DTR"

def _year(tg_by_day_of_year):
    return daily(tg_by_day_of_year, start="2001-01-01")

def test_gsl_nh_hand_case():
    tg = np.full(365, 0.0)
    tg[99:] = 10.0                      # warm from index 99 (10 Apr) ...
    tg[250:] = 0.0                      # ... cold from index 250 on
    s = _year(tg)
    r = ti.growing_season_length(s, s, latitude=45)
    assert r.iloc[0] == 250 - 99

def test_gsl_requires_6_days_and_zero_when_no_start():
    tg = np.full(365, 0.0); tg[100:105] = 10.0          # only 5 warm days
    s = _year(tg)
    assert ti.growing_season_length(s, s, 45).iloc[0] == 0

def test_gsl_no_cold_spell_runs_to_year_end():
    tg = np.full(365, 0.0); tg[100:] = 10.0
    s = _year(tg)
    assert ti.growing_season_length(s, s, 45).iloc[0] == 365 - 100

def test_gsl_cold_span_before_july_ignored():
    tg = np.full(365, 10.0); tg[20:40] = 0.0            # winter cold span; start at 0; end only searched after Jul 1
    s = _year(tg)
    assert ti.growing_season_length(s, s, 45).iloc[0] == 365

def test_gsl_south_hemisphere():
    idx = pd.date_range("2001-01-01", "2003-12-31")
    tg = pd.Series(0.0, index=idx)
    tg[pd.Timestamp("2001-09-01"):pd.Timestamp("2002-04-01")] = 10.0   # SH season starting 1 Sep 2001
    r = ti.growing_season_length(tg, tg, latitude=-30)
    d_start = (pd.Timestamp("2001-09-01") - pd.Timestamp("2001-07-01")).days
    d_end = (pd.Timestamp("2002-04-02") - pd.Timestamp("2001-07-01")).days
    assert r.loc[2001] == d_end - d_start

def test_gsl_missing_gives_nan():
    tg = np.full(365, 10.0); s = _year(tg); s.iloc[:20] = np.nan
    assert np.isnan(ti.growing_season_length(s, s, 45).iloc[0])
