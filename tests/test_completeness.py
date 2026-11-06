import numpy as np
from rclimpy.common import ANNUAL, MONTHLY
from rclimpy.completeness import CompletenessPolicy, completeness_table, n_missing
from rclimpy.indices import temperature_indices as ti
from conftest import daily

def test_monthly_limit_3_ok_4_nan():
    tn = daily(-1.0, years=(2001, 2001))
    tn.iloc[0:3] = np.nan                       # 3 missing in Jan -> still valid
    tn.iloc[31:35] = np.nan                     # 4 missing in Feb -> invalid
    r = ti.txn(tn, MONTHLY)
    assert r.iloc[0] == -1.0 and np.isnan(r.iloc[1]) and r.iloc[2] == -1.0

def test_annual_limit_15_ok_16_nan():
    for k, expect_nan in ((15, False), (16, True)):
        tx = daily(10.0, years=(2001, 2001))
        tx.iloc[40:40 + k] = np.nan   # spread irrelevant for the annual rule
        r = ti.txx(tx, ANNUAL, CompletenessPolicy(annual_month_rule="none"))   # isolate the 15-day rule
        assert np.isnan(r.iloc[0]) == expect_nan

def test_partial_year_counts_as_missing():
    tx = daily(10.0, start="2001-03-01", years=None) if False else daily([10.0] * 306, start="2001-03-01")
    from rclimpy.common import full_calendar
    full = full_calendar(tx)
    assert n_missing(full.notna(), ANNUAL).iloc[0] == 59
    assert np.isnan(ti.txx(full, ANNUAL).iloc[0])

def test_table():
    tx = daily(10.0, years=(2001, 2002)); tx.iloc[:5] = np.nan
    t = completeness_table(tx.notna(), ANNUAL, CompletenessPolicy())
    assert t.loc[2001, "n_missing"] == 5 and t.loc[2002, "n_missing"] == 0 and t["n_days"].tolist() == [365, 365]

def test_annual_requires_every_month_complete_manual_rule():
    # 5 missing in Feb only: annual total 5 <= 15 but Feb violates the monthly (<=3) rule -> annual NaN
    tx = daily(10.0, years=(2001, 2001)); tx.iloc[31:36] = np.nan
    assert np.isnan(ti.txx(tx, ANNUAL).iloc[0])
    from rclimpy.completeness import CompletenessPolicy
    assert ti.txx(tx, ANNUAL, CompletenessPolicy(annual_month_rule="none")).iloc[0] == 10.0
    # 3 missing in each of 4 months = 12 <= 15 and every month ok -> valid
    tx = daily(10.0, years=(2001, 2001))
    for m0 in (0, 31, 59, 90):
        tx.iloc[m0:m0 + 3] = np.nan
    assert ti.txx(tx, ANNUAL).iloc[0] == 10.0
