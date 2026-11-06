import numpy as np
from rclimpy.completeness import CompletenessPolicy
from rclimpy.indices import spell_indices as sp
from conftest import daily

def test_first_span():
    f = np.array([0, 1, 1, 1, 0, 1, 1, 1, 1], bool)
    assert sp.first_span(f, 3) == 1 and sp.first_span(f, 4) == 5 and sp.first_span(f, 5) == -1
    assert sp.first_span(f, 3, lo=2) == 5

def test_days_in_spells_threshold_exact_6_vs_5_and_cross_year():
    s = daily(0.0, years=(2001, 2002))
    flag = s > 1
    flag.loc["2001-12-29":"2002-01-03"] = True       # 3 days in 2001 + 3 in 2002 = 6 -> counts
    flag.loc["2002-05-01":"2002-05-05"] = True       # 5 days -> does not count
    valid = s.notna()
    r = sp.days_in_spells(flag, valid, 6, "WSDI", CompletenessPolicy())
    assert r.loc[2001] == 0 and r.loc[2002] == 6      # whole spell credited to the year it ends
