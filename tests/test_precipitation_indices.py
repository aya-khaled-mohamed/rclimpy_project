import numpy as np
from rclimpy.common import ANNUAL, MONTHLY
from rclimpy.indices import precipitation_indices as pr
from conftest import daily

def base():
    return daily(0.0, years=(2001, 2001))

def test_r10_r20_inclusive():
    p = base(); p.iloc[:4] = [9.9, 10.0, 19.9, 20.0]
    assert pr.r10mm(p).iloc[0] == 3 and pr.r20mm(p).iloc[0] == 1
    assert pr.rnn(p, 19.9).iloc[0] == 2

def test_wet_day_boundary_and_prcptot_sdii():
    p = base(); p.iloc[:4] = [0.99, 1.0, 3.0, 6.0]
    assert pr.prcptot(p).iloc[0] == 10.0                 # 0.99 not wet
    assert pr.sdii(p).iloc[0] == 10.0 / 3

def test_sdii_nan_without_wet_day_prcptot_zero():
    p = base()
    assert np.isnan(pr.sdii(p).iloc[0]) and pr.prcptot(p).iloc[0] == 0.0

def test_rx1day_rx5day_assigned_to_end_month():
    p = base()
    p.loc["2001-01-30":"2001-02-03"] = [1, 2, 3, 4, 5]          # window ends 3 Feb
    m = pr.rx5day(p, MONTHLY)
    assert m.iloc[0] == 3.0 and m.iloc[1] == 15.0   # Jan: window ending 31 Jan holds only 1+2
    assert pr.rx5day(p, ANNUAL).iloc[0] == 15.0
    assert pr.rx1day(p, MONTHLY).iloc[0] == 2 and pr.rx1day(p, MONTHLY).iloc[1] == 5

def test_rx5day_cross_year_window():
    p = daily(0.0, years=(2001, 2002)); p.loc["2001-12-30":"2002-01-03"] = 10.0
    a = pr.rx5day(p, ANNUAL)
    assert a.loc[2002] == 50.0 and a.loc[2001] == 20.0   # window ending 2002-01-03 credited to 2002

def test_rx5day_missing_breaks_window():
    p = base(); p.loc["2001-03-01":"2001-03-05"] = 10.0; p.loc["2001-03-03"] = np.nan
    assert pr.rx5day(p, MONTHLY).iloc[2] == 20.0   # no 5-day window is complete AND covers all four wet days

def test_cdd_cwd_basic_and_missing_breaks_run():
    p = daily(5.0, years=(2001, 2001)); p.iloc[10:25] = 0.0; p.iloc[30:40] = 0.0
    assert pr.cdd(p).iloc[0] == 15 and pr.cwd(p).iloc[0] == 365 - 40
    p2 = p.copy(); p2.iloc[17] = np.nan
    assert pr.cdd(p2).iloc[0] == 10

def test_cdd_cross_year_boundary():
    p = daily(5.0, years=(2001, 2002)); p.loc["2001-12-25":"2002-01-10"] = 0.0   # 7 + 10 = 17 days
    assert pr.cdd(p, span="end_year").loc[2002] == 17 and pr.cdd(p, span="end_year").loc[2001] == 0
    s = pr.cdd(p, span="split")
    assert s.loc[2001] == 7 and s.loc[2002] == 10

def test_zero_precip_all_dry():
    p = base()
    assert pr.cdd(p).iloc[0] == 365 and pr.cwd(p).iloc[0] == 0
