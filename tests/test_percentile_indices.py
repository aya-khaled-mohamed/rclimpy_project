import numpy as np, pandas as pd, pytest
from rclimpy.common import ANNUAL, MONTHLY, ClimateIndexError, calendar_bin
from rclimpy.completeness import CompletenessPolicy
from rclimpy.indices import percentile_indices as pi
from conftest import daily

POL = CompletenessPolicy()

@pytest.mark.parametrize("qtype,method", [(5, "hazen"), (6, "weibull"), (7, "linear"), (8, "median_unbiased"), (9, "normal_unbiased")])
def test_hf_quantile_matches_numpy(qtype, method):
    rng = np.random.default_rng(1)
    for n in (1, 2, 5, 17, 150):
        x = np.sort(rng.normal(size=n))
        for p in (0.01, 0.1, 0.5, 0.9, 0.95, 0.99):
            got = pi.hf_quantile(x[None, :], np.array([n]), p, qtype)[0]
            assert got == pytest.approx(np.quantile(x, p, method=method), abs=1e-12)

def test_hf_quantile_nan_padding_and_empty():
    x = np.array([[1.0, 2.0, 3.0, np.nan, np.nan], [np.nan] * 5])
    out = pi.hf_quantile(x, np.array([3, 0]), 0.5, 8)
    assert out[0] == 2.0 and np.isnan(out[1])

def test_calendar_bin_leap_handling():
    b = calendar_bin(pd.DatetimeIndex(["2001-02-28", "2001-03-01", "2000-02-29", "2000-03-01", "2000-12-31", "2001-12-31"]))
    assert b.tolist() == [58, 60, 59, 60, 365, 365]

def test_constant_series_thresholds_and_zero_exceedance():
    x = daily(10.0, years=(2001, 2005))
    thr = pi.compute_daily_thresholds(x, 2001, 2005, 90, POL)
    assert np.allclose(thr.values, 10.0)                         # incl. Feb 29 bin (leap year 2004 in base)
    r = pi.percentile_index(x, "TX90p", 90, "above", 2001, 2005, ANNUAL, POL)
    assert (r == 0.0).all()                                      # strict '>' -> never exceeds a constant

def test_feb29_bin_unavailable_without_leap_year_in_base():
    x = daily(10.0, years=(2001, 2003))
    thr = pi.compute_daily_thresholds(x, 2001, 2003, 90, POL)
    assert np.isnan(thr.values[59]) and np.isfinite(thr.values[58]) and np.isfinite(thr.values[60])

def test_70_percent_availability_rule():
    x = daily(10.0, years=(2001, 2010))
    # make 40 % of the pooled sample for Jun 15 missing: window 5 days x 10 years = 50 values
    for y in range(2001, 2005):
        x.loc[f"{y}-06-13":f"{y}-06-17"] = np.nan   # 4 of 10 years fully missing -> 60 % available < 70 %
    thr = pi.compute_daily_thresholds(x, 2001, 2010, 90, POL)
    b = calendar_bin(pd.DatetimeIndex(["2001-06-15"]))[0]
    assert np.isnan(thr.values[b])
    x2 = daily(10.0, years=(2001, 2010))
    for y in range(2001, 2004):
        x2.loc[f"{y}-06-13":f"{y}-06-17"] = np.nan  # 70 % available -> allowed (>=)
    assert np.isfinite(pi.compute_daily_thresholds(x2, 2001, 2010, 90, POL).values[b])

def test_base_period_outside_data_raises():
    x = daily(10.0, years=(2001, 2003))
    with pytest.raises(ClimateIndexError):
        pi.compute_daily_thresholds(x, 1990, 2003, 90, POL)

def _naive_bootstrap(x, base_years, p, direction):
    """Independent loop implementation working on dates/dicts (no shared code with the vectorised version)."""
    val = {d: v for d, v in x.items()}
    lo, hi = min(base_years), max(base_years)
    inrange = lambda d: lo <= d.year <= hi
    out = {}
    def pool(md, years):
        vals = []
        for y in years:
            try:
                d = pd.Timestamp(year=y, month=md[0], day=md[1])
            except ValueError:
                continue
            for k in range(-2, 3):
                dd = d + pd.Timedelta(days=k)
                if inrange(dd) and np.isfinite(val.get(dd, np.nan)):
                    vals.append(val[dd])
        return vals
    mds = sorted({(d.month, d.day) for d in x.index})
    for Y in base_years:
        others = [y for y in base_years if y != Y]
        acc = {md: [] for md in mds}
        for Z in others:
            for md in mds:
                try:
                    dY = pd.Timestamp(year=Y, month=md[0], day=md[1])
                except ValueError:
                    continue
                v = val.get(dY, np.nan)
                s = pool(md, others + [Z])
                if not np.isfinite(v) or len(s) == 0:
                    continue
                t = np.quantile(s, p / 100, method="median_unbiased")
                acc[md].append(float(v > t) if direction == "above" else float(v < t))
        for md in mds:
            try:
                dY = pd.Timestamp(year=Y, month=md[0], day=md[1])
            except ValueError:
                continue
            out[dY] = np.mean(acc[md]) if acc[md] else np.nan
    return pd.Series(out)

@pytest.mark.parametrize("direction,p", [("above", 90), ("below", 10)])
def test_bootstrap_matches_independent_naive_implementation(direction, p):
    rng = np.random.default_rng(42)
    x = daily(0.0, years=(1999, 2002))                     # contains leap year 2000
    x[:] = rng.normal(15, 5, len(x))
    x.iloc[100:110] = np.nan                                # include some missing data
    thr = pi.compute_daily_thresholds(x, 1999, 2002, p, CompletenessPolicy(min_base_fraction=0.0))
    got = pi.daily_exceedance(x, thr, direction, CompletenessPolicy(min_base_fraction=0.0), bootstrap=True)
    ref = _naive_bootstrap(x, [1999, 2000, 2001, 2002], p, direction)
    comp = pd.concat([got, ref], axis=1, keys=["got", "ref"])
    assert len(comp) == len(x)
    both_nan = comp.isna().all(axis=1)
    assert (comp.isna().any(axis=1) == both_nan).all()
    assert np.allclose(comp["got"][~both_nan], comp["ref"][~both_nan], atol=1e-12)

def test_out_of_base_years_use_plain_threshold_and_base_years_use_bootstrap():
    rng = np.random.default_rng(7)
    x = daily(0.0, years=(2001, 2008)); x[:] = rng.normal(0, 1, len(x))
    thr = pi.compute_daily_thresholds(x, 2001, 2005, 90, POL)
    ex = pi.daily_exceedance(x, thr, "above", POL, bootstrap=True)
    oob = x.index.year >= 2006
    plain = (x.to_numpy() > thr.values[calendar_bin(x.index)]).astype(float)
    assert np.allclose(ex[oob], plain[oob], equal_nan=True)
    inb = ~oob
    frac = ex[inb].to_numpy()
    assert ((frac > 0) & (frac < 1)).any()                  # bootstrap yields fractional rates in base years

def test_bootstrap_base_period_rate_close_to_nominal():
    rng = np.random.default_rng(3)
    x = daily(0.0, years=(1961, 1990)); x[:] = rng.normal(0, 1, len(x))
    r = pi.percentile_index(x, "TX90p", 90, "above", 1961, 1990, ANNUAL, POL)
    assert r.mean() == pytest.approx(10.0, abs=0.7)

def test_monthly_vs_annual_shapes_and_incomplete_nan():
    rng = np.random.default_rng(5)
    x = daily(0.0, years=(2001, 2006)); x[:] = rng.normal(0, 1, len(x)); x.iloc[0:10] = np.nan
    m = pi.percentile_index(x, "TN10p", 10, "below", 2001, 2006, MONTHLY, POL)
    a = pi.percentile_index(x, "TN10p", 10, "below", 2001, 2006, ANNUAL, POL)
    assert len(m) == 72 and len(a) == 6 and np.isnan(m.iloc[0]) and np.isnan(a.iloc[0] if False else m.iloc[0])
    assert np.isnan(a.iloc[0]) and a.iloc[1:].notna().all()  # 10 missing in Jan: <=15 total but Jan violates monthly rule -> no annual value (manual App. C)

def test_wsdi_and_csdi_hand_case():
    rng = np.random.default_rng(11)
    x = daily(0.0, years=(2001, 2010)); x[:] = rng.normal(20, 1, len(x))
    x.loc["2008-07-01":"2008-07-07"] = 40.0                 # 7-day warm spell
    x.loc["2008-08-01":"2008-08-05"] = 40.0                 # 5-day: must not count
    w = pi.spell_duration_indicator(x, "WSDI", 90, "above", 2001, 2005, POL)
    assert w.loc[2008] >= 7 and w.loc[2008] < 7 + 40
    y = x.copy(); y.loc["2008-12-29":"2009-01-03"] = -40.0  # 6-day cold spell across the year boundary
    c = pi.spell_duration_indicator(y, "CSDI", 10, "below", 2001, 2005, POL)
    assert c.loc[2009] >= 6 and c.loc[2008] < 6   # spell ending 3 Jan 2009 counts toward 2009 (manual App. C)

def test_wsdi_exact_counts_with_deterministic_thresholds():
    x = daily(20.0, years=(2001, 2003)); x.loc["2003-07-01":"2003-07-06"] = 30.0   # base 2001-2002 constant 20
    w = pi.spell_duration_indicator(x, "WSDI", 90, "above", 2001, 2002, POL)
    assert w.loc[2003] == 6 and w.loc[2001] == 0
    x.loc["2003-07-06"] = 20.0                                                     # shrink to 5 days
    assert pi.spell_duration_indicator(x, "WSDI", 90, "above", 2001, 2002, POL).loc[2003] == 0

def test_r95p_r99p_hand_case_strict_greater():
    p = daily(0.0, years=(2001, 2002))
    wet = np.arange(1, 101, dtype=float)                    # wet days 1..100 mm in base year 2001
    p.iloc[:100] = wet
    thr95 = np.quantile(wet, 0.95, method="median_unbiased")
    assert pi.wet_day_threshold(p, 2001, 2001, 95) == pytest.approx(thr95)
    p.iloc[365:368] = [thr95, thr95 + 1, 500.0]             # 2002: equal not counted, others counted
    r = pi.very_wet_precip(p, "R95p", 95, 2001, 2001)
    assert r.loc[2002] == pytest.approx(thr95 + 1 + 500.0)
    assert r.loc[2001] == pytest.approx(wet[wet > thr95].sum())

def test_r95p_no_wet_days_in_base_gives_nan():
    p = daily(0.0, years=(2001, 2003))
    assert pi.very_wet_precip(p, "R95p", 95, 2001, 2002).isna().all()

def test_short_dataset_bootstrap_falls_back_with_warning():
    x = daily(10.0, years=(2001, 2001))
    thr = pi.compute_daily_thresholds(x, 2001, 2001, 90, CompletenessPolicy())
    with pytest.warns(UserWarning):
        pi.daily_exceedance(x, thr, "above", CompletenessPolicy(), bootstrap=True)
