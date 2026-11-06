import numpy as np, pandas as pd
from rclimpy.validation import compare_trend_stats
from rclimpy.trend_analysis import ols_trend

def test_ols_trend_exact_line():
    s = pd.Series(2.0 * np.arange(1950, 1980) + 5.0, index=range(1950, 1980))
    t = ols_trend(s)
    assert abs(t["slope"] - 2.0) < 1e-10 and abs(t["r2_pct"] - 100) < 1e-9 and t["slope_se"] < 1e-8

def test_compare_flags_difference_and_match():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=40) + 0.05 * np.arange(40), index=range(1970, 2010))
    t = ols_trend(s)
    ref = {"X": [round(t["r2_pct"], 1), round(t["p_value"], 3), round(t["slope"], 3), round(t["slope_se"], 3)],
           "Y": [1.0, 0.5, 9.9, 0.1]}
    out = compare_trend_stats({"X": s, "Y": s}, ref).set_index("index")
    assert out.loc["X", "status"] == "validated" and out.loc["Y", "status"] == "numerical_difference"
    assert compare_trend_stats({"X": s}, {"Z": [1, 1, 1, 1]}).status.iloc[0] == "not_computed"
