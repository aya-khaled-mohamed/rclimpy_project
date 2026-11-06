"""Linear least-squares trend of an annual index series (RClimDex plot statistics)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def ols_trend(series: pd.Series) -> dict:
    """OLS of value on year (NaN years skipped). Returns slope (per year), intercept, slope_se, r2 (%), p (two-sided
    t-test on slope, assumes independent normal residuals), n. Descriptive only - no causal interpretation."""
    s = series.dropna()
    if len(s) < 3:
        return dict(slope=np.nan, intercept=np.nan, slope_se=np.nan, r2_pct=np.nan, p_value=np.nan, n=len(s))
    r = stats.linregress(s.index.to_numpy(float), s.to_numpy(float))
    return dict(slope=r.slope, intercept=r.intercept, slope_se=r.stderr, r2_pct=100 * r.rvalue ** 2, p_value=r.pvalue, n=len(s))
