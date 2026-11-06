import numpy as np, pandas as pd, pytest

def daily(values, start="2001-01-01", years=None):
    """Full-calendar daily Series; if `years` given, series covers whole years filled with `values` scalar."""
    if years:
        idx = pd.date_range(f"{start[:4]}-01-01", periods=1, freq="D")
        idx = pd.date_range(f"{years[0]}-01-01", f"{years[1]}-12-31")
        return pd.Series(float(values), index=idx)
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(np.asarray(values, float), index=idx)

@pytest.fixture
def year_const():
    return lambda v, y0=2001, y1=2001: daily(v, years=(y0, y1))
