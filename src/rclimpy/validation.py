"""Numerical comparison of computed annual indices against RClimDex reference statistics.

Reference here = trend statistics printed on the RClimDex plots (R2 %, p, slope, slope error), transcribed by hand
into validation/reference_trend_stats.json. A comparison is 'validated' only when ALL four printed numbers are
reproduced at the printed precision; nothing is tuned to force agreement. Annual values themselves are not available
as numbers in the supplied material (only plotted), so value-by-value validation remains pending."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .trend_analysis import ols_trend

PRINT_DECIMALS = (1, 3, 3, 3)       # r2_pct, p_value, slope, slope_se as printed by RClimDex


def compare_trend_stats(annual: dict, reference: dict) -> pd.DataFrame:
    rows = []
    for name, ref in reference.items():
        if name.startswith("_"):
            continue
        if name not in annual:
            rows.append(dict(index=name, status="not_computed"))
            continue
        st = ols_trend(annual[name])
        ours = [round(float(st[k]), nd) for k, nd in zip(("r2_pct", "p_value", "slope", "slope_se"), PRINT_DECIMALS)]
        diff = [o - r for o, r in zip(ours, ref)]
        rows.append(dict(index=name, ref_r2=ref[0], our_r2=ours[0], ref_p=ref[1], our_p=ours[1], ref_slope=ref[2], our_slope=ours[2],
                         ref_se=ref[3], our_se=ours[3], d_slope=diff[2], max_abs_diff=max(abs(x) for x in diff),
                         status="validated" if all(abs(x) < 1e-9 for x in diff) else "numerical_difference"))
    return pd.DataFrame(rows)


def load_reference(path: str) -> dict:
    return json.load(open(path))
