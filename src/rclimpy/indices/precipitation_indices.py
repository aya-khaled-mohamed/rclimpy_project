"""Precipitation indices (daily PRCP in mm, NaN = missing). Wet day: PRCP >= wet_threshold (default 1 mm)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..common import ANNUAL, MONTHLY
from ..completeness import CompletenessPolicy, mask_incomplete, period_key
from .spell_indices import SPAN_END_YEAR, max_spell_by_year

_P = CompletenessPolicy()


def rx1day(prcp, freq=MONTHLY, policy=_P):
    """RX1day: max 1-day precipitation per month/year (mm)."""
    res = prcp.groupby(period_key(prcp.index, freq)).max()
    return mask_incomplete(res, prcp.notna(), freq, policy).rename("RX1day")


def rx5day(prcp, freq=MONTHLY, policy=_P):
    """RX5day: max 5-day running sum (mm). A window is credited to the period containing its LAST day;
    windows may span month/year boundaries; any missing day in a window -> window undefined."""
    s5 = prcp.rolling(5, min_periods=5).sum()
    res = s5.groupby(period_key(s5.index, freq)).max()
    return mask_incomplete(res, prcp.notna(), freq, policy).rename("RX5day")


def _count_ge(prcp, thr, name, policy):
    cnt = (prcp >= thr).astype(float).groupby(period_key(prcp.index, ANNUAL)).sum()
    return mask_incomplete(cnt, prcp.notna(), ANNUAL, policy).rename(name)


def r10mm(prcp, policy=_P):
    """R10mm: days with PRCP >= 10 mm (days)."""
    return _count_ge(prcp, 10.0, "R10", policy)


def r20mm(prcp, policy=_P):
    """R20mm: days with PRCP >= 20 mm (days)."""
    return _count_ge(prcp, 20.0, "R20", policy)


def rnn(prcp, threshold=25.0, policy=_P):
    """Rnn: days with PRCP >= user threshold (days)."""
    return _count_ge(prcp, threshold, "Rnn", policy)


def prcptot(prcp, wet_threshold=1.0, policy=_P):
    """PRCPTOT: annual total precipitation on wet days (mm). 0 if no wet day in a complete year."""
    wet = prcp.where(prcp >= wet_threshold, 0.0).where(prcp.notna())
    res = wet.groupby(period_key(prcp.index, ANNUAL)).sum(min_count=1)
    return mask_incomplete(res, prcp.notna(), ANNUAL, policy).rename("PRCPTOT")


def sdii(prcp, wet_threshold=1.0, policy=_P):
    """SDII: annual mean precipitation on wet days (mm/day). NaN if the year has no wet day (undefined)."""
    isw = prcp >= wet_threshold
    key = period_key(prcp.index, ANNUAL)
    tot = prcp.where(isw).groupby(key).sum(min_count=1)
    n = isw.astype(float).groupby(key).sum()
    res = tot / n.where(n > 0)
    return mask_incomplete(res, prcp.notna(), ANNUAL, policy).rename("SDII")


def cdd(prcp, wet_threshold=1.0, policy=_P, span=SPAN_END_YEAR):
    """CDD: longest run of days with PRCP < wet_threshold (days); missing day breaks a run."""
    return max_spell_by_year(prcp < wet_threshold, prcp.notna(), "CDD", policy, span)


def cwd(prcp, wet_threshold=1.0, policy=_P, span=SPAN_END_YEAR):
    """CWD: longest run of days with PRCP >= wet_threshold (days); missing day breaks a run."""
    return max_spell_by_year(prcp >= wet_threshold, prcp.notna(), "CWD", policy, span)
