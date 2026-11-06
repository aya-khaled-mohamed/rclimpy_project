"""Non-percentile temperature indices. Inputs: daily pandas Series on a full calendar (NaN = missing)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..common import ANNUAL, MONTHLY
from ..completeness import CompletenessPolicy, mask_incomplete, period_key
from .spell_indices import first_span

_P = CompletenessPolicy()


def _count(flag: pd.Series, valid: pd.Series, name: str, freq: str, policy: CompletenessPolicy) -> pd.Series:
    cnt = flag.astype(float).groupby(period_key(flag.index, freq)).sum()
    return mask_incomplete(cnt, valid, freq, policy).rename(name)


def frost_days(tn: pd.Series, threshold: float = 0.0, freq: str = ANNUAL, policy: CompletenessPolicy = _P) -> pd.Series:
    """FD: days with TN < threshold (strict). Unit: days."""
    return _count(tn < threshold, tn.notna(), f"FD{threshold:g}", freq, policy)


def summer_days(tx: pd.Series, threshold: float = 25.0, freq: str = ANNUAL, policy: CompletenessPolicy = _P) -> pd.Series:
    """SU: days with TX > threshold (strict). Unit: days."""
    return _count(tx > threshold, tx.notna(), f"SU{threshold:g}", freq, policy)


def ice_days(tx: pd.Series, threshold: float = 0.0, freq: str = ANNUAL, policy: CompletenessPolicy = _P) -> pd.Series:
    """ID: days with TX < threshold (strict). Unit: days."""
    return _count(tx < threshold, tx.notna(), f"ID{threshold:g}", freq, policy)


def tropical_nights(tn: pd.Series, threshold: float = 20.0, freq: str = ANNUAL, policy: CompletenessPolicy = _P) -> pd.Series:
    """TR: days with TN > threshold (strict). Unit: days."""
    return _count(tn > threshold, tn.notna(), f"TR{threshold:g}", freq, policy)


def _extreme(x: pd.Series, how: str, name: str, freq: str, policy: CompletenessPolicy) -> pd.Series:
    g = x.groupby(period_key(x.index, freq))
    res = g.max() if how == "max" else g.min()
    return mask_incomplete(res, x.notna(), freq, policy).rename(name)


def txx(tx, freq=MONTHLY, policy=_P):
    """TXx: max of daily TX per month/year. Unit: degC."""
    return _extreme(tx, "max", "TXx", freq, policy)


def tnx(tn, freq=MONTHLY, policy=_P):
    """TNx: max of daily TN per month/year. Unit: degC."""
    return _extreme(tn, "max", "TNx", freq, policy)


def txn(tx, freq=MONTHLY, policy=_P):
    """TXn: min of daily TX per month/year. Unit: degC."""
    return _extreme(tx, "min", "TXn", freq, policy)


def tnn(tn, freq=MONTHLY, policy=_P):
    """TNn: min of daily TN per month/year. Unit: degC."""
    return _extreme(tn, "min", "TNn", freq, policy)


def diurnal_temperature_range(tx: pd.Series, tn: pd.Series, freq: str = MONTHLY,
                              policy: CompletenessPolicy = _P) -> pd.Series:
    """DTR: mean of daily (TX - TN) per period; a day is valid only if both are present. Unit: degC."""
    dtr = tx - tn
    res = dtr.groupby(period_key(dtr.index, freq)).mean()
    return mask_incomplete(res, dtr.notna(), freq, policy).rename("DTR")


def _season(arr: np.ndarray, split: int, thr: float, min_span: int) -> float:
    warm, cold = arr > thr, arr < thr          # NaN compares False -> breaks spans
    start = first_span(warm, min_span, 0, split)
    if start < 0:
        return 0.0                               # no growing season
    end = first_span(cold, min_span, split, len(arr))
    if end < 0:
        end = len(arr)                           # no cold spell: season runs to end of period
    return float(end - start)


def growing_season_length(tx: pd.Series, tn: pd.Series, latitude: float = 0.0,
                          policy: CompletenessPolicy = _P, tg_threshold: float = 5.0,
                          min_span: int = 6, sh_label: str = "start") -> pd.Series:
    """GSL (days). TG=(TX+TN)/2. NH (lat>=0): start = first run of >= min_span days with TG > 5
    beginning before 1 Jul; end = first run of >= min_span days with TG < 5 beginning on/after 1 Jul
    (else 31 Dec). GSL = end_index - start_index (days from first warm day up to the day before the
    cold span); 0 if no start. SH (lat<0): season 1 Jul(y)..30 Jun(y+1), split at 1 Jan, labelled
    by start year (sh_label='start') or end year ('end'). Period with > max_missing_year missing -> NaN.
    ASSUMPTION: boundary conventions to be verified against the manual (see docs)."""
    tg = (tx + tn) / 2.0
    years = sorted(set(tg.index.year))
    out: dict[int, float] = {}
    for y in years:
        if latitude >= 0:
            seg = tg[tg.index.year == y]
            split = (pd.Timestamp(y, 7, 1) - pd.Timestamp(y, 1, 1)).days
            label = y
        else:
            seg = tg.reindex(pd.date_range(f"{y}-07-01", f"{y + 1}-06-30"))
            split = (pd.Timestamp(y + 1, 1, 1) - pd.Timestamp(y, 7, 1)).days
            label = y if sh_label == "start" else y + 1
        arr = seg.to_numpy(float)
        out[label] = np.nan if np.isnan(arr).sum() > policy.max_missing_year else _season(arr, split, tg_threshold, min_span)
    return pd.Series(out, name="GSL").rename_axis("year").sort_index()


def annual_mean(x: pd.Series, name: str, freq: str = ANNUAL, policy: CompletenessPolicy = _P) -> pd.Series:
    """TMAXmean / TMINmean: mean of daily TX / TN per period (degC). RClimDex extra output (not one of the 27)."""
    res = x.groupby(period_key(x.index, freq)).mean()
    return mask_incomplete(res, x.notna(), freq, policy).rename(name)
