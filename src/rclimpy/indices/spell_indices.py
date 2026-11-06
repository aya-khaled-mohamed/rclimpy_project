"""Run/spell helpers and spell-based indices (WSDI, CSDI, CDD, CWD)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..common import ANNUAL, ClimateIndexError
from ..completeness import CompletenessPolicy, mask_incomplete

SPAN_END_YEAR = "end_year"   # run measured across year boundaries, credited to the year it ends
SPAN_SPLIT = "split"         # run cut at 31 Dec / 1 Jan


def first_span(flag: np.ndarray, min_len: int, lo: int = 0, hi: int | None = None) -> int:
    """Index of the first run start (in [lo, hi)) of >= min_len consecutive True; -1 if none."""
    flag = np.asarray(flag, dtype=bool)
    hi = len(flag) if hi is None else hi
    if len(flag) < min_len:
        return -1
    c = np.convolve(flag.astype(np.int64), np.ones(min_len, dtype=np.int64), mode="valid")
    starts = np.flatnonzero(c == min_len)
    starts = starts[(starts >= lo) & (starts < hi)]
    return int(starts[0]) if starts.size else -1


def _runs(flag: np.ndarray, years: np.ndarray | None = None):
    """Return (starts, ends_inclusive) of True-runs; if years given, runs are cut at year changes."""
    f = np.asarray(flag, dtype=bool)
    if f.size == 0:
        return np.array([], int), np.array([], int)
    change = np.r_[True, f[1:] != f[:-1]]
    if years is not None:
        change |= np.r_[True, years[1:] != years[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:] - 1, f.size - 1]
    keep = f[starts]
    return starts[keep], ends[keep]


def max_spell_by_year(flag: pd.Series, valid: pd.Series, name: str,
                      policy: CompletenessPolicy, span: str = SPAN_END_YEAR) -> pd.Series:
    """Longest run of True per year. Missing (NaN) days must already be False in `flag`,
    so a missing day breaks a run. Years with valid data but no run give 0."""
    if span not in (SPAN_END_YEAR, SPAN_SPLIT):
        raise ClimateIndexError(f"span must be '{SPAN_END_YEAR}' or '{SPAN_SPLIT}'")
    yrs = flag.index.year.to_numpy()
    s, e = _runs(flag.to_numpy(), yrs if span == SPAN_SPLIT else None)
    all_years = pd.Index(sorted(set(yrs)), name="year")
    out = pd.Series(0.0, index=all_years)
    if s.size:
        lens = pd.Series(e - s + 1, index=yrs[e])
        mx = lens.groupby(level=0).max()
        out.loc[mx.index] = mx.astype(float)
    return mask_incomplete(out, valid, ANNUAL, policy).rename(name)


def days_in_spells(flag: pd.Series, valid: pd.Series, min_len: int, name: str,
                   policy: CompletenessPolicy) -> pd.Series:
    """WSDI/CSDI: days per year in runs of >= min_len True days. Per the manual (Appendix C) a spell
    may continue into the next year and its WHOLE length is counted against the year in which it ENDS."""
    s, e = _runs(flag.to_numpy())
    yrs = flag.index.year.to_numpy()
    all_years = pd.Index(sorted(set(yrs)), name="year")
    out = pd.Series(0.0, index=all_years)
    for a, b in zip(s, e):
        if b - a + 1 >= min_len:
            out.loc[yrs[b]] += b - a + 1
    return mask_incomplete(out, valid, ANNUAL, policy).rename(name)
