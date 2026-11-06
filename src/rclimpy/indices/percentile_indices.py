"""Percentile-based indices: TN10p, TX10p, TN90p, TX90p, WSDI/CSDI thresholds, R95p, R99p.

Method (Zhang et al. 2005, as used by RClimDex): per calendar day, pool values from a 5-day
window (+-2 actual dates) over all base-period years; percentile of that pool = daily threshold;
thresholds need >= 70 % of expected values. For base-period years the exceedance rate uses the
bootstrap (replace year Y by each other base year Z in turn, recompute thresholds, average).
Leap handling: 366 calendar bins (Feb 29 has its own bin, fed by leap years only).
ASSUMPTIONS flagged in docs/ASSUMPTIONS_AND_PENDING.md (quantile type 8, 366 bins, strict >,<).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..common import ANNUAL, MONTHLY, ClimateIndexError, calendar_bin
from ..completeness import CompletenessPolicy, mask_incomplete, period_key
from .spell_indices import days_in_spells

_P = CompletenessPolicy()
_M = {4: lambda p: 0.0, 5: lambda p: 0.5, 6: lambda p: p, 7: lambda p: 1 - p,
      8: lambda p: (p + 1) / 3, 9: lambda p: p / 4 + 3 / 8}   # Hyndman-Fan: h = n*p + m(p)


def hf_quantile(sorted_vals: np.ndarray, n, p: float, qtype: int = 8) -> np.ndarray:
    """Hyndman-Fan quantile along the last axis. sorted_vals ascending with NaN padding after the
    first n entries; n = number of valid values per row. Type 8 == numpy method='median_unbiased'."""
    if qtype not in _M:
        raise ClimateIndexError(f"Unsupported quantile type {qtype}")
    n = np.asarray(n)
    nn = np.maximum(n, 1)
    h = n * p + _M[qtype](p)
    lo = np.clip(np.floor(h), 1, nn).astype(int)
    hi = np.clip(lo + 1, 1, nn)
    frac = np.clip(h - lo, 0.0, 1.0)
    v_lo = np.take_along_axis(sorted_vals, (lo - 1)[..., None], axis=-1)[..., 0]
    v_hi = np.take_along_axis(sorted_vals, (hi - 1)[..., None], axis=-1)[..., 0]
    return np.where(n > 0, v_lo + frac * (v_hi - v_lo), np.nan)


@dataclass
class DailyPercentileThresholds:
    p: float
    values: np.ndarray        # (366,) threshold per calendar bin (NaN if unavailable)
    n_valid: np.ndarray       # (366,) pooled valid values
    n_expected: np.ndarray    # (366,) pooled expected values
    base_start: int
    base_end: int
    qtype: int
    window: int

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"bin": np.arange(366), "threshold": self.values,
                             "n_valid": self.n_valid, "n_expected": self.n_expected})


def _check_base(x: pd.Series, base_start: int, base_end: int) -> pd.Series:
    if base_start > base_end:
        raise ClimateIndexError("Base period start year is after end year.")
    if base_start < x.index.min().year or base_end > x.index.max().year:
        raise ClimateIndexError(f"Base period {base_start}-{base_end} is not inside the data range "
                                f"{x.index.min().year}-{x.index.max().year}.")
    return x[(x.index.year >= base_start) & (x.index.year <= base_end)]


def _window_arrays(xb: pd.Series, window: int):
    """A (366,N,W): values at actual dates t+k (k=-W//2..W//2); E: slot exists inside base period."""
    if window % 2 == 0 or window < 1:
        raise ClimateIndexError("Percentile window must be odd.")
    T, half = len(xb), window // 2
    vals = xb.to_numpy(float)
    sh = np.full((T, window), np.nan)
    ex = np.zeros((T, window), bool)
    for j, k in enumerate(range(-half, half + 1)):
        if k < 0:
            sh[-k:, j], ex[-k:, j] = vals[:T + k], True
        elif k > 0:
            sh[:T - k, j], ex[:T - k, j] = vals[k:], True
        else:
            sh[:, j], ex[:, j] = vals, True
    yidx = xb.index.year.to_numpy() - xb.index.year.min()
    b = calendar_bin(xb.index)
    N = int(yidx.max()) + 1
    A = np.full((366, N, window), np.nan)
    E = np.zeros((366, N, window), bool)
    A[b, yidx, :], E[b, yidx, :] = sh, ex
    return A, E


def compute_daily_thresholds(x: pd.Series, base_start: int, base_end: int, p: float,
                             policy: CompletenessPolicy = _P, window: int = 5, qtype: int = 8
                             ) -> DailyPercentileThresholds:
    xb = _check_base(x, base_start, base_end)
    A, E = _window_arrays(xb, window)
    flat = A.reshape(366, -1)
    n_valid = np.isfinite(flat).sum(1)
    n_exp = E.reshape(366, -1).sum(1)
    thr = hf_quantile(np.sort(flat, axis=1), n_valid, p / 100.0, qtype)
    thr = np.where(n_valid >= policy.min_base_fraction * n_exp, thr, np.nan)
    return DailyPercentileThresholds(p, thr, n_valid, n_exp, base_start, base_end, qtype, window)


def bootstrap_exceedance(A: np.ndarray, E: np.ndarray, p: float, direction: str,
                         policy: CompletenessPolicy, qtype: int = 8) -> np.ndarray:
    """(366,N) exceedance rate in [0,1] for each base year Y (NaN where undefined): average over
    Z != Y of the exceedance of Y's days against thresholds from {all years except Y} + copy of Z."""
    W = A.shape[2]
    center = A[:, :, W // 2]
    N = A.shape[1]
    has = np.isfinite(center).any(axis=0)
    res = np.full((366, N), np.nan)
    for Y in range(N):
        if not has[Y]:
            continue
        others = [i for i in range(N) if i != Y and has[i]]
        if not others:
            continue
        S = np.sort(A[:, others, :].reshape(366, -1), axis=1)
        n_o = np.isfinite(S).sum(1)
        e_o = E[:, others, :].reshape(366, -1).sum(1)
        acc, cnt = np.zeros(366), np.zeros(366)
        cy = center[:, Y]
        for Z in others:
            merged = np.sort(np.concatenate([S, A[:, Z, :]], axis=1), axis=1)
            n = n_o + np.isfinite(A[:, Z, :]).sum(1)
            ee = e_o + E[:, Z, :].sum(1)
            thr = hf_quantile(merged, n, p / 100.0, qtype)
            thr = np.where(n >= policy.min_base_fraction * ee, thr, np.nan)
            ok = np.isfinite(cy) & np.isfinite(thr)
            exc = (cy > thr) if direction == "above" else (cy < thr)
            acc += np.where(ok, exc, 0)
            cnt += ok
        res[:, Y] = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return res


def daily_exceedance(x: pd.Series, thr: DailyPercentileThresholds, direction: str,
                     policy: CompletenessPolicy = _P, bootstrap: bool = True) -> pd.Series:
    """Daily exceedance indicator (0/1; fractional in base years when bootstrapped; NaN if undefined)."""
    if direction not in ("above", "below"):
        raise ClimateIndexError("direction must be 'above' or 'below'")
    b = calendar_bin(x.index)
    t = thr.values[b]
    v = x.to_numpy(float)
    ok = np.isfinite(v) & np.isfinite(t)
    exc = np.where(ok, ((v > t) if direction == "above" else (v < t)).astype(float), np.nan)
    if bootstrap:
        xb = _check_base(x, thr.base_start, thr.base_end)
        A, E = _window_arrays(xb, thr.window)
        if np.isfinite(A[:, :, thr.window // 2]).any(axis=0).sum() < 2:
            warnings.warn("Bootstrap needs >= 2 base years with data; using plain thresholds.")
        else:
            boot = bootstrap_exceedance(A, E, thr.p, direction, policy, thr.qtype)
            inb = (x.index.year >= thr.base_start) & (x.index.year <= thr.base_end)
            yi = x.index.year.to_numpy()[inb] - thr.base_start
            exc[inb] = boot[b[inb], yi]
    return pd.Series(exc, index=x.index)


def percentile_index(x: pd.Series, name: str, p: float, direction: str, base_start: int, base_end: int,
                     freq: str = MONTHLY, policy: CompletenessPolicy = _P, window: int = 5,
                     qtype: int = 8, bootstrap: bool = True) -> pd.Series:
    """% of days in the period exceeding the daily percentile threshold (denominator = days with a
    defined exceedance)."""
    thr = compute_daily_thresholds(x, base_start, base_end, p, policy, window, qtype)
    exc = daily_exceedance(x, thr, direction, policy, bootstrap)
    key = period_key(x.index, freq)
    res = 100.0 * exc.groupby(key).sum() / exc.notna().groupby(key).sum().replace(0, np.nan)
    return mask_incomplete(res, exc.notna(), freq, policy).rename(name)


PERCENTILE_SPECS = {"TX10p": ("tmax", 10, "below"), "TX90p": ("tmax", 90, "above"),
                    "TN10p": ("tmin", 10, "below"), "TN90p": ("tmin", 90, "above")}


def spell_duration_indicator(x: pd.Series, name: str, p: float, direction: str, base_start: int,
                             base_end: int, policy: CompletenessPolicy = _P, min_span: int = 6,
                             window: int = 5, qtype: int = 8) -> pd.Series:
    """WSDI (TX > 90th, 'above') / CSDI (TN < 10th, 'below'): days per year in spells of >= min_span
    consecutive days. Plain (non-bootstrapped) daily thresholds; runs cross year boundaries."""
    thr = compute_daily_thresholds(x, base_start, base_end, p, policy, window, qtype)
    t = thr.values[calendar_bin(x.index)]
    flag = (x > t) if direction == "above" else (x < t)     # NaN compares False -> breaks runs
    valid = x.notna() & pd.Series(np.isfinite(t), index=x.index)
    return days_in_spells(flag, valid, min_span, name, policy)


def wet_day_threshold(prcp: pd.Series, base_start: int, base_end: int, p: float,
                      wet_threshold: float = 1.0, qtype: int = 8) -> float:
    """p-th percentile of wet-day (PRCP >= wet_threshold) amounts in the base period."""
    base = _check_base(prcp, base_start, base_end)
    w = np.sort(base[base >= wet_threshold].dropna().to_numpy())
    if w.size == 0:
        return float("nan")
    return float(hf_quantile(w[None, :], np.array([w.size]), p / 100.0, qtype)[0])


def very_wet_precip(prcp: pd.Series, name: str, p: float, base_start: int, base_end: int,
                    wet_threshold: float = 1.0, policy: CompletenessPolicy = _P, qtype: int = 8) -> pd.Series:
    """R95p / R99p: annual sum of PRCP on days with PRCP > base-period wet-day p-th percentile (mm)."""
    thr = wet_day_threshold(prcp, base_start, base_end, p, wet_threshold, qtype)
    key = period_key(prcp.index, ANNUAL)
    if np.isnan(thr):
        return pd.Series(np.nan, index=sorted(set(prcp.index.year)), name=name).rename_axis("year")
    res = prcp.where(prcp > thr).groupby(key).sum(min_count=0)
    res = res.where(prcp.notna().groupby(key).sum() > 0)
    return mask_incomplete(res, prcp.notna(), ANNUAL, policy).rename(name)
