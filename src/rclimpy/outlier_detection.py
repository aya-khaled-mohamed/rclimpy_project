"""Temperature outlier detection - the documented RClimDex method as a separate, identifiable QC method.

Manual (3.2): outliers are daily values outside [mean - n*std, mean + n*std], where mean and std are the
climatology of THAT calendar day (n user-defined, default 3; 4 suggested if 3 flags too many).
Verified from the supplied reference file sampledata_tepstdQC.csv: bounds are identical for all years of the same
calendar day, and a day is listed when TMAX, TMIN or DTR (=TMAX-TMIN) lies outside its bounds.
NOT stated in the manual (configurable, to be checked against reference numbers): std degrees of freedom (ddof=1,
as R sd()), whether neighbouring days are pooled (halfwidth, default 0 = same calendar day only), handling of
small samples and zero std (explicit below).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .common import calendar_bin

VARS = ("tmax", "tmin", "dtr")


def daily_climatology(x: pd.Series, halfwidth: int = 0, ddof: int = 1) -> pd.DataFrame:
    """Per calendar bin (0..365; Feb 29 = bin 59): mean, std, n. halfwidth>0 pools +-halfwidth neighbouring bins."""
    b = calendar_bin(x.index)
    v = x.to_numpy(float)
    ok = np.isfinite(v)
    n = np.bincount(b[ok], minlength=366).astype(float)
    s1 = np.bincount(b[ok], weights=v[ok], minlength=366)
    s2 = np.bincount(b[ok], weights=v[ok] ** 2, minlength=366)
    if halfwidth > 0:
        sh = range(-halfwidth, halfwidth + 1)
        n, s1, s2 = (sum(np.roll(a, k) for k in sh) for a in (n, s1, s2))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n
        var = (s2 - n * mean ** 2) / (n - ddof)
    var = np.where(n > ddof, np.maximum(var, 0.0), np.nan)
    return pd.DataFrame({"mean": np.where(n > 0, mean, np.nan), "std": np.sqrt(var), "n": n.astype(int)})


@dataclass
class OutlierResult:
    table: pd.DataFrame                 # one row per flagged date (RClimDex tepstdQC layout + flag columns)
    bounds: pd.DataFrame                # per date: low/up for each variable
    notes: list = field(default_factory=list)
    n_std: float = 3.0
    halfwidth: int = 0
    min_samples: int = 10


def detect_temperature_outliers(df: pd.DataFrame, n_std: float = 3.0, min_samples: int = 10,
                                halfwidth: int = 0) -> OutlierResult:
    """df: daily frame with DatetimeIndex and tmax/tmin (NaN = missing / already invalidated)."""
    if n_std <= 0:
        raise ValueError("n_std must be positive")
    d = pd.DataFrame({"tmax": df["tmax"], "tmin": df["tmin"]})
    d["dtr"] = d["tmax"] - d["tmin"]
    b = calendar_bin(d.index)
    notes, bounds, flags = [], pd.DataFrame(index=d.index), pd.DataFrame(index=d.index)
    for v in VARS:
        clim = daily_climatology(d[v], halfwidth)
        usable = (clim["n"] >= min_samples) & (clim["std"] > 0)
        small = clim.index[(clim["n"] < min_samples)]
        zero = clim.index[(clim["n"] >= min_samples) & ~(clim["std"] > 0)]
        if len(small):
            notes.append(f"{v}: {len(small)} calendar bin(s) have fewer than {min_samples} observations - not tested (bins {small.tolist()[:10]}).")
        if len(zero):
            notes.append(f"{v}: {len(zero)} calendar bin(s) have zero standard deviation - not tested.")
        mean, sd = clim["mean"].to_numpy()[b], clim["std"].to_numpy()[b]
        ok = usable.to_numpy()[b]
        bounds[v + "low"] = np.where(ok, mean - n_std * sd, np.nan)
        bounds[v + "up"] = np.where(ok, mean + n_std * sd, np.nan)
        x = d[v].to_numpy()
        flags[v] = ok & np.isfinite(x) & ((x < bounds[v + "low"]) | (x > bounds[v + "up"]))
    any_flag = flags.any(axis=1)
    t = pd.DataFrame({"year": d.index.year, "month": d.index.month, "day": d.index.day}, index=d.index)
    for v in VARS:
        t[v + "low"], t[v], t[v + "up"] = bounds[v + "low"], d[v], bounds[v + "up"]
    for v in VARS:
        t["flag_" + v] = flags[v]
    table = t[any_flag].reset_index(drop=True)
    return OutlierResult(table, bounds, notes, n_std, halfwidth, min_samples)


REFERENCE_COLUMNS = ["year", "month", "day", "tmaxlow", "tmax", "tmaxup", "tminlow", "tmin", "tminup", "dtrlow", "dtr", "dtrup"]


def to_rclimdex_tepstd(table: pd.DataFrame) -> pd.DataFrame:
    """tepstdQC.csv layout (2 decimals as in the reference file)."""
    return table[REFERENCE_COLUMNS].round(2)
