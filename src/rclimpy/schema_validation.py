"""Schema / calendar validation. Reports problems; never drops or edits rows."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .common import ClimateIndexError
from .data_import import ImportResult


@dataclass
class SchemaReport:
    errors: list = field(default_factory=list)       # block QC
    warnings: list = field(default_factory=list)
    invalid_dates: pd.DataFrame = None
    duplicate_dates: pd.DataFrame = None
    non_numeric: pd.DataFrame = None
    missing_date_ranges: pd.DataFrame = None
    units_check_needed: bool = False

    @property
    def ok(self) -> bool:
        return not self.errors


def build_dates(imp: ImportResult) -> tuple[pd.Series, pd.DataFrame]:
    """Return (dates with NaT for invalid, numeric frame prcp/tmax/tmin with NaN for non-numeric)."""
    m, raw = imp.column_map, imp.raw
    if "date" in m:
        dates = pd.to_datetime(raw[m["date"]], errors="coerce")
    else:
        y, mo, d = (pd.to_numeric(raw[m[k]], errors="coerce") for k in ("year", "month", "day"))
        dates = pd.to_datetime(pd.DataFrame({"year": y, "month": mo, "day": d}), errors="coerce")
    nums = pd.DataFrame({k: pd.to_numeric(raw[m[k]], errors="coerce") for k in ("prcp", "tmax", "tmin")})
    return dates, nums


def validate_schema(imp: ImportResult, missing_code: float = -99.9, gap_warn_days: int = 30) -> SchemaReport:
    rep = SchemaReport()
    dates, nums = build_dates(imp)
    raw, m = imp.raw, imp.column_map
    # invalid calendar dates (e.g. 30 Feb)
    bad = dates.isna()
    rep.invalid_dates = raw[bad].assign(row=np.flatnonzero(bad.to_numpy()) + 1)
    if bad.any():
        rep.errors.append(f"{int(bad.sum())} row(s) have an invalid or unparsable date (first rows: "
                          f"{rep.invalid_dates['row'].head(5).tolist()}).")
    # non-numeric measurements (text that is not a recognised missing marker)
    nn = pd.DataFrame({k: pd.to_numeric(raw[m[k]], errors="coerce").isna() & raw[m[k]].notna() for k in ("prcp", "tmax", "tmin")})
    rep.non_numeric = raw[nn.any(axis=1)].assign(row=np.flatnonzero(nn.any(axis=1).to_numpy()) + 1)
    if nn.any().any():
        rep.errors.append(f"{int(nn.any(axis=1).sum())} row(s) contain non-numeric measurements.")
    ok = dates.dropna()
    dup = dates[dates.duplicated(keep=False) & dates.notna()]
    rep.duplicate_dates = raw.loc[dup.index].assign(row=dup.index + 1)
    if len(dup):
        rep.errors.append(f"{dup.nunique()} date(s) occur more than once ({len(dup)} rows). Decide which record is correct "
                          "(duplicates are never removed automatically).")
    if len(ok) and not ok.is_monotonic_increasing:
        rep.warnings.append("Records are not in calendar order; they will be sorted in the working copy (logged).")
    # calendar gaps
    u = pd.DatetimeIndex(sorted(set(ok)))
    if len(u) > 1:
        gaps = pd.Series(u[1:] - u[:-1], index=u[:-1]).dt.days
        gaps = gaps[gaps > 1]
        rep.missing_date_ranges = pd.DataFrame({"gap_start": gaps.index + pd.Timedelta(days=1),
                                                "gap_end": gaps.index + pd.to_timedelta(gaps.to_numpy() - 1, unit="D"),
                                                "missing_days": (gaps - 1).to_numpy()}).reset_index(drop=True)
        big = rep.missing_date_ranges[rep.missing_date_ranges.missing_days >= gap_warn_days]
        rep.warnings.append(f"{len(rep.missing_date_ranges)} calendar gap(s) in dates "
                            f"({int(rep.missing_date_ranges.missing_days.sum())} days); {len(big)} gap(s) >= {gap_warn_days} days.")
    else:
        rep.missing_date_ranges = pd.DataFrame(columns=["gap_start", "gap_end", "missing_days"])
    # unit plausibility (heuristic, only a warning that requires user confirmation)
    v = nums.mask(np.isclose(nums, missing_code))
    sus = []
    for t in ("tmax", "tmin"):
        if v[t].notna().any() and (v[t].max() > 60 or v[t].min() < -90):
            sus.append(f"{t} range [{v[t].min():g}, {v[t].max():g}] is not plausible in degC (Kelvin or Fahrenheit?)")
    if v["prcp"].notna().any() and v["prcp"].max() > 2000:
        sus.append(f"prcp max {v['prcp'].max():g} is implausible in mm (tenths of mm or inches?)")
    if sus:
        rep.units_check_needed = True
        rep.warnings.extend(sus)
    if imp.units_source == "unspecified":
        rep.units_check_needed = True
        rep.warnings.append("Units are not specified by the file format: confirm PRCP in mm and TMAX/TMIN in degC.")
    return rep
