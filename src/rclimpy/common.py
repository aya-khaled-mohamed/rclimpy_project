"""Shared constants and helpers (independent of any UI)."""
from __future__ import annotations

import numpy as np
import pandas as pd

MISSING_CODE = -99.9          # documented RClimDex missing-value code
ANNUAL = "annual"
MONTHLY = "monthly"

# cumulative days before each month in a LEAP year (Feb 29 -> bin 59)
_CUM_LEAP = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])


class ClimateIndexError(ValueError):
    """Raised for invalid input to an index calculation (message is user-facing)."""


def calendar_bin(index: pd.DatetimeIndex) -> np.ndarray:
    """Map dates to a 0..365 calendar-day bin of a leap-year calendar.

    Jan 1 -> 0, Feb 28 -> 58, Feb 29 -> 59, Mar 1 -> 60, Dec 31 -> 365.
    """
    return _CUM_LEAP[index.month.to_numpy() - 1] + index.day.to_numpy() - 1


def full_calendar(s: pd.Series, first_year: int | None = None,
                  last_year: int | None = None) -> pd.Series:
    """Reindex a daily series to every calendar day Jan 1 (first) .. Dec 31 (last).

    Days not present become NaN (i.e. *missing*); nothing is interpolated.
    """
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ClimateIndexError("Series must have a DatetimeIndex.")
    if s.index.has_duplicates:
        raise ClimateIndexError("Duplicate dates found; resolve them in the QC step first.")
    y0 = int(first_year if first_year is not None else s.index.min().year)
    y1 = int(last_year if last_year is not None else s.index.max().year)
    idx = pd.date_range(f"{y0}-01-01", f"{y1}-12-31", freq="D")
    return s.astype(float).reindex(idx)


def daily_frame_from_ymd(df: pd.DataFrame, missing_code: float = MISSING_CODE) -> pd.DataFrame:
    """Build the internal daily frame (columns prcp, tmax, tmin) from Year/Month/Day columns.

    The -99.9 code is converted to NaN *in the returned copy only*; the input is untouched.
    The full import/validation/logging layer (data_import.py) is a later stage.
    """
    cols = {c.strip().lower(): c for c in df.columns}
    need = ["year", "month", "day", "prcp", "tmax", "tmin"]
    miss = [c for c in need if c not in cols]
    if miss:
        raise ClimateIndexError(f"Missing required columns: {miss}")
    dates = pd.to_datetime(dict(year=df[cols["year"]], month=df[cols["month"]], day=df[cols["day"]]),
                           errors="raise")
    out = pd.DataFrame({k: pd.to_numeric(df[cols[k]], errors="raise").astype(float).to_numpy()
                        for k in ("prcp", "tmax", "tmin")}, index=pd.DatetimeIndex(dates, name="date"))
    out = out.sort_index()
    out = out.mask(np.isclose(out, missing_code))
    if out.index.has_duplicates:
        raise ClimateIndexError("Duplicate dates found; resolve them in the QC step first.")
    full = pd.date_range(f"{out.index.min().year}-01-01", f"{out.index.max().year}-12-31", freq="D", name="date")
    return out.reindex(full)
