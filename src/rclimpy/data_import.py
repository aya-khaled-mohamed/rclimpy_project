"""Import station files (CSV / whitespace TXT / XLSX) WITHOUT altering values.

RClimDex input convention (manual, Appendix B): ASCII, space delimited, columns Year Month Day PRCP TMAX TMIN,
PRCP in mm, temperature in degC, missing = -99.9, calendar-date order, missing dates allowed, NO header line.
RClimDex itself fails on a header line (see "scan() expected 'a real'"); this importer accepts one.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .common import ClimateIndexError

RCLIMDEX_COLUMNS = ["year", "month", "day", "prcp", "tmax", "tmin"]
_ALIASES = {
    "year": ["year", "yr", "yyyy"], "month": ["month", "mon", "mm", "mo"], "day": ["day", "dd", "dy"],
    "date": ["date", "datetime", "time", "dates"],
    "prcp": ["prcp", "precip", "precipitation", "pr", "rain", "rr"],
    "tmax": ["tmax", "tx", "tmx", "maxtemp", "tmax_c"],
    "tmin": ["tmin", "tn", "tmn", "mintemp", "tmin_c"],
}


@dataclass
class ImportResult:
    raw: pd.DataFrame                  # values exactly as read (-99.9 still present); never modified later
    source_path: str
    sha256: str
    had_header: bool
    column_map: dict
    units_source: str                  # 'rclimdex_convention' | 'unspecified'
    notes: list = field(default_factory=list)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _looks_numeric(tok) -> bool:
    try:
        float(str(tok).strip())
        return True
    except ValueError:
        return False


def _auto_map(columns) -> dict:
    low = {str(c).strip().lower(): c for c in columns}
    out = {}
    for key, names in _ALIASES.items():
        for n in names:
            if n in low:
                out[key] = low[n]
                break
    return out


def read_station_file(path: str | Path, column_map: dict | None = None, sheet_name=0) -> ImportResult:
    """Read a file. `column_map` maps our names (year, month, day | date, prcp, tmax, tmin) to file columns
    and is REQUIRED when names cannot be recognised automatically. The file on disk is never modified."""
    path = Path(path)
    if not path.exists():
        raise ClimateIndexError(f"File not found: {path}")
    ext = path.suffix.lower()
    notes: list = []
    if ext in (".xlsx", ".xlsm"):
        df = pd.read_excel(path, sheet_name=sheet_name, header=0, engine="openpyxl")
        had_header = True
    elif ext in (".csv", ".txt", ".dat", ".tsv", ""):
        sep = "," if ext == ".csv" else r"\s+|,|;"
        first = path.read_text(errors="replace").splitlines()[:1]
        toks = pd.Series(first[0].replace(",", " ").replace(";", " ").split()) if first else pd.Series([], dtype=str)
        had_header = len(toks) > 0 and not all(_looks_numeric(t) for t in toks)
        df = pd.read_csv(path, sep=sep, engine="python", header=0 if had_header else None,
                         skipinitialspace=True, na_values=["", " ", "NA", "NaN", "nan", "NULL", "null", "-"])
    else:
        raise ClimateIndexError(f"Unsupported file type '{ext}'. Use .csv, .txt or .xlsx.")
    df.columns = [str(c).strip() for c in df.columns] if had_header else range(df.shape[1])
    if had_header:
        notes.append("Header line detected and used (RClimDex itself would reject it).")
        cmap = dict(column_map) if column_map else _auto_map(df.columns)
        units_source = "unspecified"
    else:
        if df.shape[1] < 6:
            raise ClimateIndexError(f"Expected 6 columns (Year Month Day PRCP TMAX TMIN), found {df.shape[1]}.")
        cmap = dict(column_map) if column_map else dict(zip(RCLIMDEX_COLUMNS, list(df.columns)[:6]))
        units_source = "rclimdex_convention"
    has_ymd = all(k in cmap for k in ("year", "month", "day"))
    if not (has_ymd or "date" in cmap) or not all(k in cmap for k in ("prcp", "tmax", "tmin")):
        raise ClimateIndexError("Could not identify the required columns (year/month/day or date, prcp, tmax, tmin). "
                                f"Columns found: {list(df.columns)}. Provide column_map explicitly.")
    bad = [v for v in cmap.values() if v not in df.columns]
    if bad:
        raise ClimateIndexError(f"column_map refers to columns that do not exist: {bad}")
    if column_map and had_header:
        units_source = "unspecified"
    return ImportResult(raw=df.copy(deep=True), source_path=str(path), sha256=_sha256(path),
                        had_header=had_header, column_map=cmap, units_source=units_source, notes=notes)
