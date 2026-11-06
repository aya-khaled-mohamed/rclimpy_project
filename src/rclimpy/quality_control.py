"""Quality control per the RClimDex manual (section 3.2) + the extra checks requested in the project brief.

Manual rules applied: (1) -99.9 -> NA; (2) PRCP < 0 -> NA; (3) TMAX < TMIN -> NA ("unreasonable values");
(4) outliers in TMAX/TMIN flagged by mean +- n*std of the calendar day - FLAGGED ONLY, values NOT changed.
Brief-specific (not in the 1.0 manual): PRCP upper-limit flag (RClimDex GUI shows 'upper limit for PRCP = 200 mm';
FLAG ONLY), suspicious gap report. Original data are kept in an immutable copy; every change is in the audit log.
Which of TMAX/TMIN is set to NA when TMAX < TMIN is not stated; default 'both' (configurable).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .common import ClimateIndexError, MISSING_CODE
from .completeness import CompletenessPolicy, completeness_table
from .data_import import ImportResult
from .outlier_detection import OutlierResult, detect_temperature_outliers, to_rclimdex_tepstd
from .schema_validation import SchemaReport, build_dates, validate_schema

MONTH_NAMES = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
AUDIT_COLUMNS = ["timestamp", "step", "rule", "date", "variable", "original_value", "new_value", "action",
                 "reason", "affected_analysis"]


@dataclass
class QCConfig:
    missing_code: float = MISSING_CODE
    n_std: float = 3.0                   # manual default 3 (4 suggested if too many flags)
    prcp_upper_limit: float | None = 200.0   # flag only; None disables
    tmax_lt_tmin_action: str = "both"    # 'both' | 'none'
    outlier_min_samples: int = 10
    outlier_halfwidth_days: int = 0
    gap_warn_days: int = 30
    units_confirmed: bool = False        # must be True for files whose units are not defined by the format


@dataclass
class Decision:
    date: str
    variable: str                         # prcp | tmax | tmin
    action: str                           # accept | investigate | exclude | replace
    reason: str
    new_value: float | None = None
    analysis: str = "all"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


@dataclass
class QCResult:
    _original: pd.DataFrame               # stored raw daily frame (incl. -99.9); exposed only as copies via .original
    qc_data: pd.DataFrame                 # daily frame after manual rules (NaN = missing/invalid); flagged outliers untouched
    schema: SchemaReport
    outliers: OutlierResult
    audit: pd.DataFrame
    temp_qc: pd.DataFrame                 # '<station>tempQC.csv' layout
    prcp_qc: pd.DataFrame                 # '<station>prcpQC.csv' layout
    prcp_flags: pd.DataFrame              # PRCP above upper limit (flagged, unchanged)
    na_statistic: pd.DataFrame            # 'nastatistic.csv' layout
    completeness_monthly: pd.DataFrame
    completeness_annual: pd.DataFrame
    gaps: pd.DataFrame
    config: QCConfig
    source_sha256: str = ""

    @property
    def original(self) -> pd.DataFrame:
        """Raw values incl. -99.9. A fresh copy is returned on every access, so the stored original cannot be altered."""
        return self._original.copy(deep=True)

    def summary(self) -> dict:
        return {"rows_original": len(self.original), "temp_unreasonable": len(self.temp_qc), "prcp_negative": len(self.prcp_qc),
                "prcp_above_limit": len(self.prcp_flags), "outlier_days": len(self.outliers.table),
                "audit_entries": len(self.audit)}


def _log(rows, step, rule, date, var, orig, new, action, reason, analysis="all"):
    rows.append(dict(timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"), step=step, rule=rule,
                     date=pd.Timestamp(date).date().isoformat(), variable=var, original_value=orig, new_value=new,
                     action=action, reason=reason, affected_analysis=analysis))


def _na_statistic(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var, title in (("prcp", "PRCP"), ("tmax", "TMAX"), ("tmin", "TMIN")):
        na = daily[var].isna()
        g = na.groupby([daily.index.year, daily.index.month]).sum().unstack(fill_value=0).reindex(columns=range(1, 13), fill_value=0)
        for y, r in g.iterrows():
            rows.append([title, y, *r.astype(int).tolist(), int(r.sum())])
    return pd.DataFrame(rows, columns=["TITLE", "YEAR", *MONTH_NAMES, "ANN"])


def run_qc(imp: ImportResult, config: QCConfig | None = None, policy: CompletenessPolicy | None = None) -> QCResult:
    cfg, policy = config or QCConfig(), policy or CompletenessPolicy()
    if imp.units_source == "unspecified" and not cfg.units_confirmed:
        raise ClimateIndexError("Units are not defined by this file format. Confirm that PRCP is in mm and TMAX/TMIN in degC "
                                "(QCConfig.units_confirmed=True) - units are never assumed.")
    schema = validate_schema(imp, cfg.missing_code, cfg.gap_warn_days)
    if not schema.ok:
        raise ClimateIndexError("Input validation failed; fix these problems first:\n- " + "\n- ".join(schema.errors))
    dates, nums = build_dates(imp)
    audit: list = []
    if not dates.is_monotonic_increasing:
        _log(audit, "import", "sorted", dates.min(), "all", None, None, "reordered", "records were not in calendar order; sorted in working copy")
    order = np.argsort(dates.to_numpy(), kind="stable")
    raw = pd.DataFrame(nums.to_numpy()[order], index=pd.DatetimeIndex(dates.to_numpy()[order], name="date"), columns=["prcp", "tmax", "tmin"])
    full_idx = pd.date_range(f"{raw.index.min().year}-01-01", f"{raw.index.max().year}-12-31", freq="D", name="date")
    original = raw.reindex(full_idx)                       # missing calendar dates become NaN (reported in gaps)
    qc = original.copy(deep=True)
    # 1) missing code -> NaN (every conversion logged)
    code = np.isclose(qc, cfg.missing_code)
    for c in qc.columns:
        for dt in qc.index[code[:, list(qc.columns).index(c)]]:
            _log(audit, "qc1_missing", "missing_code", dt, c, cfg.missing_code, np.nan, "set_NA", f"{cfg.missing_code} is the documented missing-value code")
    qc = qc.mask(code)
    # 2) physically invalid values -> NA (manual)
    neg = qc["prcp"] < 0
    prcp_qc = qc[neg].reset_index()
    prcp_qc = pd.DataFrame({"year": prcp_qc.date.dt.year, "month": prcp_qc.date.dt.month, "day": prcp_qc.date.dt.day,
                            "prcp": prcp_qc.prcp, "tmax": prcp_qc.tmax, "tmin": prcp_qc.tmin})
    for dt in qc.index[neg]:
        _log(audit, "qc2_physical", "prcp_negative", dt, "prcp", qc.at[dt, "prcp"], np.nan, "set_NA", "daily precipitation < 0")
    qc.loc[neg, "prcp"] = np.nan
    bad_t = qc["tmax"] < qc["tmin"]
    tq = qc[bad_t].reset_index()
    temp_qc = pd.DataFrame({"year": tq.date.dt.year, "month": tq.date.dt.month, "day": tq.date.dt.day, "prcp": tq.prcp,
                            "tmax": tq.tmax, "tmin": tq.tmin, "tmax-tmin": tq.tmax - tq.tmin})
    if cfg.tmax_lt_tmin_action == "both":
        for dt in qc.index[bad_t]:
            for c in ("tmax", "tmin"):
                _log(audit, "qc2_physical", "tmax_lt_tmin", dt, c, qc.at[dt, c], np.nan, "set_NA", "daily maximum temperature < daily minimum temperature")
        qc.loc[bad_t, ["tmax", "tmin"]] = np.nan
    else:
        for dt in qc.index[bad_t]:
            _log(audit, "qc2_physical", "tmax_lt_tmin", dt, "tmax,tmin", None, None, "flag_only", "TMAX < TMIN (action='none')")
    # 3) PRCP upper limit: flag only
    if cfg.prcp_upper_limit is not None:
        hi = qc["prcp"] > cfg.prcp_upper_limit
        prcp_flags = qc[hi].reset_index()
        prcp_flags = pd.DataFrame({"year": prcp_flags.date.dt.year, "month": prcp_flags.date.dt.month, "day": prcp_flags.date.dt.day, "prcp": prcp_flags.prcp})
        for dt in qc.index[hi]:
            _log(audit, "qc3_suspicious", "prcp_above_limit", dt, "prcp", qc.at[dt, "prcp"], qc.at[dt, "prcp"], "flag_only",
                 f"PRCP > {cfg.prcp_upper_limit:g} mm: statistically unusual, not proven erroneous")
    else:
        prcp_flags = pd.DataFrame(columns=["year", "month", "day", "prcp"])
    # 4) temperature outliers: flag only
    out = detect_temperature_outliers(qc, cfg.n_std, cfg.outlier_min_samples, cfg.outlier_halfwidth_days)
    for _, r in out.table.iterrows():
        dt = f"{int(r.year):04d}-{int(r.month):02d}-{int(r.day):02d}"
        for v in ("tmax", "tmin", "dtr"):
            if r["flag_" + v]:
                _log(audit, "qc4_outlier", f"temp_outlier_{v}", dt, v, r[v], r[v], "flag_only",
                     f"outside [{r[v + 'low']:.2f}, {r[v + 'up']:.2f}] (mean +- {cfg.n_std:g} std of the calendar day); unusual, not proven erroneous")
    # reports
    valid_any = qc.notna().all(axis=1)
    comp_m = completeness_table(valid_any, "monthly", policy)
    comp_a = completeness_table(valid_any, "annual", policy)
    gaps = schema.missing_date_ranges
    return QCResult(original, qc, schema, out, pd.DataFrame(audit, columns=AUDIT_COLUMNS), temp_qc, prcp_qc, prcp_flags,
                    _na_statistic(qc), comp_m, comp_a, gaps, cfg, imp.sha256)


def apply_decisions(res: QCResult, decisions: list[Decision], analysis: str = "all") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the FINAL analysis dataset for `analysis` from qc_data + user decisions. Returns (final_data, audit_rows).
    accept/investigate leave the value; exclude -> NaN for this analysis only; replace -> user-approved value.
    The original and qc_data are never modified."""
    final, rows = res.qc_data.copy(deep=True), []
    for d in decisions:
        if d.action not in ("accept", "investigate", "exclude", "replace"):
            raise ClimateIndexError(f"Unknown action '{d.action}'. Use accept | investigate | exclude | replace.")
        if d.variable not in ("prcp", "tmax", "tmin"):
            raise ClimateIndexError(f"Unknown variable '{d.variable}'.")
        if not d.reason or not d.reason.strip():
            raise ClimateIndexError("Every decision needs a reason (it is written to the audit log).")
        dt = pd.Timestamp(d.date)
        if dt not in final.index:
            raise ClimateIndexError(f"Date {d.date} is outside the dataset.")
        if d.analysis not in ("all", analysis):
            continue
        orig = final.at[dt, d.variable]
        new = orig
        if d.action == "exclude":
            new = np.nan
        elif d.action == "replace":
            if d.new_value is None or not np.isfinite(d.new_value):
                raise ClimateIndexError("action 'replace' requires a finite new_value.")
            new = float(d.new_value)
        final.at[dt, d.variable] = new
        rows.append(dict(timestamp=d.timestamp, step="review", rule="user_decision", date=dt.date().isoformat(), variable=d.variable,
                         original_value=orig, new_value=new, action=d.action, reason=d.reason, affected_analysis=d.analysis))
    return final, pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def to_indcal_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """'<station>indcal.csv' layout: year, month, day, prcp, tmax, tmin with NA as blanks/NaN."""
    return pd.DataFrame({"year": daily.index.year, "month": daily.index.month, "day": daily.index.day,
                         "prcp": daily.prcp.to_numpy(), "tmax": daily.tmax.to_numpy(), "tmin": daily.tmin.to_numpy()})
