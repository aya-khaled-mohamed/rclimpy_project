"""Index registry + orchestration. UI-independent. Returns results AND an explicit status per index."""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .common import ANNUAL, MONTHLY, ClimateIndexError
from .indices import percentile_indices as pi
from .indices import precipitation_indices as pr
from .indices import temperature_indices as ti
from .parameters import IndexParameters

log = logging.getLogger(__name__)


@dataclass
class IndexSpec:
    name: str
    long_name: str
    unit: str
    freqs: tuple
    requires: tuple
    group: str                       # RClimDex GUI check-box group
    compute: Callable                # (daily_df, params, freq) -> Series


@dataclass
class IndexRun:
    results: dict = field(default_factory=dict)       # (name, freq) -> Series
    status: list = field(default_factory=list)        # dicts: index, freq, status, reason
    thresholds: dict = field(default_factory=dict)    # name -> DataFrame (366 bins)

    def status_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.status)


def build_specs(p: IndexParameters) -> list[IndexSpec]:
    c = p.completeness
    S = IndexSpec
    specs = [
        S("FD0", "Frost days", "days", (ANNUAL,), ("tmin",), "SU25, FD0, TR20, ID0", lambda d, p, f: ti.frost_days(d.tmin, 0.0, f, c)),
        S("SU25", "Summer days", "days", (ANNUAL,), ("tmax",), "SU25, FD0, TR20, ID0", lambda d, p, f: ti.summer_days(d.tmax, 25.0, f, c)),
        S("ID0", "Ice days", "days", (ANNUAL,), ("tmax",), "SU25, FD0, TR20, ID0", lambda d, p, f: ti.ice_days(d.tmax, 0.0, f, c)),
        S("TR20", "Tropical nights", "days", (ANNUAL,), ("tmin",), "SU25, FD0, TR20, ID0", lambda d, p, f: ti.tropical_nights(d.tmin, 20.0, f, c)),
        S("GSL", "Growing season length", "days", (ANNUAL,), ("tmax", "tmin"), "GSL", lambda d, p, f: ti.growing_season_length(d.tmax, d.tmin, p.latitude, c, 5.0, p.min_spell)),
        S("TXx", "Max of daily max temperature", "degC", (MONTHLY, ANNUAL), ("tmax",), "TXx, TXn, TNx, TNn", lambda d, p, f: ti.txx(d.tmax, f, c)),
        S("TNx", "Max of daily min temperature", "degC", (MONTHLY, ANNUAL), ("tmin",), "TXx, TXn, TNx, TNn", lambda d, p, f: ti.tnx(d.tmin, f, c)),
        S("TXn", "Min of daily max temperature", "degC", (MONTHLY, ANNUAL), ("tmax",), "TXx, TXn, TNx, TNn", lambda d, p, f: ti.txn(d.tmax, f, c)),
        S("TNn", "Min of daily min temperature", "degC", (MONTHLY, ANNUAL), ("tmin",), "TXx, TXn, TNx, TNn", lambda d, p, f: ti.tnn(d.tmin, f, c)),
        S("DTR", "Diurnal temperature range", "degC", (MONTHLY, ANNUAL), ("tmax", "tmin"), "DTR", lambda d, p, f: ti.diurnal_temperature_range(d.tmax, d.tmin, f, c)),
        S("RX1day", "Max 1-day precipitation", "mm", (MONTHLY, ANNUAL), ("prcp",), "Rx1day", lambda d, p, f: pr.rx1day(d.prcp, f, c)),
        S("RX5day", "Max 5-day precipitation", "mm", (MONTHLY, ANNUAL), ("prcp",), "Rx5day", lambda d, p, f: pr.rx5day(d.prcp, f, c)),
        S("SDII", "Simple daily intensity index", "mm/day", (ANNUAL,), ("prcp",), "SDII", lambda d, p, f: pr.sdii(d.prcp, p.wet_day_threshold, c)),
        S("R10", "Heavy precipitation days (>=10 mm)", "days", (ANNUAL,), ("prcp",), "R10mm", lambda d, p, f: pr.r10mm(d.prcp, c)),
        S("R20", "Very heavy precipitation days (>=20 mm)", "days", (ANNUAL,), ("prcp",), "R20mm", lambda d, p, f: pr.r20mm(d.prcp, c)),
        S("Rnn", "Days >= user precipitation threshold", "days", (ANNUAL,), ("prcp",), "Rnnmm", lambda d, p, f: pr.rnn(d.prcp, p.rnn_threshold, c)),
        S("CDD", "Consecutive dry days", "days", (ANNUAL,), ("prcp",), "CDD, CWD", lambda d, p, f: pr.cdd(d.prcp, p.wet_day_threshold, c, p.spell_span)),
        S("CWD", "Consecutive wet days", "days", (ANNUAL,), ("prcp",), "CDD, CWD", lambda d, p, f: pr.cwd(d.prcp, p.wet_day_threshold, c, p.spell_span)),
        S("R95p", "Very wet day precipitation", "mm", (ANNUAL,), ("prcp",), "R95p, R99p, PRCPTOT", lambda d, p, f: pi.very_wet_precip(d.prcp, "R95p", 95, p.base_start, p.base_end, p.wet_day_threshold, c, p.quantile_type)),
        S("R99p", "Extremely wet day precipitation", "mm", (ANNUAL,), ("prcp",), "R95p, R99p, PRCPTOT", lambda d, p, f: pi.very_wet_precip(d.prcp, "R99p", 99, p.base_start, p.base_end, p.wet_day_threshold, c, p.quantile_type)),
        S("PRCPTOT", "Wet-day precipitation total", "mm", (ANNUAL,), ("prcp",), "R95p, R99p, PRCPTOT", lambda d, p, f: pr.prcptot(d.prcp, p.wet_day_threshold, c)),
        S("WSDI", "Warm spell duration indicator", "days", (ANNUAL,), ("tmax",), "WSDI", lambda d, p, f: pi.spell_duration_indicator(d.tmax, "WSDI", 90, "above", p.base_start, p.base_end, c, p.min_spell, p.percentile_window, p.quantile_type)),
        S("CSDI", "Cold spell duration indicator", "days", (ANNUAL,), ("tmin",), "CSDI", lambda d, p, f: pi.spell_duration_indicator(d.tmin, "CSDI", 10, "below", p.base_start, p.base_end, c, p.min_spell, p.percentile_window, p.quantile_type)),
    ]
    specs += [  # RClimDex extra outputs (NOT part of the 27 core indices)
        S("TMAXmean", "Mean daily maximum temperature", "degC", (ANNUAL,), ("tmax",), "extra", lambda d, p, f: ti.annual_mean(d.tmax, "TMAXmean", f, c)),
        S("TMINmean", "Mean daily minimum temperature", "degC", (ANNUAL,), ("tmin",), "extra", lambda d, p, f: ti.annual_mean(d.tmin, "TMINmean", f, c)),
    ]
    for name, (var, pct, dirn) in pi.PERCENTILE_SPECS.items():
        longn = {"TX10p": "Cool days", "TX90p": "Warm days", "TN10p": "Cool nights", "TN90p": "Warm nights"}[name]
        specs.append(S(name, longn, "% of days", (MONTHLY, ANNUAL), (var,), "TX10p, TX90p, TN10p, TN90p",
                       lambda d, p, f, var=var, pct=pct, dirn=dirn, name=name: pi.percentile_index(
                           d[var], name, pct, dirn, p.base_start, p.base_end, f, c, p.percentile_window,
                           p.quantile_type, p.bootstrap)))
    core = {s.name for s in specs}
    user = [("FD", p.fd_threshold, ti.frost_days, "tmin"), ("SU", p.su_threshold, ti.summer_days, "tmax"),
            ("TR", p.tr_threshold, ti.tropical_nights, "tmin"), ("ID", p.id_threshold, ti.ice_days, "tmax")]
    for base, thr, fn, var in user:
        nm = f"{base}{thr:g}"
        if nm in core:
            continue          # identical to a core index; not duplicated
        specs.append(S(nm, f"User-defined {base}", "days", (ANNUAL,), (var,), "User Defined SU, FD, TR, ID",
                       lambda d, p, f, fn=fn, thr=thr, var=var: fn(d[var], thr, f, c)))
    return specs


def compute_indices(daily: pd.DataFrame, params: IndexParameters, selected: list[str] | None = None,
                    progress: Callable[[float, str], None] | None = None) -> IndexRun:
    """Compute selected indices (None = all). A failure in one index never stops the others; the
    reason is recorded in IndexRun.status (valid result vs. unavailable result are distinguished)."""
    params.validate(int(daily.index.min().year), int(daily.index.max().year))
    specs = [s for s in build_specs(params) if selected is None or s.name in selected]
    run = IndexRun()
    for i, s in enumerate(specs):
        if progress:
            progress(i / max(len(specs), 1), s.name)
        missing_vars = [v for v in s.requires if v not in daily or daily[v].notna().sum() == 0]
        if missing_vars:
            for f in s.freqs:
                run.status.append(dict(index=s.name, freq=f, status="unavailable", reason=f"no valid data for {missing_vars}"))
            continue
        for f in s.freqs:
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    res = s.compute(daily, params, f)
                res = res.rename(s.name)
                run.results[(s.name, f)] = res
                n_ok = int(res.notna().sum())
                run.status.append(dict(index=s.name, freq=f, status="ok" if n_ok else "unavailable",
                                       reason="; ".join(str(x.message) for x in w) or ("" if n_ok else "all periods incomplete or undefined"),
                                       n_periods=len(res), n_valid=n_ok, unit=s.unit))
            except ClimateIndexError as e:
                run.status.append(dict(index=s.name, freq=f, status="unavailable", reason=str(e)))
            except Exception as e:                      # noqa: BLE001 - recorded, never silent
                log.exception("Index %s failed", s.name)
                run.status.append(dict(index=s.name, freq=f, status="error", reason=f"{type(e).__name__}: {e}"))
    if progress:
        progress(1.0, "done")
    return run
