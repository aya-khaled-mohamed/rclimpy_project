"""Missing-data / completeness rules.

Defaults follow the commonly documented RClimDex 1.x rules (monthly value requires
<= 3 missing days, annual value <= 15 missing days, 70 % data availability for
percentile thresholds). VERIFY against the attached manual; all are configurable.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .common import ANNUAL, MONTHLY, ClimateIndexError


@dataclass(frozen=True)
class CompletenessPolicy:
    max_missing_month: int = 3
    max_missing_year: int = 15
    min_base_fraction: float = 0.70
    # Manual App. C: "No annual value will be calculated if any one month's data are missing".
    # Interpreted as: every month of the year must satisfy the monthly rule (INTERPRETATION - verify
    # with reference output). Options: all_months_complete | no_empty_month | none
    annual_month_rule: str = "all_months_complete"

    def limit(self, freq: str) -> int:
        if freq == MONTHLY:
            return self.max_missing_month
        if freq == ANNUAL:
            return self.max_missing_year
        raise ClimateIndexError(f"Unknown frequency '{freq}'")


def period_key(index: pd.DatetimeIndex, freq: str):
    """Grouping key: int year for annual, monthly Period for monthly."""
    if freq == ANNUAL:
        return pd.Index(index.year, name="year")
    if freq == MONTHLY:
        return index.to_period("M").rename("month")
    raise ClimateIndexError(f"Unknown frequency '{freq}'")


def n_missing(valid: pd.Series, freq: str) -> pd.Series:
    """Missing calendar days per period (valid must cover whole calendar years)."""
    g = valid.astype(bool).groupby(period_key(valid.index, freq))
    return g.size() - g.sum().astype(int)


def is_complete(valid: pd.Series, freq: str, policy: CompletenessPolicy) -> pd.Series:
    ok = n_missing(valid, freq) <= policy.limit(freq)
    if freq == ANNUAL and policy.annual_month_rule != "none":
        m = valid.astype(bool).groupby([valid.index.year, valid.index.month])
        size = m.size()
        miss = size - m.sum().astype(int)
        if policy.annual_month_rule == "all_months_complete":
            bad = miss > policy.max_missing_month
        elif policy.annual_month_rule == "no_empty_month":
            bad = miss >= size
        else:
            raise ClimateIndexError("annual_month_rule must be all_months_complete | no_empty_month | none")
        ok = ok & ~bad.groupby(level=0).any().reindex(ok.index).fillna(False)
    return ok


def mask_incomplete(result: pd.Series, valid: pd.Series, freq: str,
                    policy: CompletenessPolicy) -> pd.Series:
    """Set result to NaN in periods that violate the completeness rule."""
    ok = is_complete(valid, freq, policy).reindex(result.index)
    return result.where(ok.fillna(False))


def completeness_table(valid: pd.Series, freq: str, policy: CompletenessPolicy) -> pd.DataFrame:
    g = valid.astype(bool).groupby(period_key(valid.index, freq))
    t = pd.DataFrame({"n_days": g.size(), "n_valid": g.sum().astype(int)})
    t["n_missing"] = t["n_days"] - t["n_valid"]
    t["pct_missing"] = 100.0 * t["n_missing"] / t["n_days"]
    t["complete"] = is_complete(valid, freq, policy).reindex(t.index)
    return t
