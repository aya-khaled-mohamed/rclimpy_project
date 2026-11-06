"""User-defined analysis parameters (mirrors the RClimDex 'User defined parameters' dialog)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .common import ClimateIndexError
from .completeness import CompletenessPolicy


@dataclass
class IndexParameters:
    base_start: int = 1961
    base_end: int = 1990
    latitude: float = 0.0
    longitude: float = 0.0
    su_threshold: float = 25.0     # user-defined upper threshold of TX  -> SU
    id_threshold: float = 0.0      # user-defined lower threshold of TX  -> ID
    tr_threshold: float = 20.0     # user-defined upper threshold of TN  -> TR
    fd_threshold: float = 0.0      # user-defined lower threshold of TN  -> FD
    rnn_threshold: float = 25.0    # user-defined daily precipitation threshold -> Rnn
    wet_day_threshold: float = 1.0
    min_spell: int = 6
    percentile_window: int = 5
    quantile_type: int = 8
    bootstrap: bool = True
    spell_span: str = "end_year"   # CDD/CWD: 'end_year' or 'split'
    completeness: CompletenessPolicy = field(default_factory=CompletenessPolicy)

    def validate(self, first_year: int, last_year: int) -> None:
        if not (-90 <= self.latitude <= 90) or not (-180 <= self.longitude <= 360):
            raise ClimateIndexError("Latitude must be in [-90, 90] and longitude in [-180, 360].")
        if self.base_start > self.base_end:
            raise ClimateIndexError("Base period start year is after end year.")
        if self.base_start < first_year or self.base_end > last_year:
            raise ClimateIndexError(f"Base period {self.base_start}-{self.base_end} lies outside the data "
                                    f"range {first_year}-{last_year}.")
        if self.min_spell < 2:
            raise ClimateIndexError("Minimum spell length must be >= 2.")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "IndexParameters":
        d = dict(d)
        d["completeness"] = CompletenessPolicy(**d.get("completeness", {}))
        return cls(**d)
