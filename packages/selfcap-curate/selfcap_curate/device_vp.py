"""Fixed image-measured device focals with explicit initial-calibration fallback."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dataforge.records import read_json
from serde import serde


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class VPSample:
    segment: str
    """Raw catalog segment containing the measured room edges."""
    focal_px: float
    """Centre-fixed focal from the selected Manhattan directions."""
    orthogonality_deg: float | None
    """Maximum departure from orthogonality; null for the older bootstrap record."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class VPDevice:
    focal_px: float
    """Median image-only focal, never optimized against hand or lens labels."""
    low_px: float
    """Lower uncertainty envelope; its interpretation is stated in method."""
    high_px: float
    """Upper uncertainty envelope; its interpretation is stated in method."""
    samples: tuple[VPSample, ...]
    """Accepted source measurements; rejected non-Manhattan triples are excluded."""
    method: str
    """Estimator and uncertainty provenance."""

    def __post_init__(self) -> None:
        if not self.samples or not all(np.isfinite(x) and x > 0.0 for x in (self.focal_px, self.low_px, self.high_px)):
            raise ValueError('VP focal requires finite positive values and source measurements')
        if not self.low_px <= self.focal_px <= self.high_px:
            raise ValueError('VP focal must lie inside its uncertainty envelope')


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class VPTable:
    width: int
    """Source image width."""
    height: int
    """Source image height."""
    cx: float
    """Fixed image-centre principal point."""
    cy: float
    """Fixed image-centre principal point."""
    k1: float
    """Fixed radial coefficient."""
    k2: float
    """Fixed radial coefficient."""
    devices: dict[str, VPDevice]
    """Measurements keyed by complete device identifier."""


def read_vp_table(path: Path | None = None) -> VPTable:
    """Read the typed image-only measurement record."""
    return read_json(path if path is not None else Path(__file__).with_name('device_vp.json'), VPTable)
