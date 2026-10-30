"""Per-segment fit-residual screening, not independent accuracy measurement."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import rerun as rr
from jaxtyping import Float64
from numpy import ndarray
from serde import serde
from serde.json import to_json


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class SomaQuality:
    body_median_mm: float | None
    shoulders_median_mm: float | None
    ears_median_mm: float | None
    hands_median_mm: float | None
    status: Literal["pass", "flag"]
    reasons: tuple[str, ...]
    body_threshold_mm: float = 40.0
    shoulders_threshold_mm: float = 50.0
    ears_threshold_mm: float = 30.0
    hands_threshold_mm: float = 15.0
    definition: str = "v1: finite-point training residuals; body COCO 5-12, shoulders 5-6, ears 3-4 vs exo; hands 91-132 vs Quest"


def assess_quality(
    fitted: Float64[ndarray, "n 133 3"], exo: Float64[ndarray, "n 133 3"], quest: Float64[ndarray, "n 133 3"]
) -> SomaQuality:
    """Flag missing metrics or medians above fixed screening thresholds."""
    medians: list[float | None] = []
    reasons: list[str] = []
    for name, selection, target, threshold in (
        ("body", slice(5, 13), exo, 40.0),
        ("shoulders", slice(5, 7), exo, 50.0),
        ("ears", slice(3, 5), exo, 30.0),
        ("hands", slice(91, 133), quest, 15.0),
    ):
        distances = np.linalg.norm(fitted[:, selection] - target[:, selection], axis=-1)
        finite = distances[np.isfinite(distances)]
        median = float(np.median(finite) * 1000.0) if finite.size else None
        medians.append(median)
        if median is None:
            reasons.append(f"{name}_missing")
        elif median > threshold:
            reasons.append(f"{name}_above_threshold")
    return SomaQuality(medians[0], medians[1], medians[2], medians[3], "flag" if reasons else "pass", tuple(reasons))


def log_quality(recording: rr.RecordingStream, quality: SomaQuality) -> None:
    """Expose a static catalog property; JSON preserves missing values as null."""
    values = {
        "record_json": to_json(quality),
        "status": quality.status,
        "definition": quality.definition,
        "body_threshold_mm": quality.body_threshold_mm,
        "shoulders_threshold_mm": quality.shoulders_threshold_mm,
        "ears_threshold_mm": quality.ears_threshold_mm,
        "hands_threshold_mm": quality.hands_threshold_mm,
    }
    for name in ("body", "shoulders", "ears", "hands"):
        value = getattr(quality, f"{name}_median_mm")
        if value is not None:
            values[f"{name}_median_mm"] = value
    recording.log("/__properties/soma_quality", rr.AnyValues(**values), static=True)
