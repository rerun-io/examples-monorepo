"""Per-segment fit-residual screening, not independent accuracy measurement."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import rerun as rr
from jaxtyping import Float64
from numpy import ndarray
from serde import serde
from serde.json import to_json

from selfcap_soma.anthropometry import SCALE_MAX, SCALE_MIN, plausible_ears


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
    ear_span_surviving_frames: int = 0
    ear_span_rejected_frames: int = 0
    head_torso_rotation_fallback: bool = False
    scale_bound_active: bool = False
    scale_min: float = SCALE_MIN
    scale_max: float = SCALE_MAX
    ear_span_min_mm: float = 110.0
    ear_span_max_mm: float = 200.0


def assess_quality(
    fitted: Float64[ndarray, "n 133 3"],
    exo: Float64[ndarray, "n 133 3"],
    quest: Float64[ndarray, "n 133 3"],
    scale: float | None = None,
    confidence: Float64[ndarray, "n 133"] | None = None,
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
    pairs = plausible_ears(exo)
    if confidence is not None:
        pairs &= (confidence[:, 3:5] > 0.0).any(1)
    surviving = int(pairs.sum())
    fallback = surviving < 3
    bound = scale is not None and (scale <= SCALE_MIN + 1e-6 or scale >= SCALE_MAX - 1e-6)
    if fallback:
        reasons.append("insufficient_plausible_ears")
    if bound:
        reasons.append("scale_bound_active")
    return SomaQuality(
        medians[0], medians[1], medians[2], medians[3], "flag" if reasons else "pass", tuple(reasons),
        ear_span_surviving_frames=surviving, ear_span_rejected_frames=len(exo) - surviving,
        head_torso_rotation_fallback=fallback, scale_bound_active=bound,
    )


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
        "ear_span_surviving_frames": quality.ear_span_surviving_frames,
        "ear_span_rejected_frames": quality.ear_span_rejected_frames,
        "head_torso_rotation_fallback": quality.head_torso_rotation_fallback,
        "scale_bound_active": quality.scale_bound_active,
        "scale_min": quality.scale_min,
        "scale_max": quality.scale_max,
        "ear_span_min_mm": quality.ear_span_min_mm,
        "ear_span_max_mm": quality.ear_span_max_mm,
    }
    for name in ("body", "shoulders", "ears", "hands"):
        value = getattr(quality, f"{name}_median_mm")
        if value is not None:
            values[f"{name}_median_mm"] = value
    recording.log("/__properties/soma_quality", rr.AnyValues(**values), static=True)
