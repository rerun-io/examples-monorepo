"""Save fitted geometry and numerical diagnostics without catalog registration."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
from dataforge.writing import atomic_write
from jaxtyping import Float64, Int64
from numpy import ndarray
from serde import serde
from serde.json import to_json

from selfcap_soma.catalog import Observations
from selfcap_soma.fitting import FitResult
from selfcap_soma.hand_pose import HandOrientationSource
from selfcap_soma.observations import coco_joint_names


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Residual:
    median_mm: float | None
    p90_mm: float | None
    count: int


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Report:
    frames: int
    elapsed_s: float
    ms_per_frame: float
    peak_allocated_mib: float
    shared_scale: float
    quest_offset_ms: float
    requested_time_s: float | None
    sampled_time_s: float | None
    exo_body: Residual
    exo_hands: Residual
    quest_hands: Residual
    tracked_pose_to_anatomical_eye: Residual
    attachment_position: Residual
    ears: Residual
    head_anchor: str
    head_attachment_rotation: list[list[float]]
    head_offset_m: list[float]
    head_rotation_median_deg: float | None
    head_rotation_p90_deg: float | None
    target_fit_vs_quest: Residual | None
    target_exo_vs_quest: Residual | None
    target_fit_vs_exo: Residual | None
    limitations: tuple[str, ...]
    hand_orientation_source: HandOrientationSource | None = None
    """Per-hand measured orientation source; none uses the position-derived fallback."""


def residual(a: Float64[ndarray, "... 3"], b: Float64[ndarray, "... 3"]) -> Residual:
    """Unweighted Euclidean error on jointly finite points only."""
    distances = np.linalg.norm(a - b, axis=-1)
    valid = distances[np.isfinite(distances)] * 1000.0
    return Residual(float(np.median(valid)) if valid.size else None, float(np.percentile(valid, 90)) if valid.size else None, int(valid.size))


def mapped(result: FitResult, data: Observations) -> Float64[ndarray, "n 133 3"]:
    """SOMA joints and ear landmarks at their COCO-133 slots; other slots are NaN."""
    points = np.full_like(data.exo, np.nan)
    for key, name in coco_joint_names().items():
        points[:, key] = result.joints[:, result.names.index(name)]
    points[:, 3:5] = result.ear_landmarks
    return points


def joint_edges(parents: Int64[ndarray, "78"]) -> list[tuple[int, int]]:
    """Rig connections between nonvirtual joints, from the public parent indices."""
    return [(int(parent) - 1, child - 1) for child, parent in enumerate(parents) if child > 0 and parent > 0]


def log_joint_style(recording: rr.RecordingStream, path: str, edges: list[tuple[int, int]], joint_count: int) -> None:
    """The static rig connections and marker style of a joint entity."""
    recording.log(
        path,
        rr.AnnotationContext(rr.ClassDescription(info=rr.AnnotationInfo(id=0, color=[60, 220, 170]), keypoint_connections=edges)),
        static=True,
    )
    recording.log(path, rr.Points3D.from_fields(keypoint_ids=np.arange(joint_count), class_ids=0, colors=[60, 220, 170], radii=0.006), static=True)


def write_report(output: Path, data: Observations, result: FitResult, target_s: float | None = None, upper_body_only: bool = False) -> Report:
    """Write the strict JSON report of one fit."""
    output.mkdir(parents=True, exist_ok=True)
    fit_points: Float64[ndarray, "n 133 3"] = mapped(result, data)
    frame = None if target_s is None else int(np.clip(np.searchsorted(data.times, target_s, side="right") - 1, 0, len(data.times) - 1))
    report = Report(
        len(data.times),
        result.seconds,
        1000.0 * result.seconds / len(data.times),
        result.peak_bytes / 2**20,
        result.scale,
        data.offset_ms,
        target_s,
        None if frame is None else float(data.times[frame]),
        residual(fit_points[:, 5:13] if upper_body_only else fit_points[:, 5:17], data.exo[:, 5:13] if upper_body_only else data.exo[:, 5:17]),
        residual(fit_points[:, 91:], data.exo[:, 91:]),
        residual(fit_points[:, 91:], data.hands[:, 91:]),
        Residual(None, None, 0),
        Residual(None, None, 0),
        residual(result.ear_landmarks.astype(np.float64), data.exo[:, 3:5]),
        result.head_anchor,
        result.head_attachment.astype(float).tolist(),
        result.head_offset.astype(float).tolist(),
        float(np.median(result.head_angle_deg)) if np.isfinite(result.head_angle_deg).all() else None,
        float(np.percentile(result.head_angle_deg, 90)) if np.isfinite(result.head_angle_deg).all() else None,
        None if frame is None else residual(fit_points[frame, 91:], data.hands[frame, 91:]),
        None if frame is None else residual(data.exo[frame, 91:], data.hands[frame, 91:]),
        None if frame is None else residual(fit_points[frame, 91:], data.exo[frame, 91:]),
        (
            "Training residuals, not independent accuracy measurements.",
            "Head position is constrained only by triangulated ears; no headset translation term.",
            "Shared native identity, global scale and optional body bone ratios; no hand PCA or learned pose prior.",
            "No floor-contact term, image fitting or uncertainty estimate.",
            "Quest bone orientations constrained." if result.hand_orientations else "No hand bone orientation supervision.",
            "Low-LOD mesh; procedural twists enabled, pose correctives unavailable."
            if result.display_twists
            else "Low-LOD mesh with procedural twists and pose correctives disabled.",
            "Temporal prior is timestamp-scaled velocity change in translation and axis-angle parameters.",
        ),
        result.hand_orientation_source,
    )
    with atomic_write(output / "report.json") as staged:
        staged.write_text(to_json(report))
    return report
