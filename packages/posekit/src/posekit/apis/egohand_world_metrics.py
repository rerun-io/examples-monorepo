"""World-space hand correspondence metrics, without wrist alignment."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray
from serde import serde

from posekit.apis.egohand_comparison import EgoCamera
from posekit.apis.egohand_world import camera_world_poses
from posekit.models.egohand_geometry import SOMA_HAND21


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class DistanceSummary:
    """Matched, unaligned Euclidean residuals and their camera-depth component."""

    pairs: int
    """Finite corresponding joint pairs."""
    median_mm: float | None
    """Median 3D distance in world coordinates."""
    p90_mm: float | None
    """90th percentile 3D distance."""
    wrist_median_mm: float | None
    """Median wrist distance."""
    depth_bias_mm: float | None
    """Signed camera-Z residual; positive means EgoHand is farther away."""
    wrist_depth_bias_mm: float | None
    """Signed camera-Z residual of the wrist alone."""
    articulation_median_mm: float | None
    """Residual after independently subtracting each hand's wrist."""


def summarize_distances(
    predicted: Float32[ndarray, "n h j 3"], reference: Float32[ndarray, "n h j 3"], world_T_camera: Float64[ndarray, "n 4 4"]
) -> DistanceSummary:
    """Score matching anatomical joints and distinguish depth from articulation."""
    delta: Float32[ndarray, "n h j 3"] = predicted - reference
    valid = np.isfinite(delta).all(-1)
    if not valid.any():
        return DistanceSummary(0, None, None, None, None, None, None)
    distances = np.linalg.norm(delta, axis=-1) * 1000.0
    depth = np.einsum("nhjk,nk->nhj", delta, world_T_camera[:, :3, 2]) * 1000.0
    articulation = np.linalg.norm(delta - delta[:, :, :1], axis=-1) * 1000.0
    wrist = distances[:, :, 0]
    return DistanceSummary(
        int(valid.sum()),
        float(np.median(distances[valid])),
        float(np.percentile(distances[valid], 90)),
        float(np.nanmedian(wrist)) if np.isfinite(wrist).any() else None,
        float(np.median(depth[valid])),
        float(np.nanmedian(depth[:, :, 0])) if np.isfinite(depth[:, :, 0]).any() else None,
        float(np.nanmedian(articulation)) if np.isfinite(articulation).any() else None,
    )


def interpolate_joints(
    times: Float64[ndarray, "q"], positions: Float32[ndarray, "q h j 3"], targets: Float64[ndarray, "n"], max_gap: float
) -> Float32[ndarray, "n h j 3"]:
    """Interpolate finite joint pairs only within a bounded temporal gap."""
    right: Int64[ndarray, "n"] = np.clip(np.searchsorted(times, targets), 1, len(times) - 1)
    left: Int64[ndarray, "n"] = right - 1
    gaps = times[right] - times[left]
    weight = ((targets - times[left]) / np.maximum(gaps, 1e-12)).astype(np.float32)
    result = positions[left] * (1.0 - weight[:, None, None, None]) + positions[right] * weight[:, None, None, None]
    result[(targets < times[0]) | (targets > times[-1]) | (gaps > max_gap)] = np.nan
    return result


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class WorldReport:
    """Whole-segment and cursor-time world alignment, all distances in mm."""

    frames: int
    """OAK RGB frame count."""
    valid_hands: int
    """Hand observations with a calibrated world pose."""
    timestamp_s: float
    """Viewer cursor used for interpolated snapshot metrics."""
    whole_segment: dict[str, DistanceSummary]
    """Quest and SOMA residuals, pooled and split by side."""
    at_cursor: dict[str, DistanceSummary]
    """Same metrics after time interpolation to the viewer cursor."""


def compare_world_layer(parts: dict[str, Path], quest_path: Path, soma_path: Path, camera: EgoCamera, timestamp_s: float) -> WorldReport:
    """Compare the layer's 21 shared joints against latency-corrected references."""
    hand_arrays: list[Float32[ndarray, "n 25 3"]] = []
    times = np.empty(0, np.float64)
    for side in ("left", "right"):
        table = pq.read_table(parts[side], columns=["video_time", "hand_joints"])
        times = table["video_time"].to_numpy().astype(np.float64) * 1e-9
        hand_arrays.append(np.asarray([row if row else np.full((25, 3), np.nan) for row in table["hand_joints"].to_pylist()], np.float32))
    predicted = np.stack(hand_arrays, axis=1)[:, :, list(SOMA_HAND21)]
    quest = pq.read_table(quest_path)
    qt = quest["video_time"].to_numpy().astype(np.float64) * 1e-9
    qp = np.asarray(quest["joints"].to_pylist(), np.float32)[:, 91:133].reshape(-1, 2, 21, 3)
    soma = pq.read_table(soma_path)
    column: str = "/world/pred/soma_x/joints:Points3D:positions"
    soma = soma.filter(soma[column].is_valid())
    st = soma["video_time"].to_numpy().astype(np.float64) * 1e-9
    sp = np.asarray(soma[column].to_pylist(), np.float32)
    # SOMA-77 omits Root: LeftHand is 14; RightHand is 42, each followed by 24 finger joints.
    sp = np.stack([sp[:, 14:39], sp[:, 42:67]], axis=1)[:, :, list(SOMA_HAND21)]
    poses, valid = camera_world_poses(camera, times)
    predicted[~valid] = np.nan
    target = np.array([timestamp_s], dtype=np.float64)
    snapshot = interpolate_joints(times, predicted, target, 0.1)
    snapshot_poses, _ = camera_world_poses(camera, target)
    whole: dict[str, DistanceSummary] = {}
    cursor: dict[str, DistanceSummary] = {}
    for name, source_times, points, gap in (("quest", qt, qp, 0.04), ("soma", st, sp, 0.1)):
        reference = interpolate_joints(source_times, points, times, gap)
        at = interpolate_joints(source_times, points, target, gap)
        whole[name] = summarize_distances(predicted, reference, poses)
        cursor[name] = summarize_distances(snapshot, at, snapshot_poses)
        for side, label in enumerate(("left", "right")):
            whole[f"{name}_{label}"] = summarize_distances(predicted[:, side : side + 1], reference[:, side : side + 1], poses)
            cursor[f"{name}_{label}"] = summarize_distances(snapshot[:, side : side + 1], at[:, side : side + 1], snapshot_poses)
    return WorldReport(len(times), int(np.isfinite(predicted[:, :, 0]).all(-1).sum()), timestamp_s, whole, cursor)
