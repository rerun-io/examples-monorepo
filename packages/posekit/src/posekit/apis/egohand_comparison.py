"""Same-frame model comparison using SelfCap's camera and clock conventions.

The numerical algorithm in project_ego is ported from selfcap-opt/selfcap_curate/keypoints.py.
It interpolates camera pose, applies the full lens, and rejects gaps and folds.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import pyarrow as pa
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray
from serde import serde
from serde.json import from_json
from simplecv.rrd_query_utils import first_valid_value

from posekit.models.egohand_geometry import OAKCamera


@dataclass(frozen=True, slots=True)
class EgoCamera:
    """Calibrated OAK pose track in the reference implementation's layout."""

    rig: int
    """OAK rig index."""
    cam: int
    """Camera index, zero for RGB."""
    K: Float64[ndarray, "3 3"]
    """Pixel intrinsics."""
    distortion: Float64[ndarray, "14"]
    """Full OpenCV lens model."""
    size: tuple[int, int]
    """Image width and height."""
    rig_T_cam: Float64[ndarray, "4 4"]
    """RGB camera is the OAK rig origin."""
    times_s: Float64[ndarray, "p"]
    """Calibrated pose timestamps on video_time."""
    world_T_rig: Float64[ndarray, "p 4 4"]
    """OAK-to-world pose track."""


EGO_POSE_MAX_GAP_S: float = 0.1


@serde
@dataclass(frozen=True, slots=True)
class ComparisonClock:
    """Partial schema for the existing SelfCap clock record."""

    offsets_ms: dict[str, float | None]
    """Frozen capture-to-video offsets."""
    flags: list[str]
    """Flags that may invalidate the Quest clock."""
    hand_latency_ms: float = 48.0
    """Hand latency used by ego_quest_hands."""


def project_ego(camera: EgoCamera, times_s: Float64[ndarray, "n"], xyz: Float32[ndarray, "n j 3"]) -> Float32[ndarray, "n j 2"]:
    """World points at each time through the rig pose (lerp + slerp) and the full OAK lens model; missing when unprojectable.

    Points behind the camera or outside the calibrated field (beyond the image corners' rays, where the rational model can fold
    back into the image) stay NaN, as do times outside the pose track or across a pose gap.
    """
    from scipy.spatial.transform import Rotation, Slerp

    right: Int64[ndarray, "n"] = np.clip(np.searchsorted(camera.times_s, times_s), 1, len(camera.times_s) - 1)
    left: Int64[ndarray, "n"] = right - 1
    gap: Float64[ndarray, "n"] = camera.times_s[right] - camera.times_s[left]
    usable: Bool[ndarray, "n"] = (times_s >= camera.times_s[0]) & (times_s <= camera.times_s[-1]) & (gap <= EGO_POSE_MAX_GAP_S)
    clipped: Float64[ndarray, "n"] = np.clip(times_s, camera.times_s[0], camera.times_s[-1])
    alpha = ((clipped - camera.times_s[left]) / np.maximum(gap, 1e-12))[:, None]
    translation: Float64[ndarray, "n 3"] = (1.0 - alpha) * camera.world_T_rig[left, :3, 3] + alpha * camera.world_T_rig[right, :3, 3]
    rotation: Float64[ndarray, "n 3 3"] = Slerp(camera.times_s, Rotation.from_matrix(camera.world_T_rig[:, :3, :3]))(clipped).as_matrix()
    world_T_cam_r: Float64[ndarray, "n 3 3"] = rotation @ camera.rig_T_cam[:3, :3]
    world_T_cam_t: Float64[ndarray, "n 3"] = translation + np.einsum("nij,j->ni", rotation, camera.rig_T_cam[:3, 3])
    local: Float64[ndarray, "n j 3"] = np.einsum(
        "njk,nkl->njl", xyz.astype(np.float64) - world_T_cam_t[:, None, :], world_T_cam_r
    )  # R^T (x - t), row vectors
    width, height = camera.size
    corners: Float64[ndarray, "4 1 2"] = np.array([[0.0, 0.0], [width - 1.0, 0.0], [0.0, height - 1.0], [width - 1.0, height - 1.0]]).reshape(
        -1, 1, 2
    )
    max_radius: float = 1.05 * float(np.linalg.norm(cv2.undistortPoints(corners, camera.K, camera.distortion).reshape(-1, 2), axis=1).max())
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized: Float64[ndarray, "n j 2"] = local[..., :2] / local[..., 2:3]
    good: Bool[ndarray, "n j"] = (
        usable[:, None] & (local[..., 2] > 1e-3) & np.isfinite(normalized).all(-1) & (np.linalg.norm(normalized, axis=-1) <= max_radius)
    )
    uv: Float32[ndarray, "n j 2"] = np.full((*xyz.shape[:2], 2), np.nan, dtype=np.float32)
    if good.any():
        pixels, _ = cv2.projectPoints(local[good].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), camera.K, camera.distortion)
        uv[good] = pixels.reshape(-1, 2)
    return uv


def quest_pixels(table: pa.Table, clock_table: pa.Table, camera: OAKCamera, times_s: Float64[ndarray, "n"]) -> Float32[ndarray, "n 2 21 2"]:
    """Interpolate raw Quest hands at frame times, then use project_ego.

    Matches ego_quest_hands: capture time + quest_pose offset - 48 ms.
    Hands interpolate only across valid neighbours at most 40 ms apart.
    """
    clock: ComparisonClock = from_json(ComparisonClock, first_valid_value(clock_table["property:time_sync:record_json"])[0])
    offset: float | None = clock.offsets_ms.get("quest_pose")
    if offset is None or {"quest_unsynced_kept_as_cut", "quest_pose_unsynced"} & set(clock.flags):
        raise ValueError("Quest pose clock is not calibrated for this segment")
    hand_column: str = "/world/gt/coco133_xyz:Points3D:positions"
    hands: pa.Table = table.filter(table[hand_column].is_valid())
    hand_times: Float64[ndarray, "q"] = (
        hands["capture_time"].to_numpy().view(np.int64).astype(np.float64) / 1e9 + (offset - clock.hand_latency_ms) / 1000.0
    )
    positions: Float32[ndarray, "q 42 3"] = np.asarray(hands[hand_column].to_pylist(), dtype=np.float32)[:, 91:133]
    positions[(positions == 0.0).all(-1)] = np.nan
    right: Int64[ndarray, "n"] = np.clip(np.searchsorted(hand_times, times_s), 1, len(hand_times) - 1)
    left: Int64[ndarray, "n"] = right - 1
    gap: Float64[ndarray, "n"] = hand_times[right] - hand_times[left]
    valid = (times_s >= hand_times[0]) & (times_s <= hand_times[-1]) & (gap > 0.0) & (gap <= 0.04)
    alpha = ((times_s - hand_times[left]) / np.maximum(gap, 1e-12)).astype(np.float32)
    interpolated: Float32[ndarray, "n 42 3"] = (1.0 - alpha[:, None, None]) * positions[left] + alpha[:, None, None] * positions[right]
    interpolated[~valid] = np.nan
    poses: pa.Table = table.filter(table["/world/rig_04:Transform3D:mat3x3"].is_valid())
    pose_times: Float64[ndarray, "p"] = poses["video_time"].to_numpy().view(np.int64).astype(np.float64) / 1e9
    world_T_rig: Float64[ndarray, "p 4 4"] = np.repeat(np.eye(4)[None], len(poses), axis=0)
    world_T_rig[:, :3, :3] = np.asarray(poses["/world/rig_04:Transform3D:mat3x3"].to_pylist(), dtype=np.float64).reshape(-1, 3, 3).transpose(0, 2, 1)
    world_T_rig[:, :3, 3] = np.asarray(poses["/world/rig_04:Transform3D:translation"].to_pylist(), dtype=np.float64).reshape(-1, 3)
    model: EgoCamera = EgoCamera(4, 0, camera.intrinsics, camera.distortion, (camera.width, camera.height), np.eye(4), pose_times, world_T_rig)
    return project_ego(model, times_s, interpolated).reshape(-1, 2, 21, 2)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ErrorSummary:
    """Pixel errors over an explicitly matched set of joints."""

    pairs: int
    """Common finite model/Quest joint observations."""
    egohand_median_px: float | None
    """Median EgoHand distance."""
    egohand_p90_px: float | None
    """90th-percentile EgoHand distance."""
    sapiens_median_px: float | None
    """Median Sapiens2 distance."""
    sapiens_p90_px: float | None
    """90th-percentile Sapiens2 distance."""


def summarize(
    ego: Float32[ndarray, "n 2 21 2"],
    sapiens: Float32[ndarray, "n 2 21 2"],
    quest: Float32[ndarray, "n 2 21 2"],
    selection: Bool[ndarray, "n 2 21"] | Bool[ndarray, "n 2 1"] | None = None,
) -> ErrorSummary:
    """Use identical finite joints for both errors; never replace missing points."""
    valid = np.isfinite(ego).all(-1) & np.isfinite(sapiens).all(-1) & np.isfinite(quest).all(-1)
    if selection is not None:
        valid &= selection
    first: Float32[ndarray, "p"] = np.linalg.norm(ego - quest, axis=-1)[valid]
    second: Float32[ndarray, "p"] = np.linalg.norm(sapiens - quest, axis=-1)[valid]
    return ErrorSummary(
        len(first),
        float(np.median(first)) if len(first) else None,
        float(np.percentile(first, 90)) if len(first) else None,
        float(np.median(second)) if len(second) else None,
        float(np.percentile(second, 90)) if len(second) else None,
    )
