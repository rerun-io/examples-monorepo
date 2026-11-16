"""Rigid trajectory alignment and the rig trajectories the benchmark scores.

Everything here answers one of two questions: *where did the rig go* (`RigTrack`,
built either from a `frames_meta.json` or from a TUM `ground_truth.txt`) and *how
far apart are two answers* (`match_timestamps` to join them, `align_rigid` to fit
one onto the other).

The fit itself lives in `colsfm.rigid_fit` and is re-exported here, which is the
import path every caller already uses. It is a separate module because the `demo`
package runs in the *default* pixi environment, which has no pycolmap, and
`demo.poses` used to carry its own copy of the same SVD under the name
`umeyama_rigid`. Alignment defaults to rigid, with the scale **held at 1.0** and
the scale a similarity fit would have chosen reported separately;
`align_rigid(..., estimate_scale=True)` is the same fit with the scale let in, for
callers asking whether the *shape* is right rather than the size.

The rig trajectory comes from `FramesMeta.rig_frames()` rather than being
re-derived from COLMAP camera poses and the *input* `rig_T_cam` the way
`cusfm_rig_trajectory` does. Both answer the same question; the metadata route is
the one the export spec defines (`world_T_vehicle = world_T_cam * vehicle_T_cam^-1`,
reference = lowest keyframe id of the sample) and it does not silently depend on
bundle adjustment having left the extrinsics alone. The rigidity spread — the
disagreement between the per-camera estimates of one sample's rig pose — is still
computed exactly as `cusfm_rig_trajectory` computes it, just from the metadata's
poses.

This module deliberately imports nothing from `colsfm.benchmark`: the benchmark
is one consumer of these primitives, and `tools/audit` is another.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pycolmap
from jaxtyping import Bool, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from colsfm.frames_meta import FramesMeta, KeyframeMeta, RigFrame
from colsfm.geometry import MILLIMETRES_PER_METRE, TumPose, parse_tum_line
from colsfm.rigid_fit import (
    DEGENERATE_VARIANCE,
    Positions,
    RigidAlignment,
    Rotations,
    Timestamps,
    align_rigid,
    path_length_meters,
)

__all__ = [
    "DEGENERATE_VARIANCE",
    "GROUND_TRUTH_FILE_NAME",
    "GROUND_TRUTH_TOLERANCE_MICROSECONDS",
    "Positions",
    "RigTrack",
    "RigidAlignment",
    "Rotations",
    "Timestamps",
    "align_rigid",
    "match_timestamps",
    "path_length_meters",
    "read_ground_truth",
    "rig_rigidity_spread_millimeters",
    "rig_track_from_frames_meta",
    "rig_track_of",
]
"""What this module offers, its own and `colsfm.rigid_fit`'s.

The fit is re-exported rather than moved out of reach: `colsfm.alignment` is the
import path the benchmark, the audits and the tests already use, and which module
a name is defined in is not their business."""

GROUND_TRUTH_FILE_NAME: Final[str] = "ground_truth.txt"
"""TUM ground truth shipped beside `frames_meta.json`; only r2b_galileo has one."""

GROUND_TRUTH_TOLERANCE_MICROSECONDS: Final[int] = 20
"""0.02 ms, the join window NOTES.md uses to match keyframes to `ground_truth.txt`."""

# ─────────────────────────────────────────────────────────────────────────────
# Trajectories
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RigTrack:
    """A rig trajectory: one `world_T_rig` per synchronised sample, in time order."""

    timestamps_microseconds: Timestamps
    """Reference keyframe timestamp of each sample, ascending."""
    world_t_rig: Positions
    """Rig position in the world frame, metres."""
    world_R_rig: Rotations
    """Rig orientation in the world frame."""

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.timestamps_microseconds)

    def take(self, indices: Int64[ndarray, "m"]) -> RigTrack:
        """Select a subset of samples.

        Args:
            indices: Int64 sample indices with shape `[m]`.

        Returns:
            A track holding only those samples, in the given order.
        """
        return RigTrack(
            timestamps_microseconds=self.timestamps_microseconds[indices],
            world_t_rig=self.world_t_rig[indices],
            world_R_rig=self.world_R_rig[indices],
        )


def rig_track_of(poses: Sequence[tuple[int, pycolmap.Rigid3d]]) -> RigTrack:
    """Stack timestamped rig poses into a trajectory.

    The one place `(timestamp, world_T_vehicle)` becomes three parallel arrays, so
    that a track derived from the metadata and one derived from a reconstruction are
    the same object built the same way. `colsfm.trajectory` carried a second copy of
    this stacking.

    Args:
        poses: `(timestamp in microseconds, world_T_vehicle)` per sample, in the
            order the track should hold them.

    Returns:
        The trajectory; empty arrays of the right rank when nothing was given.
    """
    return RigTrack(
        timestamps_microseconds=np.asarray([timestamp for timestamp, _ in poses], dtype=np.int64),
        world_t_rig=np.asarray([pose.translation for _, pose in poses], dtype=np.float64).reshape(-1, 3),
        world_R_rig=np.asarray(
            [Rotation.from_quat(pose.rotation.quat).as_matrix() for _, pose in poses], dtype=np.float64
        ).reshape(-1, 3, 3),
    )


def rig_track_from_frames_meta(frames_meta: FramesMeta, keep_image_names: set[str] | None = None) -> RigTrack:
    """Build the rig trajectory a `frames_meta.json` implies.

    Uses `FramesMeta.rig_frames()`, so the rig pose of each sample is
    `world_T_cam * vehicle_T_cam^-1` evaluated at the sample's **lowest keyframe
    id** — the rule that reproduces cuSFM's own vehicle metadata (pose_graph_main.md §4).

    Args:
        frames_meta: A parsed collection.
        keep_image_names: When given, keep only samples with at least one member
            keyframe whose `image_name` is in this set — i.e. the samples the
            reconstruction actually registered.

    Returns:
        The trajectory, ordered by `synced_sample_id`.
    """
    by_id: dict[int, KeyframeMeta] = frames_meta.keyframe_by_id()
    frames: list[RigFrame] = []
    for rig_frame in frames_meta.rig_frames():
        if keep_image_names is not None:
            names: set[str] = {by_id[keyframe_id].image_name for keyframe_id in rig_frame.keyframe_ids}
            if names.isdisjoint(keep_image_names):
                continue
        frames.append(rig_frame)
    return rig_track_of([(frame.timestamp_microseconds, frame.world_T_vehicle) for frame in frames])


def rig_rigidity_spread_millimeters(frames_meta: FramesMeta) -> float:
    """Worst disagreement between a sample's per-camera estimates of the rig pose.

    Every camera of one `synced_sample_id` yields `world_T_cam * vehicle_T_cam^-1`;
    with `ba_frame_type=vehicle_rig` those must coincide, and the spread between
    them is what `demo_rerun.cusfm_rig_trajectory` returns as its rigidity check.
    A non-zero value means the exoego schema's single `world_T_rig` per sample is
    a lie.

    Args:
        frames_meta: A parsed collection.

    Returns:
        The largest per-sample spread across all samples, in millimetres.
    """
    by_id: dict[int, KeyframeMeta] = frames_meta.keyframe_by_id()
    worst_meters: float = 0.0
    for rig_frame in frames_meta.rig_frames():
        if len(rig_frame.keyframe_ids) < 2:
            continue
        estimates: Positions = np.asarray(
            [frames_meta.world_T_vehicle(by_id[keyframe_id]).translation for keyframe_id in rig_frame.keyframe_ids],
            dtype=np.float64,
        ).reshape(-1, 3)
        worst_meters = max(worst_meters, float(np.linalg.norm(estimates - estimates.mean(axis=0), axis=1).max()))
    return MILLIMETRES_PER_METRE * worst_meters


def read_ground_truth(input_dir: Path) -> RigTrack | None:
    """Read `ground_truth.txt` beside the input metadata, if the dataset ships one.

    The file is TUM. Its body frame is the **vehicle (FLU) frame in the same world
    frame the input `camera_to_world` uses**, so no conversion is needed on this
    side: NOTES.md gotcha 10 verified that `camera_to_world @ inv(sensor_to_vehicle_transform)`
    — exactly `FramesMeta.world_T_vehicle` — reproduces these positions to ~1.6 mm.
    Timestamps are seconds and come back as rounded microseconds.

    Args:
        input_dir: Directory holding the dataset's `frames_meta.json`.

    Returns:
        The ground-truth rig trajectory, or None when the dataset ships none.
    """
    path: Path = input_dir / GROUND_TRUTH_FILE_NAME
    if not path.is_file():
        return None
    poses: list[TumPose] = [parse_tum_line(line) for line in path.read_text().splitlines() if line.strip()]
    if not poses:
        return None
    order: Int64[ndarray, "n"] = np.argsort([pose.timestamp_microseconds for pose in poses]).astype(np.int64)
    ordered: list[TumPose] = [poses[index] for index in order]
    return RigTrack(
        timestamps_microseconds=np.asarray([pose.timestamp_microseconds for pose in ordered], dtype=np.int64),
        world_t_rig=np.asarray([pose.world_T_body.translation for pose in ordered], dtype=np.float64).reshape(-1, 3),
        world_R_rig=np.asarray(
            [Rotation.from_quat(pose.world_T_body.rotation.quat).as_matrix() for pose in ordered], dtype=np.float64
        ).reshape(-1, 3, 3),
    )


def match_timestamps(
    query_microseconds: Timestamps, reference_microseconds: Timestamps, tolerance_microseconds: int
) -> tuple[Int64[ndarray, "m"], Int64[ndarray, "m"]]:
    """Nearest-neighbour join of two timestamp streams.

    Args:
        query_microseconds: Int64 timestamps to match, shape `[n]`, ascending.
        reference_microseconds: Int64 timestamps to match against, shape `[k]`, ascending.
        tolerance_microseconds: Largest accepted gap; 0 demands an exact hit.

    Returns:
        Index arrays `(query_indices, reference_indices)` of the accepted pairs.
    """
    if len(query_microseconds) == 0 or len(reference_microseconds) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    insertion: Int64[ndarray, "n"] = np.searchsorted(reference_microseconds, query_microseconds)
    left: Int64[ndarray, "n"] = np.clip(insertion - 1, 0, len(reference_microseconds) - 1)
    right: Int64[ndarray, "n"] = np.clip(insertion, 0, len(reference_microseconds) - 1)
    left_gap: Int64[ndarray, "n"] = np.abs(reference_microseconds[left] - query_microseconds)
    right_gap: Int64[ndarray, "n"] = np.abs(reference_microseconds[right] - query_microseconds)
    nearest: Int64[ndarray, "n"] = np.where(left_gap <= right_gap, left, right)
    gap: Int64[ndarray, "n"] = np.minimum(left_gap, right_gap)
    accepted: Bool[ndarray, "n"] = gap <= tolerance_microseconds
    query_indices: Int64[ndarray, "m"] = np.flatnonzero(accepted).astype(np.int64)
    return query_indices, nearest[accepted].astype(np.int64)
