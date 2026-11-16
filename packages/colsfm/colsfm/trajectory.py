# SPDX-License-Identifier: Apache-2.0
"""Scoring a trajectory against a reference, under both alignments at once.

`colsfm.alignment` owns the fit itself — one Umeyama solve, rigid or scale-free
— and `colsfm.benchmark` owns the run-to-run report. Between them sits the
question every audit asks: how far is this trajectory from that one, under an
SE(3) fit *and* under a SIM(3) fit, and what scale would the free fit have
chosen? The audit tools grew their own answer; this module is the answer.

The two alignments are kept distinct on purpose, and both are always reported:

- **SE(3)**, the scale held at 1.0. The metric the harness quotes. A trajectory
  whose scale is fixed by a calibrated rig must be scored this way, or a wrong
  scale hides inside the fit.
- **SIM(3)**, the scale estimated. The only fair metric for a monocular
  reconstruction, whose scale is free by construction. `would_be_scale` says
  what the free fit chose, so the SE(3) number can be read knowing whether
  metric scale survived at all.

Nothing here reads a path or names a dataset; dataset selection stays with the
tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pycolmap
from jaxtyping import Int64
from numpy import ndarray

from colsfm.alignment import (
    GROUND_TRUTH_TOLERANCE_MICROSECONDS,
    Positions,
    RigidAlignment,
    RigTrack,
    align_rigid,
    match_timestamps,
    rig_track_of,
)
from colsfm.frames_meta import CameraParams, FramesMeta, KeyframeMeta
from colsfm.geometry import MILLIMETRES_PER_METRE, Vector3

MIN_ALIGNMENT_SAMPLES: Final[int] = 3
"""Below three matched samples a three-dimensional fit is not determined."""


@dataclass(frozen=True, slots=True)
class TrajectoryScore:
    """ATE of one trajectory against a reference, both alignments reported."""

    num_matched: int
    """Samples that joined the reference."""
    rigid_rmse_millimeters: float
    """RMSE after a rigid fit with the scale held at 1.0 — the harness's metric."""
    rigid_max_millimeters: float
    """Largest residual of that rigid fit."""
    would_be_scale: float
    """Scale a similarity fit would have chosen; 1.0 means metric scale survived."""
    similarity_rmse_millimeters: float
    """RMSE after a scale-free SIM(3) fit — the only fair metric for monocular COLMAP."""

    @staticmethod
    def empty(num_matched: int) -> TrajectoryScore:
        """The score of a sample set too small to fit.

        `align_rigid` raises below one sample and is undetermined below three, so
        the guard lives here once rather than in each of this module's callers.

        Args:
            num_matched: How many samples did join, for the report.

        Returns:
            A score whose every residual is NaN.
        """
        return TrajectoryScore(
            num_matched=num_matched,
            rigid_rmse_millimeters=float("nan"),
            rigid_max_millimeters=float("nan"),
            would_be_scale=float("nan"),
            similarity_rmse_millimeters=float("nan"),
        )


def score_positions_with_similarity_fit(source_xyz: Positions, target_xyz: Positions) -> tuple[TrajectoryScore, RigidAlignment]:
    """Score matched positions and hand back the similarity fit that produced the number.

    Both fits are computed here, so a caller that goes on to use the SIM(3) fit
    is using the very fit the reported SIM(3) RMSE came from.

    Args:
        source_xyz: Float64 positions to score, shape `[n, 3]`.
        target_xyz: Float64 reference positions, shape `[n, 3]`.

    Returns:
        The score and the scale-estimating fit.

    Raises:
        ValueError: When fewer than `MIN_ALIGNMENT_SAMPLES` positions were given.
    """
    if len(source_xyz) < MIN_ALIGNMENT_SAMPLES:
        raise ValueError(f"only {len(source_xyz)} matched positions; a three-dimensional fit needs {MIN_ALIGNMENT_SAMPLES}")
    rigid: RigidAlignment = align_rigid(source_xyz, target_xyz)
    similarity: RigidAlignment = align_rigid(source_xyz, target_xyz, estimate_scale=True)
    score: TrajectoryScore = TrajectoryScore(
        num_matched=len(source_xyz),
        rigid_rmse_millimeters=MILLIMETRES_PER_METRE * rigid.rmse_meters,
        rigid_max_millimeters=MILLIMETRES_PER_METRE * rigid.max_error_meters,
        would_be_scale=rigid.would_be_scale,
        similarity_rmse_millimeters=MILLIMETRES_PER_METRE * similarity.rmse_meters,
    )
    return score, similarity


def score_positions(source_xyz: Positions, target_xyz: Positions) -> TrajectoryScore:
    """Score one set of positions against a matched reference set.

    Args:
        source_xyz: Float64 positions to score, shape `[n, 3]`.
        target_xyz: Float64 reference positions, shape `[n, 3]`.

    Returns:
        Both alignments' residual statistics, in millimetres; an empty score when
        too few samples matched to determine a fit.
    """
    if len(source_xyz) < MIN_ALIGNMENT_SAMPLES:
        return TrajectoryScore.empty(len(source_xyz))
    return score_positions_with_similarity_fit(source_xyz, target_xyz)[0]


def score_track_against_ground_truth(track: RigTrack, ground_truth: RigTrack) -> TrajectoryScore:
    """Join a rig trajectory to ground truth on timestamp and score it.

    Uses exactly the join `colsfm.benchmark.compute_run_metrics` uses for its
    `ATE vs ground truth` row: nearest timestamp within 20 microseconds.

    Args:
        track: The trajectory to score.
        ground_truth: The reference trajectory.

    Returns:
        The score.
    """
    track_indices, reference_indices = match_timestamps(
        track.timestamps_microseconds, ground_truth.timestamps_microseconds, GROUND_TRUTH_TOLERANCE_MICROSECONDS
    )
    return score_positions(track.world_t_rig[track_indices], ground_truth.world_t_rig[reference_indices])


def rig_track_from_reconstruction(
    reconstruction: pycolmap.Reconstruction, frames_meta: FramesMeta, alignment: RigidAlignment | None = None
) -> RigTrack:
    """Derive a rig trajectory from a plain (rig-free) COLMAP reconstruction.

    Follows `FramesMeta.rig_frames`' rule — the rig pose of a sample comes from its
    lowest keyframe id — restricted to the keyframes the reconstruction actually
    registered, since a monocular COLMAP model registers an arbitrary subset.

    Args:
        reconstruction: A model whose image names are the dataset's `image_name`s.
        frames_meta: The input collection supplying `synced_sample_id`, timestamps
            and the calibrated `vehicle_T_cam`.
        alignment: When given, applied to every camera pose first, so the rig track
            comes out in the ground-truth frame at metric scale.

    Returns:
        The rig trajectory over the samples with at least one registered keyframe.
    """
    world_T_cam_by_name: dict[str, pycolmap.Rigid3d] = {}
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        pose: pycolmap.Rigid3d = image.cam_from_world().inverse()
        world_T_cam_by_name[image.name] = alignment.apply_pose(pose) if alignment is not None else pose

    members: dict[int, list[KeyframeMeta]] = {}
    for keyframe in frames_meta.keyframes:
        if keyframe.image_name in world_T_cam_by_name:
            members.setdefault(keyframe.synced_sample_id, []).append(keyframe)

    poses: list[tuple[int, pycolmap.Rigid3d]] = []
    for synced_sample_id in sorted(members):
        reference: KeyframeMeta = min(members[synced_sample_id], key=lambda item: item.keyframe_id)
        camera: CameraParams = frames_meta.cameras[reference.camera_params_id]
        poses.append(
            (
                reference.timestamp_microseconds,
                world_T_cam_by_name[reference.image_name] * camera.vehicle_T_cam.inverse(),
            )
        )
    # The stacking is `colsfm.alignment`'s: a track built from a reconstruction and one
    # built from the metadata have to be the same three arrays, built the same way.
    return rig_track_of(poses)


def image_center_score(
    reconstruction: pycolmap.Reconstruction, gt_center_by_image_name: dict[str, Vector3]
) -> tuple[TrajectoryScore, RigidAlignment]:
    """Score a reconstruction's camera centres against per-image ground truth.

    The only reference a monocular reconstruction can be scored against without
    first inventing a rig. The similarity fit is computed once and both returned
    and used for the reported RMSE, so the caller's alignment is the very fit the
    number came from.

    Args:
        reconstruction: A model whose image names are the dataset's `image_name`s.
        gt_center_by_image_name: Ground-truth camera centre per image name, metres.

    Returns:
        The score and the scale-estimating fit it used, so callers can reuse the fit.

    Raises:
        ValueError: When fewer than three registered images have ground truth.
    """
    # One pass over the images: `find_image_with_name` is a linear scan, so calling it
    # per name inside a comprehension is quadratic in the model size.
    center_by_name: dict[str, Vector3] = {
        image.name: np.asarray(image.cam_from_world().inverse().translation, dtype=np.float64)
        for image in reconstruction.images.values()
        if image.has_pose and image.name in gt_center_by_image_name
    }
    names: list[str] = sorted(center_by_name)
    if len(names) < MIN_ALIGNMENT_SAMPLES:
        raise ValueError(f"only {len(names)} registered images carry ground truth")
    source_xyz: Positions = np.asarray([center_by_name[name] for name in names], dtype=np.float64).reshape(-1, 3)
    target_xyz: Positions = np.asarray([gt_center_by_image_name[name] for name in names], dtype=np.float64).reshape(-1, 3)
    return score_positions_with_similarity_fit(source_xyz, target_xyz)


def indices_of_timestamps(track: RigTrack, keep_microseconds: set[int] | None) -> Int64[ndarray, "m"]:
    """Select a track's samples by timestamp, so several tracks share one sample set.

    Args:
        track: The trajectory to select from.
        keep_microseconds: Timestamps to keep, or None to keep every sample.

    Returns:
        Indices into `track`, in track order.
    """
    if keep_microseconds is None:
        return np.arange(len(track), dtype=np.int64)
    return np.asarray(
        [index for index, stamp in enumerate(track.timestamps_microseconds) if int(stamp) in keep_microseconds],
        dtype=np.int64,
    )
