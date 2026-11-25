"""Which of the loop stage's two-view inliers the mapper may triangulate from.

The loop stage matches every image pair its shortlisted rig pairs need, and the matcher verifies each pair with a free
epipolar model. Under `loop_pairs_in_mapper="shortlisted"` all of those inliers reach the mapper, including the inliers
of rig pairs whose measurement failed and aliased matches (a ceiling 180 deg off) that a free model explains. Every other
system in the survey (cuSFM, ORB-SLAM3, RootSLAM, GLOMAP's rotation filter) lets only pose-consistent loop
correspondences into the map. `pose_checked_inliers` is that check for colsfm, one rule for every dataset:

* an observation pair of a rig pair with a verified candidate (one that passed the prior gate where the profile sets
  one) keeps the inliers within `MEASURED_TOLERANCE_PX` of the candidate's measured `source_T_target`, whether or not
  the candidate became an accepted edge (the kitti profile's edge gates are 0.0 and accept none);
* an observation pair of any other shortlisted rig pair keeps the inliers within `PRIOR_TOLERANCE_PX` of the PRIOR
  relative pose, a tolerance wide enough for the drift a loop exists to correct (KITTI 06 reaches its revisits only
  through these pairs: removing them all cost 42 % ATE);
* local-map pairs are never touched.

The error is the Sampson error of each match through the rig extrinsics, on unit bearings, in pixels of the mean focal
length. Offline on the 2026-11-24 DGX Spark base runs against ground truth, this kept 99.5-100 % of the right loop
inliers on EuRoC V1_01, MSD MGO05 and KITTI 06; the pair-level and the "verified only" variants lost 32 % of KITTI's.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final, TypeAlias

import numpy as np
import pycolmap
from jaxtyping import Bool, Float64, UInt32
from numpy import ndarray

from colsfm.database import ImagePair
from colsfm.loop_closure import LoopCandidate, LoopSearchPlan
from colsfm.loop_pairs import observation_pairs
from colsfm.loop_pose import bearings_of
from colsfm.pairs import normalise_pair

MEASURED_TOLERANCE_PX: Final[float] = 8.0
"""Largest Sampson error a match may have under its rig pair's measured pose: twice the matcher's 4 px verification
threshold, because the measurement itself carries centimetres of error (4 px lost 1.3 % of MSD's right inliers)."""

PRIOR_TOLERANCE_PX: Final[float] = 32.0
"""Largest Sampson error under the prior relative pose, for rig pairs without a verified candidate: wide enough for the
prior's drift across a revisit (16 px lost 1 % of KITTI 06's right inliers), narrow against aliased matches."""

PURE_ROTATION_BASELINE_M: Final[float] = 1e-6
"""Below this baseline the epipolar constraint is undefined and the error is the rotation's transfer angle."""

InlierReader: TypeAlias = Callable[[int, int], UInt32[ndarray, "n_inliers 2"]]
"""`read(image_id_a, image_id_b)`: a pair's stored inlier matches, columns in `(min id, max id)` order."""


def sampson_error_px(
    b_from_a: pycolmap.Rigid3d, bearings_a: Float64[ndarray, "n_matches 3"], bearings_b: Float64[ndarray, "n_matches 3"], focal_px: float
) -> Float64[ndarray, "n_matches"]:
    """Sampson (first-order geometric) epipolar error of matched bearings under a relative pose, in pixels.

    Computed on the sphere, with the gradient projected onto each bearing's tangent plane, so it holds for fisheye and
    wide cube faces as for pinholes; pixels are the angle times `focal_px`.

    Args:
        b_from_a: Pose of camera a in camera b's frame.
        bearings_a: Float64 bearings in camera a with shape `[n_matches, 3]`; any length.
        bearings_b: Float64 bearings in camera b with shape `[n_matches, 3]`; any length.
        focal_px: Focal length the angle is converted with.

    Returns:
        Float64 errors with shape `[n_matches]`.
    """
    unit_a: Float64[ndarray, "n_matches 3"] = bearings_a / np.linalg.norm(bearings_a, axis=1, keepdims=True)
    unit_b: Float64[ndarray, "n_matches 3"] = bearings_b / np.linalg.norm(bearings_b, axis=1, keepdims=True)
    rotation: Float64[ndarray, "3 3"] = np.asarray(b_from_a.rotation.matrix(), dtype=np.float64)
    translation: Float64[ndarray, "3"] = np.asarray(b_from_a.translation, dtype=np.float64)
    baseline_m: float = float(np.linalg.norm(translation))
    if baseline_m < PURE_ROTATION_BASELINE_M:
        cosine: Float64[ndarray, "n_matches"] = np.clip(np.sum((unit_a @ rotation.T) * unit_b, axis=1), -1.0, 1.0)
        return np.arccos(cosine) * focal_px
    direction: Float64[ndarray, "3"] = translation / baseline_m
    skew: Float64[ndarray, "3 3"] = np.array(
        [[0.0, -direction[2], direction[1]], [direction[2], 0.0, -direction[0]], [-direction[1], direction[0], 0.0]], dtype=np.float64
    )
    essential: Float64[ndarray, "3 3"] = skew @ rotation
    line_b: Float64[ndarray, "n_matches 3"] = unit_a @ essential.T
    line_a: Float64[ndarray, "n_matches 3"] = unit_b @ essential
    residual: Float64[ndarray, "n_matches"] = np.sum(unit_b * line_b, axis=1)
    gradient_a: Float64[ndarray, "n_matches 3"] = line_a - unit_a * np.sum(unit_a * line_a, axis=1, keepdims=True)
    gradient_b: Float64[ndarray, "n_matches 3"] = line_b - unit_b * np.sum(unit_b * line_b, axis=1, keepdims=True)
    norm: Float64[ndarray, "n_matches"] = np.sqrt(np.sum(gradient_a**2, axis=1) + np.sum(gradient_b**2, axis=1))
    return np.abs(residual) / np.maximum(norm, np.finfo(np.float64).tiny) * focal_px


def pose_checked_inliers(
    plan: LoopSearchPlan, candidates: Sequence[LoopCandidate], pairs: Sequence[ImagePair], read_inliers: InlierReader
) -> dict[ImagePair, UInt32[ndarray, "n_inliers 2"]]:
    """The inliers the mapper should use instead of the stored ones, for the observation pairs that fail the check.

    Args:
        plan: The plan the pairs were matched for.
        candidates: The verified candidates that passed the prior gate (`LoopClosureResult.candidates`).
        pairs: Image pairs to check, normalised; pairs outside every shortlisted rig pair's observation walk (local-map
            pairs, pairs of other stages) are left alone.
        read_inliers: The stored inlier matches of a pair.

    Returns:
        Per pair `(min id, max id)` that loses inliers, the kept rows (possibly none), columns in that id order. A pair
        that passes whole is absent.
    """
    observed_pairs: set[ImagePair] = {
        normalise_pair(observation.anchor, observation.observed)
        for rig_pair in plan.rig_pairs
        for observation in observation_pairs(plan.rig_index, plan.geometry, plan.keypoints, rig_pair)
    }
    # `low_T_high` per unordered rig pair: `source_T_target` read with the lower rig id as source.
    measured: dict[tuple[int, int], pycolmap.Rigid3d] = {
        normalise_pair(candidate.hit.source_rig_id, candidate.hit.target_rig_id): (
            candidate.estimate.source_T_target
            if candidate.hit.source_rig_id < candidate.hit.target_rig_id
            else candidate.estimate.source_T_target.inverse()
        )
        for candidate in candidates
    }
    rig_camera_of: dict[int, tuple[int, int]] = {image_id: key for key, image_id in plan.rig_index.keyframe_by_rig_camera.items()}

    overrides: dict[ImagePair, UInt32[ndarray, "n_inliers 2"]] = {}
    for pair in pairs:
        if pair not in observed_pairs:
            continue
        rows: UInt32[ndarray, "n_inliers 2"] = read_inliers(*pair)
        if len(rows) == 0:
            continue
        rig_a, camera_a = rig_camera_of[pair[0]]
        rig_b, camera_b = rig_camera_of[pair[1]]
        low_T_high: pycolmap.Rigid3d | None = measured.get(normalise_pair(rig_a, rig_b))
        tolerance_px: float = MEASURED_TOLERANCE_PX
        if low_T_high is None:
            low_T_high = plan.rig_index.world_T_rig[min(rig_a, rig_b)].inverse() * plan.rig_index.world_T_rig[max(rig_a, rig_b)]
            tolerance_px = PRIOR_TOLERANCE_PX
        rig_b_from_rig_a: pycolmap.Rigid3d = low_T_high.inverse() if rig_a < rig_b else low_T_high
        position_a: int = plan.geometry.position_of(camera_a)
        position_b: int = plan.geometry.position_of(camera_b)
        b_from_a: pycolmap.Rigid3d = plan.geometry.cams_from_rig[position_b] * rig_b_from_rig_a * plan.geometry.cams_from_rig[position_a].inverse()
        camera_of_a: pycolmap.Camera = plan.geometry.cameras[position_a]
        camera_of_b: pycolmap.Camera = plan.geometry.cameras[position_b]
        bearings_a: Float64[ndarray, "n_inliers 3"] = bearings_of(camera_of_a, np.asarray(plan.keypoints[pair[0]])[rows[:, 0].astype(np.int64)])
        bearings_b: Float64[ndarray, "n_inliers 3"] = bearings_of(camera_of_b, np.asarray(plan.keypoints[pair[1]])[rows[:, 1].astype(np.int64)])
        focal_px: float = 0.5 * float(camera_of_a.mean_focal_length() + camera_of_b.mean_focal_length())
        keep: Bool[ndarray, "n_inliers"] = sampson_error_px(b_from_a, bearings_a, bearings_b, focal_px) <= tolerance_px
        if not keep.all():
            overrides[pair] = np.ascontiguousarray(rows[keep], dtype=np.uint32)
    return overrides
