"""How far apart two rig frames of this scene can be and still see common structure: the covisibility radius.

The prior bound (`colsfm.loop_closure.prior_distance_bound`, `T + P` on the cube profile) is exact for loop EDGES, but
the loop stage's matches also feed the mapper, and there the reach is the scene's, not the edge gate's. Measured on the
2026-11-24 DGX Spark base runs (`loop-and-quality/a5/why-and-rule.md`):

* TartanAir V2 WesternDesertTown P004 (cube faces, outdoor): the 4.5 m bound removed 215 k of the 313 k loop inliers
  that are right against ground truth, 91 % of them 4.5-15 m apart in the prior, and the ATE rose from 14.5 to 16-17 mm.
* Casa Loma 007 (cube faces, indoor): the loop pairs beyond 4.5 m are 92.6 % empty.

The difference is the scene depth. A 90 deg face shares points with a face d metres away when the points lie about d
metres or further. The 90th percentile of the stage-4 pairs' median depth (triangulated with the prior poses) is
26.8 m on WesternDesertTown and 5.2 m on Casa Loma 007; as the matching radius it keeps 99.6 % of WesternDesertTown's
far right inliers and still drops 99.2 % of Casa Loma's empty far pairs. It is learned in each run from pairs that
exist before the loop stage, so it is one default for every dataset, not a profile key.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final

import numpy as np
import pycolmap
from jaxtyping import Bool, Float64, UInt32
from numpy import ndarray

from colsfm.database import ImagePair
from colsfm.loop_pose import Keypoints, bearings_of
from colsfm.rig_geometry import RigFrameIndex, RigGeometry

MIN_BASELINE_M: Final[float] = 0.05
"""Shortest prior baseline between the two cameras a pair needs before its depths count: below it the triangulation of
anything beyond a few metres is noise."""


def triangulated_depths_m(
    b_from_a: pycolmap.Rigid3d, bearings_a: Float64[ndarray, "n_matches 3"], bearings_b: Float64[ndarray, "n_matches 3"]
) -> Float64[ndarray, "n_matches"]:
    """Depth along each bearing of camera a of the midpoint triangulation of matched bearings.

    Solves `s_a R f_a + t = s_b f_b` in least squares per match.

    Args:
        b_from_a: Pose of camera a in camera b's frame.
        bearings_a: Unit bearings in camera a.
        bearings_b: Unit bearings in camera b, matched row by row.

    Returns:
        `s_a` in metres (the unit of `b_from_a`), NaN where the point lies behind either camera.
    """
    rotated_a: Float64[ndarray, "n_matches 3"] = bearings_a @ np.asarray(b_from_a.rotation.matrix(), dtype=np.float64).T
    translation: Float64[ndarray, "3"] = np.asarray(b_from_a.translation, dtype=np.float64)
    system: Float64[ndarray, "n_matches 3 2"] = np.stack([rotated_a, -bearings_b], axis=2)
    normal: Float64[ndarray, "n_matches 2 2"] = np.einsum("nki,nkj->nij", system, system) + 1e-12 * np.eye(2)
    right_side: Float64[ndarray, "n_matches 2"] = np.einsum("nki,k->ni", system, -translation)
    scales: Float64[ndarray, "n_matches 2"] = np.linalg.solve(normal, right_side[..., None])[..., 0]
    in_front: Bool[ndarray, "n_matches"] = (scales[:, 0] > 0.0) & (scales[:, 1] > 0.0)
    return np.where(in_front, scales[:, 0], np.nan)


def spans_two_rig_frames(rig_index: RigFrameIndex) -> Callable[[ImagePair], bool]:
    """Whether an image pair joins two different rig frames, the only pairs `covisibility_radius_m` learns from.

    Handed to `colsfm.database.read_inlier_sample`, so its stride runs over these pairs only.

    Args:
        rig_index: Rig frames with their keyframes.

    Returns:
        The test for one pair `(min id, max id)`; False for an image outside the index.
    """
    rig_camera_of: dict[int, tuple[int, int]] = rig_index.rig_camera_by_keyframe()

    def spans(pair: ImagePair) -> bool:
        rig_camera_a: tuple[int, int] | None = rig_camera_of.get(pair[0])
        rig_camera_b: tuple[int, int] | None = rig_camera_of.get(pair[1])
        return rig_camera_a is not None and rig_camera_b is not None and rig_camera_a[0] != rig_camera_b[0]

    return spans


def covisibility_radius_m(
    inliers_by_pair: Mapping[ImagePair, UInt32[ndarray, "n_inliers 2"]],
    rig_index: RigFrameIndex,
    geometry: RigGeometry,
    keypoints: Mapping[int, Keypoints],
    quantile: float,
) -> float:
    """The `quantile` of the per-pair median scene depth over pairs of two different rig frames.

    Each pair's inliers are triangulated with the PRIOR relative pose through the rig extrinsics; pairs within one rig
    frame (stereo), pairs whose cameras are less than `MIN_BASELINE_M` apart in the prior and pairs with no point in
    front of both cameras are skipped.

    Args:
        inliers_by_pair: Verified inliers per image pair `(min id, max id)`, columns in that order (stage 4's pairs).
        rig_index: Rig frames with their prior poses.
        geometry: The rig's cameras and extrinsics.
        keypoints: Keypoint pixels per image id.
        quantile: Which quantile of the per-pair medians, in [0, 1].

    Returns:
        The radius in metres; 0.0 when no pair qualifies.
    """
    rig_camera_of: dict[int, tuple[int, int]] = rig_index.rig_camera_by_keyframe()
    medians_m: list[float] = []
    for (image_a, image_b), rows in inliers_by_pair.items():
        if len(rows) == 0 or image_a not in rig_camera_of or image_b not in rig_camera_of or image_a not in keypoints or image_b not in keypoints:
            continue
        rig_a, camera_a = rig_camera_of[image_a]
        rig_b, camera_b = rig_camera_of[image_b]
        if rig_a == rig_b:
            continue
        position_a: int = geometry.position_of(camera_a)
        position_b: int = geometry.position_of(camera_b)
        b_from_a: pycolmap.Rigid3d = (
            geometry.cams_from_rig[position_b]
            * rig_index.world_T_rig[rig_b].inverse()
            * rig_index.world_T_rig[rig_a]
            * geometry.cams_from_rig[position_a].inverse()
        )
        if float(np.linalg.norm(b_from_a.translation)) < MIN_BASELINE_M:
            continue
        bearings_a: Float64[ndarray, "n_inliers 3"] = bearings_of(
            geometry.cameras[position_a], np.asarray(keypoints[image_a])[rows[:, 0].astype(np.int64)]
        )
        bearings_b: Float64[ndarray, "n_inliers 3"] = bearings_of(
            geometry.cameras[position_b], np.asarray(keypoints[image_b])[rows[:, 1].astype(np.int64)]
        )
        depths_m: Float64[ndarray, "n_inliers"] = triangulated_depths_m(b_from_a, bearings_a, bearings_b)
        if np.isfinite(depths_m).any():
            medians_m.append(float(np.nanmedian(depths_m)))
    if not medians_m:
        return 0.0
    return float(np.quantile(np.asarray(medians_m, dtype=np.float64), quantile))
