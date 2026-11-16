"""Helpers shared by the loop-closure and loop-pose test scenes.

Both scenes build a synthetic rig, project known 3-D points into pinhole cameras and hand
the resulting correspondences to a `match_fn`. That scaffolding was written twice, once per
file, so it lives here instead; what stays in each test module is the part that differs —
the rig shape, the trajectory and what the test asserts about them.

What is shared with *every* synthetic-scene test — the projection and its in-image mask —
stays in `conftest.py`; this module only adds what a loop scene needs on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pycolmap
from conftest import project_and_mask as conftest_project_and_mask
from jaxtyping import Bool, Float64, Int
from numpy import ndarray

from colsfm.loop_pose import Matches, MatchFunction


@dataclass(frozen=True, slots=True)
class ProjectedPoints:
    """Where a set of world points landed in one camera."""

    pixels: Float64[ndarray, "n_points 2"]
    """Pixel coordinates, valid only where `visible`."""
    visible: Bool[ndarray, "n_points"]
    """Which points are in front of the camera, in range and on the sensor."""
    depth_m: Float64[ndarray, "n_points"]
    """Depth along the optical axis, in metres; negative behind the camera."""


def project_and_mask(
    cam_T_world: pycolmap.Rigid3d,
    camera: pycolmap.Camera,
    points_xyz: Float64[ndarray, "n_points 3"],
    min_depth_m: float,
    max_depth_m: float,
) -> ProjectedPoints:
    """Project world points into one camera, keeping only those in a usable depth band.

    Wraps `conftest.project_and_mask`, which every synthetic-scene test shares, and adds the
    two things a loop scene needs on top of "is it inside the image": a depth band, because a
    point one metre behind the rig and a point a hundred metres away are both useless to
    triangulate from, and the depths themselves, because a resection is constrained along the
    optical axis by its *near* points and the scenes pick those first.

    Args:
        cam_T_world: Pose of the world in the camera frame.
        camera: The calibrated camera, whose model does the projection.
        points_xyz: Float64 world points with shape `[n_points, 3]`.
        min_depth_m: Nearest depth a point may have and still count as visible.
        max_depth_m: Farthest depth a point may have and still count as visible.

    Returns:
        The pixels, the visibility mask and the depths.
    """
    pixels, in_image = conftest_project_and_mask(camera, cam_T_world, points_xyz)
    rotation: Float64[ndarray, "3 3"] = np.asarray(cam_T_world.rotation.matrix(), dtype=np.float64)
    depth_m: Float64[ndarray, "n_points"] = points_xyz @ rotation[2] + float(cam_T_world.translation[2])
    visible: Bool[ndarray, "n_points"] = in_image & (depth_m > min_depth_m) & (depth_m < max_depth_m)
    return ProjectedPoints(pixels=pixels, visible=visible, depth_m=depth_m)


def matcher_from_point_indices(
    point_index_by_image: dict[int, Int[ndarray, "n_keypoints"]], outlier_ratio: float, seed: int
) -> MatchFunction:
    """Build a `match_fn` from known correspondences, with gross outliers added.

    Two keypoints match when they observe the same 3-D point. The outliers are what stops a
    test passing on a RANSAC that never rejects anything.

    The outlier draw is seeded from the image pair, not from a running generator, so a pair
    always gets the same wrong matches however many other pairs were asked for first. A
    stateful generator here would make every test in the file depend on the estimator's call
    order, and a refactor that merely stopped asking for the same pair twice would silently
    change the scene.

    Args:
        point_index_by_image: Which 3-D point each keypoint of each image observes.
        outlier_ratio: Wrong matches added, as a fraction of the true ones.
        seed: Seed for the outlier draw.

    Returns:
        A callable `match_fn(image_id_a, image_id_b) -> Int[ndarray, "n_matches 2"]`.
    """

    def match_fn(image_id_a: int, image_id_b: int) -> Matches:
        """Return true correspondences plus a fixed fraction of random wrong pairs."""
        generator: np.random.Generator = np.random.default_rng((seed, image_id_a, image_id_b))
        points_a: Int[ndarray, "n_a"] = point_index_by_image[image_id_a]
        points_b: Int[ndarray, "n_b"] = point_index_by_image[image_id_b]
        position_in_b: dict[int, int] = {int(point): position for position, point in enumerate(points_b)}
        true_matches: list[tuple[int, int]] = [
            (position, position_in_b[int(point)]) for position, point in enumerate(points_a) if int(point) in position_in_b
        ]
        outliers: list[tuple[int, int]] = [
            (int(generator.integers(len(points_a))), int(generator.integers(len(points_b)))) for _ in range(round(outlier_ratio * len(true_matches)))
        ]
        return np.array(true_matches + outliers, dtype=np.int64).reshape(-1, 2)

    return match_fn
