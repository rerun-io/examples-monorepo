"""The rational FULL_OPENCV pole guard: where the pole is, and what the extrinsic gather leaves out there.

The guard covers that one singularity. It is not a validity test of the camera model, and the last test pins one
fold it deliberately does not cover.
"""

from __future__ import annotations

import numpy as np
import pyceres
import pycolmap
import pytest

from colsfm.ceres_pose import pose_parameter_blocks
from colsfm.extrinsic_observations import CameraObservations
from colsfm.extrinsic_solve import _add_reprojection_blocks
from colsfm.point_filters import before_rational_pole, rational_pole_angle

MGO07_CAM2_PARAMS: list[float] = [271.73053, 271.61353, 231.32161, 314.52106, 0.16062, -0.46199, -6e-05, -0.00012, 0.02577, 0.4509, -0.58754, 0.00296]
"""MSD MGO07's cam 2 (FULL_OPENCV, 480x640): its denominator nearly cancels the numerator, leaving a narrow pole inside the image."""

POLE_RADIUS: float = 1.32422
"""tan(angle) of that pole, where the projection jumps to -1.9e10 px."""


def _cam2() -> pycolmap.Camera:
    return pycolmap.Camera(model="FULL_OPENCV", width=480, height=640, params=MGO07_CAM2_PARAMS)


def test_the_guard_stops_at_the_first_root_of_the_rational_denominator() -> None:
    """The analytic root finds a spike 2048 samples step over. No pole, no limit: a pinhole, a pole-free FULL_OPENCV."""
    assert np.tan(rational_pole_angle(_cam2())) == pytest.approx(POLE_RADIUS, abs=1e-4)
    pinhole = pycolmap.Camera(model="PINHOLE", width=640, height=480, params=[500.0, 500.0, 320.0, 240.0])
    assert rational_pole_angle(pinhole) == pytest.approx(0.5 * np.pi)
    pole_free = pycolmap.Camera(model="FULL_OPENCV", width=480, height=640, params=[*MGO07_CAM2_PARAMS[:9], 0.1, 0.0, 0.0])
    assert rational_pole_angle(pole_free) == pytest.approx(0.5 * np.pi)


def test_points_behind_the_camera_or_past_the_pole_are_left_out() -> None:
    angle = rational_pole_angle(_cam2())
    points = np.array([[0.0, 0.5, 1.0], [0.0, POLE_RADIUS - 1e-3, 1.0], [0.0, POLE_RADIUS + 1e-3, 1.0], [0.0, 0.1, -1.0]])
    assert before_rational_pole(points, angle).tolist() == [True, True, False, False]


def test_the_extrinsic_gather_leaves_out_what_it_cannot_evaluate() -> None:
    """A point past the pole is left out by the pole guard; a NaN pixel by the finite-evaluation backstop.

    Either one used to fail the extrinsics solve on its first evaluation.
    """
    camera = _cam2()
    rng = np.random.default_rng(0)
    inside = np.column_stack([rng.uniform(-0.5, 0.5, 20), rng.uniform(-0.5, 0.5, 20), np.full(20, 2.0)])
    points = np.vstack([inside, [[0.0, (POLE_RADIUS + 0.01) * 2.0, 2.0]]])
    pixels = np.asarray(camera.img_from_cam(points[:20]), dtype=np.float64)
    pixels = np.vstack([pixels, [[231.0, 600.0]]])
    pixels[3] = np.nan
    observations = {1: CameraObservations(camera_params_id=1, camera=camera, points_in_vehicle=points, observed_px=pixels)}
    poses = {0: pose_parameter_blocks(pycolmap.Rigid3d()), 1: pose_parameter_blocks(pycolmap.Rigid3d())}
    costs: list[pyceres.CostFunction] = []
    blocks, residuals, outside, non_finite = _add_reprojection_blocks(pyceres.Problem(), poses, observations, 0, 1.0, costs)
    assert (blocks, residuals, outside, non_finite) == (1, 2 * 19, 1, 1)


def test_a_tangential_fold_between_the_image_axes_is_out_of_the_guards_scope() -> None:
    """The review's counterexample: OPENCV with p1 = p2 = 0.1 and no radial terms folds along the negative diagonal.

    Its image radius there is `r - 3 sqrt(2) 0.1 r^2`, whose derivative vanishes at r = 1.17851, so a ray at r = 1.5
    projects back inside the fold. There is no rational pole, so the guard leaves the camera unlimited: folds are out of
    its scope (`rational_pole_angle`), and the extrinsic backstop only catches non-finite results.
    """
    camera = pycolmap.Camera(model="OPENCV", width=640, height=640, params=[300.0, 300.0, 320.0, 320.0, 0.0, 0.0, 0.1, 0.1])
    radii = np.array([1.0, 1.17851, 1.5])
    rays = np.column_stack([-radii / np.sqrt(2.0), -radii / np.sqrt(2.0), np.ones(3)])
    image_radius = np.hypot(*(np.asarray(camera.img_from_cam(rays), dtype=np.float64) - 320.0).T) / 300.0
    assert image_radius == pytest.approx(radii - 3.0 * np.sqrt(2.0) * 0.1 * radii**2)
    assert image_radius[2] < image_radius[1]
    assert rational_pole_angle(camera) == pytest.approx(0.5 * np.pi)
    assert before_rational_pole(rays, rational_pole_angle(camera)).all()
