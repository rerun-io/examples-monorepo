"""A camera model's valid domain: the rational FULL_OPENCV pole, and the observations every solve leaves out there."""

from __future__ import annotations

import numpy as np
import pyceres
import pycolmap
import pytest

from colsfm.ceres_pose import pose_parameter_blocks
from colsfm.extrinsic_observations import CameraObservations
from colsfm.extrinsic_solve import _add_reprojection_blocks
from colsfm.point_filters import camera_domain_angle, in_camera_domain

MGO07_CAM2_PARAMS: list[float] = [271.73053, 271.61353, 231.32161, 314.52106, 0.16062, -0.46199, -6e-05, -0.00012, 0.02577, 0.4509, -0.58754, 0.00296]
"""MSD MGO07's cam 2 (FULL_OPENCV, 480x640): its denominator nearly cancels the numerator, leaving a narrow pole inside the image."""

POLE_RADIUS: float = 1.32422
"""tan(angle) of that pole, where the projection jumps to -1.9e10 px."""


def _cam2() -> pycolmap.Camera:
    return pycolmap.Camera(model="FULL_OPENCV", width=480, height=640, params=MGO07_CAM2_PARAMS)


def test_the_domain_ends_at_the_first_pole_of_the_rational_model() -> None:
    """2048 samples step over the spike; the analytic root does not. A pinhole keeps the whole front half-space."""
    assert np.tan(camera_domain_angle(_cam2())) == pytest.approx(POLE_RADIUS, abs=1e-4)
    pinhole = pycolmap.Camera(model="PINHOLE", width=640, height=480, params=[500.0, 500.0, 320.0, 240.0])
    assert camera_domain_angle(pinhole) == pytest.approx(0.5 * np.pi)


def test_points_behind_the_camera_or_past_the_pole_are_outside() -> None:
    angle = camera_domain_angle(_cam2())
    points = np.array([[0.0, 0.5, 1.0], [0.0, POLE_RADIUS - 1e-3, 1.0], [0.0, POLE_RADIUS + 1e-3, 1.0], [0.0, 0.1, -1.0]])
    assert in_camera_domain(points, angle).tolist() == [True, True, False, False]


def test_the_extrinsic_gather_leaves_out_what_it_cannot_evaluate() -> None:
    """A point past the pole is outside the domain; a NaN pixel is caught by the finite-evaluation backstop.

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
