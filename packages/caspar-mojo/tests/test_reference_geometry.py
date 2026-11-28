"""The float64 camera models and pose math against central finite differences."""

from __future__ import annotations

import numpy as np
import pytest
from reference.cameras import FOCAL_AND_EXTRA_COLUMNS, project
from reference.poses import camera_point, quaternion_matrix, retract_pose
from reference.synthetic import default_intrinsics

from caspar_mojo.problem import CameraModel

STEP: float = 1e-6
POINT_STEP: float = 1e-7
"""Small against the fisheye's sqrt(1e-6) norm regularisation, which sets the curvature scale near the axis."""


def _central_difference(f, x: np.ndarray, step: float) -> np.ndarray:
    """d f / d x for row-batched f: (n, m) -> (n, k), returned as (n, k, m)."""
    columns = []
    for j in range(x.shape[1]):
        dx = np.zeros_like(x)
        dx[:, j] = step
        columns.append((f(x + dx) - f(x - dx)) / (2 * step))
    return np.stack(columns, axis=-1)


def _points(case: str, rng: np.random.Generator) -> np.ndarray:
    n = 37
    if case == "front":
        return np.column_stack([rng.uniform(-2, 2, (n, 2)), rng.uniform(1, 6, n)])
    if case == "near_axis":
        return np.column_stack([rng.uniform(-1e-4, 1e-4, (n, 2)), rng.uniform(1, 6, n)])
    if case == "wide":
        return np.column_stack([rng.uniform(-3, 3, (n, 2)), rng.uniform(0.6, 1.0, n)])
    # Behind the image plane: only the fisheye's ray form is meant for it.
    return np.column_stack([rng.uniform(-2, 2, (n, 2)), rng.uniform(-3, -0.5, n)])


CASES = [(model, case) for model in CameraModel for case in ("front", "near_axis", "wide")] + [(CameraModel.OPENCV_FISHEYE, "behind")]


def _assert_jacobian(analytic: np.ndarray, numeric: np.ndarray) -> None:
    """Agreement to 1e-6 of each column's largest entry, plus 1e-6 px per unit for the finite-difference noise."""
    column_scale = np.max(np.abs(numeric), axis=(0, 1), keepdims=True)
    excess = np.abs(analytic - numeric) - (1e-6 * column_scale + 1e-6)
    assert np.max(excess) < 0, f"worst excess {np.max(excess):.3g} at {np.unravel_index(np.argmax(excess), excess.shape)}"


@pytest.mark.parametrize(("model", "case"), CASES)
def test_projection_jacobians(model: CameraModel, case: str) -> None:
    rng = np.random.default_rng(int(model) * 10 + len(case))
    points = _points(case, rng)
    params = np.tile(default_intrinsics(model), (len(points), 1))
    projection = project(model, points, params)
    assert np.all(np.isfinite(projection.pixel))
    _assert_jacobian(projection.d_point, _central_difference(lambda p: project(model, p, params).pixel, points, POINT_STEP))

    columns = list(FOCAL_AND_EXTRA_COLUMNS[model])

    def pixel_of(focal_and_extra: np.ndarray) -> np.ndarray:
        moved = params.copy()
        moved[:, columns] = focal_and_extra
        return project(model, points, moved).pixel

    step = STEP * np.maximum(np.abs(params[0, columns]), 1.0)
    numeric = np.stack(
        [
            (pixel_of(params[:, columns] + np.eye(len(columns))[j] * step[j]) - pixel_of(params[:, columns] - np.eye(len(columns))[j] * step[j]))
            / (2 * step[j])
            for j in range(len(columns))
        ],
        axis=-1,
    )
    _assert_jacobian(projection.d_focal_and_extra, numeric)


def test_pinhole_matches_closed_form() -> None:
    params = default_intrinsics(CameraModel.PINHOLE)[None]
    pixel = project(CameraModel.PINHOLE, np.array([[1.0, -2.0, 4.0]]), params).pixel
    np.testing.assert_allclose(pixel, [[900.0 * 0.25 / (1 + 0.25e-6) + 640.0, 905.0 * -0.5 / (1 + 0.25e-6) + 480.0]], rtol=1e-15)


def test_fisheye_behind_camera_does_not_mirror() -> None:
    params = default_intrinsics(CameraModel.OPENCV_FISHEYE)[None].repeat(2, 0)
    pixel = project(CameraModel.OPENCV_FISHEYE, np.array([[1.0, 0.0, 1.0], [1.0, 0.0, -1.0]]), params).pixel
    # theta = 45 and 135 degrees: both land right of the centre, the second farther out.
    assert 640.0 < pixel[0, 0] < pixel[1, 0]


def _random_poses(rng: np.random.Generator, n: int) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return np.concatenate([q, rng.normal(size=(n, 3))], axis=1)


def test_quaternion_matrix_is_rotation() -> None:
    rotation = quaternion_matrix(_random_poses(np.random.default_rng(1), 11)[:, :4])
    np.testing.assert_allclose(rotation @ rotation.transpose(0, 2, 1), np.broadcast_to(np.eye(3), (11, 3, 3)), atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-14)


def test_retract_rotates_on_the_right() -> None:
    pose = _random_poses(np.random.default_rng(2), 1)
    tangent = np.array([[0.0, 0.0, np.pi / 2, 1.0, 2.0, 3.0]])
    moved = retract_pose(pose, tangent)
    expected = quaternion_matrix(pose[:, :4])[0] @ np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(quaternion_matrix(moved[:, :4])[0], expected, atol=1e-12)
    np.testing.assert_allclose(moved[0, 4:], pose[0, 4:] + [1.0, 2.0, 3.0])


def test_camera_point_jacobians_through_retraction() -> None:
    rng = np.random.default_rng(3)
    n = 29
    rig, sensor, point = _random_poses(rng, n), _random_poses(rng, n), rng.normal(size=(n, 3)) * 3
    analytic = camera_point(rig, sensor, point)
    zero = np.zeros((n, 6))
    numeric_rig = _central_difference(lambda t: camera_point(retract_pose(rig, t), sensor, point).point_cam, zero, STEP)
    numeric_sensor = _central_difference(lambda t: camera_point(rig, retract_pose(sensor, t), point).point_cam, zero, STEP)
    numeric_point = _central_difference(lambda x: camera_point(rig, sensor, x).point_cam, point, STEP)
    np.testing.assert_allclose(analytic.d_rig_from_world, numeric_rig, atol=1e-8)
    np.testing.assert_allclose(analytic.d_sensor_from_rig, numeric_sensor, atol=1e-8)
    np.testing.assert_allclose(analytic.d_point, numeric_point, atol=1e-8)


def test_camera_point_jacobians_away_from_zero_tangent() -> None:
    """The Jacobian at a retracted state is the Jacobian of the retracted pose, so linearising there is exact."""
    rng = np.random.default_rng(4)
    n = 7
    rig, sensor, point = _random_poses(rng, n), _random_poses(rng, n), rng.normal(size=(n, 3))
    base = rng.normal(size=(n, 6)) * 0.3
    moved = retract_pose(rig, base)
    analytic = camera_point(moved, sensor, point).d_rig_from_world
    zero = np.zeros((n, 6))
    numeric = _central_difference(lambda t: camera_point(retract_pose(moved, t), sensor, point).point_cam, zero, STEP)
    np.testing.assert_allclose(analytic, numeric, atol=1e-8)
