"""OAK overlays are projected by us through the full 14-coefficient lens model; the viewer's Pinhole has no distortion."""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from selfcap_curate.keypoints import EgoCamera, project_ego

K = np.array([[572.0, 0.0, 648.0], [0.0, 572.0, 357.0], [0.0, 0.0, 1.0]])
OAK_MONO = np.array([5.535, 1.663, 0.0, 0.0, 0.026, 5.908, 3.36, 0.226, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # d31bf1da OAK left


def camera(distortion: np.ndarray, poses: list[tuple[float, np.ndarray]]) -> EgoCamera:
    times = np.array([t for t, _ in poses])
    return EgoCamera(rig=4, cam=1, K=K, distortion=distortion, size=(1280, 720), rig_T_cam=np.eye(4), times_s=times, world_T_rig=np.stack([p for _, p in poses]))


def pose(rotvec: tuple[float, float, float], translation: tuple[float, float, float]) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = Rotation.from_rotvec(rotvec).as_matrix(), translation
    return matrix


def test_zero_distortion_is_the_pinhole() -> None:
    cam = camera(np.zeros(14), [(0.0, pose((0, 0, 0), (0, 0, 0))), (0.05, pose((0, 0, 0), (0, 0, 0)))])
    xyz = np.array([[[0.1, -0.2, 1.0], [0.0, 0.0, 2.0]]], dtype=np.float32)
    uv = project_ego(cam, np.array([0.025]), xyz)
    np.testing.assert_allclose(uv[0], [[648.0 + 57.2, 357.0 - 114.4], [648.0, 357.0]], atol=1e-3)


def test_full_lens_model_matches_opencv_at_a_pose_sample() -> None:
    world_T_rig = pose((0.1, -0.2, 0.05), (0.3, -0.1, 0.2))
    cam = camera(OAK_MONO, [(0.0, world_T_rig), (0.05, world_T_rig)])
    xyz = np.array([[[0.4, 0.1, 1.5], [0.1, 0.3, 0.9], [0.6, -0.2, 1.1]]], dtype=np.float32)
    rig_T_world = np.linalg.inv(world_T_rig)
    expected, _ = cv2.projectPoints(xyz[0].astype(np.float64), cv2.Rodrigues(rig_T_world[:3, :3])[0], rig_T_world[:3, 3], K, OAK_MONO)
    np.testing.assert_allclose(project_ego(cam, np.array([0.0]), xyz)[0], expected.reshape(-1, 2), atol=1e-2)


def test_pose_is_interpolated_between_samples() -> None:
    cam = camera(np.zeros(14), [(0.0, pose((0, 0, 0), (0, 0, 0))), (0.05, pose((0, 0.2, 0), (0.2, 0, 0)))])
    xyz = np.array([[[0.0, 0.0, 2.0]]], dtype=np.float32)
    halfway = pose((0, 0.1, 0), (0.1, 0, 0))
    q = np.linalg.inv(halfway) @ np.r_[xyz[0, 0], 1.0]
    np.testing.assert_allclose(project_ego(cam, np.array([0.025]), xyz)[0, 0], K[:2, :2] @ (q[:2] / q[2]) + K[:2, 2], atol=1e-3)


def test_points_behind_or_beyond_the_calibrated_field_are_missing() -> None:
    cam = camera(OAK_MONO, [(0.0, pose((0, 0, 0), (0, 0, 0))), (0.05, pose((0, 0, 0), (0, 0, 0)))])
    xyz = np.array([[[0.0, 0.0, -1.0], [50.0, 0.0, 1.0], [0.1, 0.1, 1.0]]], dtype=np.float32)  # behind, ~89 degrees off axis, fine
    uv = project_ego(cam, np.array([0.025]), xyz)[0]
    assert np.isnan(uv[0]).all() and np.isnan(uv[1]).all() and np.isfinite(uv[2]).all()


def test_times_outside_the_pose_track_or_across_a_gap_are_missing() -> None:
    cam = camera(np.zeros(14), [(0.0, pose((0, 0, 0), (0, 0, 0))), (0.05, pose((0, 0, 0), (0, 0, 0))), (1.0, pose((0, 0, 0), (0, 0, 0)))])
    uv = project_ego(cam, np.array([-0.5, 1.5, 0.5]), np.full((3, 1, 3), [0.0, 0.0, 1.0], dtype=np.float32))
    assert np.isnan(uv).all()
