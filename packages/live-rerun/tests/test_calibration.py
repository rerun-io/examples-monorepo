"""Device-free tests for the OAK calibration -> simplecv pinhole conversion.

These guard the unit-prone bits: cm->m on extrinsic translation, the K matrix
flowing through at the encoded resolution, reference-camera identity pose, and
distortion handling. No OAK device required.
"""

from __future__ import annotations

import numpy as np

from live_rerun.calibration import OakCameraCalib, oak_calibration_to_pinholes


def _k(fx: float = 800.0, fy: float = 800.0, cx: float = 640.0, cy: float = 360.0) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)


def test_reference_camera_has_identity_pose() -> None:
    rgb = OakCameraCalib(label="rgb", width=1920, height=1080, k_matrix=_k(), distortion=[], ref_T_cam_cm=None)
    pinholes = oak_calibration_to_pinholes([rgb])

    pose = pinholes["rgb"].extrinsics
    assert pose.cam_t_world is not None
    np.testing.assert_allclose(pose.world_T_cam, np.eye(4))
    np.testing.assert_allclose(pose.cam_t_world, np.zeros(3))
    assert pinholes["rgb"].name == "oak_rgb"


def test_extrinsic_translation_converted_cm_to_m() -> None:
    ref_T_cam_cm = np.eye(4)
    ref_T_cam_cm[:3, 3] = [7.5, -2.0, 0.0]  # centimetres, as DepthAI reports
    left = OakCameraCalib(label="left", width=1280, height=800, k_matrix=_k(), distortion=[], ref_T_cam_cm=ref_T_cam_cm)

    pose = oak_calibration_to_pinholes([left])["left"].extrinsics
    assert pose.world_t_cam is not None
    np.testing.assert_allclose(pose.world_t_cam, [0.075, -0.02, 0.0])


def test_intrinsics_use_encoded_resolution_and_k() -> None:
    k = _k(fx=1234.0, fy=1240.0, cx=960.0, cy=540.0)
    rgb = OakCameraCalib(label="rgb", width=1920, height=1080, k_matrix=k, distortion=[], ref_T_cam_cm=None)
    intr = oak_calibration_to_pinholes([rgb])["rgb"].intrinsics

    assert (intr.width, intr.height) == (1920, 1080)
    assert intr.fl_x == 1234.0 and intr.fl_y == 1240.0
    assert intr.k_matrix is not None
    np.testing.assert_allclose(intr.k_matrix, k)


def test_distortion_built_from_coeffs_and_skipped_when_too_few() -> None:
    coeffs = [0.1, -0.2, 0.001, -0.002, 0.05] + [0.0] * 9  # 14 DepthAI coeffs
    with_dist = OakCameraCalib(label="rgb", width=1920, height=1080, k_matrix=_k(), distortion=coeffs, ref_T_cam_cm=None)
    dist = oak_calibration_to_pinholes([with_dist])["rgb"].distortion
    assert dist is not None
    assert (dist.k1, dist.k2, dist.p1, dist.p2, dist.k3) == (0.1, -0.2, 0.001, -0.002, 0.05)

    no_dist = OakCameraCalib(label="rgb", width=1920, height=1080, k_matrix=_k(), distortion=[0.1, 0.2], ref_T_cam_cm=None)
    assert oak_calibration_to_pinholes([no_dist])["rgb"].distortion is None
