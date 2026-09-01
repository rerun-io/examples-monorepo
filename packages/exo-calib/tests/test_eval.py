import numpy as np
from jaxtyping import Float64
from numpy import ndarray

from exo_calib.eval import CameraErrors, RigAlignment, align_rigs, camera_errors, evaluate_rig


def test_evaluate_rig_reports_zero_for_identical_rigs() -> None:
    centers_v3: Float64[ndarray, "4 3"] = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=np.float64
    )
    cam_T_world_v44: Float64[ndarray, "4 4 4"] = np.repeat(np.eye(4, dtype=np.float64)[None], 4, axis=0)
    cam_T_world_v44[:, :3, 3] = -centers_v3

    errors: CameraErrors = evaluate_rig(cam_T_world_v44, cam_T_world_v44, mode="se3")

    np.testing.assert_allclose(errors.rotation_error_deg_v, np.zeros(4, dtype=np.float64), atol=1e-10)
    np.testing.assert_allclose(errors.translation_error_cm_v, np.zeros(4, dtype=np.float64), atol=1e-10)
    assert errors.alignment_scale == 1.0


def test_align_rigs_recovers_known_se3_world_transform() -> None:
    gt_centers_v3: Float64[ndarray, "4 3"] = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [-0.3, 1.7, 0.4], [0.2, -0.5, 2.1]], dtype=np.float64
    )
    gt_cam_T_world_v44: Float64[ndarray, "4 4 4"] = np.repeat(np.eye(4, dtype=np.float64)[None], 4, axis=0)
    gt_cam_T_world_v44[:, :3, 3] = -gt_centers_v3
    angle_rad: float = np.deg2rad(31.0)
    rotation_33: Float64[ndarray, "3 3"] = np.array(
        [[np.cos(angle_rad), -np.sin(angle_rad), 0.0], [np.sin(angle_rad), np.cos(angle_rad), 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    translation_3: Float64[ndarray, "3"] = np.array([1.2, -0.7, 0.4], dtype=np.float64)
    pred_centers_v3: Float64[ndarray, "4 3"] = np.einsum("ij,vj->vi", rotation_33.T, gt_centers_v3 - translation_3)
    pred_cam_T_world_v44: Float64[ndarray, "4 4 4"] = np.repeat(np.eye(4, dtype=np.float64)[None], 4, axis=0)
    pred_cam_T_world_v44[:, :3, :3] = rotation_33
    pred_cam_T_world_v44[:, :3, 3] = -np.einsum("vij,vj->vi", pred_cam_T_world_v44[:, :3, :3], pred_centers_v3)

    alignment: RigAlignment = align_rigs(pred_cam_T_world_v44, gt_cam_T_world_v44)
    errors: CameraErrors = evaluate_rig(pred_cam_T_world_v44, gt_cam_T_world_v44, mode="se3")

    np.testing.assert_allclose(alignment.rotation_33, rotation_33, atol=1e-12)
    np.testing.assert_allclose(alignment.translation_3, translation_3, atol=1e-12)
    assert alignment.scale == 1.0
    np.testing.assert_allclose(errors.rotation_error_deg_v, np.zeros(4, dtype=np.float64), atol=1e-10)
    np.testing.assert_allclose(errors.translation_error_cm_v, np.zeros(4, dtype=np.float64), atol=1e-10)


def test_align_rigs_recovers_known_sim3_scale() -> None:
    gt_centers_v3: Float64[ndarray, "4 3"] = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [-0.3, 1.7, 0.4], [0.2, -0.5, 2.1]], dtype=np.float64
    )
    gt_cam_T_world_v44: Float64[ndarray, "4 4 4"] = np.repeat(np.eye(4, dtype=np.float64)[None], 4, axis=0)
    gt_cam_T_world_v44[:, :3, 3] = -gt_centers_v3
    expected_scale: float = 1.37
    translation_3: Float64[ndarray, "3"] = np.array([-0.8, 1.1, 0.3], dtype=np.float64)
    pred_centers_v3: Float64[ndarray, "4 3"] = (gt_centers_v3 - translation_3) / expected_scale
    pred_cam_T_world_v44: Float64[ndarray, "4 4 4"] = np.repeat(np.eye(4, dtype=np.float64)[None], 4, axis=0)
    pred_cam_T_world_v44[:, :3, 3] = -pred_centers_v3

    alignment: RigAlignment = align_rigs(pred_cam_T_world_v44, gt_cam_T_world_v44, with_scale=True)
    errors: CameraErrors = evaluate_rig(pred_cam_T_world_v44, gt_cam_T_world_v44, mode="sim3")

    np.testing.assert_allclose(alignment.rotation_33, np.eye(3, dtype=np.float64), atol=1e-12)
    np.testing.assert_allclose(alignment.translation_3, translation_3, atol=1e-12)
    assert np.isclose(alignment.scale, expected_scale, atol=1e-12)
    assert np.isclose(errors.alignment_scale, expected_scale, atol=1e-12)
    np.testing.assert_allclose(errors.rotation_error_deg_v, np.zeros(4, dtype=np.float64), atol=1e-10)
    np.testing.assert_allclose(errors.translation_error_cm_v, np.zeros(4, dtype=np.float64), atol=1e-10)


def test_camera_errors_reports_known_single_camera_rotation() -> None:
    centers_v3: Float64[ndarray, "4 3"] = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [-0.3, 1.7, 0.4], [0.2, -0.5, 2.1]], dtype=np.float64
    )
    gt_cam_T_world_v44: Float64[ndarray, "4 4 4"] = np.repeat(np.eye(4, dtype=np.float64)[None], 4, axis=0)
    gt_cam_T_world_v44[:, :3, 3] = -centers_v3
    pred_cam_T_world_v44: Float64[ndarray, "4 4 4"] = gt_cam_T_world_v44.copy()
    expected_angle_deg: float = 7.5
    angle_rad: float = np.deg2rad(expected_angle_deg)
    perturbed_rotation_33: Float64[ndarray, "3 3"] = np.array(
        [[np.cos(angle_rad), -np.sin(angle_rad), 0.0], [np.sin(angle_rad), np.cos(angle_rad), 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    pred_cam_T_world_v44[2, :3, :3] = perturbed_rotation_33
    pred_cam_T_world_v44[2, :3, 3] = -perturbed_rotation_33 @ centers_v3[2]
    identity_alignment: RigAlignment = RigAlignment(
        rotation_33=np.eye(3, dtype=np.float64), translation_3=np.zeros(3, dtype=np.float64), scale=1.0
    )

    errors: CameraErrors = camera_errors(pred_cam_T_world_v44, gt_cam_T_world_v44, alignment=identity_alignment)

    np.testing.assert_allclose(errors.rotation_error_deg_v, np.array([0.0, 0.0, expected_angle_deg, 0.0], dtype=np.float64), atol=1e-10)
    np.testing.assert_allclose(errors.translation_error_cm_v, np.zeros(4, dtype=np.float64), atol=1e-10)
