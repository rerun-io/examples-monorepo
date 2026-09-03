import numpy as np
from jaxtyping import Float64
from numpy import ndarray
from simplecv.ops.umeyama import SimilarityTransform

from exo_calib.eval import CameraErrors, RigGeometry, align_rigs, camera_errors, evaluate_rig, rig_geometry


def test_evaluate_rig_reports_zero_for_identical_rigs() -> None:
    centers: Float64[ndarray, "4 3"] = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=np.float64
    )
    cam_T_world: Float64[ndarray, "4 4 4"] = np.repeat(np.eye(4, dtype=np.float64)[None], 4, axis=0)
    cam_T_world[:, :3, 3] = -centers

    errors: CameraErrors = evaluate_rig(cam_T_world, cam_T_world, mode="se3")

    np.testing.assert_allclose(errors.rotation_error_deg, np.zeros(4, dtype=np.float64), atol=1e-10)
    np.testing.assert_allclose(errors.translation_error_cm, np.zeros(4, dtype=np.float64), atol=1e-10)
    assert errors.alignment_scale == 1.0


def test_align_rigs_recovers_known_se3_world_transform() -> None:
    gt_centers: Float64[ndarray, "4 3"] = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [-0.3, 1.7, 0.4], [0.2, -0.5, 2.1]], dtype=np.float64
    )
    gt_cam_T_world: Float64[ndarray, "4 4 4"] = np.repeat(np.eye(4, dtype=np.float64)[None], 4, axis=0)
    gt_cam_T_world[:, :3, 3] = -gt_centers
    angle_rad: float = np.deg2rad(31.0)
    rotation: Float64[ndarray, "3 3"] = np.array(
        [[np.cos(angle_rad), -np.sin(angle_rad), 0.0], [np.sin(angle_rad), np.cos(angle_rad), 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    translation: Float64[ndarray, "3"] = np.array([1.2, -0.7, 0.4], dtype=np.float64)
    pred_centers: Float64[ndarray, "4 3"] = np.einsum("ij,vj->vi", rotation.T, gt_centers - translation)
    pred_cam_T_world: Float64[ndarray, "4 4 4"] = np.repeat(np.eye(4, dtype=np.float64)[None], 4, axis=0)
    pred_cam_T_world[:, :3, :3] = rotation
    pred_cam_T_world[:, :3, 3] = -np.einsum("vij,vj->vi", pred_cam_T_world[:, :3, :3], pred_centers)

    alignment: SimilarityTransform = align_rigs(pred_cam_T_world, gt_cam_T_world)
    errors: CameraErrors = evaluate_rig(pred_cam_T_world, gt_cam_T_world, mode="se3")

    np.testing.assert_allclose(alignment.dst_R_src, rotation, atol=1e-12)
    np.testing.assert_allclose(alignment.dst_t_src, translation, atol=1e-12)
    assert alignment.scale == 1.0
    np.testing.assert_allclose(errors.rotation_error_deg, np.zeros(4, dtype=np.float64), atol=1e-10)
    np.testing.assert_allclose(errors.translation_error_cm, np.zeros(4, dtype=np.float64), atol=1e-10)


def test_align_rigs_recovers_known_sim3_scale() -> None:
    gt_centers: Float64[ndarray, "4 3"] = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [-0.3, 1.7, 0.4], [0.2, -0.5, 2.1]], dtype=np.float64
    )
    gt_cam_T_world: Float64[ndarray, "4 4 4"] = np.repeat(np.eye(4, dtype=np.float64)[None], 4, axis=0)
    gt_cam_T_world[:, :3, 3] = -gt_centers
    expected_scale: float = 1.37
    translation: Float64[ndarray, "3"] = np.array([-0.8, 1.1, 0.3], dtype=np.float64)
    pred_centers: Float64[ndarray, "4 3"] = (gt_centers - translation) / expected_scale
    pred_cam_T_world: Float64[ndarray, "4 4 4"] = np.repeat(np.eye(4, dtype=np.float64)[None], 4, axis=0)
    pred_cam_T_world[:, :3, 3] = -pred_centers

    alignment: SimilarityTransform = align_rigs(pred_cam_T_world, gt_cam_T_world, with_scale=True)
    errors: CameraErrors = evaluate_rig(pred_cam_T_world, gt_cam_T_world, mode="sim3")

    np.testing.assert_allclose(alignment.dst_R_src, np.eye(3, dtype=np.float64), atol=1e-12)
    np.testing.assert_allclose(alignment.dst_t_src, translation, atol=1e-12)
    assert np.isclose(alignment.scale, expected_scale, atol=1e-12)
    assert np.isclose(errors.alignment_scale, expected_scale, atol=1e-12)
    np.testing.assert_allclose(errors.rotation_error_deg, np.zeros(4, dtype=np.float64), atol=1e-10)
    np.testing.assert_allclose(errors.translation_error_cm, np.zeros(4, dtype=np.float64), atol=1e-10)


def test_camera_errors_reports_known_single_camera_rotation() -> None:
    centers: Float64[ndarray, "4 3"] = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [-0.3, 1.7, 0.4], [0.2, -0.5, 2.1]], dtype=np.float64
    )
    gt_cam_T_world: Float64[ndarray, "4 4 4"] = np.repeat(np.eye(4, dtype=np.float64)[None], 4, axis=0)
    gt_cam_T_world[:, :3, 3] = -centers
    pred_cam_T_world: Float64[ndarray, "4 4 4"] = gt_cam_T_world.copy()
    expected_angle_deg: float = 7.5
    angle_rad: float = np.deg2rad(expected_angle_deg)
    perturbed_rotation: Float64[ndarray, "3 3"] = np.array(
        [[np.cos(angle_rad), -np.sin(angle_rad), 0.0], [np.sin(angle_rad), np.cos(angle_rad), 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    pred_cam_T_world[2, :3, :3] = perturbed_rotation
    pred_cam_T_world[2, :3, 3] = -perturbed_rotation @ centers[2]
    identity: SimilarityTransform = SimilarityTransform(dst_R_src=np.eye(3, dtype=np.float64), dst_t_src=np.zeros(3, dtype=np.float64), scale=1.0)

    errors: CameraErrors = camera_errors(pred_cam_T_world, gt_cam_T_world, gt_T_pred=identity)

    np.testing.assert_allclose(errors.rotation_error_deg, np.array([0.0, 0.0, expected_angle_deg, 0.0], dtype=np.float64), atol=1e-10)
    np.testing.assert_allclose(errors.translation_error_cm, np.zeros(4, dtype=np.float64), atol=1e-10)


def test_rig_geometry_reports_pairwise_distances_and_floor_heights() -> None:
    cam_T_world: Float64[ndarray, "3 4 4"] = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    centers: Float64[ndarray, "3 3"] = np.array(
        [[0.0, 0.0, 1.0], [3.0, 0.0, 2.0], [0.0, 4.0, 3.0]], dtype=np.float64
    )
    cam_T_world[:, :3, 3] = -centers

    geometry: RigGeometry = rig_geometry(cam_T_world)

    np.testing.assert_allclose(geometry.camera_centers_m, centers)
    np.testing.assert_allclose(
        geometry.pairwise_distance_m,
        np.array([[0.0, np.sqrt(10.0), np.sqrt(20.0)], [np.sqrt(10.0), 0.0, np.sqrt(26.0)], [np.sqrt(20.0), np.sqrt(26.0), 0.0]]),
    )
    np.testing.assert_allclose(geometry.height_above_floor_m, np.array([1.0, 2.0, 3.0]))
