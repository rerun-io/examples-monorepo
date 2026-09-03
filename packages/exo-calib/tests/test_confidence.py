import numpy as np
from jaxtyping import Float64
from numpy import ndarray

from exo_calib.confidence import (
    drop_face_confidences,
    kineo_point_confidence,
    median_filter_keypoints,
    modulate_crop_edge_confidence,
    threshold_rescale_confidence,
)


def test_modulate_crop_edge_confidence_handles_edges_outside_and_nan() -> None:
    kp_xy: Float64[ndarray, "6 2"] = np.array(
        [[50.0, 40.0], [5.0, 40.0], [95.0, 40.0], [-1.0, 40.0], [50.0, 81.0], [np.nan, 40.0]], dtype=np.float64
    )
    conf: Float64[ndarray, "6"] = np.array([0.8, 0.8, 0.8, 0.8, 0.8, np.nan], dtype=np.float64)

    actual: Float64[ndarray, "6"] = modulate_crop_edge_confidence(kp_xy, conf, crop_wh=(100, 80), margin_px=10.0)

    expected: Float64[ndarray, "6"] = np.array([0.8, 0.4, 0.4, 0.0, 0.0, 0.0], dtype=np.float64)
    np.testing.assert_allclose(actual, expected)


def test_threshold_rescale_confidence_applies_cutoff_and_linear_scale() -> None:
    conf: Float64[ndarray, "5"] = np.array([0.0, 0.14, 0.15, 0.575, 1.0], dtype=np.float64)

    actual: Float64[ndarray, "5"] = threshold_rescale_confidence(conf, tau=0.15)

    expected: Float64[ndarray, "5"] = np.array([0.0, 0.0, 0.0, 0.5, 1.0], dtype=np.float64)
    np.testing.assert_allclose(actual, expected)


def test_median_filter_keypoints_uses_truncated_windows_and_ignores_nan() -> None:
    kp_xy: Float64[ndarray, "5 2 2"] = np.array(
        [
            [[0.0, 10.0], [np.nan, 0.0]],
            [[100.0, 10.0], [np.nan, np.nan]],
            [[2.0, 10.0], [np.nan, 4.0]],
            [[3.0, 10.0], [np.nan, 6.0]],
            [[4.0, 10.0], [np.nan, 8.0]],
        ],
        dtype=np.float64,
    )

    actual: Float64[ndarray, "5 2 2"] = median_filter_keypoints(kp_xy, window=3)

    expected: Float64[ndarray, "5 2 2"] = np.array(
        [
            [[50.0, 10.0], [np.nan, 0.0]],
            [[2.0, 10.0], [np.nan, 2.0]],
            [[3.0, 10.0], [np.nan, 5.0]],
            [[3.0, 10.0], [np.nan, 6.0]],
            [[3.5, 10.0], [np.nan, 7.0]],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_kineo_point_confidence_averages_all_view_pairs() -> None:
    points_xyz: Float64[ndarray, "1 3"] = np.array([[0.0, 0.0, 5.0]], dtype=np.float64)
    intrinsics: Float64[ndarray, "2 3 3"] = np.repeat(np.diag([100.0, 100.0, 1.0])[None], 2, axis=0)
    cam_T_world: Float64[ndarray, "2 4 4"] = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    cam_T_world[1, 0, 3] = -1.0
    kp_xy: Float64[ndarray, "2 1 2"] = np.array([[[0.0, 0.0]], [[-20.0, 0.0]]], dtype=np.float64)
    conf: Float64[ndarray, "2 1"] = np.ones((2, 1), dtype=np.float64)

    perfect_conf: Float64[ndarray, "1"] = kineo_point_confidence(points_xyz, kp_xy, conf, intrinsics, cam_T_world, lambda_decay=1.0)

    np.testing.assert_allclose(perfect_conf, np.array([1.0], dtype=np.float64))

    points_xyz = np.repeat(points_xyz, 2, axis=0)
    intrinsics = np.repeat(np.diag([100.0, 100.0, 1.0])[None], 3, axis=0)
    cam_T_world = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    cam_T_world[1, 0, 3] = -1.0
    cam_T_world[2, 0, 3] = 1.0
    kp_xy = np.array(
        [[[0.0, 0.0], [0.0, 0.0]], [[-20.0, 0.0], [-20.0, 0.0]], [[30.0, 0.0], [20.0, 0.0]]], dtype=np.float64
    )
    conf = np.array([[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float64)

    mixed_conf: Float64[ndarray, "2"] = kineo_point_confidence(points_xyz, kp_xy, conf, intrinsics, cam_T_world, lambda_decay=1.0)

    expected_conf: Float64[ndarray, "2"] = np.array([0.9674862827, 0.0], dtype=np.float64)
    np.testing.assert_allclose(mixed_conf, expected_conf, atol=1e-10)


def test_kineo_point_confidence_invalidates_a_view_behind_the_camera() -> None:
    points_xyz: Float64[ndarray, "1 3"] = np.array([[0.0, 0.0, 5.0]], dtype=np.float64)
    intrinsics: Float64[ndarray, "3 3 3"] = np.repeat(np.diag([100.0, 100.0, 1.0])[None], 3, axis=0)
    cam_T_world: Float64[ndarray, "3 4 4"] = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    cam_T_world[1, 0, 3] = -1.0
    cam_T_world[2, :3, :3] = np.diag([-1.0, 1.0, -1.0])
    kp_xy: Float64[ndarray, "3 1 2"] = np.array([[[0.0, 0.0]], [[-20.0, 0.0]], [[0.0, 0.0]]], dtype=np.float64)
    conf: Float64[ndarray, "3 1"] = np.ones((3, 1), dtype=np.float64)

    actual_conf: Float64[ndarray, "1"] = kineo_point_confidence(points_xyz, kp_xy, conf, intrinsics, cam_T_world, lambda_decay=1.0)

    np.testing.assert_allclose(actual_conf, np.array([1.0 / 3.0], dtype=np.float64))


def test_drop_face_confidences_keeps_body_feet_and_hands() -> None:
    conf: Float64[ndarray, "1 1 133"] = np.ones((1, 1, 133), dtype=np.float64)

    selected: Float64[ndarray, "1 1 133"] = drop_face_confidences(conf)

    assert np.all(selected[..., :23] == 1.0)
    assert np.all(selected[..., 23:91] == 0.0)
    assert np.all(selected[..., 91:] == 1.0)
    assert np.all(conf == 1.0)
