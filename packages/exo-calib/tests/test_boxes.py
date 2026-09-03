import numpy as np
from jaxtyping import Bool, Float64
from numpy import ndarray

from exo_calib.boxes import boxes_from_projected_keypoints, boxes_needing_detection, extrapolate_keypoints
from exo_calib.triangulation import triangulate_frame_keypoints


def test_boxes_from_projected_keypoints_filters_confidence_and_pads_about_center() -> None:
    points_xyz: Float64[ndarray, "4 3"] = np.array(
        [[-1.0, -0.5, 5.0], [1.0, 0.5, 5.0], [10.0, 10.0, 5.0], [0.0, 0.0, -1.0]], dtype=np.float64
    )
    conf: Float64[ndarray, "4"] = np.array([0.9, 0.8, 0.1, 1.0], dtype=np.float64)
    camera_intrinsics: Float64[ndarray, "3 3"] = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    camera_T_world: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)

    box_xyxy: Float64[ndarray, "4"] = boxes_from_projected_keypoints(
        points_xyz,
        conf,
        camera_intrinsics,
        camera_T_world,
        image_wh=(100, 80),
        pad=1.25,
        min_joints=2,
        min_confidence=0.5,
    )

    np.testing.assert_allclose(box_xyxy, np.array([25.0, 27.5, 75.0, 52.5], dtype=np.float64))


def test_boxes_from_projected_keypoints_clips_padding_and_rejects_insufficient_joints() -> None:
    camera_intrinsics: Float64[ndarray, "3 3"] = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    camera_T_world: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
    clipped_points_xyz: Float64[ndarray, "2 3"] = np.array([[-2.25, -1.75, 5.0], [-1.75, -1.25, 5.0]], dtype=np.float64)

    clipped_xyxy: Float64[ndarray, "4"] = boxes_from_projected_keypoints(
        clipped_points_xyz,
        np.ones(2, dtype=np.float64),
        camera_intrinsics,
        camera_T_world,
        image_wh=(100, 80),
        pad=2.0,
        min_joints=2,
        min_confidence=0.15,
    )
    missing_xyxy: Float64[ndarray, "4"] = boxes_from_projected_keypoints(
        np.array([[0.0, 0.0, 5.0], [20.0, 0.0, 5.0]], dtype=np.float64),
        np.ones(2, dtype=np.float64),
        camera_intrinsics,
        camera_T_world,
        image_wh=(100, 80),
        min_joints=2,
        pad=1.25,
        min_confidence=0.15,
    )

    np.testing.assert_allclose(clipped_xyxy, np.array([0.0, 0.0, 20.0, 20.0], dtype=np.float64))
    assert np.isnan(missing_xyxy).all()


def test_boxes_needing_detection_flags_missing_and_image_boundary_boxes() -> None:
    boxes: Float64[ndarray, "5 4"] = np.array(
        [
            [np.nan, np.nan, np.nan, np.nan],
            [0.0, 10.0, 80.0, 70.0],
            [10.0, 0.0, 80.0, 70.0],
            [10.0, 10.0, 100.0, 80.0],
            [10.0, 10.0, 80.0, 70.0],
        ],
        dtype=np.float64,
    )

    needs_detection: Bool[ndarray, " n"] = boxes_needing_detection(boxes, image_wh=(100, 80))

    np.testing.assert_array_equal(needs_detection, np.array([True, True, True, True, False]))


def test_extrapolate_keypoints_uses_velocity_where_available_and_latest_elsewhere() -> None:
    latest_points_xyz: Float64[ndarray, "3 3"] = np.array(
        [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [np.nan, np.nan, np.nan]], dtype=np.float64
    )
    previous_points_xyz: Float64[ndarray, "3 3"] = np.array(
        [[1.0, 1.0, 1.0], [np.nan, np.nan, np.nan], [1.0, 2.0, 3.0]], dtype=np.float64
    )

    predicted_points_xyz: Float64[ndarray, "3 3"] = extrapolate_keypoints(latest_points_xyz, previous_points_xyz, step_ratio=2.0)

    expected_points_xyz: Float64[ndarray, "3 3"] = np.array(
        [[4.0, 7.0, 10.0], [5.0, 6.0, 7.0], [np.nan, np.nan, np.nan]], dtype=np.float64
    )
    np.testing.assert_allclose(predicted_points_xyz, expected_points_xyz, equal_nan=True)


def test_triangulate_frame_keypoints_preserves_joint_rows_and_confidence() -> None:
    points_xyz: Float64[ndarray, "3 3"] = np.array(
        [[0.0, 0.0, 5.0], [0.5, 0.2, 4.0], [-0.2, 0.1, 6.0]], dtype=np.float64
    )
    intrinsics: Float64[ndarray, "3 3 3"] = np.repeat(
        np.array([[[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]], dtype=np.float64), 3, axis=0
    )
    cam_T_world: Float64[ndarray, "3 4 4"] = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    cam_T_world[1, 0, 3] = -1.0
    cam_T_world[2, 1, 3] = -1.0
    points_homo: Float64[ndarray, "3 4"] = np.column_stack((points_xyz, np.ones(3, dtype=np.float64)))
    points_cam: Float64[ndarray, "3 3 3"] = np.einsum("vij,nj->vni", cam_T_world[:, :3], points_homo)
    projected: Float64[ndarray, "3 3 3"] = np.einsum("vij,vnj->vni", intrinsics, points_cam)
    kp_xy: Float64[ndarray, "3 3 2"] = projected[:, :, :2] / projected[:, :, 2:3]
    conf: Float64[ndarray, "3 3"] = np.full((3, 3), 0.9, dtype=np.float64)
    conf[1:, 2] = 0.1

    triangulated_xyz, triangulated_conf = triangulate_frame_keypoints(
        kp_xy,
        conf,
        intrinsics,
        cam_T_world,
        min_confidence=0.5,
        reproj_threshold_px=1.0,
    )

    np.testing.assert_allclose(triangulated_xyz[:2], points_xyz[:2], atol=1e-8)
    assert np.isnan(triangulated_xyz[2]).all()
    np.testing.assert_allclose(triangulated_conf, np.array([0.9, 0.9, 0.0], dtype=np.float64))
