"""Fisheye stereo rectification for the RoboCap front camera pair."""

import cv2
import numpy as np
from jaxtyping import Float64, UInt8
from numpy import ndarray

from monopriors.models.stereo_depth.rectify import (
    FisheyeCamera,
    StereoRectification,
    fisheye_stereo_rectify,
    rectified_rig_extrinsics,
    rectify_pair,
)


def _robocap_front_pair() -> tuple[FisheyeCamera, FisheyeCamera, Float64[ndarray, "4 4"]]:
    """Return the real RoboCap front-pair calibration and a representative rotation.

    Returns:
        Left camera, right camera, and ``cam1_T_cam0`` as ``Float64[ndarray, "4 4"]``.
    """
    # Kalibr camchain, device f408193e6447b3b0.
    cam0: FisheyeCamera = FisheyeCamera(
        K_33=np.array([[636.4360979504252, 0.0, 956.2034051233109], [0.0, 634.7124672858736, 525.4381269616423], [0.0, 0.0, 1.0]]),
        dist_4=np.array([0.06166616664989166, -0.02109092590818281, 0.037163389786906705, -0.013518619346447233]),
        width=1920,
        height=1080,
    )
    cam1: FisheyeCamera = FisheyeCamera(
        K_33=np.array([[630.6917887212685, 0.0, 946.6721836113659], [0.0, 628.7775762673645, 539.5312757485085], [0.0, 0.0, 1.0]]),
        dist_4=np.array([0.07725944509266558, -0.0625834150766394, 0.08006517968619031, -0.028795758317964704]),
        width=1920,
        height=1080,
    )
    rotation_result: tuple[Float64[ndarray, "3 3"], Float64[ndarray, "3 9"]] = cv2.Rodrigues(np.array([0.002, -0.003, 0.001]))
    cam1_T_cam0: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
    cam1_T_cam0[:3, :3] = rotation_result[0]
    cam1_T_cam0[:3, 3] = [-0.08617986581746322, -0.0014843392131672361, -0.0009221497579335349]
    return cam0, cam1, cam1_T_cam0


def test_real_front_pair_rectifies_projected_points() -> None:
    cam0: FisheyeCamera
    cam1: FisheyeCamera
    cam1_T_cam0: Float64[ndarray, "4 4"]
    cam0, cam1, cam1_T_cam0 = _robocap_front_pair()
    rect: StereoRectification = fisheye_stereo_rectify(cam0, cam1, cam1_T_cam0)

    rng: np.random.Generator = np.random.default_rng(7)
    points_cam0_n13: Float64[ndarray, "n 1 3"] = np.empty((200, 1, 3), dtype=np.float64)
    points_cam0_n13[:, 0, 2] = rng.uniform(1.0, 8.0, size=200)
    points_cam0_n13[:, 0, :2] = rng.uniform(-0.1, 0.1, size=(200, 2)) * points_cam0_n13[:, :, 2]
    zero_3: Float64[ndarray, "3"] = np.zeros(3, dtype=np.float64)
    pixels0_result: tuple[Float64[ndarray, "n 1 2"], Float64[ndarray, "2n 15"]] = cv2.fisheye.projectPoints(
        points_cam0_n13, zero_3, zero_3, cam0.K_33, cam0.dist_4.reshape(4, 1)
    )
    rvec1_result: tuple[Float64[ndarray, "3 1"], Float64[ndarray, "9 3"]] = cv2.Rodrigues(cam1_T_cam0[:3, :3])
    cam1_t_cam0_31: Float64[ndarray, "3 1"] = np.ascontiguousarray(cam1_T_cam0[:3, 3]).reshape(3, 1)
    pixels1_result: tuple[Float64[ndarray, "n 1 2"], Float64[ndarray, "2n 15"]] = cv2.fisheye.projectPoints(
        points_cam0_n13, rvec1_result[0], cam1_t_cam0_31, cam1.K_33, cam1.dist_4.reshape(4, 1)
    )
    rectified0_n12: Float64[ndarray, "n 1 2"] = cv2.fisheye.undistortPoints(
        pixels0_result[0], cam0.K_33, cam0.dist_4.reshape(4, 1), R=rect.R0_33, P=rect.K_rect_33
    )
    rectified1_n12: Float64[ndarray, "n 1 2"] = cv2.fisheye.undistortPoints(
        pixels1_result[0], cam1.K_33, cam1.dist_4.reshape(4, 1), R=rect.R1_33, P=rect.K_rect_33
    )

    np.testing.assert_allclose(rectified0_n12[:, 0, 1], rectified1_n12[:, 0, 1], atol=0.05, rtol=0.0)
    disparity_n: Float64[ndarray, "n"] = rectified0_n12[:, 0, 0] - rectified1_n12[:, 0, 0]
    expected_disparity_n: Float64[ndarray, "n"] = rect.K_rect_33[0, 0] * rect.baseline_m / points_cam0_n13[:, 0, 2]
    assert np.all(disparity_n > 0.0)
    np.testing.assert_allclose(disparity_n, expected_disparity_n, rtol=0.005, atol=0.0)


def test_rectify_pair_remaps_images_deterministically() -> None:
    cam0: FisheyeCamera
    cam1: FisheyeCamera
    cam1_T_cam0: Float64[ndarray, "4 4"]
    cam0, cam1, cam1_T_cam0 = _robocap_front_pair()
    rect: StereoRectification = fisheye_stereo_rectify(cam0, cam1, cam1_T_cam0)
    yx_2hw: ndarray = np.indices((cam0.height, cam0.width))
    y_hw: ndarray = yx_2hw[0]
    x_hw: ndarray = yx_2hw[1]
    checker_hw: UInt8[ndarray, "h w"] = ((x_hw // 32 + y_hw // 32) % 2 * 160).astype(np.uint8)
    image_hw3: UInt8[ndarray, "h w 3"] = np.stack((checker_hw + 32, x_hw.astype(np.uint8), y_hw.astype(np.uint8)), axis=-1)

    first_pair: tuple[UInt8[ndarray, "h w 3"], UInt8[ndarray, "h w 3"]] = rectify_pair(rect, image_hw3, image_hw3)
    second_pair: tuple[UInt8[ndarray, "h w 3"], UInt8[ndarray, "h w 3"]] = rectify_pair(rect, image_hw3, image_hw3)

    for image_rect_hw3 in first_pair:
        assert image_rect_hw3.shape == (cam0.height, cam0.width, 3)
        assert image_rect_hw3.dtype == np.uint8
        assert np.any(image_rect_hw3[cam0.height // 4 : 3 * cam0.height // 4, cam0.width // 4 : 3 * cam0.width // 4])
    np.testing.assert_array_equal(first_pair[0], second_pair[0])
    np.testing.assert_array_equal(first_pair[1], second_pair[1])


def test_rectified_rig_extrinsics_preserve_point_mapping() -> None:
    cam_rotation_result: tuple[Float64[ndarray, "3 3"], Float64[ndarray, "3 9"]] = cv2.Rodrigues(np.array([0.1, -0.04, 0.02]))
    rect_rotation_result: tuple[Float64[ndarray, "3 3"], Float64[ndarray, "3 9"]] = cv2.Rodrigues(np.array([-0.01, 0.03, 0.005]))
    cam_R_rig: Float64[ndarray, "3 3"] = cam_rotation_result[0]
    cam_t_rig: Float64[ndarray, "3"] = np.array([0.2, -0.1, 0.04], dtype=np.float64)
    R_rect: Float64[ndarray, "3 3"] = rect_rotation_result[0]
    point_rig_3: Float64[ndarray, "3"] = np.array([0.7, -0.2, 2.5], dtype=np.float64)

    result: tuple[Float64[ndarray, "3 3"], Float64[ndarray, "3"]] = rectified_rig_extrinsics(cam_R_rig, cam_t_rig, R_rect)
    point_rect_direct_3: Float64[ndarray, "3"] = R_rect @ (cam_R_rig @ point_rig_3 + cam_t_rig)
    point_rect_composed_3: Float64[ndarray, "3"] = result[0] @ point_rig_3 + result[1]

    np.testing.assert_allclose(point_rect_composed_3, point_rect_direct_3, rtol=0.0, atol=1e-12)


def test_rectification_reports_metric_baseline_and_homogeneous_intrinsics() -> None:
    cam0: FisheyeCamera
    cam1: FisheyeCamera
    cam1_T_cam0: Float64[ndarray, "4 4"]
    cam0, cam1, cam1_T_cam0 = _robocap_front_pair()

    rect: StereoRectification = fisheye_stereo_rectify(cam0, cam1, cam1_T_cam0)

    assert rect.baseline_m == float(np.linalg.norm(cam1_T_cam0[:3, 3]))
    assert rect.K_rect_33[2, 2] == 1.0
