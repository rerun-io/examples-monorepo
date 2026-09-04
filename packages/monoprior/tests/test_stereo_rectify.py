"""Fisheye stereo rectification for the RoboCap front camera pair, on simplecv camera types."""

import cv2
import numpy as np
from jaxtyping import Float64, UInt8
from numpy import ndarray
from simplecv.camera_parameters import Extrinsics, Fisheye62Parameters, Intrinsics, KannalaBrandtDistortion

from monopriors.models.stereo_depth.rectify import StereoRectification, fisheye_stereo_rectify, kannala_brandt_4, rectify_pair


def _robocap_front_pair() -> tuple[Fisheye62Parameters, Fisheye62Parameters]:
    """The real RoboCap front-pair calibration (Kalibr camchain, device f408193e6447b3b0), extrinsics as ``cam_T_rig``."""
    cam0: Fisheye62Parameters = Fisheye62Parameters(
        name="left_front",
        extrinsics=Extrinsics(cam_R_world=np.eye(3), cam_t_world=np.zeros(3)),
        intrinsics=Intrinsics.from_focal_principal_point(
            camera_conventions="RDF", fl_x=636.4360979504252, fl_y=634.7124672858736, cx=956.2034051233109, cy=525.4381269616423, height=1080, width=1920
        ),
        distortion=KannalaBrandtDistortion(k1=0.06166616664989166, k2=-0.02109092590818281, k3=0.037163389786906705, k4=-0.013518619346447233),
    )
    rotation_result: tuple[Float64[ndarray, "3 3"], Float64[ndarray, "3 9"]] = cv2.Rodrigues(np.array([0.002, -0.003, 0.001]))
    cam1: Fisheye62Parameters = Fisheye62Parameters(
        name="right_front",
        extrinsics=Extrinsics(cam_R_world=rotation_result[0], cam_t_world=np.array([-0.08617986581746322, -0.0014843392131672361, -0.0009221497579335349])),
        intrinsics=Intrinsics.from_focal_principal_point(
            camera_conventions="RDF", fl_x=630.6917887212685, fl_y=628.7775762673645, cx=946.6721836113659, cy=539.5312757485085, height=1080, width=1920
        ),
        distortion=KannalaBrandtDistortion(k1=0.07725944509266558, k2=-0.0625834150766394, k3=0.08006517968619031, k4=-0.028795758317964704),
    )
    return cam0, cam1


def test_real_front_pair_rectifies_projected_points() -> None:
    cam0, cam1 = _robocap_front_pair()
    rect: StereoRectification = fisheye_stereo_rectify(cam0, cam1)
    cam1_T_cam0: Float64[ndarray, "4 4"] = cam1.extrinsics.cam_T_world  # cam0 is the rig origin
    K_rect_33: Float64[ndarray, "3 3"] = np.asarray(rect.left_rect.intrinsics.k_matrix)

    rng: np.random.Generator = np.random.default_rng(7)
    points_cam0_n13: Float64[ndarray, "n 1 3"] = np.empty((200, 1, 3), dtype=np.float64)
    points_cam0_n13[:, 0, 2] = rng.uniform(1.0, 8.0, size=200)
    points_cam0_n13[:, 0, :2] = rng.uniform(-0.1, 0.1, size=(200, 2)) * points_cam0_n13[:, :, 2]
    zero_3: Float64[ndarray, "3"] = np.zeros(3, dtype=np.float64)
    K0_33: Float64[ndarray, "3 3"] = np.asarray(cam0.intrinsics.k_matrix)
    K1_33: Float64[ndarray, "3 3"] = np.asarray(cam1.intrinsics.k_matrix)
    pixels0_result: tuple[Float64[ndarray, "n 1 2"], Float64[ndarray, "2n 15"]] = cv2.fisheye.projectPoints(points_cam0_n13, zero_3, zero_3, K0_33, kannala_brandt_4(cam0))
    rvec1_result: tuple[Float64[ndarray, "3 1"], Float64[ndarray, "9 3"]] = cv2.Rodrigues(np.ascontiguousarray(cam1_T_cam0[:3, :3]))
    cam1_t_cam0_31: Float64[ndarray, "3 1"] = np.ascontiguousarray(cam1_T_cam0[:3, 3]).reshape(3, 1)
    pixels1_result: tuple[Float64[ndarray, "n 1 2"], Float64[ndarray, "2n 15"]] = cv2.fisheye.projectPoints(
        points_cam0_n13, rvec1_result[0], cam1_t_cam0_31, K1_33, kannala_brandt_4(cam1)
    )
    rectified0_n12: Float64[ndarray, "n 1 2"] = cv2.fisheye.undistortPoints(pixels0_result[0], K0_33, kannala_brandt_4(cam0), R=rect.R0_33, P=K_rect_33)
    rectified1_n12: Float64[ndarray, "n 1 2"] = cv2.fisheye.undistortPoints(pixels1_result[0], K1_33, kannala_brandt_4(cam1), R=rect.R1_33, P=K_rect_33)

    np.testing.assert_allclose(rectified0_n12[:, 0, 1], rectified1_n12[:, 0, 1], atol=0.05, rtol=0.0)
    disparity_n: Float64[ndarray, "n"] = rectified0_n12[:, 0, 0] - rectified1_n12[:, 0, 0]
    expected_disparity_n: Float64[ndarray, "n"] = K_rect_33[0, 0] * rect.baseline_m / points_cam0_n13[:, 0, 2]
    assert np.all(disparity_n > 0.0)
    np.testing.assert_allclose(disparity_n, expected_disparity_n, rtol=0.005, atol=0.0)


def test_rectified_cameras_compose_rig_extrinsics() -> None:
    cam0, cam1 = _robocap_front_pair()
    rect: StereoRectification = fisheye_stereo_rectify(cam0, cam1)
    point_rig_3: Float64[ndarray, "3"] = np.array([0.7, -0.2, 2.5], dtype=np.float64)
    for camera, R_rect, rectified in ((cam0, rect.R0_33, rect.left_rect), (cam1, rect.R1_33, rect.right_rect)):
        cam_T_rig: Float64[ndarray, "4 4"] = camera.extrinsics.cam_T_world
        direct_3: Float64[ndarray, "3"] = R_rect @ (cam_T_rig[:3, :3] @ point_rig_3 + cam_T_rig[:3, 3])
        composed_3: Float64[ndarray, "3"] = rectified.extrinsics.cam_T_world[:3, :3] @ point_rig_3 + rectified.extrinsics.cam_T_world[:3, 3]
        np.testing.assert_allclose(composed_3, direct_3, rtol=0.0, atol=1e-12)
    # both rectified cameras share intrinsics and sit one baseline apart
    np.testing.assert_allclose(rect.left_rect.intrinsics.k_matrix, rect.right_rect.intrinsics.k_matrix)
    centres_delta_3: Float64[ndarray, "3"] = rect.right_rect.extrinsics.world_t_cam - rect.left_rect.extrinsics.world_t_cam
    assert abs(np.linalg.norm(centres_delta_3) - rect.baseline_m) < 1e-9
    assert rect.baseline_m == float(np.linalg.norm(cam1.extrinsics.cam_T_world[:3, 3]))


def test_rectify_pair_remaps_images_deterministically() -> None:
    cam0, cam1 = _robocap_front_pair()
    rect: StereoRectification = fisheye_stereo_rectify(cam0, cam1)
    height: int = cam0.intrinsics.height
    width: int = cam0.intrinsics.width
    yx_2hw: ndarray = np.indices((height, width))
    checker_hw: UInt8[ndarray, "h w"] = ((yx_2hw[1] // 32 + yx_2hw[0] // 32) % 2 * 160).astype(np.uint8)
    image_hw3: UInt8[ndarray, "h w 3"] = np.stack((checker_hw + 32, yx_2hw[1].astype(np.uint8), yx_2hw[0].astype(np.uint8)), axis=-1)

    first_pair: tuple[UInt8[ndarray, "h w 3"], UInt8[ndarray, "h w 3"]] = rectify_pair(rect, image_hw3, image_hw3)
    second_pair: tuple[UInt8[ndarray, "h w 3"], UInt8[ndarray, "h w 3"]] = rectify_pair(rect, image_hw3, image_hw3)
    for image_rect_hw3 in first_pair:
        assert image_rect_hw3.shape == (height, width, 3) and image_rect_hw3.dtype == np.uint8
        assert np.any(image_rect_hw3[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4])
    np.testing.assert_array_equal(first_pair[0], second_pair[0])
    np.testing.assert_array_equal(first_pair[1], second_pair[1])
