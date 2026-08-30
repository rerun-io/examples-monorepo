"""Fisheye stereo rectification for calibrated camera pairs."""

from dataclasses import dataclass

import cv2
import numpy as np
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray


@dataclass(slots=True, frozen=True)
class FisheyeCamera:
    """OpenCV fisheye camera calibration."""

    K_33: Float64[ndarray, "3 3"]
    """Camera intrinsic matrix as ``Float64[ndarray, "3 3"]``."""
    dist_4: Float64[ndarray, "4"]
    """Kannala-Brandt equidistant coefficients ``k1..k4`` as ``Float64[ndarray, "4"]``."""
    width: int
    """Image width in pixels."""
    height: int
    """Image height in pixels."""


@dataclass(slots=True)
class StereoRectification:
    """Shared pinhole geometry and OpenCV remap tables for a stereo pair."""

    map0_x: Float32[ndarray, "h w"]
    """Camera 0 horizontal source-coordinate map as ``Float32[ndarray, "h w"]``."""
    map0_y: Float32[ndarray, "h w"]
    """Camera 0 vertical source-coordinate map as ``Float32[ndarray, "h w"]``."""
    map1_x: Float32[ndarray, "h w"]
    """Camera 1 horizontal source-coordinate map as ``Float32[ndarray, "h w"]``."""
    map1_y: Float32[ndarray, "h w"]
    """Camera 1 vertical source-coordinate map as ``Float32[ndarray, "h w"]``."""
    R0_33: Float64[ndarray, "3 3"]
    """Rotation from camera 0 coordinates to rectified camera 0 coordinates."""
    R1_33: Float64[ndarray, "3 3"]
    """Rotation from camera 1 coordinates to rectified camera 1 coordinates."""
    K_rect_33: Float64[ndarray, "3 3"]
    """Shared rectified pinhole intrinsics from camera 0's projection matrix."""
    baseline_m: float
    """Euclidean distance between the two camera centers in metres."""
    width: int
    """Rectified image width in pixels."""
    height: int
    """Rectified image height in pixels."""


def fisheye_stereo_rectify(
    cam0: FisheyeCamera,
    cam1: FisheyeCamera,
    cam1_T_cam0: Float64[ndarray, "4 4"],
    *,
    focal_scale: float = 1.0,
) -> StereoRectification:
    """Rectify an OpenCV-fisheye stereo pair into a shared pinhole geometry.

    ``cam1_T_cam0`` maps camera-0 coordinates into camera-1 coordinates as
    ``p1 = cam1_T_cam0[:3, :3] @ p0 + cam1_T_cam0[:3, 3]``. Its rotation and
    translation go directly to OpenCV as camera 1 with respect to camera 0.
    Returned ``R0_33`` and ``R1_33`` are ``rectified_R_cam`` for each camera.

    Args:
        cam0: Camera 0 fisheye calibration.
        cam1: Camera 1 fisheye calibration.
        cam1_T_cam0: Camera-0-to-camera-1 rigid transform as ``Float64[ndarray, "4 4"]``.
        focal_scale: Rectified focal length as a multiple of ``cam0``'s fx (smaller keeps more of the fisheye FOV).

    Returns:
        Rectification rotations, shared intrinsics, baseline, and Float32 remap tables.
    """
    image_size: tuple[int, int] = (cam0.width, cam0.height)
    D0_41: Float64[ndarray, "4 1"] = cam0.dist_4.reshape(4, 1)
    D1_41: Float64[ndarray, "4 1"] = cam1.dist_4.reshape(4, 1)
    R_33: Float64[ndarray, "3 3"] = np.ascontiguousarray(cam1_T_cam0[:3, :3])
    T_31: Float64[ndarray, "3 1"] = np.ascontiguousarray(cam1_T_cam0[:3, 3]).reshape(3, 1)
    stereo_result: tuple[Float64[ndarray, "3 3"], Float64[ndarray, "3 3"], Float64[ndarray, "3 4"], Float64[ndarray, "3 4"], Float64[ndarray, "4 4"]] = (
        cv2.fisheye.stereoRectify(
            cam0.K_33, D0_41, cam1.K_33, D1_41, image_size, R_33, T_31,
            flags=cv2.CALIB_ZERO_DISPARITY,
        )
    )
    R0_33: Float64[ndarray, "3 3"] = stereo_result[0]
    R1_33: Float64[ndarray, "3 3"] = stereo_result[1]
    # cv2.fisheye.stereoRectify's estimated P is unusable for wide fisheyes (it put cx at 2721 px on a
    # 1920 px robocap frame); keep its R0/R1 and define the shared rectified pinhole ourselves.
    K_rect_33: Float64[ndarray, "3 3"] = np.array(
        [[cam0.K_33[0, 0] * focal_scale, 0.0, cam0.width / 2.0], [0.0, cam0.K_33[0, 0] * focal_scale, cam0.height / 2.0], [0.0, 0.0, 1.0]]
    )
    map0_result: tuple[Float32[ndarray, "h w"], Float32[ndarray, "h w"]] = cv2.fisheye.initUndistortRectifyMap(
        cam0.K_33, D0_41, R0_33, K_rect_33, image_size, cv2.CV_32FC1
    )
    map1_result: tuple[Float32[ndarray, "h w"], Float32[ndarray, "h w"]] = cv2.fisheye.initUndistortRectifyMap(
        cam1.K_33, D1_41, R1_33, K_rect_33, image_size, cv2.CV_32FC1
    )
    return StereoRectification(
        map0_x=map0_result[0],
        map0_y=map0_result[1],
        map1_x=map1_result[0],
        map1_y=map1_result[1],
        R0_33=R0_33,
        R1_33=R1_33,
        K_rect_33=K_rect_33,
        baseline_m=float(np.linalg.norm(cam1_T_cam0[:3, 3])),
        width=cam0.width,
        height=cam0.height,
    )


def rectified_rig_extrinsics(
    cam_R_rig: Float64[ndarray, "3 3"], cam_t_rig: Float64[ndarray, "3"], R_rect: Float64[ndarray, "3 3"]
) -> tuple[Float64[ndarray, "3 3"], Float64[ndarray, "3"]]:
    """Compose rig-to-camera extrinsics with a camera-to-rectified-camera rotation.

    Args:
        cam_R_rig: Rig-to-camera rotation as ``Float64[ndarray, "3 3"]``.
        cam_t_rig: Rig-to-camera translation as ``Float64[ndarray, "3"]``.
        R_rect: Camera-to-rectified-camera rotation as ``Float64[ndarray, "3 3"]``.

    Returns:
        Rig-to-rectified-camera rotation and translation. These map parent rig
        points into the child camera, matching Rerun ``Transform3D(from_parent=True)``.
    """
    cam_rect_R_rig: Float64[ndarray, "3 3"] = R_rect @ cam_R_rig
    cam_rect_t_rig: Float64[ndarray, "3"] = R_rect @ cam_t_rig
    return cam_rect_R_rig, cam_rect_t_rig


def rectify_pair(
    rect: StereoRectification, left_hw3: UInt8[ndarray, "h w 3"], right_hw3: UInt8[ndarray, "h w 3"]
) -> tuple[UInt8[ndarray, "h w 3"], UInt8[ndarray, "h w 3"]]:
    """Remap a fisheye stereo image pair into the rectified pinhole views.

    Args:
        rect: Precomputed stereo rectification and Float32 remap tables.
        left_hw3: Camera 0 image as ``UInt8[ndarray, "h w 3"]``.
        right_hw3: Camera 1 image as ``UInt8[ndarray, "h w 3"]``.

    Returns:
        Rectified camera 0 and camera 1 images as ``UInt8[ndarray, "h w 3"]``.
    """
    left_rect_hw3: UInt8[ndarray, "h w 3"] = cv2.remap(left_hw3, rect.map0_x, rect.map0_y, cv2.INTER_LINEAR)
    right_rect_hw3: UInt8[ndarray, "h w 3"] = cv2.remap(right_hw3, rect.map1_x, rect.map1_y, cv2.INTER_LINEAR)
    return left_rect_hw3, right_rect_hw3
