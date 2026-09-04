"""Stereo rectification of a Kannala-Brandt fisheye pair into a shared pinhole, on simplecv camera types.

Input cameras are :class:`simplecv.camera_parameters.Fisheye62Parameters` whose extrinsics are ``cam_T_rig``
(the rig frame plays the role of "world", as the exoego rig writers use it). The output holds the remap tables
and the two rectified virtual cameras as :class:`PinholeParameters` with composed ``cam_rect_T_rig`` extrinsics,
so they slot straight into ``CameraSensor`` / ``log_rig_static``.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from jaxtyping import Float32, Float64
from numpy import ndarray
from simplecv.camera_parameters import Extrinsics, Fisheye62Parameters, Intrinsics, PinholeParameters


@dataclass(slots=True)
class StereoRectification:
    """Remap tables plus the rectified virtual cameras of a stereo pair."""

    map0_x: Float32[ndarray, "h w"]
    """Camera 0 remap x-coordinates for ``cv2.remap``."""
    map0_y: Float32[ndarray, "h w"]
    """Camera 0 remap y-coordinates."""
    map1_x: Float32[ndarray, "h w"]
    """Camera 1 remap x-coordinates."""
    map1_y: Float32[ndarray, "h w"]
    """Camera 1 remap y-coordinates."""
    R0_33: Float64[ndarray, "3 3"]
    """Camera 0 rectification rotation, ``rect0_R_cam0`` (cv2's R1)."""
    R1_33: Float64[ndarray, "3 3"]
    """Camera 1 rectification rotation, ``rect1_R_cam1`` (cv2's R2)."""
    left_rect: PinholeParameters
    """Rectified camera 0: shared pinhole intrinsics, extrinsics ``cam_rect_T_rig``."""
    right_rect: PinholeParameters
    """Rectified camera 1, same intrinsics, extrinsics ``cam_rect_T_rig``."""
    baseline_m: float
    """Distance between the two camera centres in metres."""


def kannala_brandt_4(camera: Fisheye62Parameters) -> Float64[ndarray, "4 1"]:
    """The ``k1..k4`` column ``cv2.fisheye`` expects; the model has no k5/k6/p1/p2 so those must be zero."""
    if camera.distortion is None:
        raise ValueError(f"{camera.name}: fisheye camera without Kannala-Brandt distortion")
    d = camera.distortion
    if any(abs(v) > 0.0 for v in (d.k5, d.k6, d.p1, d.p2)):
        raise ValueError(f"{camera.name}: cv2.fisheye supports k1..k4 only; got k5={d.k5} k6={d.k6} p1={d.p1} p2={d.p2}")
    return np.array([[d.k1], [d.k2], [d.k3], [d.k4]], dtype=np.float64)


def fisheye_stereo_rectify(cam0: Fisheye62Parameters, cam1: Fisheye62Parameters, *, focal_scale: float = 1.0) -> StereoRectification:
    """Rectify a Kannala-Brandt fisheye pair whose extrinsics are ``cam_T_rig``.

    ``cv2.fisheye.stereoRectify`` (``CALIB_ZERO_DISPARITY``) supplies the rotations R0/R1; the shared rectified
    pinhole is defined here — cv2's estimated ``P`` is unusable for wide fisheyes (it put ``cx`` at 2721 px on a
    1920 px robocap frame): principal point at the image centre, focal ``focal_scale * fx0``.

    Args:
        cam0: Left camera with ``extrinsics.cam_T_world`` = ``cam0_T_rig``, ``Float[ndarray, "4 4"]``.
        cam1: Right camera, same convention.
        focal_scale: Rectified focal length as a multiple of ``cam0``'s fx (smaller keeps more of the fisheye FOV).

    Returns:
        The remap tables, rotations, and both rectified cameras as :class:`PinholeParameters`.
    """
    width: int = cam0.intrinsics.width
    height: int = cam0.intrinsics.height
    K0_33: Float64[ndarray, "3 3"] = np.asarray(cam0.intrinsics.k_matrix, dtype=np.float64)
    K1_33: Float64[ndarray, "3 3"] = np.asarray(cam1.intrinsics.k_matrix, dtype=np.float64)
    cam0_T_rig: Float64[ndarray, "4 4"] = cam0.extrinsics.cam_T_world
    cam1_T_rig: Float64[ndarray, "4 4"] = cam1.extrinsics.cam_T_world
    cam1_T_cam0: Float64[ndarray, "4 4"] = cam1_T_rig @ np.linalg.inv(cam0_T_rig)
    # cv2 wants contiguous (3,3) R, (3,1) T and (4,1) D arrays.
    R_33: Float64[ndarray, "3 3"] = np.ascontiguousarray(cam1_T_cam0[:3, :3])
    T_31: Float64[ndarray, "3 1"] = np.ascontiguousarray(cam1_T_cam0[:3, 3]).reshape(3, 1)
    stereo_result: tuple[Float64[ndarray, "3 3"], Float64[ndarray, "3 3"], Float64[ndarray, "3 4"], Float64[ndarray, "3 4"], Float64[ndarray, "4 4"]] = (
        cv2.fisheye.stereoRectify(K0_33, kannala_brandt_4(cam0), K1_33, kannala_brandt_4(cam1), (width, height), R_33, T_31, flags=cv2.CALIB_ZERO_DISPARITY)
    )
    R0_33: Float64[ndarray, "3 3"] = stereo_result[0]
    R1_33: Float64[ndarray, "3 3"] = stereo_result[1]
    focal: float = float(K0_33[0, 0]) * focal_scale
    K_rect_33: Float64[ndarray, "3 3"] = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])
    map0: tuple[Float32[ndarray, "h w"], Float32[ndarray, "h w"]] = cv2.fisheye.initUndistortRectifyMap(K0_33, kannala_brandt_4(cam0), R0_33, K_rect_33, (width, height), cv2.CV_32FC1)
    map1: tuple[Float32[ndarray, "h w"], Float32[ndarray, "h w"]] = cv2.fisheye.initUndistortRectifyMap(K1_33, kannala_brandt_4(cam1), R1_33, K_rect_33, (width, height), cv2.CV_32FC1)

    def rectified(camera: Fisheye62Parameters, R_rect: Float64[ndarray, "3 3"]) -> PinholeParameters:
        # rect_T_rig = [R_rect, 0] @ cam_T_rig — the rectifying rotation is applied in the camera frame.
        cam_T_rig: Float64[ndarray, "4 4"] = camera.extrinsics.cam_T_world
        return PinholeParameters(
            name=f"{camera.name}_rectified",
            extrinsics=Extrinsics(cam_R_world=R_rect @ cam_T_rig[:3, :3], cam_t_world=R_rect @ cam_T_rig[:3, 3]),
            intrinsics=Intrinsics.from_k_matrix(camera_conventions="RDF", k_matrix=K_rect_33, height=height, width=width),
        )

    return StereoRectification(
        map0_x=map0[0],
        map0_y=map0[1],
        map1_x=map1[0],
        map1_y=map1[1],
        R0_33=R0_33,
        R1_33=R1_33,
        left_rect=rectified(cam0, R0_33),
        right_rect=rectified(cam1, R1_33),
        baseline_m=float(np.linalg.norm(cam1_T_cam0[:3, 3])),
    )


def rectify_pair(rect: StereoRectification, left_hw3: np.ndarray, right_hw3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remap a fisheye stereo image pair into the rectified pinhole views (``cv2.INTER_LINEAR``)."""
    return cv2.remap(left_hw3, rect.map0_x, rect.map0_y, cv2.INTER_LINEAR), cv2.remap(right_hw3, rect.map1_x, rect.map1_y, cv2.INTER_LINEAR)
