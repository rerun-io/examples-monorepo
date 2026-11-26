"""RoboCap camera names and its factory Kalibr calibration (``*-camchain-imucam.yaml``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float32
from numpy import ndarray
from serde import serde
from serde.yaml import from_yaml
from simplecv.camera_parameters import Extrinsics, Fisheye62Parameters, Intrinsics, KannalaBrandtDistortion

VIDEO_SUFFIX_TO_CAM_NAME: dict[str, str] = {
    "left-front": "left_front",
    "right-front": "right_front",
    "left-eye": "left_eye",
    "right-eye": "right_eye",
    "left": "left",
    "right": "right",
}
"""Video filename suffix -> canonical camera name."""

CAMERA_DISPLAY_ORDER: list[str] = ["left_front", "right_front", "left_eye", "right_eye", "left", "right"]
"""Camera order of the 2D panes, and of the rig's camera indices."""

CAM_TO_CALIB_INFO: dict[str, tuple[str, int]] = {
    "left_front": ("imus_cam_lr_front_extrinsic", 0),
    "right_front": ("imus_cam_lr_front_extrinsic", 1),
    "left_eye": ("imus_cam_lr_eye_extrinsic", 0),
    "right_eye": ("imus_cam_lr_eye_extrinsic", 1),
    "left": ("imus_cam_l_extrinsic", 0),
    "right": ("imus_cam_r_extrinsic", 0),
}
"""Camera name -> (factory calibration folder, ``camN`` index in its camchain file)."""


@serde
@dataclass(frozen=True, slots=True)
class KalibrCamWithExtrinsic:
    """One camera entry in a Kalibr camchain-imucam YAML; keys not read here are ignored."""

    camera_model: str
    """Camera model, e.g. 'pinhole'."""
    distortion_model: str
    """Distortion model, e.g. 'equidistant' (Kannala-Brandt)."""
    intrinsics: list[float]
    """[fx, fy, cx, cy]."""
    distortion_coeffs: list[float]
    """4 coefficients for equidistant."""
    resolution: list[int]
    """[width, height]."""
    T_cam_imu: list[list[float]]
    """4x4 camera <- IMU transform (cam_T_imu)."""
    timeshift_cam_imu: float | None = None
    """Seconds: t_imu = t_camera + offset; missing means unknown, not zero."""


def load_kalibr_camchain_imucam(yaml_path: Path) -> dict[str, KalibrCamWithExtrinsic]:
    """Cameras of a Kalibr camchain-imucam YAML, keyed ``cam0``, ``cam1``, ..."""
    return from_yaml(dict[str, KalibrCamWithExtrinsic], yaml_path.read_text())


def kalibr_cam_to_fisheye62(cam_data: KalibrCamWithExtrinsic, name: str) -> Fisheye62Parameters:
    """A Kalibr equidistant camera as Fisheye62Parameters, with the IMU frame as world."""
    fx, fy, cx, cy = cam_data.intrinsics
    width, height = cam_data.resolution
    k1, k2, k3, k4 = cam_data.distortion_coeffs[:4]
    intrinsics: Intrinsics = Intrinsics.from_focal_principal_point(camera_conventions="RDF", fl_x=fx, fl_y=fy, cx=cx, cy=cy, width=width, height=height)
    cam_T_imu: Float32[ndarray, "4 4"] = np.array(cam_data.T_cam_imu, dtype=np.float32)
    return Fisheye62Parameters(
        name=name,
        intrinsics=intrinsics,
        extrinsics=Extrinsics(cam_R_world=cam_T_imu[:3, :3], cam_t_world=cam_T_imu[:3, 3]),
        distortion=KannalaBrandtDistortion(k1=k1, k2=k2, k3=k3, k4=k4),
    )
