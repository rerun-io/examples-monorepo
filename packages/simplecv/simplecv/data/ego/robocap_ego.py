from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Float32
from numpy import ndarray
from serde import serde
from serde.yaml import from_yaml

from simplecv.camera_parameters import (
    Extrinsics,
    Fisheye62Parameters,
    Intrinsics,
    KannalaBrandtDistortion,
)
from simplecv.data.ego.base_ego import BaseEgoSequence, CameraParam, EgoData

if TYPE_CHECKING:
    from simplecv.data.exoego.robocap import RobocapConfig
else:
    from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig as RobocapConfig

# Camera name mapping from video filename suffix to canonical name
VIDEO_SUFFIX_TO_CAM_NAME: dict[str, str] = {
    "left-front": "left_front",
    "right-front": "right_front",
    "left-eye": "left_eye",
    "right-eye": "right_eye",
    "left": "left",
    "right": "right",
}

# Preferred display order for 2D video panels in Rerun viewer
CAMERA_DISPLAY_ORDER: list[str] = [
    "left_front",
    "right_front",
    "left_eye",
    "right_eye",
    "left",
    "right",
]

# Mapping from camera name to calibration folder and cam index
CAM_TO_CALIB_INFO: dict[str, tuple[str, int]] = {
    "left_front": ("imus_cam_lr_front_extrinsic", 0),
    "right_front": ("imus_cam_lr_front_extrinsic", 1),
    "left_eye": ("imus_cam_lr_eye_extrinsic", 0),
    "right_eye": ("imus_cam_lr_eye_extrinsic", 1),
    "left": ("imus_cam_l_extrinsic", 0),
    "right": ("imus_cam_r_extrinsic", 0),
}


# ─────────────────────────────────────────────────────────────────────────────
# Kalibr YAML serde dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@serde
class KalibrCamWithExtrinsic:
    """Single camera entry in a Kalibr camchain-imucam YAML (with T_cam_imu)."""

    camera_model: str
    """Camera model, e.g. 'pinhole'."""
    distortion_model: str
    """Distortion model, e.g. 'equidistant' (Kannala-Brandt)."""
    intrinsics: list[float]
    """Intrinsics as [fx, fy, cx, cy]."""
    distortion_coeffs: list[float]
    """Distortion coefficients, 4 for equidistant."""
    resolution: list[int]
    """Resolution as [width, height]."""
    T_cam_imu: list[list[float]]
    """4x4 transformation matrix: camera ← IMU (cam_T_imu)."""
    cam_overlaps: list[int] = field(default_factory=list)
    """Overlapping camera indices."""
    rostopic: str = ""
    """ROS topic name (optional)."""
    timeshift_cam_imu: float = 0.0
    """Time offset between camera and IMU."""
    T_cn_cnm1: list[list[float]] | None = None
    """Optional stereo transform to previous camera."""


def _load_kalibr_camchain_imucam(yaml_path: Path) -> dict[str, KalibrCamWithExtrinsic]:
    """Load a Kalibr camchain-imucam YAML file and return camera dict.

    Uses pyserde's from_yaml with dict[str, KalibrCamWithExtrinsic] to directly
    deserialize the dynamic camera keys (cam0, cam1, etc.).
    """
    yaml_content: str = yaml_path.read_text()
    result: dict[str, KalibrCamWithExtrinsic] = from_yaml(
        dict[str, KalibrCamWithExtrinsic], yaml_content
    )
    return result


def _kalibr_cam_to_fisheye62(
    cam_data: KalibrCamWithExtrinsic,
    name: str,
) -> Fisheye62Parameters:
    """Convert Kalibr camera data to Fisheye62Parameters."""
    fx, fy, cx, cy = cam_data.intrinsics
    width, height = cam_data.resolution
    k1, k2, k3, k4 = cam_data.distortion_coeffs[:4]

    intrinsics = Intrinsics.from_focal_principal_point(
        camera_conventions="RDF",
        fl_x=fx,
        fl_y=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
    )

    distortion = KannalaBrandtDistortion(k1=k1, k2=k2, k3=k3, k4=k4)

    # T_cam_imu is camera ← IMU, so cam_R_imu and cam_t_imu
    # We treat IMU frame as "world" for static ego assumption
    t_cam_imu: Float32[ndarray, "4 4"] = np.array(cam_data.T_cam_imu, dtype=np.float32)
    cam_R_world: Float32[ndarray, "3 3"] = t_cam_imu[:3, :3]
    cam_t_world: Float32[ndarray, "3"] = t_cam_imu[:3, 3]

    extrinsics = Extrinsics(cam_R_world=cam_R_world, cam_t_world=cam_t_world)

    return Fisheye62Parameters(
        name=name,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        distortion=distortion,
    )


class RobocapEgoSequence(BaseEgoSequence["RobocapConfig"]):
    """Egocentric view loader for Robocap dataset (6 synchronized cameras)."""

    def __init__(self, cfg: "RobocapConfig") -> None:
        self._calibration_data: dict[str, Fisheye62Parameters] = {}
        super().__init__(cfg)

    def load_video_paths(self) -> list[Path]:
        """Load video paths from session directory."""
        session_dir: Path = (
            self.config.root_directory
            / f"{self.config.device_id}_session_{self.config.session_id}"
        )
        if not session_dir.exists():
            raise FileNotFoundError(f"Session directory not found: {session_dir}")

        pattern: str = f"video_*_session{self.config.session_id}_segment{self.config.segment_id}_*.mp4"
        video_paths: list[Path] = sorted(session_dir.glob(pattern))

        if not video_paths:
            raise FileNotFoundError(
                f"No videos found matching pattern '{pattern}' in {session_dir}"
            )

        return video_paths

    def load_ego_cams(self) -> dict[str, list[CameraParam]]:
        """Load camera parameters from factory calibration YAML files."""
        calib_dir: Path = (
            self.config.root_directory
            / f"0factory-calibration-{self.config.device_id}"
        )
        if not calib_dir.exists():
            raise FileNotFoundError(f"Calibration directory not found: {calib_dir}")

        ego_cam_dict: dict[str, list[CameraParam]] = {}

        for cam_name, (calib_folder, cam_idx) in CAM_TO_CALIB_INFO.items():
            calib_folder_path: Path = calib_dir / calib_folder
            if not calib_folder_path.exists():
                continue

            # Find the camchain-imucam.yaml file
            imucam_files: list[Path] = list(
                calib_folder_path.glob("*-camchain-imucam.yaml")
            )
            if not imucam_files:
                continue

            imucam_path: Path = imucam_files[0]
            cam_dict: dict[str, KalibrCamWithExtrinsic] = _load_kalibr_camchain_imucam(
                imucam_path
            )

            cam_key: str = f"cam{cam_idx}"
            if cam_key not in cam_dict:
                continue

            cam_data: KalibrCamWithExtrinsic = cam_dict[cam_key]
            cam_params: Fisheye62Parameters = _kalibr_cam_to_fisheye62(
                cam_data, name=cam_name
            )

            # Store calibration for later lookup
            self._calibration_data[cam_name] = cam_params

            # Return list with single static camera params (no per-frame variation)
            ego_cam_dict[cam_name] = [cam_params]

        return ego_cam_dict

    def align_cams_and_videos(
        self,
        video_path_list: list[Path],
        ego_cam_dict: dict[str, list[CameraParam]],
    ) -> tuple[dict[str, list[CameraParam]], dict[str, Path]]:
        """Align cameras and videos by matching video filename suffixes to camera names.

        Returns dicts ordered by CAMERA_DISPLAY_ORDER for consistent 2D panel layout.
        """
        # First pass: collect all video-to-camera mappings
        cam_to_video: dict[str, Path] = {}
        for video_path in video_path_list:
            # Extract camera suffix from filename like "video_dev0_session11_segment1_right-eye.mp4"
            stem: str = video_path.stem
            parts: list[str] = stem.split("_")
            if len(parts) < 2:
                continue

            video_suffix: str = parts[-1]  # e.g., "right-eye"
            cam_name: str | None = VIDEO_SUFFIX_TO_CAM_NAME.get(video_suffix)

            if cam_name is not None and cam_name in ego_cam_dict:
                cam_to_video[cam_name] = video_path

        # Second pass: build ordered dicts according to CAMERA_DISPLAY_ORDER
        aligned_cam_dict: dict[str, list[CameraParam]] = {}
        video_map: dict[str, Path] = {}
        for cam_name in CAMERA_DISPLAY_ORDER:
            if cam_name in cam_to_video:
                aligned_cam_dict[cam_name] = ego_cam_dict[cam_name]
                video_map[cam_name] = cam_to_video[cam_name]

        return aligned_cam_dict, video_map

    def __len__(self) -> int:
        return len(self.ego_video_readers)

    def __getitem__(self, idx: int) -> EgoData:
        cam_params_list: list[CameraParam] = [
            cam_list[min(idx, len(cam_list) - 1)]
            for cam_list in self._ego_cam_dict.values()
        ]
        return EgoData(
            cam_params_list=cam_params_list,
            bgr_list=self.ego_video_readers[idx],
        )

    @property
    def image_plane_distance(self) -> int | float:
        """Image plane distance for visualization frustum rendering."""
        return 0.025
