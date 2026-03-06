"""Robocap dataset implementation for cuVSLAM tracking."""

from dataclasses import dataclass, field
from pathlib import Path

import cuvslam
import numpy as np
import rerun as rr
from jaxtyping import Int, UInt8
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.camera_parameters import Fisheye62Parameters
from simplecv.data.ego.robocap_ego import RobocapEgoSequence
from simplecv.data.exoego.robocap import RobocapConfig
from simplecv.video_io import VideoReader

from robocap_slam.data.base import BaseTrackDataset, BaseTrackDatasetConfig, CameraParam

# Stereo pair definitions: (left_cam, right_cam)
# 3 pairs from 4 physical cameras (left_front and right_front are shared)
STEREO_PAIRS: list[tuple[str, str]] = [
    ("left", "left_front"),
    ("left_front", "right_front"),
    ("right_front", "right"),
]


def fisheye62_to_cuvslam_camera(params: Fisheye62Parameters) -> cuvslam.Camera:
    """Convert simplecv Fisheye62Parameters to a cuvslam.Camera.

    Args:
        params: Fisheye62 camera parameters with intrinsics, extrinsics, and distortion.

    Returns:
        cuvslam.Camera configured for the given parameters.
    """
    cam = cuvslam.Camera()
    cam.size = (params.intrinsics.width, params.intrinsics.height)
    cam.focal = (params.intrinsics.fl_x, params.intrinsics.fl_y)
    cam.principal = (params.intrinsics.cx, params.intrinsics.cy)

    # Distortion: Kannala-Brandt k1-k4 -> cuvslam Fisheye model
    if params.distortion is not None:
        k1: float = params.distortion.k1
        k2: float = params.distortion.k2
        k3: float = params.distortion.k3
        k4: float = params.distortion.k4
        cam.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Fisheye, [k1, k2, k3, k4])

    # rig_from_camera = imu_T_cam = world_T_cam (IMU frame treated as world)
    # Extrinsics stores cam_T_world, so we need its inverse
    world_T_cam = params.extrinsics.world_T_cam
    quat_xyzw = Rotation.from_matrix(world_T_cam[:3, :3]).as_quat()
    cam.rig_from_camera = cuvslam.Pose(rotation=quat_xyzw, translation=world_T_cam[:3, 3])

    return cam


@dataclass
class RobocapTrackConfig(BaseTrackDatasetConfig):
    """Configuration for tracking Robocap headset sequences."""

    _target: type = field(default_factory=lambda: RobocapTrackDataset)
    """Target class to instantiate."""
    root_directory: Path = Path("data/robocap")
    """Root directory containing robocap data."""
    device_id: str = "f408193e6447b3b0"
    """Device identifier."""
    session_id: int = 14
    """Session number."""
    segment_id: int = 1
    """Segment number."""
    pairs: list[int] = field(default_factory=lambda: [0, 2])
    """Stereo pair indices: 0=left|left_front, 1=left_front|right_front, 2=right_front|right."""


class RobocapTrackDataset(BaseTrackDataset):
    """Robocap dataset providing frames and calibrations for cuVSLAM tracking."""

    def __init__(self, cfg: RobocapTrackConfig) -> None:
        # Build RobocapEgoSequence which loads calibrations and discovers videos
        ego_cfg: RobocapConfig = RobocapConfig(
            root_directory=cfg.root_directory,
            device_id=cfg.device_id,
            session_id=cfg.session_id,
            segment_id=cfg.segment_id,
        )
        ego: RobocapEgoSequence = RobocapEgoSequence(ego_cfg)

        # Extract Fisheye62Parameters from ego cam dict
        ego_cam_dict: dict[str, list[CameraParam]] = ego.ego_cam_dict
        fisheye_params: dict[str, Fisheye62Parameters] = {}
        for name, param_list in ego_cam_dict.items():
            param: CameraParam = param_list[0]
            if isinstance(param, Fisheye62Parameters):
                fisheye_params[name] = param

        # Build camera order from selected stereo pairs
        selected_pairs: list[tuple[str, str]] = [STEREO_PAIRS[i] for i in cfg.pairs]
        cam_order: list[str] = []
        for left, right in selected_pairs:
            cam_order.extend([left, right])
        self._cam_names: list[str] = cam_order

        # Convert to cuvslam cameras
        self._cameras: list[cuvslam.Camera] = [fisheye62_to_cuvslam_camera(fisheye_params[name]) for name in cam_order]

        # Store camera parameters
        self._cam_params: dict[str, CameraParam] = {name: fisheye_params[name] for name in cam_order}

        # Get video paths from ego sequence
        ego_video_dict: dict[str, Path] = {}
        for name, path in zip(ego.ego_video_names, ego.ego_video_paths, strict=True):
            ego_video_dict[name] = path
        self._video_paths: dict[str, Path] = {name: ego_video_dict[name] for name in dict.fromkeys(cam_order)}

        # Create VideoReader instances for unique physical cameras
        physical_cams: list[str] = list(dict.fromkeys(cam_order))
        self._readers: dict[str, VideoReader] = {name: VideoReader(self._video_paths[name]) for name in physical_cams}

        # Frame count = minimum across all readers
        self._n_frames: int = min(reader.frame_cnt for reader in self._readers.values())

        # Get video timestamps from first video using AssetVideo
        first_video_path: Path = next(iter(self._video_paths.values()))
        self._video_timestamps_ns: Int[ndarray, "n_frames"] = rr.AssetVideo(path=first_video_path).read_frame_timestamps_nanos()

        # Store for reference
        self._image_plane_distance: float = ego.image_plane_distance

        # Rig mesh for visualization
        mesh_file: Path = cfg.root_directory / "robocap-mesh" / "3DModel.glb"
        if mesh_file.exists():
            self._mesh_path: Path | None = mesh_file
        else:
            self._mesh_path = None
            print(f"Rig mesh not found at {mesh_file}, mesh visualization disabled.")

        print(f"Stereo pairs: {selected_pairs}")
        print(f"Camera order ({len(self._cameras)} slots): {cam_order}")
        print(f"Frames: {self._n_frames}")

    @property
    def cameras(self) -> list[cuvslam.Camera]:
        return self._cameras

    @property
    def cam_names(self) -> list[str]:
        return self._cam_names

    @property
    def cam_params(self) -> dict[str, CameraParam]:
        return self._cam_params

    @property
    def video_paths(self) -> dict[str, Path]:
        return self._video_paths

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def video_timestamps_ns(self) -> Int[ndarray, "n_frames"]:
        return self._video_timestamps_ns

    def get_frame(self, frame_idx: int) -> list[UInt8[np.ndarray, "h w 3"]]:
        """Read frames for all cameras at the given index."""
        bgrs: dict[str, UInt8[np.ndarray, "h w 3"]] = {
            name: reader.get_frame(frame_idx) for name, reader in self._readers.items()
        }
        return [bgrs[name] for name in self._cam_names]

    @property
    def image_plane_distance(self) -> float:
        return self._image_plane_distance

    @property
    def mesh_path(self) -> Path | None:
        return self._mesh_path
