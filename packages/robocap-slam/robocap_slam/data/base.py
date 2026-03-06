"""Base dataset interface for cuVSLAM tracking pipelines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cuvslam
import numpy as np
from jaxtyping import Int, UInt8
from numpy import ndarray
from simplecv.camera_parameters import Fisheye62Parameters, PinholeParameters

from robocap_slam.configs.base_config import InstantiateConfig

CameraParam = PinholeParameters | Fisheye62Parameters


@dataclass
class BaseTrackDatasetConfig(InstantiateConfig):
    """Base configuration for datasets used with cuVSLAM tracking."""


class BaseTrackDataset(ABC):
    """Abstract base providing data for cuVSLAM tracking pipelines."""

    @property
    @abstractmethod
    def cameras(self) -> list[cuvslam.Camera]:
        """cuVSLAM camera objects for the rig."""
        ...

    @property
    @abstractmethod
    def cam_names(self) -> list[str]:
        """Ordered camera names matching the cameras list."""
        ...

    @property
    @abstractmethod
    def cam_params(self) -> dict[str, CameraParam]:
        """Camera parameters keyed by camera name."""
        ...

    @property
    @abstractmethod
    def video_paths(self) -> dict[str, Path]:
        """Video file paths keyed by camera name."""
        ...

    @property
    @abstractmethod
    def n_frames(self) -> int:
        """Total number of frames available."""
        ...

    @property
    @abstractmethod
    def video_timestamps_ns(self) -> Int[ndarray, "n_frames"]:
        """Video frame timestamps in nanoseconds."""
        ...

    @abstractmethod
    def get_frame(self, frame_idx: int) -> list[UInt8[np.ndarray, "h w 3"]]:
        """Read frames for all cameras at the given index."""
        ...

    @property
    @abstractmethod
    def image_plane_distance(self) -> float:
        """Distance for Rerun frustum visualization."""
        ...

    @property
    def mesh_path(self) -> Path | None:
        """Optional path to a 3D mesh representing the capture rig."""
        return None
