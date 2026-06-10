"""The single input type the streaming engine consumes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mamma.calibration.npz_contract import CameraCalibration


@dataclass(frozen=True, slots=True)
class MultiViewSequence:
    """A synchronized multiview capture: per-camera calibration + video files.

    Every dataset adapter (mamma NPZ, HOCap, Assembly101) produces this type;
    the streaming engine consumes only this.
    """

    name: str
    """Sequence identifier used for output naming and Rerun recording paths."""
    cameras: list[CameraCalibration]
    """Per-camera calibration at native video resolution, same order as ``video_paths``."""
    video_paths: list[Path]
    """One video file per camera, frame-synchronized, same order as ``cameras``."""
    fps: float
    """Frame rate shared by all cameras."""
    frame_count: int
    """Number of synchronized frames available across all cameras."""

    def __post_init__(self) -> None:
        if len(self.cameras) != len(self.video_paths):
            raise ValueError(f"cameras ({len(self.cameras)}) and video_paths ({len(self.video_paths)}) must align")
        if len(self.cameras) == 0:
            raise ValueError("MultiViewSequence needs at least one camera")

    @property
    def camera_names(self) -> list[str]:
        """Camera names in capture order."""
        return [cam.name for cam in self.cameras]

    def scaled_cameras(self, resize_hw: tuple[int, int]) -> list[CameraCalibration]:
        """Calibrations rescaled to the engine resolution ``(height, width)``."""
        return [cam.scaled_to(height=resize_hw[0], width=resize_hw[1]) for cam in self.cameras]
