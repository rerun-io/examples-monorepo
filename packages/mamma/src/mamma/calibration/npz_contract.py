"""Reader for the MAMMA ``ma_cap`` per-camera NPZ calibration contract.

The original pipeline's ``ma_cap`` stage writes one NPZ per camera plus a
``global.npz``. The crossing_arms dataset ships the same format under
``meta/``. Fields used here:

- ``cam_name``: camera name (e.g. ``"A001"``)
- ``cam_int``:  3x3 K matrix (pixels, native resolution)
- ``cam_ext``:  4x4 world->camera transform (``x_cam = cam_ext @ x_world``;
  projection in the original code is ``K @ cam_ext``)
- ``cam_img_w`` / ``cam_img_h``: native image size in pixels

Translations are normalized to meters following the original loader: any
``|t| > 200`` is assumed to be millimeters and divided by 1000
(``optimization/run_ma_3d.py`` in the source repo).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from jaxtyping import Float64
from numpy import ndarray
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    """Single-camera pinhole calibration in meters, native pixel units."""

    name: str
    """Camera name, e.g. ``"A001"``."""
    k_matrix: Float64[ndarray, "3 3"]
    """Pinhole intrinsics K in pixels at ``(width, height)`` resolution."""
    world_to_cam: Float64[ndarray, "4 4"]
    """World->camera rigid transform; ``x_cam = world_to_cam @ x_world``."""
    width: int
    """Image width in pixels that ``k_matrix`` is expressed in."""
    height: int
    """Image height in pixels that ``k_matrix`` is expressed in."""

    def scaled_to(self, height: int, width: int) -> "CameraCalibration":
        """Return a copy with K rescaled to a new ``height x width`` grid."""
        sx: float = width / self.width
        sy: float = height / self.height
        k_scaled: Float64[ndarray, "3 3"] = self.k_matrix.copy()
        k_scaled[0, :] *= sx
        k_scaled[1, :] *= sy
        return replace(self, k_matrix=k_scaled, width=width, height=height)

    def to_pinhole_parameters(self) -> PinholeParameters:
        """Convert to simplecv ``PinholeParameters`` (RDF/OpenCV convention)."""
        intrinsics: Intrinsics = Intrinsics.from_k_matrix(
            camera_conventions="RDF",
            k_matrix=self.k_matrix,
            height=self.height,
            width=self.width,
        )
        extrinsics: Extrinsics = Extrinsics(
            cam_R_world=self.world_to_cam[:3, :3],
            cam_t_world=self.world_to_cam[:3, 3],
        )
        return PinholeParameters(name=self.name, extrinsics=extrinsics, intrinsics=intrinsics)


def load_camera_npz(npz_path: Path) -> CameraCalibration:
    """Load one ``ma_cap``-format per-camera NPZ into a :class:`CameraCalibration`."""
    data = np.load(npz_path, allow_pickle=True)
    name: str = str(data["cam_name"]) if "cam_name" in data.files else npz_path.stem
    k_matrix: Float64[ndarray, "3 3"] = np.asarray(data["cam_int"], dtype=np.float64)
    world_to_cam: Float64[ndarray, "4 4"] = np.asarray(data["cam_ext"], dtype=np.float64)
    # mm -> m normalization, mirroring the original ma_3d loader.
    if np.abs(world_to_cam[:3, 3]).max() > 200:
        world_to_cam = world_to_cam.copy()
        world_to_cam[:3, 3] /= 1000.0
    width: int = int(data["cam_img_w"])
    height: int = int(data["cam_img_h"])
    return CameraCalibration(name=name, k_matrix=k_matrix, world_to_cam=world_to_cam, width=width, height=height)


@dataclass(frozen=True, slots=True)
class GlobalCaptureInfo:
    """Sequence-level metadata from ``global.npz``."""

    sequence_name: str
    """Sequence name, e.g. ``"crossing_arms"``."""
    fps: float
    """Capture frame rate."""
    frame_start: int
    """First frame index of the capture."""
    frame_end: int
    """One-past-last frame index of the capture."""


def load_global_npz(npz_path: Path) -> GlobalCaptureInfo:
    """Load the sequence-level ``global.npz``."""
    data = np.load(npz_path, allow_pickle=True)
    return GlobalCaptureInfo(
        sequence_name=str(data["seq_name"]),
        fps=float(data["fps"]),
        frame_start=int(data["frame_start"]),
        frame_end=int(data["frame_end"]),
    )
