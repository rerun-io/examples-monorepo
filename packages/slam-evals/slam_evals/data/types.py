"""Dataclasses + enums for parsed VSLAM-LAB sequences."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Modality(str, Enum):
    """Modality tag matching VSLAM-LAB's mono/stereo/rgbd[-vi] taxonomy.

    Inherits from ``str`` so members survive JSON round-trips as their raw
    string value. Python 3.11's ``StrEnum`` would be equivalent but we target
    3.10 to stay consistent with the rest of the monorepo.
    """

    MONO = "mono"
    MONO_VI = "mono-vi"
    STEREO = "stereo"
    STEREO_VI = "stereo-vi"
    RGBD = "rgbd"
    RGBD_VI = "rgbd-vi"

    def __str__(self) -> str:
        return self.value

    @property
    def has_imu(self) -> bool:
        return self.value.endswith("-vi")

    @property
    def has_stereo(self) -> bool:
        return self.value.startswith("stereo")

    @property
    def has_depth(self) -> bool:
        return self.value.startswith("rgbd")


@dataclass(frozen=True, slots=True)
class Sequence:
    """One benchmark sequence rooted at ``<benchmark_root>/<dataset>/<name>/``."""

    dataset: str
    name: str
    root: Path
    modality: Modality
    has_calibration: bool

    @property
    def slug(self) -> str:
        return f"{self.dataset}/{self.name}"

    @property
    def recording_id(self) -> str:
        return f"{self.dataset}__{self.name}"

    # Per-stream presence checks — drive ingest off what's actually on disk
    # rather than the modality enum, so adding new sensor types later is just a
    # new helper + a new layer writer (no enum surgery). The modality enum
    # under-reports content on stereo-rgbd sequences (e.g. OPENLORIS-D400 is
    # classified as STEREO/STEREO_VI but also ships depth_0 / depth_1).

    def has_camera(self, idx: int) -> bool:
        if idx == 0:
            return True  # rgb_0 is part of the required-files triad
        return (self.root / f"rgb_{idx}").is_dir()

    def has_depth(self, idx: int) -> bool:
        return (self.root / f"depth_{idx}").is_dir()

    def has_imu(self, idx: int) -> bool:
        return (self.root / f"imu_{idx}.csv").is_file()


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole intrinsics + body-sensor extrinsic ``T_BS`` (row-major 4x4)."""

    cam_name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_type: str | None
    distortion_coefficients: tuple[float, ...]
    fps: float | None
    t_bs: tuple[float, ...]  # 16 floats, row-major 4x4; (1,0,0,0, 0,1,0,0, ...) if missing
    depth_factor: float | None = None  # rgbd only; divisor that maps uint16 -> metres


@dataclass(frozen=True, slots=True)
class Calibration:
    cameras: tuple[CameraIntrinsics, ...]
    # IMU fields parsed verbatim from calibration.yaml's `imus[0]` block.
    # Values can be str (imu_name), float (fps, noise terms), or list[float] (T_BS).
    # None entries are filtered out by the parser.
    imu_params: dict[str, float | list[float] | str] | None = None
