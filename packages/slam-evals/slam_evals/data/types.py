"""Dataclasses + enums for parsed VSLAM-LAB sequences."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from simplecv.camera_parameters import Fisheye62Parameters, PinholeParameters

from slam_evals.data.datasets import DatasetSpec
from slam_evals.data.datasets import lookup as _dataset_lookup


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

    @property
    def dataset_spec(self) -> DatasetSpec | None:
        """Static frame-convention metadata for this sequence's dataset.

        Returns ``None`` for datasets without an entry in
        ``slam_evals.data.datasets`` — caller should treat that as "use
        viewer defaults" (i.e. don't log world ViewCoordinates).
        """
        return _dataset_lookup(self.dataset)


@dataclass(frozen=True, slots=True)
class CameraSpec:
    """One camera's full calibration, modelled on simplecv's parameter classes.

    Holds simplecv's ``PinholeParameters`` (or ``Fisheye62Parameters`` for
    fisheye) directly — that's where intrinsics, extrinsics, distortion,
    and the derived projection matrix live. The two extra fields here
    (``fps``, ``depth_factor``) are VSLAM-LAB calibration-yaml metadata
    that simplecv doesn't carry, so we keep them alongside.

    Use ``parameters`` when handing the camera to simplecv helpers
    (``log_pinhole``, etc.); the convenience properties below let
    callers read ``cam.cam_name`` instead of ``cam.parameters.name``.
    """

    parameters: PinholeParameters | Fisheye62Parameters
    fps: float | None = None
    depth_factor: float | None = None  # rgbd only; divisor that maps uint16 → metres

    @property
    def cam_name(self) -> str:
        return self.parameters.name


@dataclass(frozen=True, slots=True)
class Calibration:
    cameras: tuple[CameraSpec, ...]
    # Modelled IMU fields parsed from calibration.yaml's ``imus[0]`` block:
    # ``fps`` (float) and ``T_BS`` (list[float]). Other YAML fields
    # (``imu_name``, per-IMU noise terms like ``a_max``, ``sigma_g_c``) are
    # dropped at parse time. None entries are filtered out.
    imu_params: dict[str, float | list[float]] | None = None
