"""Pooled device intrinsics, RGB/IMU rotations and curated camera-50 borrowing."""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
from dataforge.records import read_json
from jaxtyping import Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from serde import serde

from selfcap_curate.records import GyroFit, OakFit

GRID: tuple[float, ...] = (450.0, 550.0, 650.0, 750.0, 850.0, 1000.0, 1200.0, 1450.0, 1750.0)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class FocalProfile:
    """Partial adapter for scratch f_profile results, permitting error-only sessions."""

    session: str  # Date__sid8 source session.
    curves: dict[str, tuple[float, ...]] = field(default_factory=dict)  # Nine BA costs per device on GRID.

    error: str | None = None  # Reason a session cannot produce curves.

    def __post_init__(self) -> None:
        if any(len(c) != 9 or not np.isfinite(c).all() for c in self.curves.values()):
            raise ValueError("focal profiles require nine finite costs per device")


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class DeviceIntrinsics:
    """Pooled focal and reproducible machine-record provenance."""

    f: float  # Square-pixel focal in 1920×1080 images; centre (960,540), no distortion.
    source_sessions: tuple[str, ...]  # Sessions contributing normalized focal profiles.
    date: str  # ISO date the table was generated.

    def __post_init__(self) -> None:
        date.fromisoformat(self.date)
        if not np.isfinite(self.f) or self.f <= 0.0 or not self.source_sessions:
            raise ValueError("device focal must be positive and have source sessions")


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class DeviceTable:
    """Packaged machine JSON; never catalog facts copied into a manifest."""

    devices: dict[str, DeviceIntrinsics]  # Full phone device ids and their pooled intrinsics.

    @property
    def focals(self) -> dict[str, float]:
        """Square-pixel focal per phone."""
        return {p: device.f for p, device in self.devices.items()}


def read_table(path: Path | None = None) -> DeviceTable:
    """Decode our strict table once, naming the source on a schema failure."""
    return read_json(path if path is not None else Path(__file__).with_name("device_intrinsics.json"), DeviceTable)


def pool(profiles: list[FocalProfile], generated_date: str) -> DeviceTable:
    """Sum (cost-min)/max(median,1) curves, refining their minimum by parabola."""
    accumulated: dict[str, list[tuple[str, tuple[float, ...]]]] = {}
    if len({p.session for p in profiles}) != len(profiles):
        raise ValueError("duplicate focal-profile session")
    for profile in profiles:
        for phone, curve in profile.curves.items():
            accumulated.setdefault(phone, []).append((profile.session, curve))
    devices: dict[str, DeviceIntrinsics] = {}
    grid = np.asarray(GRID)
    for phone, entries in accumulated.items():
        curves = [np.asarray(c) for _, c in entries]
        total = np.sum([(c - c.min()) / max(float(np.median(c)), 1.0) for c in curves], axis=0)
        k = int(np.argmin(total))
        focal = float(grid[k])
        if 0 < k < len(grid) - 1:
            coefficients = np.polyfit(grid[k - 1 : k + 2], total[k - 1 : k + 2], 2)
            if coefficients[0] > 0.0:
                focal = float(np.clip(-coefficients[1] / (2.0 * coefficients[0]), grid[k - 1], grid[k + 1]))
        devices[phone] = DeviceIntrinsics(focal, tuple(sorted(session for session, _ in entries)), generated_date)
    return DeviceTable(devices)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Camera50Borrow:
    device_id: str  # Full Quest identity; the healthy source must match it.
    healthy_segment: str  # Named catalog segment whose raw factory value carries camera 50 extrinsics.
    rig_T_device_m: Float64[ndarray, "3"]  # Measured rig/device lever arm in metres, retained at full precision.

    def __post_init__(self) -> None:
        if not self.healthy_segment.startswith("selfcap__") or not np.isfinite(self.rig_T_device_m).all():
            raise ValueError("camera-50 provenance requires a segment key and finite lever arm")


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Camera50Table:
    date: str  # ISO date this static machine table was generated.
    sessions: dict[str, Camera50Borrow]  # Faulty date__sid8 sessions, never duplicated catalog calibration documents.


def read_camera50_table(path: Path | None = None) -> Camera50Table:
    """Read strict static borrowing facts; calibration itself is read from the catalog."""
    return read_json(path if path is not None else Path(__file__).with_name("borrowed_camera50.json"), Camera50Table)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class DeviceRotation:
    rgb_R_imu: Float64[ndarray, "3 3"]  # OAK IMU axes to RGB camera axes.
    source_sessions: tuple[str, ...]  # Eligible sessions before robust rejection.
    used_sessions: tuple[str, ...]  # Sessions retained by max(3 degrees, 2.5×median) rotation trim.
    date: str  # ISO date the machine table was generated.


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RotationTable:
    devices: dict[str, DeviceRotation]  # Full OAK device ids, never session-dependent estimated factory matrices.


def read_rotations(path: Path | None = None) -> RotationTable:
    """Read the checked-in strict machine table."""
    return read_json(path if path is not None else Path(__file__).with_name("oak_device_rgb_R_imu.json"), RotationTable)


def pool_rotations(free: list[OakFit], gyros: dict[str, GyroFit], generated_date: str) -> RotationTable:
    """Require gyro spread <=1°, hand median <=20 px, and >=50% retained hands."""
    date.fromisoformat(generated_date)
    eligible: dict[str, list[tuple[str, Float64[ndarray, "3 3"]]]] = {}
    for fit in free:
        gyro = gyros.get(fit.session)
        if (
            not fit.ok
            or gyro is None
            or not gyro.ok
            or gyro.max_episode_dev_deg is None
            or gyro.max_episode_dev_deg > 1.0
            or fit.median_px is None
            or fit.median_px > 20.0
            or fit.hands_kept_frac is None
            or fit.hands_kept_frac < 0.5
        ):
            continue
        assert fit.questrig_T_rgb is not None and gyro.R_head_from_imu is not None and fit.oak_device_id is not None
        eligible.setdefault(fit.oak_device_id, []).append((fit.session, fit.questrig_T_rgb[:3, :3].T @ gyro.R_head_from_imu))
    output: dict[str, DeviceRotation] = {}
    for device, entries in eligible.items():
        rotations = Rotation.from_matrix(np.array([r for _, r in entries]))
        deviation = np.degrees((rotations.mean().inv() * rotations).magnitude())
        good = deviation <= max(3.0, 2.5 * float(np.median(deviation)))
        output[device] = DeviceRotation(
            rotations[good].mean().as_matrix(),
            tuple(sorted(s for s, _ in entries)),
            tuple(sorted(s for (s, _), keep in zip(entries, good, strict=True) if keep)),
            generated_date,
        )
    return RotationTable(output)
