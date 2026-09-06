"""dataforge:v1 logging schema — the exoego:v2 conventions, as code. Stdlib-only.

The authoritative prose spec is ``packages/simplecv/docs/exoego_schema.md``:
one ``video_time`` timestamp timeline everywhere, world-anchored rigs at
``/world/rig_NN``, cameras at ``.../cam_MM/pinhole/video``, and IMUs at the
(previously reserved) ``.../imu_MM/{gyro,accel}``. dataforge is the first
emitter of the IMU section (§8) and of the magnetometer section (§9), whose
``.../mag_MM/{field,heading}`` is the same peer-sensor shape.
"""

from __future__ import annotations

TIMELINE: str = "video_time"
"""The single timestamp timeline every dataforge stream logs on."""

EXOEGO_SCHEMA_VERSION: str = "exoego:v2"
"""Value of the ``schema_version`` AnyValue on every rig node."""

DATAFORGE_SCHEMA_VERSION: str = "dataforge:v1"
"""Value of the ``property:capture:schema`` recording property."""

GT_RUN_SOURCE: str = "gt"
"""Run source of the ground-truth trajectory and trail under ``/world/runs/``.

Deliberately equal to ``paths.GT_LAYER``: the layer a converter writes and the
run source it logs under name the same thing to a reader, even though one is a
directory and the other an entity path component.
"""


def _index(value: int, kind: str) -> str:
    if value < 0:
        raise ValueError(f"{kind} index must be non-negative, got {value}")
    return f"{value:02d}"


def rig_path(rig: int) -> str:
    """``/world/rig_NN`` — a static exo rig or the moving ego rig."""
    return f"/world/rig_{_index(rig, 'rig')}"


def cam_path(rig: int, cam: int) -> str:
    """``/world/rig_NN/cam_MM`` — carries the static ``rig_T_cam`` transform."""
    return f"{rig_path(rig)}/cam_{_index(cam, 'cam')}"


def pinhole_path(rig: int, cam: int) -> str:
    """``.../cam_MM/pinhole`` — the (distorted) pinhole projection node."""
    return f"{cam_path(rig, cam)}/pinhole"


def video_path(rig: int, cam: int) -> str:
    """``.../pinhole/video`` — the VideoStream entity on ``video_time``."""
    return f"{pinhole_path(rig, cam)}/video"


def imu_path(rig: int, imu: int) -> str:
    """``/world/rig_NN/imu_MM`` — carries the static ``rig_T_imu`` transform."""
    return f"{rig_path(rig)}/imu_{_index(imu, 'imu')}"


def gyro_path(rig: int, imu: int) -> str:
    """``.../imu_MM/gyro`` — rad/s Scalars on ``video_time``."""
    return f"{imu_path(rig, imu)}/gyro"


def accel_path(rig: int, imu: int) -> str:
    """``.../imu_MM/accel`` — m/s^2 Scalars on ``video_time``."""
    return f"{imu_path(rig, imu)}/accel"


def mag_path(rig: int, mag: int) -> str:
    """``/world/rig_NN/mag_MM`` — carries the static ``rig_T_mag`` transform."""
    return f"{rig_path(rig)}/mag_{_index(mag, 'mag')}"


def field_path(rig: int, mag: int) -> str:
    """``.../mag_MM/field`` — 3-component Scalars in the sensor's native units."""
    return f"{mag_path(rig, mag)}/field"


def heading_path(rig: int, mag: int) -> str:
    """``.../mag_MM/heading`` — the field direction as a fixed-length Arrows3D."""
    return f"{mag_path(rig, mag)}/heading"


def control_points_path() -> str:
    """``/world/gt/control_points`` — a sequence's surveyed points, in the world frame."""
    return "/world/gt/control_points"


def cp_uv_path(rig: int, cam: int) -> str:
    """``.../cam_MM/pinhole/cp_uv`` — control-point detections in that camera's image."""
    return f"{pinhole_path(rig, cam)}/cp_uv"


def run_path(source: str) -> str:
    """``/world/runs/<source>`` — derived outputs from one processing source."""
    return f"/world/runs/{source}"


def trajectory_path(source: str) -> str:
    """``.../runs/<source>/trajectory`` — the source's full rig trajectory."""
    return f"{run_path(source)}/trajectory"


def trail_path(source: str) -> str:
    """``.../runs/<source>/trail`` — the source's cursor-relative trajectory trail."""
    return f"{run_path(source)}/trail"


def capture_property(name: str) -> str:
    """``property:capture:<name>`` — recording-level capture metadata key."""
    return f"property:capture:{name}"
