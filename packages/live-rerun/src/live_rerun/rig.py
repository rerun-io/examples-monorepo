"""Generic, system-agnostic rig schema (COLMAP-style), shared by the logging
core and any sensor backend under ``sources/``.

A **rig** is a set of sensors with fixed relative poses; one **reference
sensor** defines the rig origin (its ``rig_T_sensor`` is identity) and every
other sensor is a fixed offset from it. A future SLAM pass drives the rig pose
(``world_T_rig``) over time; the whole rig then moves rigidly with the reference
sensor.

This mirrors the entity layout in ``packages/slam-evals/docs/schema.md``
(``/world/rig_<i>/cam_<i>``, peer ``/world/rig_<i>/imu_<i>``) so a recording made
here drops into the same mental model. Backends translate their device-specific
calibration into these generic descriptors; the core never learns a vendor name.
See ``packages/live-rerun/docs/rig_schema.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from simplecv.camera_parameters import PinholeParameters

#: Reserved sensor kinds. Only ``rgb``/``grayscale`` (cameras) are emitted today.
#: ``depth``/``imu`` are reserved so a future backend slots a new peer sensor in
#: (e.g. ``/world/rig_00/imu_00``) without a vocabulary change — no IMU exists yet
#: (it crashes the target OAK-D-W; see the README).
SensorKind = Literal["rgb", "grayscale", "depth", "imu"]

#: Zero-padded width for rig/sensor entity indices (``rig_00``, ``cam_00``,
#: ``imu_00``). Two digits so ids sort lexicographically for any realistic rig
#: (``cam_02`` before ``cam_10``); a single digit would sort ``cam_10`` first.
INDEX_WIDTH: int = 2


def entity_id(prefix: str, index: int) -> str:
    """Format a zero-padded rig/sensor entity id, e.g. ``cam_03`` / ``rig_00``."""
    return f"{prefix}_{index:0{INDEX_WIDTH}d}"


@dataclass
class CameraSensor:
    """One camera in a rig.

    ``pinhole`` carries intrinsics plus extrinsics expressed **relative to the
    rig frame** — i.e. ``rig_T_cam`` — so the reference sensor's extrinsics are
    identity. ``index`` selects the entity path ``cam_<NN>`` (zero-padded via
    :func:`entity_id`, e.g. ``cam_00``); ``name`` is the human role label (e.g.
    ``"left"``) logged as metadata; ``kind`` is the image content
    (``"rgb"`` / ``"grayscale"``).
    """

    index: int
    name: str
    kind: SensorKind
    pinhole: PinholeParameters


@dataclass
class RigCalibration:
    """A rig's cameras plus which one is the reference (identity) sensor.

    ``reference_index`` is the ``index`` of the camera whose ``rig_T_cam`` is
    identity (the rig origin); it gets a distinct frustum colour in the viewer
    and is recorded in the rig's schema metadata.
    """

    cameras: list[CameraSensor] = field(default_factory=list)
    reference_index: int = 0
