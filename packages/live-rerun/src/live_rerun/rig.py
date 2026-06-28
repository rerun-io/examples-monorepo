"""Re-export of the shared rig schema, now owned by ``simplecv``.

The generic COLMAP-style rig types (``SensorKind``, ``CameraSensor``,
``RigCalibration``, plus the exoego-level ``Rig`` / ``RigPoseStream`` and the
``entity_id`` / ``INDEX_WIDTH`` helpers) were promoted into :mod:`simplecv.rig`
so they can be shared by the exoego writer and any other system. ``live-rerun`` already depends
on ``simplecv`` (it imports ``simplecv.camera_parameters`` and
``simplecv.rerun_log_utils``), so this is a pure re-export with no new dependency
edge and zero churn at the existing call sites. See
``packages/live-rerun/docs/rig_schema.md`` and ``packages/simplecv/simplecv/rig.py``.
"""

from __future__ import annotations

from simplecv.rig import (  # noqa: F401
    INDEX_WIDTH,
    CameraSensor,
    Rig,
    RigCalibration,
    RigPoseStream,
    SensorKind,
    entity_id,
)

__all__ = [
    "INDEX_WIDTH",
    "CameraSensor",
    "Rig",
    "RigCalibration",
    "RigPoseStream",
    "SensorKind",
    "entity_id",
]
