"""Bulk (``rr.send_columns``) logging for GT trajectory + IMU timeseries.

All timelines are ``duration`` in seconds so they line up with the video's
``ts`` timeline that the PyAV encoder emits.
"""

from __future__ import annotations

import numpy as np
import rerun as rr
from jaxtyping import Float64
from numpy import ndarray

from slam_evals.data.parse import GroundTruth, ImuSamples


def log_groundtruth_columns(
    gt: GroundTruth,
    *,
    entity_path: str = "/world/body",
    timeline: str = "video_time",
    t0_ns: int | None = None,
    recording: rr.RecordingStream | None = None,
) -> None:
    """Log GT as a time-varying ``Transform3D`` stream on the body entity.

    VSLAM-LAB's groundtruth gives ``(tx, ty, tz, qx, qy, qz, qw)`` as
    ``world_T_body`` — the body's pose expressed in the world frame — which
    maps to a default (``from_parent=False``) ``rr.Transform3D`` at
    ``/world/body``. Sensor entities under ``/world/body`` (``cam_0``,
    ``cam_1``) keep their own static body-from-sensor extrinsics and ride
    this trajectory automatically via the scene graph.
    """
    anchor = int(t0_ns) if t0_ns is not None else int(gt.ts_ns[0])
    t_rel_s: Float64[ndarray, "n"] = (gt.ts_ns - anchor).astype(np.float64) * 1e-9

    rr.send_columns(
        entity_path,
        indexes=[rr.TimeColumn(timeline, duration=t_rel_s)],
        columns=rr.Transform3D.columns(
            translation=gt.translation.astype(np.float64),
            quaternion=gt.quaternion_xyzw.astype(np.float64),
        ),
        recording=recording,
    )


def log_imu_columns(
    imu: ImuSamples,
    *,
    gyro_path: str = "/imu/gyro",
    accel_path: str = "/imu/accel",
    timeline: str = "video_time",
    t0_ns: int | None = None,
    recording: rr.RecordingStream | None = None,
) -> None:
    """Log gyro + accel as 3-component ``Scalars`` streams.

    ``t0_ns`` anchors the timeline relative to a common epoch (typically the
    first RGB frame's timestamp) so video + IMU share an origin.
    """
    anchor = int(t0_ns) if t0_ns is not None else int(imu.ts_ns[0])
    t_rel_s: Float64[ndarray, "n"] = (imu.ts_ns - anchor).astype(np.float64) * 1e-9
    idx: list[rr.TimeColumn] = [rr.TimeColumn(timeline, duration=t_rel_s)]

    rr.send_columns(
        gyro_path,
        indexes=idx,
        columns=rr.Scalars.columns(scalars=imu.gyro_rads.astype(np.float64)),
        recording=recording,
    )
    rr.send_columns(
        accel_path,
        indexes=idx,
        columns=rr.Scalars.columns(scalars=imu.accel_ms2.astype(np.float64)),
        recording=recording,
    )


def trajectory_length_m(translation_xyz: Float64[ndarray, "n 3"]) -> float:
    """Sum of adjacent-pose translation norms. Used in recording properties."""
    if translation_xyz.shape[0] < 2:
        return 0.0
    diffs: Float64[ndarray, "n 3"] = np.diff(translation_xyz, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())
