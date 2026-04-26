"""Write the ``groundtruth`` layer — time-varying rig pose ``world_T_rig_0``.

What this layer contributes to the composed segment:

- time-varying ``Transform3D`` (``world_T_rig_0``) at ``/world/rig_0`` over
  the ``video_time`` timeline. When ``groundtruth.csv`` is empty (HAMLYN /
  HILTI2026 ship that way), we fall back to a static identity transform so
  child sensor entities still resolve under a parent.
- static ``LineStrips3D`` at ``/world/rig_0_path`` showing the full rig
  trajectory in world frame (one line connecting every GT translation).
  PyCuVSLAM-style: makes "where has the rig been?" obvious without scrubbing
  the timeline. Logged at world level (sibling of ``/world/rig_0``) so the
  rig's time-varying transform doesn't deform the line.

Recording properties on this layer:

- ``groundtruth`` — ``num_poses``, ``trajectory_len_m``, ``duration_s``,
  ``has_rotation`` (false when source GT has all-identity quaternions —
  e.g. VSLAM-LAB-Benchmark's EUROC has zero rotation variance, a known
  upstream conversion bug).

The ``video_time`` epoch (``t0_ns``) is anchored to the first RGB-0 frame
timestamp so all per-frame timelines (GT, depth, IMU) share an origin with
the rgb_0 video stream's PTS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rerun as rr

from slam_evals.data.parse import GroundTruth
from slam_evals.data.types import Sequence
from slam_evals.ingest.columns import log_groundtruth_columns, trajectory_length_m

# Solid colour used for the GT path linestrip (RGBA, 0-255). Green is the
# default convention for ground-truth trajectories — predictions get their
# own colours per algorithm later.
_GT_PATH_COLOR: tuple[int, int, int, int] = (40, 200, 80, 255)


def _has_meaningful_rotation(quaternion_xyzw: np.ndarray, *, tol: float = 1e-6) -> bool:
    """True iff GT quaternions actually vary across frames.

    Sequences like VSLAM-LAB's EUROC have all-zero variance on the
    quaternion columns (their conversion script dropped rotation data);
    we surface that as ``property:groundtruth:has_rotation = False`` so
    consumers can treat such GT as positional-only.
    """
    if quaternion_xyzw.shape[0] < 2:
        return False
    return bool(np.any(np.var(quaternion_xyzw, axis=0) > tol))


def write_groundtruth_layer(
    sequence: Sequence,
    *,
    groundtruth: GroundTruth,
    t0_ns: int,
    out_path: Path,
    application_id: str = "slam-evals",
) -> Path:
    """Write ``groundtruth.rrd`` for ``sequence``. Returns the output path.

    ``t0_ns`` is the timeline epoch (typically ``rgb.csv`` row 0); GT
    timestamps are converted to seconds relative to it.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rec = rr.RecordingStream(
        application_id=application_id,
        recording_id=sequence.recording_id,
        send_properties=True,
    )
    has_gt = groundtruth.ts_ns.shape[0] > 0

    with rec:
        if has_gt:
            # Time-varying world_T_rig_0 at /world/rig_0.
            log_groundtruth_columns(
                groundtruth,
                entity_path="/world/rig_0",
                t0_ns=t0_ns,
                recording=rec,
            )
            duration_s = float((int(groundtruth.ts_ns[-1]) - t0_ns) * 1e-9)

            # PyCuVSLAM-style path visualisation. One LineStrips3D containing
            # all GT translations as a single connected polyline, in world
            # frame. Static — the path itself doesn't change as the timeline
            # advances, only the live frustum (which sits on /world/rig_0).
            rr.log(
                "/world/rig_0_path",
                rr.LineStrips3D(
                    [groundtruth.translation.astype(np.float64)],
                    colors=[_GT_PATH_COLOR],
                    radii=[0.01],
                ),
                static=True,
                recording=rec,
            )
        else:
            # Empty GT — log a static identity so child sensor entities
            # still have a parent pose to inherit.
            rr.log(
                "/world/rig_0",
                rr.Transform3D(translation=[0.0, 0.0, 0.0]),
                static=True,
                recording=rec,
            )
            duration_s = 0.0

        rec.send_property(
            "groundtruth",
            rr.AnyValues(
                num_poses=int(groundtruth.ts_ns.shape[0]),
                trajectory_len_m=float(trajectory_length_m(groundtruth.translation)) if has_gt else 0.0,
                duration_s=duration_s,
                has_rotation=_has_meaningful_rotation(groundtruth.quaternion_xyzw) if has_gt else False,
            ),
        )

    rec.save(str(out_path))
    return out_path
