"""Write the ``groundtruth`` layer — time-varying rig pose ``world_T_rig_0``.

What this layer contributes to the composed segment:

- time-varying ``Transform3D`` (``world_T_rig_0``) at ``/world/rig_0`` over
  the ``video_time`` timeline. When ``groundtruth.csv`` is empty (HAMLYN /
  HILTI2026 ship that way), we fall back to a static identity transform so
  child sensor entities still resolve under a parent.

Recording properties on this layer:

- ``groundtruth`` — ``num_poses``, ``trajectory_len_m``, ``duration_s``.

The ``video_time`` epoch (``t0_ns``) is anchored to the first RGB-0 frame
timestamp so all per-frame timelines (GT, depth, IMU) share an origin with
the rgb_0 video stream's PTS.
"""

from __future__ import annotations

from pathlib import Path

import rerun as rr

from slam_evals.data.parse import GroundTruth
from slam_evals.data.types import Sequence
from slam_evals.ingest.columns import log_groundtruth_columns, trajectory_length_m


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
    with rec:
        if groundtruth.ts_ns.shape[0] > 0:
            # Time-varying world_T_rig_0 at /world/rig_0.
            log_groundtruth_columns(
                groundtruth,
                entity_path="/world/rig_0",
                t0_ns=t0_ns,
                recording=rec,
            )
            duration_s = float((int(groundtruth.ts_ns[-1]) - t0_ns) * 1e-9)
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
                trajectory_len_m=float(trajectory_length_m(groundtruth.translation)) if groundtruth.ts_ns.shape[0] > 0 else 0.0,
                duration_s=duration_s,
            ),
        )

    rec.save(str(out_path))
    return out_path


