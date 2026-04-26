"""Write a ``imu_<i>`` layer — gyro + accel scalar streams.

What this layer contributes to the composed segment:

- 3-component ``Scalars`` at ``/world/rig_0/imu_<i>/gyro`` (rad/s)
- 3-component ``Scalars`` at ``/world/rig_0/imu_<i>/accel`` (m/s²)

both over the ``video_time`` timeline.

Recording properties on this layer:

- ``imu_<i>`` — num_samples, rate_hz (median), plus pass-through noise terms
  (``a_max``, ``g_max``, ``sigma_g_c``, ``sigma_a_c``, ``sigma_bg_c``,
  ``sigma_ba_c``) when present in ``calibration.yaml``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rerun as rr

from slam_evals.data.parse import ImuSamples
from slam_evals.data.types import Calibration, Sequence
from slam_evals.ingest.columns import log_imu_columns

_NOISE_PARAM_KEYS: tuple[str, ...] = (
    "a_max", "g_max", "sigma_g_c", "sigma_a_c", "sigma_bg_c", "sigma_ba_c", "fps",
)


def _imu_rate_hz(imu: ImuSamples) -> float:
    if imu.ts_ns.shape[0] < 2:
        return -1.0
    dt_ns = float(np.median(np.diff(imu.ts_ns)))
    return 1e9 / dt_ns if dt_ns > 0 else -1.0


def _noise_props(calibration: Calibration | None) -> dict[str, float]:
    """Pull pass-through scalar noise terms from calibration.imu_params."""
    if calibration is None or calibration.imu_params is None:
        return {}
    out: dict[str, float] = {}
    for key in _NOISE_PARAM_KEYS:
        val = calibration.imu_params.get(key)
        if isinstance(val, (int, float)):
            out[key] = float(val)
    return out


def write_imu_layer(
    sequence: Sequence,
    *,
    imu_idx: int,
    imu: ImuSamples,
    calibration: Calibration | None,
    t0_ns: int,
    out_path: Path,
    application_id: str = "slam-evals",
) -> Path:
    """Write ``imu_<imu_idx>.rrd`` for ``sequence``. Returns the output path.

    ``t0_ns`` is the timeline epoch (typically rgb.csv row 0); IMU
    timestamps are converted to seconds relative to it so the gyro/accel
    streams share an origin with the video.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rec = rr.RecordingStream(
        application_id=application_id,
        recording_id=sequence.recording_id,
        send_properties=True,
    )
    with rec:
        log_imu_columns(
            imu,
            gyro_path=f"/world/rig_0/imu_{imu_idx}/gyro",
            accel_path=f"/world/rig_0/imu_{imu_idx}/accel",
            t0_ns=t0_ns,
            recording=rec,
        )

        props = {
            "num_samples": int(imu.ts_ns.shape[0]),
            "rate_hz": _imu_rate_hz(imu),
        }
        props.update(_noise_props(calibration))
        rec.send_property(f"imu_{imu_idx}", rr.AnyValues(**props))

    rec.save(str(out_path))
    return out_path
