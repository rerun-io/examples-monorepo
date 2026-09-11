"""What the replay tool's own input rung writes: the inertial samples.

The estimator's rung is ``test_vio_log.py``'s and the frontend's is
``test_frontend_log.py``'s; what neither covers is the input every stage logs
whether or not it consumes it. The samples used to go in one row at a time —
``set_time`` plus two ``log`` calls each, about 171 calls a frameset — and they
now go in two columns. That is a rewrite of how the rows are written, so what is
asserted here is that they are the same rows.

The recording reader is a :mod:`conftest` fixture, which pytest injects; the
alias below is declared here rather than imported from it, because ``tests`` is
not on the typechecker's search path and every module in this directory stands
alone.
"""

from pathlib import Path

import numpy as np
import rerun as rr
from fixture_types import IMU_PERIOD_NS, Rows, RowsReader
from jaxtyping import Float64, Int64
from numpy import ndarray

from slam_rs.catalog_feed import IMU_ENTITY, TIMELINE, ImuStream
from slam_rs.vio_log import log_imu

SAMPLES: int = 33
"""One 30 Hz frameset's worth of samples."""


def stream() -> ImuStream:
    """A frameset's samples, every component different so a transposition shows."""
    t_ns: Int64[ndarray, " n_samples"] = np.arange(SAMPLES, dtype=np.int64) * IMU_PERIOD_NS
    values: Float64[ndarray, "n_samples 3"] = np.arange(3 * SAMPLES, dtype=np.float64).reshape(SAMPLES, 3)
    return ImuStream(t_ns=t_ns, gyro_rad_s=values, accel_m_s2=-values)


def written(output: Path, read_rows: RowsReader, columnar: bool) -> Rows:
    """Log one frameset's samples into a fresh recording, the new way or the old."""
    imu: ImuStream = stream()
    rr.init("slam-rs-replay-log-test", recording_id=f"replay-log-{output.stem}")
    rr.save(output)
    if columnar:
        log_imu(imu)
    else:
        for sample in range(len(imu)):
            rr.set_time(TIMELINE, duration=np.timedelta64(int(imu.t_ns[sample]), "ns"))
            rr.log(f"{IMU_ENTITY}/gyro", rr.Scalars(imu.gyro_rad_s[sample]))
            rr.log(f"{IMU_ENTITY}/accel", rr.Scalars(imu.accel_m_s2[sample]))
    rr.disconnect()
    return read_rows(output)


def test_the_columns_write_the_rows_the_per_sample_loop_wrote(tmp_path: Path, read_rows: RowsReader) -> None:
    """Same entities, same timestamps, same three components a row."""
    columns: Rows = written(tmp_path / "columns.rrd", read_rows, columnar=True)
    loop: Rows = written(tmp_path / "loop.rrd", read_rows, columnar=False)
    assert set(columns) == {f"{IMU_ENTITY}/gyro", f"{IMU_ENTITY}/accel"} == set(loop)
    for entity, rows in columns.items():
        assert rows == loop[entity], entity
        assert len(rows) == SAMPLES
        assert all(len(row.values["Scalars:scalars"]) == 3 for row in rows)


def test_a_frameset_without_samples_logs_nothing(tmp_path: Path, read_rows: RowsReader) -> None:
    """An empty column is refused by the SDK, and there is nothing to draw anyway."""
    output: Path = tmp_path / "empty.rrd"
    rr.init("slam-rs-replay-log-test", recording_id="replay-log-empty")
    rr.save(output)
    log_imu(ImuStream(t_ns=np.zeros(0, dtype=np.int64), gyro_rad_s=np.zeros((0, 3)), accel_m_s2=np.zeros((0, 3))))
    rr.disconnect()
    assert read_rows(output) == {}


