"""Every row a run's layer holds sits on its base video frame, whatever the camera-to-IMU offset.

The recording reader is a :mod:`conftest` fixture, which pytest injects; the
alias below is declared here rather than imported from it, because ``tests`` is
not on the typechecker's search path and every module in this directory stands
alone.
"""

from pathlib import Path

import numpy as np
import rerun as rr
from fixture_types import FRAME_PERIOD_NS, IMU_PERIOD_NS, CameraFactory, PipelineFactory, Rows, RowsReader, TextureFactory, gravity_batch
from jaxtyping import Int64
from numpy import ndarray

from slam_rs import _core
from slam_rs.catalog_feed import CameraCalib, Frameset, ImuStream
from slam_rs.columns import ColumnLog
from slam_rs.frontend_log import camera_entity
from slam_rs.tracking import Lockstep
from slam_rs.trajectory import empty_trajectory
from slam_rs.vio_log import VIO_RUN_ENTITY, VIO_STATS_ENTITY, VioLogger, VioStage


def test_every_layer_row_sits_on_its_base_video_frame(
    pipeline: PipelineFactory, camera: CameraFactory, texture: TextureFactory, tmp_path: Path, read_rows: RowsReader
) -> None:
    """With a camera-to-IMU offset (RoboCap's 14.9 ms) the estimator's stamps are ``video_time`` + offset; the keypoints,
    the window and the counters still land on the base video's own stamps."""
    offset_ns: int = 14_902_432
    base_video_ns: Int64[ndarray, " n"] = np.arange(12, dtype=np.int64) * FRAME_PERIOD_NS
    cameras: tuple[CameraCalib, ...] = (camera(0, 0.0), camera(1, 0.1))
    output: Path = tmp_path / "offset.rrd"
    with rr.RecordingStream("slam-rs-replay-log-test", recording_id="replay-log-offset") as recording:
        recording.save(output)
        vio: _core.Vio = pipeline(2)
        stage: VioStage = VioStage(
            Lockstep(vio),
            VioLogger(cameras=cameras, ground_truth=empty_trajectory(), frame_t_ns=base_video_ns + offset_ns,
                      tracked_resolution=tuple(vio.tracked_resolution), cam_time_offset_ns=offset_ns, columns=ColumnLog(recording=recording)),
        )
        for step, video_ns in enumerate(base_video_ns.tolist()):
            t_ns: int = video_ns + offset_ns
            samples: Int64[ndarray, " n_samples"] = np.arange(t_ns, t_ns + FRAME_PERIOD_NS, IMU_PERIOD_NS, dtype=np.int64)
            gyro, accel = gravity_batch(samples)
            stage.run(Frameset(t_ns=t_ns, images=[texture(step, 0), texture(step - 2, 0)],
                               imu=ImuStream(t_ns=samples, gyro_rad_s=gyro, accel_m_s2=accel), ground_truth=None))
        stage.flush()
    rows: Rows = read_rows(output)
    base: set[int] = set(base_video_ns.tolist())
    for entity in (f"{camera_entity(0)}/keypoints", f"{VIO_RUN_ENTITY}/window/00", f"{VIO_RUN_ENTITY}/landmarks", f"{VIO_STATS_ENTITY}/track_ms"):
        stamps: set[int] = {row.t_ns for row in rows[entity]}
        assert stamps and stamps <= base, entity
