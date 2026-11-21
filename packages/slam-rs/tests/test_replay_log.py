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
from fixture_types import FRAME_PERIOD_NS, IMU_PERIOD_NS, CameraFactory, PipelineFactory, Rows, RowsReader, TextureFactory, gravity_batch
from jaxtyping import Float64, Int64
from numpy import ndarray

from slam_rs import _core
from slam_rs.catalog_feed import IMU_ENTITY, TIMELINE, CameraCalib, Frameset, ImuStream
from slam_rs.frontend_log import camera_entity
from slam_rs.tracking import Lockstep
from slam_rs.trajectory import empty_trajectory
from slam_rs.vio_log import VIO_RUN_ENTITY, VIO_STATS_ENTITY, VioLogger, VioStage, log_frameset_inputs, log_imu

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
        log_imu(imu, 0)
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
    log_imu(ImuStream(t_ns=np.zeros(0, dtype=np.int64), gyro_rad_s=np.zeros((0, 3)), accel_m_s2=np.zeros((0, 3))), 0)
    rr.disconnect()
    assert read_rows(output) == {}


def test_the_masks_a_frameset_was_tracked_with_are_logged_with_their_scale(tmp_path: Path, read_rows: RowsReader) -> None:
    """In the mask layer's own form, under the layer's own name: native resolution plus a static scale onto the camera's
    image, logged once; a mask is logged again only when it changes, and every nonzero label draws as masked."""
    from rerun.chunk import RrdReader

    from slam_rs.vio_log import FrameMaskLog

    output: Path = tmp_path / "masks.rrd"
    rr.init("slam-rs-replay-log-test", recording_id="replay-log-masks")
    rr.save(output)
    first = np.zeros((4, 5), dtype=np.uint8)
    first[1, 2] = 1
    second = np.zeros((4, 5), dtype=np.uint8)
    second[2, 3] = 2
    mask_log = FrameMaskLog("face_mask")
    for t_ns, mask in [(1_000, first), (2_000, first.copy()), (3_000, second)]:
        rr.set_time(TIMELINE, duration=np.timedelta64(t_ns, "ns"))
        mask_log.log((0, 1), [(20, 12), (20, 12)], [mask, None])
    rr.disconnect()

    entity: str = f"{camera_entity(0)}/face_mask"
    rows: Rows = read_rows(output)
    assert set(rows) == {entity}
    assert len(rows[entity]) == 2
    statics = [chunk.to_record_batch() for chunk in RrdReader(output).stream().collect().stream() if chunk.is_static and chunk.entity_path == entity]
    assert [batch.column("Transform3D:scale").to_pylist() for batch in statics] == [[[[4.0, 3.0, 1.0]]]]  # one chunk, one row, one instance
    (context,) = [batch.column("AnnotationContext:context").to_pylist() for batch in statics]
    classes: dict[int, dict] = {entry["class_id"]: entry["class_description"]["info"] for entry in context[0][0]}  # one row, one instance
    assert classes[2]["label"] == classes[1]["label"] == "masked" and classes[2]["color"] == classes[1]["color"]


def test_every_replay_row_sits_on_its_base_video_frame(
    pipeline: PipelineFactory, camera: CameraFactory, texture: TextureFactory, tmp_path: Path, read_rows: RowsReader
) -> None:
    """With a camera-to-IMU offset (RoboCap's 14.9 ms) the estimator's stamps are ``video_time`` + offset; the replay's
    images, keypoints, run rig, counters and IMU samples still land on the base video's own stamps."""
    offset_ns: int = 14_902_432
    base_video_ns: Int64[ndarray, " n"] = np.arange(12, dtype=np.int64) * FRAME_PERIOD_NS
    cameras: tuple[CameraCalib, ...] = (camera(0, 0.0), camera(1, 0.1))
    output: Path = tmp_path / "offset.rrd"
    rr.init("slam-rs-replay-log-test", recording_id="replay-log-offset")
    rr.save(output)
    vio: _core.Vio = pipeline(2)
    stage: VioStage = VioStage(
        Lockstep(vio),
        VioLogger(cameras=cameras, ground_truth=empty_trajectory(), frame_t_ns=base_video_ns + offset_ns,
                  tracked_resolution=tuple(vio.tracked_resolution), cam_time_offset_ns=offset_ns),
    )
    imu_ns: list[int] = []
    for step, video_ns in enumerate(base_video_ns.tolist()):
        t_ns: int = video_ns + offset_ns
        samples: Int64[ndarray, " n_samples"] = np.arange(t_ns, t_ns + FRAME_PERIOD_NS, IMU_PERIOD_NS, dtype=np.int64)
        imu_ns.extend(samples.tolist())
        gyro, accel = gravity_batch(samples)
        frameset: Frameset = Frameset(t_ns=t_ns, images=[texture(step, 0), texture(step - 2, 0)],
                                      imu=ImuStream(t_ns=samples, gyro_rad_s=gyro, accel_m_s2=accel), ground_truth=None)
        log_frameset_inputs(cameras, offset_ns, frameset, "jpeg")
        stage.run(frameset)
    stage.flush()
    rr.disconnect()
    rows: Rows = read_rows(output)

    base: set[int] = set(base_video_ns.tolist())
    assert {row.t_ns for row in rows[f"{camera_entity(0)}/image"]} == base
    for entity in (f"{camera_entity(0)}/keypoints", f"{VIO_RUN_ENTITY}/rig", f"{VIO_STATS_ENTITY}/track_ms"):
        stamps: set[int] = {row.t_ns for row in rows[entity]}
        assert stamps and stamps <= base, entity
    assert [row.t_ns for row in rows[f"{IMU_ENTITY}/gyro"]] == [t_ns - offset_ns for t_ns in imu_ns]
