"""The global BA's layer (PR d2): GBA keyframes under the run beside the raw VIO, its attempts on their own timeline, and its trajectory."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.chunk as rrc
from fixture_types import RigFactory, Rows, RowsReader, TextureFactory
from jaxtyping import Float64, UInt8
from numpy import ndarray

from slam_rs import _core
from slam_rs.catalog_feed import TIMELINE
from slam_rs.columns import ColumnLog
from slam_rs.gba_log import GBA_KEYFRAMES_ENTITY, GBA_PATH_ENTITY, GBA_STATS_ENTITY, GBA_TIMELINE, keyframe_trajectory, log_map_result
from slam_rs.trajectory import Trajectory
from slam_rs.vio_log import RUN_ENTITY


def _finished(rig: RigFactory, texture: TextureFactory, run_gba: bool) -> _core.MapResult:
    """Twelve framesets of a still stereo pair with the map on (Offline), then ``finish_map``."""
    vio: _core.Vio = _core.Vio(rig(2), _core.VioConfig().with_backend("offline"), backend="cpu")
    for t_ns in range(0, 261_000_000, 1_000_000):
        vio.push_imu(t_ns, [0.0, 0.0, 0.0], [0.0, 0.0, 9.81])
    images: list[UInt8[ndarray, "h w"]] = [texture(0, 0), texture(-2, 0)]
    for index in range(12):
        vio.track(index * 20_000_000, images)
    vio.flush()
    result: _core.MapResult | None = vio.finish_map(run_gba=run_gba)
    assert result is not None
    return result


def test_the_keyframe_trajectory_is_the_map_before_or_after_the_gba(rig: RigFactory, texture: TextureFactory) -> None:
    result: _core.MapResult = _finished(rig, texture, run_gba=True)
    after: Trajectory = keyframe_trajectory(result, after_gba=True)
    before: Trajectory = keyframe_trajectory(result, after_gba=False)
    np.testing.assert_array_equal(after.t_ns, result.keyframe_t_ns)
    np.testing.assert_array_equal(after.position_m, result.keyframe_poses[:, 0:3])
    np.testing.assert_array_equal(before.position_m, result.keyframe_poses_before_gba[:, 0:3])
    # w-first, as :mod:`slam_rs.trajectory` stores rotations.
    np.testing.assert_array_equal(after.quaternion_wxyz[:, 0], result.keyframe_poses[:, 6])
    np.testing.assert_array_equal(after.quaternion_wxyz[:, 1:], result.keyframe_poses[:, 3:6])


def test_the_gba_keyframes_go_under_the_run_and_its_attempts_on_their_timeline(
    rig: RigFactory, texture: TextureFactory, tmp_path: Path, read_rows: RowsReader
) -> None:
    assert GBA_KEYFRAMES_ENTITY.startswith(f"{RUN_ENTITY}/") and GBA_PATH_ENTITY.startswith(f"{RUN_ENTITY}/")
    result: _core.MapResult = _finished(rig, texture, run_gba=True)
    gba: _core.GbaRun | None = result.gba
    assert gba is not None
    output: Path = tmp_path / "gba.rrd"
    with rr.RecordingStream("slam-rs-gba-log-test", recording_id="gba-log") as recording:
        recording.save(output)
        log_map_result(result, ColumnLog(recording=recording), 220_000_000)
    rows: Rows = read_rows(output)
    assert [row.t_ns for row in rows[GBA_KEYFRAMES_ENTITY]] == [220_000_000]
    drawn: Float64[ndarray, "k 3"] = np.asarray(rows[GBA_KEYFRAMES_ENTITY][0].values["Points3D:positions"], dtype=np.float64).reshape(-1, 3)
    np.testing.assert_allclose(drawn, result.keyframe_poses[:, 0:3], rtol=1e-6, atol=1e-6)
    assert [row.t_ns for row in rows[GBA_PATH_ENTITY]] == [220_000_000]
    costs: list[float] = []
    for chunk in rrc.RrdReader(output).stream().collect().stream():
        if chunk.entity_path == f"{GBA_STATS_ENTITY}/cost":
            batch: pa.RecordBatch = chunk.to_record_batch()
            assert GBA_TIMELINE in batch.schema.names
            costs += [value for row in batch.column("Scalars:scalars").to_pylist() for value in row]
    np.testing.assert_allclose(costs, gba.iteration_cost)


def test_nothing_is_drawn_without_the_gba(rig: RigFactory, texture: TextureFactory, tmp_path: Path, read_rows: RowsReader) -> None:
    result: _core.MapResult = _finished(rig, texture, run_gba=False)
    output: Path = tmp_path / "plain.rrd"
    with rr.RecordingStream("slam-rs-gba-log-test", recording_id="gba-log-plain") as recording:
        recording.save(output)
        log_map_result(result, ColumnLog(recording=recording), 0)
        recording.set_time(TIMELINE, duration=np.timedelta64(0, "ns"))
        recording.log("/marker", rr.Points3D([[0.0, 0.0, 0.0]]))
    rows: Rows = read_rows(output)
    assert GBA_KEYFRAMES_ENTITY not in rows and GBA_PATH_ENTITY not in rows
