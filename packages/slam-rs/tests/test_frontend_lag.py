"""Lagged result timestamps and draining through the Python boundary."""

import json
from pathlib import Path

import numpy as np
import pytest
import rerun as rr
from fixture_types import FRAME, CameraFactory, RigFactory, Rows, RowsReader, TextureFactory, gravity_batch
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray

from slam_rs import _core
from slam_rs.catalog_feed import CameraCalib, Frameset, ImuStream
from slam_rs.frontend_log import camera_entity
from slam_rs.tracking import Lockstep
from slam_rs.trajectory import empty_trajectory
from slam_rs.vio_log import VioLogger, VioStage, log_calibration, vio_blueprint


def still_frameset(index: int, images: list[UInt8[ndarray, "h w"]]) -> Frameset:
    """One stereo frameset with the following 20 ms of gravity samples."""
    times: Int64[ndarray, "n"] = np.arange(index * 20_000_000, (index + 1) * 20_000_000, 1_000_000, dtype=np.int64)
    batch: tuple[Float64[ndarray, "n 3"], Float64[ndarray, "n 3"]] = gravity_batch(times)
    return Frameset(t_ns=index * 20_000_000, images=images,
                    imu=ImuStream(t_ns=times, gyro_rad_s=batch[0], accel_m_s2=batch[1]), ground_truth=None)


@pytest.mark.parametrize("lag", [False, True])
def test_blank_start_has_no_snapshot_and_stereo_starts_the_world(rig: RigFactory, texture: TextureFactory, lag: bool) -> None:
    config: _core.VioConfig = _core.VioConfig.from_json(json.dumps({"value0": {"port.frontend_lag": lag}}))
    vio: _core.Vio = _core.Vio(rig(2), config)
    blank: UInt8[ndarray, "h w"] = np.zeros((FRAME, FRAME), dtype=np.uint8)
    for t_ns in range(0, 181_000_000, 1_000_000):
        vio.push_imu(t_ns, [0.0, 0.0, 0.0], [0.0, 0.0, 9.81])
    results: list[_core.VioResult] = []
    for index in range(8):
        images: list[UInt8[ndarray, "h w"]] = [blank, blank] if index < 3 else [texture(0, 0), texture(-2, 0)]
        result: _core.VioResult = vio.track(index * 20_000_000, images)
        if result.status != _core.VioStatus.Tracking:
            assert result.world_from_rig is None
            assert result.velocity is None
            assert result.gyro_bias is None
            assert result.accel_bias is None
        if result.status != _core.VioStatus.Buffered:
            results.append(result)
        if index < 3:
            assert vio.snapshot() is None
            assert vio.flow_frame() is not None
    final: _core.VioResult | None = vio.flush()
    if final is not None:
        results.append(final)
    assert vio.flush() is None
    assert [result.t_ns for result in results] == [index * 20_000_000 for index in range(8)]
    assert [result.status for result in results] == [_core.VioStatus.NoVisualFeatures] * 3 + [_core.VioStatus.Tracking] * 5
    assert all(result.world_from_rig is not None for result in results[3:])
    assert vio.snapshot() is not None


@pytest.mark.parametrize("lag", [False, True])
@pytest.mark.parametrize("blank_frames", [3, 8])
def test_lockstep_publishes_no_pose_until_stereo_starts(rig: RigFactory, texture: TextureFactory, lag: bool, blank_frames: int) -> None:
    config: _core.VioConfig = _core.VioConfig.from_json(json.dumps({"value0": {"port.frontend_lag": lag}}))
    lockstep: Lockstep = Lockstep(_core.Vio(rig(2), config))
    blank: UInt8[ndarray, "h w"] = np.zeros((FRAME, FRAME), dtype=np.uint8)
    timestamps: list[int] = []
    for index in range(8):
        frameset: Frameset = still_frameset(index, [blank, blank] if index < blank_frames else [texture(0, 0), texture(-2, 0)])
        timestamps.extend(result.t_ns for _, result in lockstep.push(frameset))
    timestamps.extend(result.t_ns for _, result in lockstep.flush())
    assert timestamps == [index * 20_000_000 for index in range(blank_frames, 8)]
    assert not lockstep.pending
    assert lockstep.retries == 0
    assert len(lockstep.elapsed_ms) == 8 - blank_frames
    assert list(lockstep.flush()) == []


def test_lagged_results_and_final_drain(rig: RigFactory, texture: TextureFactory) -> None:
    config: _core.VioConfig = _core.VioConfig.from_json(json.dumps({"value0": {"port.frontend_lag": True}}))
    vio: _core.Vio = _core.Vio(rig(2), config)
    assert vio.flush() is None
    for t_ns in range(0, 61_000_000, 1_000_000):
        vio.push_imu(t_ns, [0.0, 0.0, 0.0], [0.0, 0.0, 9.81])
    timestamps: list[int] = []
    for index, t_ns in enumerate([0, 20_000_000, 40_000_000]):
        result: _core.VioResult = vio.track(t_ns, [texture(index, 0), texture(index - 2, 0)])
        if index == 0:
            assert result.status == _core.VioStatus.Buffered
            assert vio.snapshot() is None
        else:
            assert result.status == _core.VioStatus.Tracking
            timestamps.append(result.t_ns)
    final: _core.VioResult | None = vio.flush()
    assert final is not None
    assert final.status == _core.VioStatus.Tracking
    timestamps.append(final.t_ns)
    assert timestamps == [0, 20_000_000, 40_000_000]
    assert vio.flush() is None


def test_lockstep_pairs_delayed_results_with_their_images_and_keypoints(rig: RigFactory, texture: TextureFactory) -> None:
    config: _core.VioConfig = _core.VioConfig.from_json(json.dumps({"value0": {"port.frontend_lag": True}}))
    lockstep: Lockstep = Lockstep(_core.Vio(rig(2), config))
    timestamps: list[int] = []
    for index in range(3):
        frameset: Frameset = still_frameset(index, [texture(index, 0), texture(index - 2, 0)])
        for held, result in lockstep.push(frameset):
            assert held.t_ns == result.t_ns
            assert lockstep.flow_frame().t_ns == result.t_ns
            timestamps.append(result.t_ns)
    for held, result in lockstep.flush():
        assert held.t_ns == result.t_ns
        assert lockstep.flow_frame().t_ns == result.t_ns
        timestamps.append(result.t_ns)
    assert timestamps == [0, 20_000_000, 40_000_000]
    assert len(lockstep.elapsed_ms) == 3
    assert lockstep.retries == 0
    assert not lockstep.pending


def test_lagged_viewer_layer_keeps_all_frame_timestamps(
    rig: RigFactory, texture: TextureFactory, camera: CameraFactory, read_rows: RowsReader, tmp_path: Path
) -> None:
    output: Path = tmp_path / "frontend-lag.rrd"
    rr.init("slam-rs-frontend-lag", recording_id="frontend-lag")
    rr.save(output)
    cameras: tuple[CameraCalib, ...] = (camera(0, 0.0), camera(1, 0.1))
    log_calibration(cameras)
    rr.send_blueprint(vio_blueprint(cameras))
    config: _core.VioConfig = _core.VioConfig.from_json(json.dumps({"value0": {"port.frontend_lag": True}}))
    timestamps: Int64[ndarray, "n"] = np.arange(0, 240_000_000, 20_000_000, dtype=np.int64)
    stage: VioStage = VioStage(
        Lockstep(_core.Vio(rig(2), config)),
        VioLogger(cameras=cameras, ground_truth=empty_trajectory(), frame_t_ns=timestamps),
    )
    for index, time in enumerate(timestamps):
        t_ns: int = int(time)
        frameset: Frameset = still_frameset(index, [texture(index, 0), texture(index - 2, 0)])
        rr.set_time("video_time", duration=np.timedelta64(t_ns, "ns"))
        for cam, pixels in enumerate(frameset.images):
            rr.log(f"{camera_entity(cam)}/image", rr.Image(pixels))
        stage.run(frameset)
    stage.flush()
    stage.logger.log_complete_paths()
    rr.disconnect()
    np.testing.assert_array_equal(stage.logger.estimated().t_ns, timestamps)
    rows: Rows = read_rows(output)
    assert [row.t_ns for row in rows["/stats/vio/track_ms"]] == timestamps.tolist()
