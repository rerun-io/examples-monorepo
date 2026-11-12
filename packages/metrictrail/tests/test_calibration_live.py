"""SharedK calibration, live: the head focals, the sweep, the DROID map and the status line, on the CPU (no ViPE engine)."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray
from rerun.chunk import RrdReader

from metrictrail import entities
from metrictrail.calibration_live import CalibrationLive, KeyframeCloud, MapSnapshot, RunPhase

K_EIGHTH: Float64[ndarray, "3 3"] = np.array([[55.0, 0.0, 32.0], [0.0, 55.0, 18.0], [0.0, 0.0, 1.0]])


def snapshot(frames: list[int], changed: list[int]) -> MapSnapshot:
    count: int = len(frames)
    poses: Float32[ndarray, "n 7"] = np.tile(np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32), (count, 1))
    poses[:, 2] = -np.arange(count, dtype=np.float32)  # walking forward along +z
    clouds: list[KeyframeCloud] = [
        KeyframeCloud(frame, np.ones((10, 3), dtype=np.float32), np.zeros((10, 3), dtype=np.uint8)) for frame in changed
    ]
    edges: Int64[ndarray, "e 2"] = np.array([[index, index + 1] for index in range(count - 1)], dtype=np.int64).reshape(-1, 2)
    return MapSnapshot(
        steps=np.array(frames, dtype=np.int64),
        world_to_cam=poses,
        edges=edges,
        k_eighth=K_EIGHTH,
        changed=clouds,
        newest_rgb=np.zeros((288, 512, 3), dtype=np.uint8),
        newest_prior_m=np.full((36, 64), 2.0, dtype=np.float32),
        newest_depth_m=np.full((36, 64), 2.1, dtype=np.float32),
    )


def make_live(tmp_path: Path, stages: list[str], cursors: list[tuple[int, RunPhase]]) -> tuple[CalibrationLive, rr.RecordingStream]:
    rec = rr.RecordingStream("metrictrail-test", recording_id="calibration-live")
    rec.save(tmp_path / "calibration.rrd")
    live = CalibrationLive(
        rec=rec,
        video_time_ns=np.arange(100, dtype=np.int64) * 33_333_333,
        target_keyframes=128,
        head_frames=32,
        on_stage=stages.append,
        on_cursor=lambda value, phase: cursors.append((value, phase)),
        min_interval_s=0.0,
    )
    return live, rec


def timeline_values(rrd: Path, timeline: str) -> dict[str, list[int]]:
    values: dict[str, list[int]] = {}
    for chunk in RrdReader(rrd).stream():
        batch = chunk.to_record_batch()
        if timeline in batch.schema.names:
            values.setdefault(str(chunk.entity_path), []).extend(batch.column(timeline).cast(pa.int64()).to_pylist())
    return values


def test_the_status_line_follows_the_head_then_the_sweep(tmp_path: Path) -> None:
    stages: list[str] = []
    cursors: list[tuple[int, RunPhase]] = []
    live, _ = make_live(tmp_path, stages, cursors)
    for focal in (450.0, 440.0, 460.0):
        live.head_focal(np.array([focal, focal, 252.0, 140.0]))
    assert stages[-1] == "SharedK calibration: MoGe-2 focal on frame 3 / 32"
    for step, source in ((0, 0), (1, 2), (2, 4)):
        live.read(step, source)
    assert cursors == [(0, "calibration"), (1, "calibration"), (2, "calibration")]  # the sweep's own timeline
    assert stages[-1] == "SharedK calibration: 3 frames read, 0 / 128 keyframes"  # no backend solve yet: no focal
    live.frontend(5)
    live.focal(np.array([431.5, 431.5, 252.0, 140.0]))
    live.read(3, 6)
    assert stages[-1] == "SharedK calibration: 4 frames read, 5 / 128 keyframes, focal 431.5 px (MoGe-2: 450.0 px)"


def test_a_stop_during_calibration_escapes_the_sweep() -> None:
    class Stop(BaseException):
        pass

    def stop(_: str) -> None:
        raise Stop

    live = CalibrationLive(rec=rr.RecordingStream("metrictrail-test"), video_time_ns=None, target_keyframes=128, head_frames=32, on_stage=stop, on_cursor=None)
    with pytest.raises(Stop):
        live.read(0, 0)


def test_the_map_sends_only_what_changed_and_clears_dropped_keyframes(tmp_path: Path) -> None:
    live, rec = make_live(tmp_path, [], [])
    live.read(3, 6)
    live.map(snapshot([0, 3, 6], changed=[0, 3, 6]), np.array([440.0, 440.0, 252.0, 140.0]))
    live.read(9, 18)
    live.map(snapshot([0, 6, 18], changed=[18]), np.array([438.0, 438.0, 252.0, 140.0]))  # the frontend dropped frame 3
    live.read(12, 24)
    live.map(snapshot([0, 6, 18], changed=[0, 6, 18]), np.array([436.0, 436.0, 252.0, 140.0]))  # a backend solve moved all
    rec.flush()
    rec.disconnect()
    rrd: Path = tmp_path / "calibration.rrd"
    paths: set[str] = set(RrdReader(rrd).store().schema().entity_paths())
    for frame in (0, 3, 6, 18):
        keyframe: str = entities.calibration_keyframe(frame)
        assert {f"/{keyframe}", f"/{keyframe}/points", f"/{keyframe}/camera"} <= paths
        assert f"/{keyframe}/camera/thumbnail" not in paths  # a 3D view reports images whose pinhole is not there yet
    assert {"/" + entities.CALIBRATION_PATH, "/" + entities.CALIBRATION_EDGES, "/" + entities.CALIBRATION_NEWEST_PRIOR, "/" + entities.CALIBRATION_NEWEST_DEPTH} <= paths
    steps: dict[str, list[int]] = timeline_values(rrd, entities.CALIBRATION_TIMELINE)
    f0: str = entities.calibration_keyframe(0)
    assert steps[f"/{f0}/points"] == [3, 12]  # re-sent only when it changed
    assert steps[f"/{f0}"] == [3, 12] and steps[f"/{f0}/camera"] == [3, 12]  # pose and pinhole too
    assert steps[f"/{entities.calibration_keyframe(18)}/points"] == [9, 12]
    assert 9 in steps["/" + entities.calibration_keyframe(3)]  # the Clear of the dropped keyframe
    video: dict[str, list[int]] = timeline_values(rrd, entities.VIDEO_TIMELINE)
    assert 18 * 33_333_333 in video["/" + entities.CALIBRATION_PATH]  # also on the input video's timeline


def test_both_depth_panels_share_the_priors_range() -> None:
    from metrictrail.calibration_live import depth_range_m

    prior: Float32[ndarray, "h w"] = np.linspace(1.0, 5.0, 100, dtype=np.float32).reshape(10, 10)
    prior[0, 0] = 0.0  # no prior there
    lo, hi = depth_range_m(prior)
    assert 1.0 < lo < 1.2 and 4.8 < hi < 5.0  # 2nd and 98th percentile of the valid prior
    assert depth_range_m(np.zeros((4, 4), dtype=np.float32)) == (0.0, 1.0)


def test_head_focals_land_on_their_own_frames_and_solves_on_the_sweeps_step(tmp_path: Path) -> None:
    live, rec = make_live(tmp_path, [], [])
    for focal in (450.0, 440.0):
        live.head_focal(np.array([focal, focal, 252.0, 140.0]))
    live.read(40, 80)
    live.frontend(12)
    live.focal(np.array([431.5, 431.5, 252.0, 140.0]))
    rec.flush()
    rec.disconnect()
    steps: dict[str, list[int]] = timeline_values(tmp_path / "calibration.rrd", entities.CALIBRATION_TIMELINE)
    assert steps["/" + entities.CALIBRATION_HEAD_FOCAL] == [0, 1]
    assert steps["/" + entities.CALIBRATION_FOCAL] == [40]
    assert steps["/" + entities.CALIBRATION_KEPT] == [40]


def test_keyframe_points_back_project_droids_disparities_with_the_eighth_resolution_k() -> None:
    torch = pytest.importorskip("torch")
    from metrictrail.calibration_live import keyframe_points

    disps = torch.full((2, 4, 6), 0.5)  # 2 m everywhere
    disps[1, 0, 0] = 0.1  # 10 m: below half the mean disparity, dropped (ViPE's single-view rule)
    masks = torch.zeros((2, 4, 6), dtype=torch.bool)
    masks[0, 3, 5] = True  # masked out
    images = torch.rand((2, 3, 32, 48))
    k4 = torch.tensor([80.0, 80.0, 24.0, 16.0])  # internal raster; /8: fx 10, cx 3, cy 2
    points, valid, colors = keyframe_points(disps, masks, images[:, :, 3::8, 3::8], k4)
    assert points.shape == (2, 4, 6, 3) and valid.shape == (2, 4, 6) and colors.shape == (2, 4, 6, 3)
    np.testing.assert_allclose(points[0, 2, 5].numpy(), [(5 - 3) / 10 * 2.0, (2 - 2) / 10 * 2.0, 2.0], atol=1e-6)
    assert not bool(valid[0, 3, 5]) and not bool(valid[1, 0, 0]) and int(valid.sum()) == 2 * 24 - 2
    assert colors.dtype == torch.uint8 and int(colors[0, 0, 0, 0]) == int(images[0, 0, 3, 3] * 255)


def test_a_live_view_failure_fails_the_run_instead_of_reaching_vipes_focal_fallback() -> None:
    from metrictrail.calibration_live import CalibrationLiveError, guarded

    def broken() -> None:
        raise NotImplementedError("a kernel the live view needs is missing")

    assert not issubclass(CalibrationLiveError, Exception)  # ViPE's sweep catches Exception and binds MoGe-2's focal
    with pytest.raises(CalibrationLiveError) as raised:
        guarded(broken)
    assert isinstance(raised.value.__cause__, NotImplementedError)
