"""The run recording's layout: entity paths, components and properties (synthetic inputs, written to a temp .rrd)."""

from pathlib import Path

import numpy as np
import pytest
import rerun as rr
from jaxtyping import Float64
from numpy import ndarray
from rerun.chunk import RrdReader

from metrictrail import entities
from metrictrail.artifacts import ColoredCloud
from metrictrail.blueprint import frame_view, log_waiting_notes, run_blueprint, time_panel, waiting_notes
from metrictrail.eval.vipe_output import MetricTrailTrajectory
from metrictrail.ground_truth import GtContext
from metrictrail.outputs import LoggedRun, LogOptions, Provenance, RunMetrics, RunOutputs, log_run_outputs


def synthetic_outputs(world_T_cam: Float64[ndarray, "n 4 4"]) -> RunOutputs:
    count: int = world_T_cam.shape[0]
    trajectory = MetricTrailTrajectory(
        name="synthetic",
        world_T_cam=world_T_cam,
        source_frame_ids=np.arange(count, dtype=np.int64),
        runtime_s=1.0,
        keyframe_rows=np.arange(0, count, 10, dtype=np.int64),
    )
    intrinsics: Float64[ndarray, "n 4"] = np.tile([400.0, 400.0, 259.0, 194.0], (count, 1))
    return RunOutputs(run_dir=Path("/nonexistent"), name="synthetic", trajectory=trajectory, intrinsics=intrinsics, record=None)


def write_recording(path: Path, outputs: RunOutputs, context: GtContext | None, cloud: ColoredCloud | None) -> RunMetrics:
    rec = rr.RecordingStream("metrictrail-test", recording_id="test", send_properties=False)
    rec.save(path)
    log_waiting_notes(rec, has_capture=False, has_gt=context is not None)
    provenance = Provenance("seq", "frames", "/run", "unit test (synthetic)", "env", "2026-11-06T00:00:00Z")
    options = LogOptions(label="ours", log_frames=False, raster_policy="unchanged", depth_range_m=(0.3, 3.0))
    logged: LoggedRun = log_run_outputs(rec, outputs, options, frames=None, context=context, cloud_world=cloud, provenance=provenance)
    rec.flush()
    rec.disconnect()
    assert logged.world_eye is not None and (logged.metric_eye is not None) == (context is not None)
    return logged.metrics


def test_run_recording_with_gt_has_every_entity_and_property(tmp_path: Path, tum_like_context: tuple[GtContext, Float64[ndarray, "n 4 4"]]) -> None:
    context, world_T_cam = tum_like_context
    cloud = ColoredCloud(
        points_xyz=np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32), colors_rgb=np.full((50, 3), 128, dtype=np.uint8)
    )
    metrics: RunMetrics = write_recording(tmp_path / "run.rrd", synthetic_outputs(world_T_cam), context, cloud)
    assert metrics.ate_sim3_m is not None and metrics.ate_sim3_m < 0.01
    assert metrics.num_gt_pairs == 60 and metrics.cloud_points == 50
    schema = RrdReader(tmp_path / "run.rrd").store().schema()
    paths: set[str] = set(schema.entity_paths())
    expected: set[str] = {
        "/world",
        "/" + entities.world_trajectory("ours"),
        "/" + entities.world_cam("ours"),
        "/" + entities.WORLD_GT_TRAJECTORY,
        "/" + entities.WORLD_GT_POSITION,
        "/" + entities.WORLD_CLOUD,
        "/" + entities.metric_trajectory("ours"),
        "/" + entities.METRIC_GT_TRAJECTORY,
        "/" + entities.METRIC_CLOUD,
        "/" + entities.ATE_SIM3_ERR,
        "/" + entities.ATE_SIM3_RUNNING,
        "/" + entities.ATE_SE3_ERR,
        "/" + entities.ATE_SE3_RUNNING,
        "/" + entities.SUMMARY,
        "/notes/waiting/dino",
    }
    assert expected <= paths
    columns: set[str] = set(schema.column_names())
    assert {
        "property:provenance:producer",
        "property:provenance:dataset_path",
        "property:metrics:ate_sim3_m",
        "property:metrics:sim3_scale",
    } <= columns
    assert "/world/ours/cam:Transform3D:translation" in columns and "/world/ours/cam:Pinhole:image_from_camera" in columns
    assert {index.name for index in schema.index_columns()} >= {"frame", "source_frame", "video_time"}


def test_run_recording_without_gt_has_no_gt_entities(tmp_path: Path, tum_like_context: tuple[GtContext, Float64[ndarray, "n 4 4"]]) -> None:
    _, world_T_cam = tum_like_context
    metrics: RunMetrics = write_recording(tmp_path / "run.rrd", synthetic_outputs(world_T_cam), None, None)
    assert metrics.ate_sim3_m is None and metrics.cloud_points is None
    paths: set[str] = set(RrdReader(tmp_path / "run.rrd").store().schema().entity_paths())
    assert not any(path.startswith(("/world/gt", "/metric", "/metrics/ate")) for path in paths)
    assert "/notes/waiting/gt" in paths


@pytest.mark.parametrize(("has_capture", "has_gt"), [(True, True), (False, True), (True, False), (False, False)])
def test_blueprint_builds_and_waiting_tiles_match_the_notes(has_capture: bool, has_gt: bool) -> None:
    blueprint = run_blueprint("upstream", has_capture=has_capture, has_gt=has_gt)
    assert isinstance(blueprint, rr.blueprint.Blueprint)
    notes: dict[str, str] = waiting_notes(has_capture, has_gt)
    assert ("dino" in notes) == (not has_capture) and ("gt" in notes) == (not has_gt)
    assert all(text.startswith("waiting for") for text in notes.values())



def test_a_video_run_frames_the_input_video_and_plays_video_time() -> None:
    with_video = frame_view(has_video=True, has_capture=True, background=rr.blueprint.Background(color=(0, 0, 0)))
    assert str(with_video.origin) == entities.INPUT and with_video.contents == [entities.INPUT_VIDEO, entities.INPUT_PATCHES]
    frame_dir = frame_view(has_video=False, has_capture=True, background=rr.blueprint.Background(color=(0, 0, 0)))
    assert str(frame_dir.origin) == entities.CAM and frame_dir.contents == [entities.CAM_RGB, entities.CAM_PATCHES]
    playing = time_panel(has_video=True)
    assert str(playing.timeline) == entities.VIDEO_TIMELINE and str(playing.play_state) == "playing" and str(playing.loop_mode) == "all"
    assert str(time_panel(has_video=False).timeline) == entities.FRAME_TIMELINE
