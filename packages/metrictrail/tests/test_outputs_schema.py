"""The run recording's layout: entity paths, components and properties (synthetic inputs, written to a temp .rrd)."""

from pathlib import Path
from typing import cast

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
    log_waiting_notes(rec, has_capture=False)
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
    assert not any(path.startswith("/notes/waiting/gt") for path in paths)  # no GT tab without GT, so nothing to wait for


@pytest.mark.parametrize(("has_capture", "has_gt"), [(True, True), (False, True), (True, False), (False, False)])
def test_blueprint_builds_and_waiting_tiles_match_the_notes(has_capture: bool, has_gt: bool) -> None:
    blueprint = run_blueprint("upstream", has_capture=has_capture, has_gt=has_gt)
    assert isinstance(blueprint, rr.blueprint.Blueprint)
    notes: dict[str, str] = waiting_notes(has_capture)
    assert ("dino" in notes) == (not has_capture) and "gt" not in notes
    assert all(text.startswith("waiting for") for text in notes.values())


def test_the_app_and_the_cli_share_one_layout_with_tabs_in_run_order() -> None:
    from metrictrail.blueprint import run_tabs

    background = rr.blueprint.Background(color=(0, 0, 0))
    # The app's run and a CLI capture run of a video: no GT.
    names: list[str] = [str(tab.name) for tab in run_tabs("ours", has_capture=True, has_gt=False, has_calibration=True, background=background)]
    assert names == ["1 · SharedK calibration", "2 · DPVO tracking", "3 · trajectory + metric surface", "4 · summary"]
    # A benchmark with GT adds the metric-scale tab; a run without SharedK (known K) has no calibration tab.
    names = [str(tab.name) for tab in run_tabs("ours", has_capture=True, has_gt=True, has_calibration=False, background=background)]
    assert names == [
        "1 · DPVO tracking",
        "2 · trajectory vs GT, Sim(3) aligned + metric surface",
        "3 · metric scale vs GT",
        "4 · summary",
    ]



def test_the_world_trajectory_grows_with_the_cursor(tmp_path: Path, tum_like_context: tuple[GtContext, Float64[ndarray, "n 4 4"]]) -> None:
    """One segment per tracked frame, at that frame's time: a view that shows everything up to the cursor draws the path
    as far as the camera has come."""
    context, world_T_cam = tum_like_context
    write_recording(tmp_path / "run.rrd", synthetic_outputs(world_T_cam), context, None)
    frames: list[int] = []
    for chunk in RrdReader(tmp_path / "run.rrd").stream():
        if str(chunk.entity_path) == "/" + entities.world_trajectory("ours"):
            batch = chunk.to_record_batch()
            assert entities.FRAME_TIMELINE in batch.schema.names, "the path is temporal, not static"
            frames += batch.column(entities.FRAME_TIMELINE).to_pylist()
    assert frames == list(range(1, world_T_cam.shape[0]))  # the segment that ends at each frame after the first


def test_the_trajectory_tab_shows_the_overview_a_follow_camera_and_the_normals_side_by_side() -> None:
    from metrictrail.blueprint import run_tabs

    background = rr.blueprint.Background(color=(0, 0, 0))
    tabs = run_tabs("ours", has_capture=True, has_gt=False, has_video=True, has_calibration=True, background=background)
    assert [str(tab.name) for tab in tabs] == ["1 · SharedK calibration", "2 · DPVO tracking", "3 · trajectory + metric surface", "4 · summary"]
    views = [view for view in walk(tabs[2]) if isinstance(view, rr.blueprint.Spatial3DView)]
    assert [str(view.name) for view in views] == ["overview: trajectory + metric surface", "follow (the camera)", "surface coloured by normals"]
    assert str(views[1].origin) == entities.DPVO_CURRENT  # the camera the video plays in, riding the final trajectory
    for view in views:
        assert "/" + entities.world_trajectory("ours") in view.visualizer_overrides  # drawn up to the cursor


def walk(node: object) -> list[object]:
    """The views under a blueprint container, depth first (a view is its own leaf)."""
    if isinstance(node, rr.blueprint.Container):
        return [leaf for child in node.contents for leaf in walk(child)]
    return [node]


def test_every_video_run_without_gt_shows_its_input_camera_in_the_world_views() -> None:
    """A live capture or a log of finished artifacts (log_run): the camera the video plays in rides the final trajectory."""
    from metrictrail.blueprint import run_tabs, shares_input_camera

    assert shares_input_camera(has_capture=False, has_video=True, has_gt=False)
    assert shares_input_camera(has_capture=True, has_video=False, has_gt=False)
    assert not shares_input_camera(has_capture=True, has_video=True, has_gt=True)  # the world is aligned to GT
    assert not shares_input_camera(has_capture=False, has_video=False, has_gt=False)  # no camera entity at all
    background = rr.blueprint.Background(color=(0, 0, 0))
    tabs = run_tabs("ours", has_capture=False, has_gt=False, has_video=True, has_calibration=False, background=background)
    assert str(tabs[1].name) == "2 · trajectory + metric surface"
    world = walk(tabs[1])[0]
    assert isinstance(world, rr.blueprint.Spatial3DView)
    contents: list[str] = list(cast(list[str], world.contents))
    assert f"/{entities.DPVO_CURRENT}/**" in contents and f"- /{entities.world_cam('ours')}/**" in contents


def test_a_video_run_frames_the_input_video_and_plays_video_time() -> None:
    with_video = frame_view(has_video=True, has_capture=True, background=rr.blueprint.Background(color=(0, 0, 0)))
    assert str(with_video.origin) == entities.INPUT and with_video.contents == [entities.INPUT_VIDEO, entities.INPUT_PATCHES]
    frame_dir = frame_view(has_video=False, has_capture=True, background=rr.blueprint.Background(color=(0, 0, 0)))
    assert str(frame_dir.origin) == entities.CAM and frame_dir.contents == [entities.CAM_RGB, entities.CAM_PATCHES]
    playing = time_panel(has_video=True)
    assert str(playing.timeline) == entities.VIDEO_TIMELINE and str(playing.play_state) == "playing" and str(playing.loop_mode) == "all"
    assert str(time_panel(has_video=False).timeline) == entities.FRAME_TIMELINE
    # A live run logs the whole video first; the app moves the cursor to each tracked frame, so the panel waits paused.
    live = time_panel(has_video=True, live=True)
    assert str(live.timeline) == entities.VIDEO_TIMELINE and str(live.play_state) == "paused"


def test_without_gt_the_world_hangs_in_dpvos_frame_and_the_live_camera_rides_the_final_trajectory(tmp_path: Path) -> None:
    import pyarrow as pa

    from metrictrail.outputs import ride_final_trajectory

    world_T_cam: Float64[ndarray, "n 4 4"] = np.tile(np.eye(4), (6, 1, 1))
    world_T_cam[:, 2, 3] = np.arange(6, dtype=np.float64)  # walking along +z
    world_T_cam[3] = np.nan  # a non-finite pose is left out
    outputs: RunOutputs = synthetic_outputs(world_T_cam)
    rec = rr.RecordingStream("metrictrail-test", recording_id="ride", send_properties=False)
    rec.save(tmp_path / "ride.rrd")
    ride_final_trajectory(rec, outputs.trajectory, video_time_ns=np.arange(6, dtype=np.int64) * 40_000_000)
    rec.flush()
    rec.disconnect()
    world_parent: list[object] = []
    frames: list[int] = []
    video: list[int] = []
    translations: list[list[list[float]]] = []
    for chunk in RrdReader(tmp_path / "ride.rrd").stream():
        batch = chunk.to_record_batch()
        if str(chunk.entity_path) == "/" + entities.WORLD_ROOT and "Transform3D:parent_frame" in batch.schema.names:
            world_parent += batch.column("Transform3D:parent_frame").to_pylist()
        if str(chunk.entity_path) == "/" + entities.DPVO_CURRENT:
            frames += batch.column(entities.FRAME_TIMELINE).to_pylist()
            video += batch.column(entities.VIDEO_TIMELINE).cast(pa.int64()).to_pylist()
            translations += batch.column("Transform3D:translation").to_pylist()
    assert world_parent == [["tf#/dpvo"]]  # ViPE's world without GT is DPVO's: the first camera's RDF frame
    assert frames == [0, 1, 2, 4, 5]  # every finite tracked frame, at its own times
    assert video == [0, 40_000_000, 80_000_000, 160_000_000, 200_000_000]
    assert [row[0][2] for row in translations] == [0.0, 1.0, 2.0, 4.0, 5.0]
