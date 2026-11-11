"""The Gradio app's builder and helpers, on the CPU (no pipeline runs)."""

from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("gradio", reason="the metrictrail envs carry gradio (common feature)")
pytest.importorskip("gradio_rerun", reason="the metrictrail envs carry gradio-rerun (common feature)")

import gradio as gr
from gradio_rerun import Rerun

from metrictrail.blueprint import SceneView, calibration_blueprint, run_blueprint
from metrictrail.gradio_ui.metrictrail_ui import app_overrides, build_app, example_videos, metrictrail_streaming_fn


def test_example_videos_list_the_iphone_clips_first(tmp_path: Path) -> None:
    for relative in ("movies/IMG_0493.mp4", "movies/IMG_0482.mp4", "iphone/pool.mp4", "movies/IMG_0493.MOV", "movies/notes.txt"):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).touch()
    assert [path.relative_to(tmp_path).as_posix() for path in example_videos(tmp_path)] == [
        "iphone/pool.mp4",
        "movies/IMG_0482.mp4",
        "movies/IMG_0493.mp4",
    ]
    assert example_videos(tmp_path / "missing") == []


def test_app_overrides_keep_upstream_values_out_of_the_command() -> None:
    assert app_overrides(1, 1500, "robust128") == []
    assert app_overrides(2, 600, "fast64") == [
        "pipeline.init.tracking_stride=2",
        "pipeline.init.input_raster.max_frames=600",
        "pipeline.init.calibration.budget=fast64",
    ]
    with pytest.raises(ValueError, match="tracking stride"):
        app_overrides(0, 1500, "robust128")
    with pytest.raises(ValueError):
        app_overrides(1, 1500, "all")  # a protocol mode, not a pipeline budget


def test_build_app_wires_a_streaming_viewer_and_the_examples(tmp_path: Path) -> None:
    (tmp_path / "iphone").mkdir()
    (tmp_path / "iphone" / "pool.mp4").touch()
    demo: gr.Blocks = build_app(tmp_path)
    components: list[object] = list(demo.blocks.values())
    viewers: list[Rerun] = [block for block in components if isinstance(block, Rerun)]
    assert len(viewers) == 1 and viewers[0].streaming
    assert any(isinstance(block, gr.File) for block in components)
    assert any(isinstance(block, gr.Dataset) and len(block.samples) == 1 for block in components)
    assert any(viewers[0]._id in dependency["outputs"] for dependency in demo.config["dependencies"])


def test_streaming_fn_without_a_video_yields_an_empty_chunk() -> None:
    assert list(metrictrail_streaming_fn(None, 1, 1500, "fast64")) == [(b"", "Upload a video or pick an example first.")]


def test_app_layouts_build_for_calibration_tracking_and_the_final_run() -> None:
    eye = SceneView(position=(1.0, 1.0, 1.0), look_target=(0.0, 0.0, 0.0), eye_up=(0.0, -1.0, 0.0))
    calibration_blueprint()
    run_blueprint("ours", has_capture=True, has_gt=False, has_video=True, has_calibration=True, tracking=True)
    run_blueprint("ours", has_capture=True, has_gt=False, has_video=True, has_calibration=True, world_eye=eye)  # the CLI's layout too


def test_a_stop_is_not_an_exception_so_vipes_calibration_fallback_cannot_swallow_it() -> None:
    from metrictrail.gradio_ui.metrictrail_ui import RunCancelled, RunStatus

    assert not issubclass(RunCancelled, Exception)
    status = RunStatus()
    status.cancel.set()
    with pytest.raises(RunCancelled):
        status.update("SharedK calibration: 3 frames read")


def test_a_mov_upload_reaches_vipe_under_an_mp4_name(tmp_path: Path) -> None:
    from metrictrail.apis.run import PreparedInput, RunSettings, prepare_input

    video: Path = tmp_path / "upload" / "pool.MOV"
    video.parent.mkdir()
    video.write_bytes(b"not decoded here")
    for _ in range(2):  # a second run on the same output replaces the link
        prepared: PreparedInput = prepare_input(RunSettings(input=video, output=tmp_path / "run"))
    assert prepared.stream_path == tmp_path / "run" / "input" / "pool.mp4"
    assert prepared.stream_path.resolve() == video.resolve() and prepared.name == "pool"
    mp4: Path = tmp_path / "clip.mp4"
    mp4.touch()
    assert prepare_input(RunSettings(input=mp4, output=tmp_path / "run2")).stream_path == mp4


def test_a_host_passes_its_run_examples_and_frame_cap_to_build_app(tmp_path: Path) -> None:
    clip: Path = tmp_path / "clip.mp4"
    clip.touch()

    def hosted_run(video_path: str | None, tracking_stride: int | float, max_frames: int | float, budget: str) -> Iterator[tuple[bytes, str]]:
        yield b"", "hosted"

    demo: gr.Blocks = build_app(run=hosted_run, examples=[clip], max_frames=300)
    components: list[object] = list(demo.blocks.values())
    assert any(isinstance(block, gr.Dataset) and len(block.samples) == 1 for block in components)
    assert any(isinstance(block, gr.Number) and block.value == 300 for block in components)
    budgets = [block for block in components if isinstance(block, gr.Dropdown)]
    assert len(budgets) == 1 and budgets[0].value == "fast64"  # half of ViPE's default robust128 sweep
    assert [choice[1] for choice in budgets[0].choices] == ["fast64", "robust128", "robust256"]
    assert any(dependency.fn is hosted_run for dependency in demo.fns.values())


def test_the_cursor_follows_the_calibration_then_the_video_plays_smoothly_behind_tracking() -> None:
    from metrictrail.gradio_ui.metrictrail_ui import RunStatus

    def payload(command: dict[str, object] | None) -> dict[str, object]:
        assert command is not None
        inner = command["_command"]
        assert isinstance(inner, dict)
        return inner

    status = RunStatus()
    assert status.cursor_command(0.0) is None and status.cursor is None  # nothing read yet
    status.at(42, "calibration")
    calibration = payload(status.cursor_command(0.0))
    assert calibration["timeline"] == "calibration_step" and calibration["time"] == "42" and not calibration["play"]
    assert status.cursor_command(0.1) is None  # sent once per change
    second: int = 1_000_000_000
    status.at(second, "tracking")
    assert status.cursor_command(1.0) is None  # less than the resume buffer tracked: wait
    status.at(3 * second, "tracking")
    start = payload(status.cursor_command(2.0))  # enough tracked: play from the start at real time
    assert start["timeline"] == "video_time" and start["time"] == "0" and start["play"]
    status.at(20 * second, "tracking")  # tracking runs ahead of the playhead: no jumps
    assert status.cursor_command(5.0) is None and status.cursor_command(9.0) is None
    status.at(21 * second, "tracking")
    pause = payload(status.cursor_command(22.5))  # the playhead (20.5 s) is within a second of the tracked frames: wait
    assert not pause["play"] and int(str(pause["time"])) <= 21 * second
    status.at(24 * second, "tracking")
    resume = payload(status.cursor_command(23.0))
    assert resume["play"] and resume["time"] == pause["time"]


def test_the_final_layout_keeps_playing_from_where_the_video_is() -> None:
    from metrictrail.gradio_ui.metrictrail_ui import RunStatus

    status = RunStatus()
    status.at(10 * 1_000_000_000, "tracking")
    status.cursor_command(0.0)  # plays from 0
    final = status.playback.finish(4.0)["_command"]
    assert isinstance(final, dict) and final["play"] and int(str(final["time"])) == 4 * 1_000_000_000


def test_the_budget_dropdown_offers_exactly_vipes_budgets() -> None:
    from metrictrail.gradio_ui.metrictrail_ui import APP_BUDGET, BUDGET_LABELS, PIPELINE_BUDGETS, UPSTREAM_BUDGET

    assert tuple(BUDGET_LABELS) == PIPELINE_BUDGETS and {APP_BUDGET, UPSTREAM_BUDGET} <= set(PIPELINE_BUDGETS)
