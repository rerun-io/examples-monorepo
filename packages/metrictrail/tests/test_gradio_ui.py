"""The Gradio app's builder and helpers, on the CPU (no pipeline runs)."""

from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("gradio", reason="the metrictrail envs carry gradio (common feature)")
pytest.importorskip("gradio_rerun", reason="the metrictrail envs carry gradio-rerun (common feature)")

import gradio as gr
from gradio_rerun import Rerun

from metrictrail.blueprint import SceneView, app_blueprint
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
    assert app_overrides(1, 1500) == []
    assert app_overrides(2, 600) == ["pipeline.init.tracking_stride=2", "pipeline.init.input_raster.max_frames=600"]
    with pytest.raises(ValueError, match="tracking stride"):
        app_overrides(0, 1500)


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
    assert list(metrictrail_streaming_fn(None, 1, 1500)) == [(b"", "Upload a video or pick an example first.")]


def test_app_blueprint_builds_for_the_live_and_the_final_layout() -> None:
    eye = SceneView(position=(1.0, 1.0, 1.0), look_target=(0.0, 0.0, 0.0), eye_up=(0.0, -1.0, 0.0))
    for final_eye in (None, eye):
        app_blueprint(final_eye)


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

    def hosted_run(video_path: str | None, tracking_stride: int | float, max_frames: int | float) -> Iterator[tuple[bytes, str]]:
        yield b"", "hosted"

    demo: gr.Blocks = build_app(run=hosted_run, examples=[clip], max_frames=300)
    components: list[object] = list(demo.blocks.values())
    assert any(isinstance(block, gr.Dataset) and len(block.samples) == 1 for block in components)
    assert any(isinstance(block, gr.Number) and block.value == 300 for block in components)
    assert any(dependency.fn is hosted_run for dependency in demo.fns.values())


def test_the_cursor_follows_the_newest_tracked_frame_once_per_change() -> None:
    from metrictrail.gradio_ui.metrictrail_ui import RunStatus

    status = RunStatus()
    assert status.cursor_command() is None  # calibrating: nothing tracked yet, the viewer stays paused
    status.tracked_at(1_500_000_000)
    command = status.cursor_command()
    assert command is not None
    payload = command["_command"]
    assert isinstance(payload, dict)
    assert payload["timeline"] == "video_time" and payload["kind"] == "duration" and payload["time"] == "1500000000" and not payload["play"]
    assert status.cursor_command() is None  # sent once per new frame
    status.tracked_at(1_533_333_333)
    assert status.cursor_command() is not None
