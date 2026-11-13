"""The hands app runs the CLI's settings: a stack name selects the same configs as the CLI subcommand, the camera the CLI's pinhole."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import tyro

pytest.importorskip("gradio_rerun", reason="the hands app needs gradio-rerun")

from posekit import hands_ui
from posekit.apis.video_hands import EXAMPLES, assumed_pinhole
from posekit.models import HandDetectorConfig, HandPose3dConfig, HandPose3dPipeline


@pytest.mark.parametrize("stack", ["wilor", "egohand-fast", "egohand-fast-bf16"])
def test_inputs_become_the_cli_settings(stack: str) -> None:
    settings = hands_ui.settings_from_inputs("/tmp/clip.mp4", stack, 32, 1000.0, None, 960.0, None, "low")
    assert settings.detector == tyro.cli(HandDetectorConfig, args=[stack])
    assert settings.pose == tyro.cli(HandPose3dConfig, args=[stack])
    assert (settings.video_path, settings.max_frames, settings.fx, settings.fy, settings.cx, settings.cy, settings.mesh_detail) == (
        Path("/tmp/clip.mp4"), 32, 1000.0, None, 960.0, None, "low"
    )
    assert hands_ui.settings_from_inputs("/tmp/clip.mp4", stack, 0, None, None, None, None, "xlo").max_frames is None


def test_choosing_a_video_fills_the_assumed_pinhole() -> None:
    assert hands_ui.fill_intrinsics(None) == (None, None, None, None)
    assert hands_ui.fill_intrinsics(str(EXAMPLES / "gsi_sewing_left_front.mp4")) == tuple(round(v, 1) for v in assumed_pinhole(1920, 1080, 65.0))


def test_examples_name_a_clip_and_a_stack_and_the_exocentric_clip_runs_wilor() -> None:
    rows = hands_ui.example_rows()
    assert all(Path(video).exists() and stack in hands_ui.STACKS for video, stack in rows)
    assert [stack for video, stack in rows if Path(video).name == "hands_video.mp4"] == ["wilor"]
    assert [Path(video).name for video, _ in rows][0] == "gsi_sewing_left_front.mp4"


def test_each_stack_is_set_up_once_and_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    # Switching between an EgoHand example and the WiLoR example must not set either stack up again.
    built: list[str] = []
    monkeypatch.setattr(hands_ui, "PIPELINES", {})
    monkeypatch.setattr(
        hands_ui, "build_pipeline", lambda settings: built.append(type(settings.pose).__name__) or HandPose3dPipeline.__new__(HandPose3dPipeline)
    )
    first = hands_ui.pipeline_for("wilor")
    hands_ui.pipeline_for("egohand-fast")
    assert hands_ui.pipeline_for("wilor") is first
    assert len(built) == 2


def test_the_layout_takes_the_upload_as_is_picks_an_examples_stack_and_stop_cancels_run() -> None:
    import gradio as gr

    from posekit.hands_app_ui.layout import build

    demo = build()
    video = next(block for block in demo.blocks.values() if block.elem_id == "source-video")
    assert isinstance(video, gr.Video)
    assert video.format is None and video.include_audio  # Gradio converts nothing on upload
    stack = next(block for block in demo.blocks.values() if isinstance(block, gr.Dropdown) and block.label == "Hand stack")
    examples = next(block for block in demo.blocks.values() if isinstance(block, gr.Dataset))
    # the examples show only the video (thumbnails), and a click on one sets the video and its stack
    assert [type(component) for component in examples._components] == [gr.Video]
    clicks = [dependency for dependency in demo.config["dependencies"] if (examples._id, "click") in dependency["targets"]]
    assert [dependency["outputs"] for dependency in clicks] == [[video._id, stack._id]]
    assert any(dependency.get("cancels") for dependency in demo.config["dependencies"])


def test_the_outputs_tab_summarises_a_run_and_a_run_without_a_video_asks_for_one() -> None:
    from posekit.apis.video_hands import VideoHandsProgress
    from posekit.hands_app_ui.layout import run_hands, summary

    table = summary(VideoHandsProgress("hands", 64, 180, 128, "assumed pinhole"), 2.0, "egohand-fast", "xlo")
    assert "| Frames | 64 / 180 |" in table and "| Hands | 128 |" in table and "32 fps" in table and "mesh `xlo`" in table
    updates = list(run_hands(None, "egohand-fast", 0, None, None, None, None, "xlo"))
    assert len(updates) == 1 and updates[0][1] == "Choose a video first."


def test_the_layout_shows_the_stacks_examples_default_and_run_it_is_given() -> None:
    import gradio as gr

    from posekit.hands_app_ui.layout import build

    def run(*inputs: object) -> Iterator[tuple[Any, str, Any, str, str]]:
        yield None, "", None, "", ""

    rows = [row for row in hands_ui.example_rows(Path("/space/examples")) if row[1] != "wilor"]
    assert rows[0][0] == "/space/examples/gsi_sewing_left_front.mp4" and len(rows) == 4
    demo = build(stacks=["egohand-fast"], examples=rows[:1], stack_default="egohand-fast", description="On this host", run=run)
    stack = next(block for block in demo.blocks.values() if isinstance(block, gr.Dropdown))
    assert [choice[1] for choice in stack.choices] == ["egohand-fast"] and stack.value == "egohand-fast"
    examples = next(block for block in demo.blocks.values() if isinstance(block, gr.Dataset))
    assert len(examples.samples) == 1
    assert any(isinstance(block, gr.Markdown) and block.value == "On this host" for block in demo.blocks.values())
    assert any(function.fn is run for function in demo.fns.values())


def test_a_first_run_says_it_prepares_the_stack_then_how_long_that_took(monkeypatch: pytest.MonkeyPatch) -> None:
    # The setup lines name the stack and no backend: WiLoR runs on torch, the EgoHand stacks on TensorRT.
    from posekit.apis.video_hands import VideoHandsProgress
    from posekit.hands_app_ui import layout

    monkeypatch.setattr(layout, "PIPELINES", {})
    monkeypatch.setattr(layout, "pipeline_for", lambda stack: None)
    monkeypatch.setattr(layout, "stream_hands", lambda settings, stack: iter([(None, VideoHandsProgress("done", 8, 8, 4, "assumed pinhole"))]))
    updates = list(layout.run_hands("/tmp/clip.mp4", "wilor", 0, None, None, None, None, "xlo"))
    assert updates[0][1] == "Preparing wilor (the first run may take several minutes)…"
    assert "TensorRT" not in updates[0][1] + updates[-1][4]
    newest, ready, preparing = updates[-1][4].splitlines()
    assert newest.endswith("processed 8 frames, 4 hands") and ready.endswith("wilor ready (0 s)") and preparing.endswith(updates[0][1])


def test_the_run_log_keeps_only_recent_lines_newest_first(monkeypatch: pytest.MonkeyPatch) -> None:
    from posekit.apis.video_hands import VideoHandsProgress
    from posekit.hands_app_ui import layout

    chunks: int = layout.RUN_LOG_LINES + 40
    monkeypatch.setattr(layout, "PIPELINES", {"wilor": None})
    monkeypatch.setattr(
        layout, "stream_hands",
        lambda settings, stack: ((None, VideoHandsProgress("hands", done, chunks, done, "assumed pinhole")) for done in range(1, chunks + 1)),
    )
    logs = [update[4].splitlines() for update in layout.run_hands("/tmp/clip.mp4", "wilor", 0, None, None, None, None, "xlo")]
    assert max(len(log) for log in logs) == layout.RUN_LOG_LINES
    assert logs[-1][0].endswith(f"Frames {chunks}/{chunks}, {chunks} hands") and logs[-1][-1].endswith(f"Frames 41/{chunks}, 41 hands")


def test_the_viewer_takes_the_height_the_header_leaves() -> None:
    # A fixed calc(100vh - ...) overflowed whenever the header wrapped taller, and inside the Space's iframe the footer's
    # height that Gradio posts to the parent grew the iframe past the window: the viewer must fill the row, the footer stay hidden.
    import gradio as gr
    from gradio_rerun import Rerun

    from posekit.hands_app_ui.layout import build
    from posekit.hands_app_ui.theme import APP_CSS

    demo = build()
    viewer = next(block for block in demo.blocks.values() if isinstance(block, Rerun))
    main_row = next(block for block in demo.blocks.values() if isinstance(block, gr.Row) and block.elem_id == "main-row")
    assert demo.fill_height and main_row.scale == 1 and viewer.height == "100%" and viewer.elem_id == "viewer"
    assert "100vh" not in APP_CSS and "footer { display: none !important; }" in APP_CSS


def test_a_host_links_the_rerun_io_example_page_from_the_header() -> None:
    """A Space links its rerun.io example page (the full walkthrough); the local app has no page to link."""
    import gradio as gr

    from posekit.hands_app_ui.layout import build

    url = "https://rerun.io/examples/spatial-computing/egohand"
    links = [b for b in build(example_url=url).blocks.values() if isinstance(b, gr.HTML) and b.elem_id == "example-link"]
    assert len(links) == 1 and f'href="{url}"' in links[0].value and "rerun.io" in links[0].value
    assert not [b for b in build().blocks.values() if getattr(b, "elem_id", None) == "example-link"]
