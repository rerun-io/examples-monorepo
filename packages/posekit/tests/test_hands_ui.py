"""The hands app runs the CLI's settings: a stack name selects the same configs as the CLI subcommand, the camera the CLI's pinhole."""

from pathlib import Path

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
