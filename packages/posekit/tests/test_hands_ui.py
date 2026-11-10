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
    settings = hands_ui.settings_from_inputs("/tmp/clip.mp4", stack, 32, 1000.0, None, 960.0, None)
    assert settings.detector == tyro.cli(HandDetectorConfig, args=[stack])
    assert settings.pose == tyro.cli(HandPose3dConfig, args=[stack])
    assert (settings.video_path, settings.max_frames, settings.fx, settings.fy, settings.cx, settings.cy) == (Path("/tmp/clip.mp4"), 32, 1000.0, None, 960.0, None)
    assert hands_ui.settings_from_inputs("/tmp/clip.mp4", stack, 0, None, None, None, None).max_frames is None


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
    monkeypatch.setattr(hands_ui, "_PIPELINES", {})
    monkeypatch.setattr(
        hands_ui, "build_pipeline", lambda settings: built.append(type(settings.pose).__name__) or HandPose3dPipeline.__new__(HandPose3dPipeline)
    )
    first = hands_ui.pipeline_for("wilor")
    hands_ui.pipeline_for("egohand-fast")
    assert hands_ui.pipeline_for("wilor") is first
    assert len(built) == 2
