"""The hands app runs the CLI's settings: a stack name selects the same configs as the CLI subcommand."""

from pathlib import Path

import pytest
import tyro

pytest.importorskip("gradio_rerun", reason="the hands app needs gradio-rerun")

from posekit import hands_ui
from posekit.models import HandDetectorConfig, HandPose3dConfig


@pytest.mark.parametrize("stack", ["wilor", "egohand-fast", "egohand-fast-bf16"])
def test_sync_config_selects_the_cli_stack(stack: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hands_ui, "_CONFIG", hands_ui.VideoHandsSettings())
    hands_ui._sync_config("/tmp/clip.mp4", None, stack, 32)
    assert hands_ui._CONFIG.detector == tyro.cli(HandDetectorConfig, args=[stack])
    assert hands_ui._CONFIG.pose == tyro.cli(HandPose3dConfig, args=[stack])
    assert (hands_ui._CONFIG.video_path, hands_ui._CONFIG.camera_path, hands_ui._CONFIG.max_frames) == (Path("/tmp/clip.mp4"), None, 32)
    hands_ui._sync_config(None, "/tmp/camera.rrd", stack, 0)
    assert hands_ui._CONFIG.video_path == Path("/tmp/clip.mp4") and hands_ui._CONFIG.max_frames is None
