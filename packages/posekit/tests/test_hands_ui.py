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


def test_a_first_run_reports_what_setup_did(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Engines already in the cache: setup loads. An engine file that appears during setup: setup built it.
    monkeypatch.setattr(hands_ui, "PIPELINES", {})
    monkeypatch.setattr(hands_ui, "SETUP_NOTES", {})
    monkeypatch.setattr(hands_ui, "DEFAULT_TRT_CACHE_DIR", tmp_path)
    (tmp_path / "detector.engine").touch()
    monkeypatch.setattr(hands_ui, "build_pipeline", lambda settings: HandPose3dPipeline.__new__(HandPose3dPipeline))
    hands_ui.pipeline_for("egohand-fast")
    assert hands_ui.SETUP_NOTES["egohand-fast"] == "Loaded egohand-fast (0 s)"
    monkeypatch.setattr(
        hands_ui, "build_pipeline", lambda settings: (tmp_path / "pose.engine").touch() or HandPose3dPipeline.__new__(HandPose3dPipeline)
    )
    hands_ui.pipeline_for("wilor")
    assert hands_ui.SETUP_NOTES["wilor"] == "Built 1 TensorRT engine for wilor (0 s)"


@pytest.mark.parametrize("failure", ["raises", "silent"])
def test_nvdec_is_tried_once_then_the_cpu_decodes_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, failure: str
) -> None:
    # NVDEC fails loudly (torchcodec raises) or silently (no libnvcuvid: torchcodec decodes on the CPU and says so in cpu_fallback).
    import torchcodec.decoders
    from simplecv import rerun_log_utils

    from posekit.apis.video_hands import VideoHandsProgress, open_decoder

    opened: list[str] = []
    cause: str = "[CUDA] Fallback status: Falling back due to: NVcuvid unavailable"

    class Decoder:
        def __init__(self, video: Path, device: str = "cpu", **_: str) -> None:
            opened.append(device)
            if device != "cpu" and failure == "raises":
                raise RuntimeError(cause)
            self.cpu_fallback: str = cause if device != "cpu" else ""

        def get_frames_in_range(self, start: int, stop: int) -> None:
            return None

    monkeypatch.setattr(torchcodec.decoders, "VideoDecoder", Decoder)
    monkeypatch.setattr(rerun_log_utils, "HARDWARE_VIDEO", {})
    monkeypatch.setattr("posekit.apis.video_hands.HARDWARE_VIDEO", rerun_log_utils.HARDWARE_VIDEO)
    with caplog.at_level("WARNING"):
        open_decoder(Path("clip.mp4"), "cuda")
        open_decoder(Path("clip.mp4"), "cuda")
    assert opened == ["cuda", "cpu", "cpu"]  # NVDEC once, then the CPU
    assert [record.levelname for record in caplog.records] == ["WARNING"] and "NVcuvid unavailable" in caplog.text
    warning = rerun_log_utils.hardware_video_warning()
    assert warning.startswith("WARNING: hardware video failed") and "decode ([CUDA] Fallback status" in warning
    assert warning in str(VideoHandsProgress("video", 0, 8, 0, "assumed pinhole", warning))  # the app's status shows it


def test_nvdec_that_works_stays_and_one_unsupported_video_is_not_remembered(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    import torchcodec.decoders
    from simplecv import rerun_log_utils

    from posekit.apis.video_hands import open_decoder

    opened: list[str] = []

    class Decoder:
        def __init__(self, video: Path, device: str = "cpu", **_: str) -> None:
            opened.append(device)
            self.cpu_fallback: str = "[CUDA] Fallback status: Falling back due to: Video not supported" if video.name == "odd.mp4" else ""

        def get_frames_in_range(self, start: int, stop: int) -> None:
            return None

    monkeypatch.setattr(torchcodec.decoders, "VideoDecoder", Decoder)
    monkeypatch.setattr(rerun_log_utils, "HARDWARE_VIDEO", {})
    monkeypatch.setattr("posekit.apis.video_hands.HARDWARE_VIDEO", rerun_log_utils.HARDWARE_VIDEO)
    with caplog.at_level("WARNING"):
        for name in ("clip.mp4", "odd.mp4", "clip.mp4"):
            open_decoder(Path(name), "cuda")
    assert opened == ["cuda", "cuda", "cuda"] and rerun_log_utils.hardware_video_warning() == ""
    assert len(caplog.records) == 1 and "odd.mp4" in caplog.text
