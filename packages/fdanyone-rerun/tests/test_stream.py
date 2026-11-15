"""The streamer's phase logic and the four layouts, without a GPU."""

from pathlib import Path

import pytest
import rerun as rr
import torch

from fdanyone_rerun.core import RunConfig, ViewLayout
from fdanyone_rerun.previews import preview_latent_index
from fdanyone_rerun.rerun_log import body_blueprint, diffusion_blueprint, motion_blueprint, result_blueprint
from fdanyone_rerun.stream import GENERATION_STAGE, RunCancelled, RunStreamer


def _streamer(tmp_path: Path) -> tuple[RunStreamer, rr.MemoryRecording]:
    recording = rr.RecordingStream("fdanyone-rerun-test")
    memory: rr.MemoryRecording = recording.memory_recording()
    config = RunConfig(video=tmp_path / "clip.mp4", output_dir=tmp_path / "run", layout=ViewLayout(views_per_layer=12))
    return RunStreamer(config, recording, preview_decoder=None), memory


@pytest.mark.parametrize("num_views", [6, 12, 24])
def test_layouts_build_for_every_rig(num_views: int) -> None:
    for blueprint in (motion_blueprint(), body_blueprint(), diffusion_blueprint(num_views), result_blueprint(num_views)):
        assert isinstance(blueprint, rr.blueprint.Blueprint)


def test_generation_stage_switches_to_the_diffusion_layout(tmp_path: Path) -> None:
    streamer, memory = _streamer(tmp_path)
    streamer.on_stage(GENERATION_STAGE, 0.45)
    streamer.on_step(0, None, torch.zeros(6, 48, 31, 2, 2))  # RCP proposals: nothing to preview
    streamer.on_step(0, (0, 1), torch.zeros(2, 48, 31, 2, 2))  # no decoder: no previews
    streamer.recording.flush()
    assert len(memory.drain_as_bytes()) > 0


def test_stop_cancels_at_the_next_callback(tmp_path: Path) -> None:
    streamer, _ = _streamer(tmp_path)
    streamer.stop.set()
    with pytest.raises(RunCancelled):
        streamer.on_stage("Generating target-view videos", 0.45)
    with pytest.raises(RunCancelled):
        streamer.on_step(1, (0,), torch.zeros(1, 48, 31, 2, 2))


def test_previews_show_the_middle_latent_frame() -> None:
    # 121 frames -> 31 latent frames; frame 15 decodes to pixel frame 60, the middle of the clip.
    assert preview_latent_index(31) == 15
