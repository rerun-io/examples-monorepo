"""The streamer's phase logic and the four layouts, without a GPU."""

from pathlib import Path

import pytest
import rerun as rr
import torch

from fdanyone_rerun.core import RunConfig, ViewLayout
from fdanyone_rerun.outputs import CameraRecord, CameraRig
from fdanyone_rerun.previews import preview_latent_index
from fdanyone_rerun.rerun_log import body_blueprint, diffusion_blueprint, motion_blueprint, person_blueprint, result_blueprint, training_blueprint
from fdanyone_rerun.stream import GENERATION_STAGE, RunCancelled, RunStreamer


def _streamer(tmp_path: Path) -> tuple[RunStreamer, rr.MemoryRecording]:
    recording = rr.RecordingStream("fdanyone-rerun-test")
    memory: rr.MemoryRecording = recording.memory_recording()
    config = RunConfig(video=tmp_path / "clip.mp4", output_dir=tmp_path / "run", layout=ViewLayout(views_per_layer=12))
    return RunStreamer(config, recording, previews=False), memory


@pytest.mark.parametrize("num_views", [6, 12, 24])
def test_layouts_build_for_every_rig(num_views: int) -> None:
    cameras: list[CameraRecord] = [
        CameraRecord(
            camera_id=view,
            layer_index=0,
            pitch=15,
            yaw=360.0 * view / num_views,
            K=[[900.0, 0.0, 352.0], [0.0, 900.0, 640.0], [0.0, 0.0, 1.0]],
            camera_to_world=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 2.5], [0.0, 0.0, 0.0, 1.0]],
            image_width=704,
            image_height=1280,
            video=f"videos/{view:02d}.mp4",
            skeleton_video=f"skeletons/{view:02d}.mp4",
        )
        for view in range(num_views)
    ]
    rig: CameraRig = CameraRig(camera_model="OPENCV", cameras=cameras, front_camera_ids=[0])
    layouts = (
        motion_blueprint(),
        body_blueprint(),
        diffusion_blueprint(num_views),
        result_blueprint(num_views),
        training_blueprint(),
        person_blueprint(rig),
    )
    for blueprint in layouts:
        assert isinstance(blueprint, rr.blueprint.Blueprint)


def test_generation_stage_switches_to_the_diffusion_layout(tmp_path: Path) -> None:
    streamer, memory = _streamer(tmp_path)
    streamer.on_stage(GENERATION_STAGE, 0.45, None)
    streamer.on_step(0, None, torch.zeros(6, 48, 31, 2, 2))  # RCP proposals: nothing to preview
    streamer.on_step(0, (0, 1), torch.zeros(2, 48, 31, 2, 2))  # previews off: nothing decoded
    streamer.recording.flush()
    assert len(memory.drain_as_bytes()) > 0


def test_stop_cancels_at_the_next_callback(tmp_path: Path) -> None:
    streamer, _ = _streamer(tmp_path)
    streamer.stop.set()
    with pytest.raises(RunCancelled):
        streamer.on_stage("Generating target-view videos", 0.45, None)
    with pytest.raises(RunCancelled):
        streamer.on_step(1, (0,), torch.zeros(1, 48, 31, 2, 2))


def test_previews_show_the_middle_latent_frame() -> None:
    # 121 frames -> 31 latent frames; frame 15 decodes to pixel frame 60, the middle of the clip.
    assert preview_latent_index(31) == 15
