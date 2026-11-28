"""The streamer's phase logic and its layouts (motion, body, diffusion, result, training, person), without a GPU."""

from collections.abc import Callable
from pathlib import Path

import pytest
import rerun as rr
import torch

from fdanyone.errors import RunCancelled
from fdanyone_rerun.core import RunConfig, ViewLayout
from fdanyone_rerun.outputs import CameraRecord, CameraRig
from fdanyone_rerun.previews import preview_latent_index
from fdanyone_rerun.rerun_log import body_blueprint, diffusion_blueprint, motion_blueprint, person_blueprint, result_blueprint, training_blueprint
from fdanyone_rerun.stream import GENERATION_STAGE, RunStreamer


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


def _fake_run(monkeypatch: pytest.MonkeyPatch, log_cameras: Callable[..., None], train_person: Callable[..., None]) -> None:
    """``RunStreamer.run`` with the views, the camera logging and the training replaced."""
    from fdanyone_rerun import stream
    from fdanyone_rerun.outputs import GenerationRecord, InputRecord, RunMetadata

    metadata = RunMetadata(
        input=InputRecord(filename="clip.mp4", fps="25/1", start_time_seconds=0.0, num_frames=121, width=704, height=1280),
        generation=GenerationRecord(
            seed=0,
            attention_backend="sdpa",
            inference_steps=4,
            elapsed_seconds={},
            total_elapsed_seconds=1.0,
            peak_vram_allocated_bytes=0,
            peak_vram_reserved_bytes=0,
        ),
    )
    monkeypatch.setattr(stream, "run_views", lambda config, **_: metadata)
    monkeypatch.setattr(stream, "read_camera_rig", lambda output_dir: None)
    monkeypatch.setattr(stream, "log_cameras", log_cameras)
    monkeypatch.setattr(stream, "train_person", train_person)


def test_cameras_log_while_the_person_trains(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    from fdanyone_rerun.person4d import PersonConfig

    training = threading.Event()
    logged = threading.Event()

    def log_cameras(rig: object, output_dir: Path, recording: rr.RecordingStream) -> None:
        assert training.wait(timeout=10), "the cameras were logged before training started, not beside it"
        logged.set()

    def train_person(output_dir: Path, config: PersonConfig, recording: rr.RecordingStream, **_: object) -> None:
        training.set()

    _fake_run(monkeypatch, log_cameras, train_person)
    streamer, _ = _streamer(tmp_path)
    streamer.person = PersonConfig()
    streamer.run()
    assert logged.is_set()  # run() returns only once the cameras are in the recording


def test_a_camera_logging_error_fails_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def log_cameras(rig: object, output_dir: Path, recording: rr.RecordingStream) -> None:
        raise RuntimeError("Mp4Reader failed")

    _fake_run(monkeypatch, log_cameras, lambda *args, **kwargs: None)
    streamer, _ = _streamer(tmp_path)
    with pytest.raises(RuntimeError, match="Mp4Reader failed"):
        streamer.run()


def test_the_source_clip_logs_beside_gvhmr_and_is_in_before_denoising(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    from fdanyone_rerun import stream

    release = threading.Event()
    logged: list[Path] = []

    def log_canonical_clip(clip: Path, recording: rr.RecordingStream) -> None:
        assert release.wait(timeout=10)
        logged.append(clip)

    monkeypatch.setattr(stream, "log_canonical_clip", log_canonical_clip)
    streamer, _ = _streamer(tmp_path)
    clip: Path = tmp_path / "canonical_clip.mp4"
    streamer.on_stage("Recovering human motion with GVHMR", 0.15, clip)  # returns while the clip is still being logged
    assert logged == []
    release.set()
    streamer.on_stage(GENERATION_STAGE, 0.45, None)  # waits for it: upstream deletes the clip when the run ends
    assert logged == [clip]
