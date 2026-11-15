"""The example run end to end (GPU, weights): upstream's outputs and our Rerun recording of them."""

from pathlib import Path

import av
import pytest
import rerun as rr
from rerun.chunk import RrdReader

from fdanyone_rerun.assets import EXAMPLE_VIDEO, model_dir
from fdanyone_rerun.outputs import read_camera_rig, read_run_metadata
from fdanyone_rerun.rerun_log import camera_entity, log_run

pytestmark = pytest.mark.integration


def test_example_run_is_the_space_policy(example_run: Path) -> None:
    metadata = read_run_metadata(example_run)
    rig = read_camera_rig(example_run)
    assert metadata.input.num_frames == 121
    assert metadata.generation.inference_steps == 4
    assert [camera.pitch for camera in rig.cameras] == [15] * 6
    assert [round(camera.yaw) for camera in rig.cameras] == [0, 60, 120, 180, 240, 300]
    for camera in rig.cameras:
        with av.open(str(example_run / camera.video)) as container:
            stream = container.streams.video[0]
            frames: int = sum(1 for _ in container.decode(stream))
            assert (stream.codec_context.width, stream.codec_context.height) == (camera.image_width, camera.image_height)
        assert frames == 121


def test_logs_the_run_to_an_rrd(example_run: Path, tmp_path: Path) -> None:
    recording = rr.RecordingStream("fdanyone-rerun-test")
    recording.save(tmp_path / "run.rrd")
    log_run(example_run, EXAMPLE_VIDEO, model_dir(), recording)
    recording.flush()
    recording.disconnect()
    entities: set[str] = {str(chunk.entity_path) for chunk in RrdReader(tmp_path / "run.rrd").stream().to_chunks()}
    expected: set[str] = {"/world/body/mesh", "/world/source", "/world/source/video"} | {f"/{camera_entity(i)}/video" for i in range(6)}
    assert expected <= entities
