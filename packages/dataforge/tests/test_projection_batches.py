"""Projection writers batch a camera's native hand clock."""

from pathlib import Path

import numpy as np
import pytest
from conftest import read_chunks, vrs_file

from dataforge import aria, schema
from dataforge.datasets.aria_gen2_pilot_layers import write_projections
from dataforge.datasets.aria_gen2_pilot_source import Camera, HandSamples, Scene
from dataforge.vrs import VrsFile
from dataforge.writing import recording_to


def test_projection_writer_batches_the_camera_clock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The writer must not return to the scalar frame/joint projection path."""
    fixture: Path = Path(__file__).parent / "fixtures/aria/gen2-pilot-clean_0-calib.json"
    calibration = aria.DeviceCalibration.from_json(fixture.read_text(), "test")
    times = np.array([0, 33_333_333, 66_666_667], dtype=np.int64)
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[-1] = np.nan
    positions = np.full((3, 133, 3), np.nan, dtype=np.float32)
    camera = calibration.camera("slam-front-left")
    positions[:2, 91] = camera.rig_T_cam.matrix()[:3, 3] + camera.rig_T_cam.matrix()[:3, 2]
    samples = HandSamples(
        times, positions, np.ones((3, 133), dtype=np.float32),
        np.full((3, 2, 4, 4), np.nan), np.full((3, 2, 2, 3), np.nan),
        np.ones((3, 2)), poses, np.arange(3, dtype=np.int64),
    )
    source: Path = tmp_path / "video.vrs"
    source.write_bytes(vrs_file({214: {}}, []))
    scene = Scene(source.parent, VrsFile(source), calibration, [Camera("1201-1", "slam-front-left", 30, camera, times, 3)],
                  aria.Trajectory(times, poses, np.ones(3)), samples, None)

    def reject_scalar(*_args, **_kwargs) -> None:
        raise AssertionError("scalar projection path")

    monkeypatch.setattr(aria, "project_to_calibration", reject_scalar)
    target: Path = tmp_path / "projections.rrd"
    with recording_to(target, recording_id="test", send_properties=False) as recording:
        write_projections(recording, scene)
    chunks = [chunk for chunk in read_chunks(target) if str(chunk.entity_path) == schema.coco133_uv_projected_path(0, 0)]
    assert any("Points2D:positions" in chunk.to_record_batch().schema.names for chunk in chunks)
