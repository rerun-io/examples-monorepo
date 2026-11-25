"""The maskers sample the framesets of the cube layer: its ``cube`` recording property is the one stride."""

from __future__ import annotations

from pathlib import Path

import pytest


def _write(path: Path, entity: str, cube_stride: int | None) -> None:
    import rerun as rr

    recording = rr.RecordingStream("dataforge", recording_id="clip")
    recording.save(path)
    if cube_stride is not None:
        recording.send_property("cube", rr.AnyValues(stride=cube_stride, framesets=4))
    recording.set_time("video_time", duration=0)
    recording.log(entity, rr.Points3D([[0.0, 0.0, 0.0]]))
    recording.flush()


def test_a_cube_layer_with_stride_7_makes_the_masker_sample_0_7_14(tmp_path: Path) -> None:
    from posekit.cube_layer import CubeSampling, read_cube_sampling, sampled_framesets

    _write(tmp_path / "cube.rrd", "/world/rig_00/cam_02", cube_stride=7)
    cube: CubeSampling | None = read_cube_sampling(tmp_path / "cube.rrd", "clip")
    assert cube == CubeSampling(stride=7, framesets=4)
    assert list(sampled_framesets("--stride", None, cube, 0, 1000)) == [0, 7, 14, 21]
    assert list(sampled_framesets("--stride", 7, cube, 7, 15)) == [7, 14]


def test_a_stride_that_disagrees_with_the_cube_layer_is_refused() -> None:
    from posekit.cube_layer import CubeSampling, sampled_framesets

    with pytest.raises(ValueError, match="cube layer's stride 7"):
        sampled_framesets("--stride", 10, CubeSampling(stride=7, framesets=4), 0, 100)
    with pytest.raises(ValueError, match="multiple of the cube stride 7"):
        sampled_framesets("--stride", None, CubeSampling(stride=7, framesets=4), 3, 100)


def test_without_a_cube_layer_the_stride_or_its_default_applies(tmp_path: Path) -> None:
    from posekit.cube_layer import read_cube_sampling, sampled_framesets

    _write(tmp_path / "base.rrd", "/world/rig_00/cam_00", cube_stride=None)
    assert read_cube_sampling(tmp_path / "base.rrd", "clip") is None
    assert list(sampled_framesets("--stride", None, None, 0, 25)) == [0, 10, 20]
    assert list(sampled_framesets("--stride", 5, None, 3, 14)) == [3, 8, 13]


def test_the_cube_layer_must_belong_to_the_recording(tmp_path: Path) -> None:
    from posekit.cube_layer import read_cube_sampling

    _write(tmp_path / "cube.rrd", "/world/rig_00/cam_02", cube_stride=7)
    with pytest.raises(ValueError, match="other"):
        read_cube_sampling(tmp_path / "cube.rrd", "other")


def test_a_cube_layer_file_overrides_the_segment_property(tmp_path: Path) -> None:
    import rerun as rr
    from rerun.catalog import DatasetEntry

    from posekit.cube_layer import CubeSampling, cube_sampling

    _write(tmp_path / "base.rrd", "/world/rig_00/cam_00", cube_stride=None)
    _write(tmp_path / "cube.rrd", "/world/rig_00/cam_02", cube_stride=7)
    with rr.server.Server(datasets={"in": [str(tmp_path / "base.rrd")]}) as server:
        dataset: DatasetEntry = server.client().get_dataset("in")
        assert cube_sampling(dataset, "clip", None) is None
        assert cube_sampling(dataset, "clip", tmp_path / "cube.rrd") == CubeSampling(stride=7, framesets=4)
