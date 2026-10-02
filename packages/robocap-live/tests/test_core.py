"""The Rust hands-layer core through its binding, with no networks (``nets="none"``): validation, zero-copy frames, the pose
store rule, the file."""

import gc
import weakref
from pathlib import Path

import numpy as np
import pytest

from robocap_live import _core
from robocap_live.apis.hands_layer import default_ort_dylib

HEIGHT: int = _core.FULL_HEIGHT
WIDTH: int = _core.FULL_WIDTH


def rig(names: list[str] | None = None) -> _core.Rig:
    count: int = 6
    cam_from_rig: np.ndarray = np.tile(np.eye(4), (count, 1, 1))
    cam_from_rig[:, 0, 3] = np.arange(count) * 0.05
    return _core.Rig(
        names=list(_core.CAMERA_NAMES) if names is None else names,
        resolution_wh=np.tile([float(WIDTH), float(HEIGHT)], (count, 1)),
        cam_from_rig=cam_from_rig,
        focal=np.full((count, 2), 600.0),
        principal=np.tile([959.5, 539.5], (count, 1)),
        fisheye62=np.zeros((count, 8)),
        source="test",
        device="test-device",
    )


def test_a_rig_keeps_its_cameras_and_refuses_another_order() -> None:
    built = rig()
    assert built.names == list(_core.CAMERA_NAMES) and built.device == "test-device"
    assert '"name":"left_front"' in built.to_json()
    with pytest.raises(ValueError, match="where left_front was expected"):
        rig(names=list(reversed(_core.CAMERA_NAMES)))


def reference(times: list[int], translations: list[float | None]) -> tuple[np.ndarray, np.ndarray]:
    """A reference pose table: one row per time, NaN where the translation is None."""
    poses: np.ndarray = np.tile(np.eye(4), (len(times), 1, 1))
    for row, x in enumerate(translations):
        if x is None:
            poses[row] = np.nan
        else:
            poses[row, 0, 3] = x
    return np.asarray(times, dtype=np.int64), poses


def test_a_layer_runs_borrowed_and_copied_frames_through_the_pipeline_and_refuses_bad_input(tmp_path: Path) -> None:
    output: Path = tmp_path / "layer.rrd"
    # Framesets 33 ms apart (a pose matches within 2 ms); the second has none.
    times, poses = reference([1_000_000_000, 1_033_000_000], [0.5, None])
    layer = _core.HandsLayer(rig(), output, "segment", times, poses, nets="none")
    assert layer.nets.startswith("none")
    frame: np.ndarray = np.full((HEIGHT, WIDTH), 80, dtype=np.uint8)
    padded: np.ndarray = np.full((HEIGHT, WIDTH + 64), 80, dtype=np.uint8)[:, :WIDTH]
    frames: list[np.ndarray | None] = [frame, padded, None, frame, frame, None]
    assert layer.push(1_000_000_000, frames, [1_000_000_000, 1_000_100_000, 0, 1_000_200_000, 1_000_300_000, 0]) == 0
    assert layer.push(1_033_000_000, [frame] * 6) == 1
    with pytest.raises(ValueError, match="does not follow"):
        layer.push(1_033_000_000, [frame] * 6)
    with pytest.raises(ValueError, match="6"):
        layer.push(1_066_000_000, [frame] * 5)
    with pytest.raises(ValueError, match="shape"):
        layer.push(1_066_000_000, [np.zeros((360, 640), dtype=np.uint8)] * 6)
    with pytest.raises(ValueError, match="uint8"):
        layer.push(1_066_000_000, [frame.astype(np.float32)] * 6)
    summary = layer.finish()
    # The second frameset has no pose of its own: the pipeline's pose store hands it the first one's.
    assert (summary.framesets, summary.with_pose, summary.held_pose) == (2, 1, 1)
    assert summary.reported == (0, 0) and not summary.scale_final
    assert summary.stage_total_ms("downsample") > 0.0
    assert output.stat().st_size > 0
    with pytest.raises(ValueError, match="finished"):
        layer.push(2_000_000_000, [frame] * 6)


def test_the_pipeline_holds_a_borrowed_frame_until_it_is_done_and_then_releases_it(tmp_path: Path) -> None:
    times, poses = reference([1_000_000_000], [0.0])
    layer = _core.HandsLayer(rig(), tmp_path / "layer.rrd", "segment", times, poses, nets="none")
    frame: np.ndarray = np.full((HEIGHT, WIDTH), 80, dtype=np.uint8)
    alive = weakref.ref(frame)
    layer.push(1_000_000_000, [frame] * 6)
    # The caller lets go while the pipeline's threads may still read the frame: the pipeline holds its own references.
    del frame
    gc.collect()
    layer.push(1_033_000_000, [np.full((HEIGHT, WIDTH), 81, dtype=np.uint8)] * 6)
    assert layer.finish().framesets == 2
    # References the pipeline's threads dropped without the GIL are released on the next call into the extension.
    assert layer.nets.startswith("none")
    gc.collect()
    assert alive() is None


def test_an_aborted_layer_takes_no_more_framesets(tmp_path: Path) -> None:
    times, poses = reference([1_000_000_000], [0.0])
    layer = _core.HandsLayer(rig(), tmp_path / "layer.rrd", "segment", times, poses, nets="none")
    layer.push(1_000_000_000, [np.full((HEIGHT, WIDTH), 80, dtype=np.uint8)] * 6)
    layer.abort()
    layer.abort()
    with pytest.raises(ValueError, match="finished"):
        layer.push(1_033_000_000, [None] * 6)


def test_the_reference_poses_need_one_4x4_per_time(tmp_path: Path) -> None:
    times, poses = reference([1_000, 34_000], [0.5, 0.6])
    with pytest.raises(ValueError, match="shape"):
        _core.HandsLayer(rig(), tmp_path / "layer.rrd", "segment", times, poses[:1], nets="none")


def test_ort_nets_need_a_models_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="models_dir"):
        _core.HandsLayer(rig(), tmp_path / "layer.rrd", "segment", *reference([0], [0.0]), nets="ort")


def test_the_onnx_runtime_library_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ORT_DYLIB_PATH", str(tmp_path / "libonnxruntime.so"))
    assert default_ort_dylib() == tmp_path / "libonnxruntime.so"
    monkeypatch.delenv("ORT_DYLIB_PATH")
    assert default_ort_dylib().name.startswith("libonnxruntime.so")
