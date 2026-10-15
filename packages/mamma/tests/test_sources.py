"""The MAMMA-format file source (its npz rig contract) and the capture graph's frameset source (unit lane, no video decode)."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import torch
from dagkit.node import Event, Fenced
from dagkit.sources import IterSource
from dagkit.time import STATIC
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from simplecv.sources.catalog_video import VideoSource
from simplecv.types.rig import RigCameras

from mamma.sources.frameset import FramesetSource
from mamma.sources.npz import load_rig
from mamma.types import Frameset


@settings(max_examples=25, deadline=None)
@given(translation_m=arrays(np.float64, (2, 3), elements=st.floats(-5.0, 5.0)), millimetres=st.booleans())
def test_npz_rig_reads_metres_and_millimetres(tmp_path_factory, translation_m: np.ndarray, millimetres: bool) -> None:  # type: ignore[no-untyped-def]
    meta: Path = tmp_path_factory.mktemp("meta")
    translation_m[:, 0] = 1.0  # at least one large component so the millimetre rule applies after x1000
    for i, name in enumerate(("A001", "B001")):
        cam_T_world: np.ndarray = np.eye(4)
        cam_T_world[:3, 3] = translation_m[i] * (1000.0 if millimetres else 1.0)
        np.savez(meta / f"{name}.npz", cam_name=name, cam_int=np.diag([900.0, 900.0, 1.0]), cam_ext=cam_T_world, cam_img_w=3840, cam_img_h=2160)
    rig = load_rig(meta, ("B001", "A001"))
    assert rig.names == ("B001", "A001")
    np.testing.assert_allclose(rig.cam_T_world[:, :3, 3], translation_m[::-1], rtol=1e-12)
    np.testing.assert_array_equal(rig.resolution_wh, [[3840, 2160], [3840, 2160]])


class _FramesReached(Exception):
    """Raised by the stub frames: the source got past its statics."""


def test_npz_source_never_opens_the_catalog(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from dagkit.sources import SegmentRef

    from mamma.sources.npz import npz_sequence_source

    (tmp_path / "meta").mkdir()
    (tmp_path / "videos").mkdir()
    np.savez(tmp_path / "meta" / "global.npz", seq_name="probe", fps=30.0, frame_start=0, frame_end=10)
    np.savez(tmp_path / "meta" / "A001.npz", cam_name="A001", cam_int=np.diag([900.0, 900.0, 1.0]), cam_ext=np.eye(4), cam_img_w=64, cam_img_h=48)
    source = npz_sequence_source(tmp_path, ("A001",), resize_hw=(48, 64), keep_native=False)

    def no_catalog(self: SegmentRef) -> None:
        raise AssertionError(f"opened the catalog for {self.dataset!r}")

    def frames_reached(view: int) -> None:
        raise _FramesReached

    monkeypatch.setattr(SegmentRef, "entry", no_catalog)
    monkeypatch.setattr(source.frames, "times_ns", frames_reached)
    events = source.events()
    assert next(events).port == "rig"
    try:
        next(events)
    except _FramesReached:
        return
    raise AssertionError("the source yielded past its statics without reading frames")


class _CountingFrames:
    """A FrameProvider whose every pixel of frame ``i`` is ``i`` (64 x 32 per camera, 30 fps)."""

    def times_ns(self, view: int) -> np.ndarray:
        return np.arange(5, dtype=np.int64) * 33_333_333

    def frames_at(self, view: int, indices: Sequence[int]) -> torch.Tensor:
        return torch.stack([torch.full((3, 32, 64), i, dtype=torch.uint8) for i in indices])

    def close(self) -> None:
        pass


@pytest.mark.parametrize("keep_native", [True, False])
def test_frameset_source_hands_each_tick_its_crop_frames_in_one_value(keep_native: bool) -> None:
    rig = Event(STATIC, "rig", "the rig")
    video = VideoSource(_CountingFrames(), ("a", "b"), IterSource({"rig": RigCameras}, lambda: iter([rig])), fenced=True, resize_hw=(16, 32), keep_native=keep_native)
    source = FramesetSource(video)
    assert source.outputs == {"frames": Fenced[Frameset], "rig": RigCameras}
    events = list(source.events())
    assert [e.port for e in events] == ["rig"] + ["frames"] * 5
    for index, event in enumerate(events[1:]):
        assert isinstance(event.value, Fenced) and isinstance(event.value.value, Frameset)
        frameset: Frameset = event.value.value
        assert event.t_ns == frameset.frames.t_ns == frameset.crops.t_ns and frameset.frames.frame_index == frameset.crops.frame_index == index
        assert tuple(frameset.frames.rgb.shape) == (2, 3, 16, 32)
        assert tuple(frameset.crops.rgb.shape) == ((2, 3, 32, 64) if keep_native else (2, 3, 16, 32))
        assert int(frameset.crops.rgb.max()) == index
