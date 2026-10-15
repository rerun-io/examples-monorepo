"""The MAMMA-format file source: its npz rig contract (unit lane, no video decode)."""

from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from mamma.sources.npz import load_rig


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

    from mamma.sources.npz import NpzSequenceSource

    (tmp_path / "meta").mkdir()
    (tmp_path / "videos").mkdir()
    np.savez(tmp_path / "meta" / "global.npz", seq_name="probe", fps=30.0, frame_start=0, frame_end=10)
    np.savez(tmp_path / "meta" / "A001.npz", cam_name="A001", cam_int=np.diag([900.0, 900.0, 1.0]), cam_ext=np.eye(4), cam_img_w=64, cam_img_h=48)
    source = NpzSequenceSource(tmp_path, ("A001",), resize_hw=(48, 64), keep_native=False)

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
