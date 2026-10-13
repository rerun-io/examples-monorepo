"""``CatalogVideoSource(decode="cpu_luma")`` on synthetic lossless H.264 (unit lane: PyAV on the CPU, the catalog read faked)."""

from fractions import Fraction

import av
import numpy as np
import pytest
from dagkit.node import Event
from dagkit.sources import SegmentRef
from jaxtyping import Int64, UInt8
from numpy import ndarray

import simplecv.sources.catalog_video as catalog_video
from simplecv.catalog_video import CatalogVideo
from simplecv.ops.framesets import group_framesets
from simplecv.types.frames import LumaFrameset

HEIGHT, WIDTH, GOP = 48, 64, 5
CAMERAS: tuple[str, ...] = ("rig_00/cam_00", "rig_00/cam_01", "rig_00/cam_02")


def encode(levels: list[int]) -> tuple[list[UInt8[ndarray, " b"]], list[bool]]:
    """Constant-luma frames as lossless H.264 (qp 0, a keyframe every GOP frames, parameter sets on every keyframe)."""
    codec = av.CodecContext.create("libx264", "w")
    codec.width, codec.height, codec.pix_fmt, codec.time_base = WIDTH, HEIGHT, "yuv420p", Fraction(1, 30)
    codec.options = {"qp": "0", "g": str(GOP), "keyint_min": str(GOP), "bf": "0", "sc_threshold": "0"}
    packets: list[av.Packet] = []
    for i, level in enumerate(levels):
        planes: UInt8[ndarray, "hh w"] = np.full((HEIGHT * 3 // 2, WIDTH), 128, dtype=np.uint8)
        planes[:HEIGHT] = level
        frame = av.VideoFrame.from_ndarray(planes, format="yuv420p")
        frame.pts = i
        packets += codec.encode(frame)
    packets += codec.encode(None)
    return [np.frombuffer(bytes(p), dtype=np.uint8) for p in packets], [bool(p.is_keyframe) for p in packets]


def camera_times() -> list[Int64[ndarray, " n"]]:
    """30 Hz with per-camera jitter; camera 2 skips frames 7 and 8 and starts 1 ms late."""
    base: Int64[ndarray, " n"] = 1_000_000_000 + np.arange(20, dtype=np.int64) * 33_000_000
    return [base, base + 1_200_000, np.delete(base, [7, 8]) + 1_000_000]


@pytest.fixture
def fake_catalog(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int] | None]:
    """Serve the synthetic cameras through ``read_catalog_videos`` (luma level = 10 + 10 x frame number); records each read window."""
    windows: list[tuple[int, int] | None] = []
    videos: list[CatalogVideo] = []
    for times in camera_times():
        frame_numbers: Int64[ndarray, " n"] = (times - 1_000_000_000) // 33_000_000
        samples, keyframes = encode([10 + 10 * int(n) for n in frame_numbers])
        videos.append(CatalogVideo(times.astype("timedelta64[ns]"), samples, keyframes, "h264"))

    def read(dataset: object, segment: str, entities: list[str], timeline: str, window_ns: tuple[int, int] | None = None) -> tuple[CatalogVideo, ...]:
        assert entities == [f"/world/{c}/pinhole/video" for c in CAMERAS] and timeline == "video_time"
        windows.append(window_ns)
        keep = [np.ones(v.t_ns.size, bool) if window_ns is None else (v.t_ns >= window_ns[0]) & (v.t_ns < window_ns[1]) for v in videos]
        return tuple(CatalogVideo(v.times[k], [s for s, ok in zip(v.samples, k, strict=True) if ok], [f for f, ok in zip(v.keyframes, k, strict=True) if ok], v.codec)
                     for v, k in zip(videos, keep, strict=True))  # fmt: skip

    monkeypatch.setattr(catalog_video, "read_catalog_videos", read)
    monkeypatch.setattr(SegmentRef, "entry", lambda self: None)
    return windows


def framesets(window_ns: tuple[int, int] | None) -> list[LumaFrameset]:
    source = catalog_video.CatalogVideoSource(SegmentRef("d", "s", window_ns=window_ns), CAMERAS, decode="cpu_luma", group_tolerance_ns=3_000_000)
    assert source.outputs["frames"] is LumaFrameset
    source.statics = {}
    events: list[Event] = list(source.events())
    assert all(e.port == "frames" and isinstance(e.value, LumaFrameset) for e in events)
    return [e.value for e in events]  # type: ignore[misc]


@pytest.mark.parametrize("window_ns", [None, (1_000_000_000 + 8 * 33_000_000, 1_000_000_000 + 16 * 33_000_000)])
def test_luma_framesets_hold_each_cameras_raw_y_plane_grouped_like_the_op(fake_catalog: list, window_ns: tuple[int, int] | None) -> None:
    got: list[LumaFrameset] = framesets(window_ns)
    lo, hi = window_ns or (0, 2**62)
    inside: list[Int64[ndarray, " n"]] = [t[(t >= lo) & (t < hi)] for t in camera_times()]
    index = group_framesets(inside, 3_000_000)
    assert [f.t_ns for f in got] == index.t_ns.tolist()  # window [lo, hi): the first frameset starts mid-GOP, decoded from its keyframe
    if window_ns is not None:
        assert fake_catalog == [(lo - catalog_video.KEYFRAME_LEAD_NS, hi)]
    for frameset, frames, stamps in zip(got, index.frame, index.cam_t_ns, strict=True):
        assert frameset.views == CAMERAS
        for c, plane in enumerate(frameset.luma):
            if frames[c] < 0:
                assert plane is None and frameset.cam_t_ns[c] is None
                continue
            assert plane is not None and plane.shape == (HEIGHT, WIDTH) and plane.dtype == np.uint8
            assert frameset.cam_t_ns[c] == int(stamps[c])
            np.testing.assert_array_equal(plane, 10 + 10 * ((int(stamps[c]) - 1_000_000_000) // 33_000_000))  # lossless: the frame's own level


def test_luma_decode_needs_grouping_and_rgb_decode_refuses_it() -> None:
    with pytest.raises(NotImplementedError):
        catalog_video.CatalogVideoSource(SegmentRef("d", "s"), CAMERAS, decode="cpu_luma")
    with pytest.raises(NotImplementedError):
        catalog_video.CatalogVideoSource(SegmentRef("d", "s"), CAMERAS, decode="cpu_rgb", group_tolerance_ns=3_000_000)
