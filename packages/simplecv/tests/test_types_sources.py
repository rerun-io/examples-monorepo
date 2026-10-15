"""simplecv shared Types and the catalog video source: serde, Rerun mapping structure, sampling (unit lane: no GPU, no catalog)."""

from collections.abc import Sequence

import numpy as np
import pyarrow as pa
import pytest
from dagkit.types import Log
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from serde import SerdeError, from_dict, to_dict

pytest.importorskip("torchcodec", reason="the catalog video source decodes with torchcodec")

import cv2  # noqa: E402
import torch  # noqa: E402
from dagkit.node import Event, Fenced  # noqa: E402
from dagkit.sources import IterSource, SegmentRef  # noqa: E402
from dagkit.time import STATIC  # noqa: E402

from simplecv.catalog_video import CatalogVideo  # noqa: E402
from simplecv.sources import catalog_video  # noqa: E402
from simplecv.sources.catalog_video import (  # noqa: E402
    SegmentFrames,
    UndistortedFrames,
    VideoSource,
    sample_indices,
    static_events,
    undistortion_maps,
)
from simplecv.types.frames import MultiViewFrames  # noqa: E402
from simplecv.types.rig import RigCameras  # noqa: E402

FINITE = st.floats(-10.0, 10.0)


@st.composite
def rigs(draw: st.DrawFn) -> RigCameras:
    v: int = draw(st.integers(1, 5))
    names: tuple[str, ...] = tuple(f"rig_{i:02d}/cam_00" for i in range(v))
    return RigCameras(
        names,
        draw(arrays(np.float64, (v, 3, 3), elements=FINITE)),
        draw(arrays(np.float64, (v, 4, 4), elements=FINITE)),
        draw(arrays(np.int64, (v, 2), elements=st.integers(1, 4096))),
    )


@settings(max_examples=50, deadline=None)
@given(rig=rigs())
def test_rig_serde_roundtrip_keeps_dtype_and_shape(rig: RigCameras) -> None:
    back: RigCameras = from_dict(RigCameras, to_dict(rig))
    assert back.names == rig.names
    for name in ("intrinsics", "cam_T_world", "resolution_wh"):
        np.testing.assert_array_equal(getattr(back, name), getattr(rig, name))
        assert getattr(back, name).dtype == getattr(rig, name).dtype


def test_rig_refuses_unknown_fields() -> None:
    with pytest.raises(SerdeError):
        from_dict(RigCameras, {"names": [], "intrinsics": [], "cam_T_world": [], "resolution_wh": [], "color": 1})


@settings(max_examples=50, deadline=None)
@given(rig=rigs(), root=st.sampled_from(["/world", "/world/exocalib/init"]))
def test_rig_to_rerun_is_static_pose_plus_pinhole_per_camera(rig: RigCameras, root: str) -> None:
    logs = rig.to_rerun(root)
    assert [log.path for log in logs] == [p for name in rig.names for p in (f"{root}/{name}", f"{root}/{name}", f"{root}/{name}/pinhole")]
    assert all(log.static for log in logs)
    assert RigCameras.contents(root) == [f"{root}/**"]


def static_row(logs: list[Log]) -> pa.Table:
    """One catalog-style static row (``<entity>:<component>`` columns) from a Type's logs, without a server."""
    return pa.table({f"{log.path}:{b.component_descriptor()}": [b.as_arrow_array().to_pylist()] for log in logs for b in log.data.as_component_batches()})


@settings(max_examples=50, deadline=None)
@given(rig=rigs(), seed=st.integers(0, 2**16))
def test_rig_mapping_reads_back_the_camera_order_it_wrote(rig: RigCameras, seed: int) -> None:
    shuffled: RigCameras = rig.select(list(np.random.default_rng(seed).permutation(rig.names)))
    row: pa.Table = static_row(shuffled.to_rerun("/world"))
    ((_, back),) = RigCameras.from_layer(row, "/world", "video_time")
    assert back.names == shuffled.names
    # the strategy draws any 4x4 matrix; a pose reads back with the bottom row [0, 0, 0, 1], and Rerun stores float32
    np.testing.assert_allclose(back.cam_T_world[:, :3], shuffled.cam_T_world[:, :3], rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(back.intrinsics, shuffled.intrinsics, rtol=1e-6, atol=1e-5)
    np.testing.assert_array_equal(back.resolution_wh, shuffled.resolution_wh)
    no_index: pa.Table = row.select([c for c in row.column_names if not c.endswith(":view_index")])  # e.g. a dataset's GT rig
    ((_, gt),) = RigCameras.from_layer(no_index, "/world", "video_time")
    assert gt.names == tuple(sorted(rig.names))


@settings(max_examples=50, deadline=None)
@given(rig=rigs(), seed=st.integers(0, 2**16))
def test_rig_select_reorders_and_names_a_missing_camera(rig: RigCameras, seed: int) -> None:
    order: list[str] = list(np.random.default_rng(seed).permutation(rig.names))
    picked: RigCameras = rig.select(order)
    for i, name in enumerate(order):
        np.testing.assert_array_equal(picked.cam_T_world[i], rig.cam_T_world[rig.names.index(name)])
    with pytest.raises(KeyError, match="missing"):
        rig.select(["missing"])


@settings(max_examples=100, deadline=None)
@given(n=st.integers(1, 500), budget=st.integers(1, 600), start=st.integers(0, 400), length=st.integers(1, 400))
def test_sample_indices_spread_a_budget_inside_the_window(n: int, budget: int, start: int, length: int) -> None:
    times = np.arange(n, dtype=np.int64) * 10
    everything = sample_indices(times, budget, None)
    assert everything.size == min(budget, n) and everything[0] == 0 and (everything[-1] == n - 1 or budget == 1) and np.all(np.diff(everything) > 0)
    np.testing.assert_array_equal(sample_indices(times, "all", None), np.arange(n))
    window = (start * 10, (start + length) * 10)
    if start >= n:
        with pytest.raises(ValueError, match="no frames"):
            sample_indices(times, budget, window)
        return
    inside = sample_indices(times, budget, window)
    assert np.all((times[inside] >= window[0]) & (times[inside] < window[1])) and inside.size == min(budget, min(n, start + length) - start)
    np.testing.assert_array_equal(sample_indices(times, (n - 1, 0, n - 1), None), np.unique([0, n - 1]))


@settings(max_examples=50, deadline=None)
@given(rig=rigs(), height=st.integers(16, 4096), width=st.integers(16, 4096))
def test_rig_scaled_to_a_size_and_back_is_the_identity(rig: RigCameras, height: int, width: int) -> None:
    scaled: RigCameras = rig.scaled_to(height, width)
    assert np.all(scaled.resolution_wh == (width, height)) and scaled.names == rig.names
    np.testing.assert_array_equal(scaled.cam_T_world, rig.cam_T_world)
    np.testing.assert_allclose(scaled.intrinsics[:, 0], rig.intrinsics[:, 0] * width / rig.resolution_wh[:, :1], rtol=1e-12)
    np.testing.assert_array_equal(scaled.intrinsics[:, 2], rig.intrinsics[:, 2])
    back: RigCameras = scaled.select(rig.names[:1]).scaled_to(int(rig.resolution_wh[0, 1]), int(rig.resolution_wh[0, 0]))
    np.testing.assert_allclose(back.intrinsics, rig.intrinsics[:1], rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(back.resolution_wh, rig.resolution_wh[:1])


NO_STATICS = IterSource({}, lambda: iter(()))


class FakeFrames:
    """Stand-in for ``SegmentFrames``: views of different constant colours, 10 ms apart."""

    def __init__(self, views: int, hw: tuple[int, int]) -> None:
        self.views, self.hw = views, hw

    def times_ns(self, view: int) -> np.ndarray:
        return np.arange(5, dtype=np.int64) * 10_000_000

    def frames_at(self, view: int, indices: Sequence[int]) -> torch.Tensor:
        return torch.full((len(indices), 3, *self.hw), 40 * view + indices[0], dtype=torch.uint8)

    def close(self) -> None:
        pass


@settings(max_examples=20, deadline=None)
@given(views=st.integers(1, 3), native_hw=st.sampled_from([(32, 48), (40, 40)]), resize_hw=st.sampled_from([(16, 24), (8, 8)]), keep=st.booleans())
def test_source_resizes_and_puts_the_native_frames_first(views: int, native_hw: tuple[int, int], resize_hw: tuple[int, int], keep: bool) -> None:
    cameras = tuple(f"rig_{v:02d}/cam_00" for v in range(views))
    source = VideoSource(FakeFrames(views, native_hw), cameras, NO_STATICS, resize_hw=resize_hw, keep_native=keep)
    events = list(source.events())
    assert [e.port for e in events] == (["frames_native", "frames"] if keep else ["frames"]) * 5
    assert ("frames_native" in source.outputs) == keep
    for event in events:
        frames: MultiViewFrames = event.value  # type: ignore[assignment]
        assert frames.rgb.shape == (views, 3, *(native_hw if event.port == "frames_native" else resize_hw)) and frames.t_ns == frames.frame_index * 10_000_000
        np.testing.assert_array_equal(frames.rgb[:, 0, 0, 0].numpy(), 40 * np.arange(views) + frames.frame_index)  # a constant image stays constant


def test_source_refuses_native_frames_without_a_resize() -> None:
    with pytest.raises(NotImplementedError, match="keep_native"):
        VideoSource(FakeFrames(1, (8, 8)), ("a",), NO_STATICS, keep_native=True)


def test_a_source_over_any_frame_provider_emits_its_statics_then_the_window() -> None:
    """A file capture (mamma's MAMMA-format directories) composes the same source: its own rig event, its own frames, no catalog."""
    rig = RigCameras(("a", "b"), np.repeat(np.eye(3)[None], 2, 0), np.repeat(np.eye(4)[None], 2, 0), np.array([[48, 32], [48, 32]]))
    statics = IterSource({"rig": RigCameras}, lambda: iter([Event(STATIC, "rig", rig)]))
    source = VideoSource(FakeFrames(2, (32, 48)), ("a", "b"), statics, window_ns=(10_000_000, 40_000_000), fenced=True)
    events = list(source.events())
    assert source.outputs == {"frames": Fenced[MultiViewFrames], "rig": RigCameras} and source.clock == "video_time"
    assert [(e.port, e.t_ns) for e in events] == [("rig", STATIC), ("frames", 10_000_000), ("frames", 20_000_000), ("frames", 30_000_000)]
    assert all(isinstance(e.value, Fenced) for e in events[1:])


@settings(max_examples=50, deadline=None)
@given(
    f=st.floats(250.0, 600.0), offset=st.tuples(st.floats(-15.0, 15.0), st.floats(-15.0, 15.0)), k1=st.floats(-0.3, 0.3), k2=st.floats(-0.1, 0.1),
    p=st.tuples(st.floats(-2e-3, 2e-3), st.floats(-2e-3, 2e-3)), uv=arrays(np.float64, (20, 2), elements=st.floats(0.1, 0.9)),
)  # fmt: skip
def test_undistortion_maps_send_each_undistorted_pixel_to_its_distorted_source(
    f: float, offset: tuple[float, float], k1: float, k2: float, p: tuple[float, float], uv: np.ndarray
) -> None:
    width, height = 160, 120  # a small image keeps the maps cheap; the intrinsics scale with it
    K = np.array([[f / 4, 0.0, (width - 1) / 2 + offset[0] / 4], [0.0, f / 4, (height - 1) / 2 + offset[1] / 4], [0.0, 0.0, 1.0]])
    coefficients = np.array([k1, k2, p[0], p[1], 0.0] + [0.0] * 9)  # the catalog's zero-padded 14-vector
    K_new, map_x, map_y = undistortion_maps(K, coefficients, width, height)
    np.testing.assert_allclose(K_new[:2, 2], [(width - 1) / 2, (height - 1) / 2])  # centred principal point
    pixels = uv * [width - 1, height - 1]  # inside the undistorted image
    rays = np.c_[(pixels - K_new[:2, 2]) / K_new[[0, 1], [0, 1]], np.ones(len(pixels))]
    distorted, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), K, coefficients)
    sampled = np.stack([cv2.remap(m, pixels[:, :1].astype(np.float32), pixels[:, 1:].astype(np.float32), cv2.INTER_LINEAR)[:, 0] for m in (map_x, map_y)], axis=1)
    np.testing.assert_allclose(sampled, distorted[:, 0], atol=0.02)


def rig_with_distortion(K: np.ndarray, width: int, height: int, coefficients: list[float] | None, model: str = "brown_conrady") -> pa.Table:
    rig = RigCameras(("rig_00/cam_00",), K[None], np.eye(4)[None], np.array([[width, height]]))
    row: pa.Table = static_row(rig.to_rerun("/world"))
    if coefficients is not None:
        row = row.append_column("/world/rig_00/cam_00/pinhole:simplecv.components.DistortionCoefficients", pa.array([[coefficients]], pa.list_(pa.list_(pa.float32()))))
        row = row.append_column("/world/rig_00/cam_00/pinhole:simplecv.components.DistortionModel", pa.array([[model]]))
    return row


class TextureFrames(FakeFrames):
    """The same fixed random texture in every frame."""

    def frames_at(self, view: int, indices: Sequence[int]) -> torch.Tensor:
        texture = np.random.default_rng(0).integers(0, 256, (3, *self.hw), dtype=np.uint8)
        return torch.from_numpy(texture)[None].repeat(len(indices), 1, 1, 1)


def test_source_undistorts_its_frames_and_emits_the_undistorted_pinhole(monkeypatch: pytest.MonkeyPatch) -> None:
    width, height = 64, 48
    K = np.array([[60.0, 0.0, 33.0], [0.0, 60.0, 22.0], [0.0, 0.0, 1.0]])
    coefficients = [-0.25, 0.05, 1e-3, -1e-3, 0.0]
    monkeypatch.setattr(catalog_video, "read_column", lambda *args, **kwargs: rig_with_distortion(K, width, height, coefficients))
    monkeypatch.setattr(SegmentRef, "entry", lambda self: None)
    segment, cameras = SegmentRef("d", "s"), ("rig_00/cam_00",)
    undistorted = UndistortedFrames(TextureFrames(1, (height, width)))  # what VideoSource.catalog(..., undistort=True) composes
    source = VideoSource(undistorted, cameras, IterSource({"rig": RigCameras}, lambda: static_events(segment, cameras, {}, undistorted)))
    list(source.events())  # a graph reads the source again after a first look (e.g. at the frame size)
    events = list(source.events())
    rig: RigCameras = events[0].value  # type: ignore[assignment]
    K_new, map_x, map_y = undistortion_maps(K, np.asarray(coefficients, dtype=np.float32).astype(np.float64), width, height)
    np.testing.assert_allclose(rig.intrinsics[0], K_new)
    np.testing.assert_array_equal(rig.resolution_wh, [[width, height]])
    native = undistorted.frames.frames_at(0, [0])[0].permute(1, 2, 0).numpy().copy()
    for event in events[1:]:
        frames: MultiViewFrames = event.value  # type: ignore[assignment]
        np.testing.assert_array_equal(frames.rgb[0].permute(1, 2, 0).numpy(), cv2.remap(native, map_x, map_y, cv2.INTER_LINEAR))


def test_source_refuses_to_undistort_without_coefficients_or_off_the_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(catalog_video, "read_column", lambda *args, **kwargs: rig_with_distortion(np.eye(3), 8, 8, None))
    monkeypatch.setattr(SegmentRef, "entry", lambda self: None)
    segment, cameras = SegmentRef("d", "s"), ("rig_00/cam_00",)
    undistorted = UndistortedFrames(FakeFrames(1, (8, 8)))
    source = VideoSource(undistorted, cameras, IterSource({"rig": RigCameras}, lambda: static_events(segment, cameras, {}, undistorted)))
    with pytest.raises(ValueError, match="DistortionCoefficients"):
        list(source.events())
    fisheye = rig_with_distortion(np.eye(3), 8, 8, [0.1, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "kannala_brandt")  # 8 terms OpenCV would misread
    monkeypatch.setattr(catalog_video, "read_column", lambda *args, **kwargs: fisheye)
    with pytest.raises(ValueError, match="brown_conrady"):
        list(source.events())
    with pytest.raises(NotImplementedError, match="undistort"):
        VideoSource.catalog(SegmentRef("d", "s"), ("a",), decode="nvdec_rgb", undistort=True)


def test_segment_frames_fetch_every_camera_in_one_read(monkeypatch: pytest.MonkeyPatch) -> None:
    reads: list[list[str]] = []

    def read(dataset: object, segment: str, entities: list[str], timeline: str, window_ns: object = None) -> tuple[CatalogVideo, ...]:
        reads.append(entities)
        times = [np.arange(3, dtype=np.int64).astype("timedelta64[ns]") * (i + 1) for i in range(len(entities))]
        return tuple(CatalogVideo(t, [np.full(2, i, np.uint8)] * 3, [True, False, False], "h264") for i, t in enumerate(times))

    monkeypatch.setattr(catalog_video, "read_catalog_videos", read)
    monkeypatch.setattr(catalog_video, "wrap_mp4", lambda samples, keyframes, fps, codec: b"".join(bytes(s) for s in samples))
    monkeypatch.setattr(SegmentRef, "entry", lambda self: None)
    frames = SegmentFrames(SegmentRef("d", "s"), ("rig_00/cam_00", "rig_01/cam_00"), "cpu")
    np.testing.assert_array_equal(frames.times_ns(1), [0, 2, 4])
    np.testing.assert_array_equal(frames.times_ns(0), [0, 1, 2])
    assert reads == [["/world/rig_00/cam_00/pinhole/video", "/world/rig_01/cam_00/pinhole/video"]]
