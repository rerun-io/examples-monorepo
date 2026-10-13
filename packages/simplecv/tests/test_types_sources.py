"""simplecv shared Types and the catalog video source: serde, Rerun mapping structure, sampling (unit lane: no GPU, no catalog)."""

import numpy as np
import pyarrow as pa
import pytest
from dagkit.types import Log
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from serde import SerdeError, from_dict, to_dict

pytest.importorskip("torchcodec", reason="the catalog video source decodes with torchcodec")

import torch  # noqa: E402
from dagkit.sources import SegmentRef  # noqa: E402

from simplecv.sources.catalog_video import CatalogVideoSource, sample_indices  # noqa: E402
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


class FakeFrames:
    """Stand-in for ``SegmentFrames``: views of different constant colours, 10 ms apart."""

    def __init__(self, views: int, hw: tuple[int, int]) -> None:
        self.views, self.hw = views, hw

    def times_ns(self, view: int) -> np.ndarray:
        return np.arange(5, dtype=np.int64) * 10_000_000

    def frames_at(self, view: int, indices: list[int]) -> torch.Tensor:
        return torch.full((len(indices), 3, *self.hw), 40 * view + indices[0], dtype=torch.uint8)

    def close(self) -> None:
        pass


@settings(max_examples=20, deadline=None)
@given(views=st.integers(1, 3), native_hw=st.sampled_from([(32, 48), (40, 40)]), resize_hw=st.sampled_from([(16, 24), (8, 8)]), keep=st.booleans())
def test_source_resizes_and_puts_the_native_frames_first(views: int, native_hw: tuple[int, int], resize_hw: tuple[int, int], keep: bool) -> None:
    cameras = tuple(f"rig_{v:02d}/cam_00" for v in range(views))
    source = CatalogVideoSource(SegmentRef("d", "s"), cameras, decode="cpu_rgb", resize_hw=resize_hw, keep_native=keep)
    source.statics, source.frames = {}, FakeFrames(views, native_hw)  # type: ignore[assignment]
    events = list(source.events())
    assert [e.port for e in events] == (["frames_native", "frames"] if keep else ["frames"]) * 5
    assert ("frames_native" in source.outputs) == keep
    for event in events:
        frames: MultiViewFrames = event.value  # type: ignore[assignment]
        assert frames.rgb.shape == (views, 3, *(native_hw if event.port == "frames_native" else resize_hw)) and frames.t_ns == frames.frame_index * 10_000_000
        np.testing.assert_array_equal(frames.rgb[:, 0, 0, 0].numpy(), 40 * np.arange(views) + frames.frame_index)  # a constant image stays constant


def test_source_refuses_native_frames_without_a_resize() -> None:
    with pytest.raises(NotImplementedError, match="keep_native"):
        CatalogVideoSource(SegmentRef("d", "s"), ("a",), decode="cpu_rgb", keep_native=True)
