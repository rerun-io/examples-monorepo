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

from simplecv.sources.catalog_video import sample_indices  # noqa: E402
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
