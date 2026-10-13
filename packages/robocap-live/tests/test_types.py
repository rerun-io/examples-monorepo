"""robocap-live Types: serde round-trips and to_rerun -> from_layer on catalog-shaped rows (unit lane: no catalog, no GPU)."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import fields

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
from dagkit.time import STATIC
from dagkit.types import Log
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from jaxtyping import Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from serde import SerdeError, from_dict, to_dict
from simplecv.types.rig import RigCameras

from robocap_live.types import RIG_ROOT, CapRig, CatalogPoses, LayerRun

FLOAT32 = st.floats(-100.0, 100.0, width=32)
NAMES: tuple[str, ...] = ("left_front", "left_side", "left_back", "right_front", "right_side", "right_back")


def catalog_rows(logs: Sequence[tuple[int, Log]]) -> pa.Table:
    """What a catalog reader returns for these logs: one row per time, ``<entity>:<component>`` columns, statics in every row."""
    cells: dict[str, dict[int, list]] = defaultdict(dict)
    statics: dict[str, list] = {}
    for t_ns, log in logs:
        for batch in log.data.as_component_batches():
            name: str = f"{log.path}:{batch.component_descriptor()}"
            if log.static:
                statics[name] = batch.as_arrow_array().to_pylist()
            else:
                cells[name][t_ns] = batch.as_arrow_array().to_pylist()
    times: list[int] = sorted({t for cell in cells.values() for t in cell}) or [0]
    columns: dict[str, list] = (
        {"video_time": times} | {n: [c.get(t) for t in times] for n, c in cells.items()} | {n: [v] * len(times) for n, v in statics.items()}
    )
    return pa.table({name: pa.array(column, type=pa.int64()) if name == "video_time" else column for name, column in columns.items()})


@st.composite
def cap_rigs(draw: st.DrawFn) -> CapRig:
    """Six cameras with float32-exact calibrations (Rerun stores them as float32)."""
    rotation: Float64[ndarray, "6 3 3"] = Rotation.from_rotvec(draw(arrays(np.float64, (6, 3), elements=st.floats(-3.0, 3.0)))).as_matrix()
    cam_from_rig: Float64[ndarray, "6 4 4"] = np.tile(np.eye(4), (6, 1, 1))
    cam_from_rig[:, :3, :3] = rotation.astype(np.float32)
    cam_from_rig[:, :3, 3] = draw(arrays(np.float32, (6, 3), elements=FLOAT32))
    intrinsics: Float64[ndarray, "6 3 3"] = np.tile(np.eye(3), (6, 1, 1))
    intrinsics[:, [0, 1], [0, 1]] = draw(arrays(np.float32, (6, 2), elements=st.floats(100.0, 2000.0, width=32)))
    intrinsics[:, :2, 2] = draw(arrays(np.float32, (6, 2), elements=st.floats(0.0, 2000.0, width=32)))
    resolution: Int64[ndarray, "6 2"] = draw(arrays(np.int64, (6, 2), elements=st.integers(16, 4096)))
    rig = RigCameras(tuple(f"cam_{i:02d}" for i in range(6)), intrinsics, cam_from_rig, resolution)
    return CapRig(rig, NAMES, draw(arrays(np.float32, (6, 8), elements=st.floats(-1.0, 1.0, width=32))).astype(np.float64))


@settings(max_examples=40, deadline=None)
@given(cap=cap_rigs())
def test_cap_rig_serde_and_catalog_rows_round_trip_exactly(cap: CapRig) -> None:
    back: CapRig = from_dict(CapRig, to_dict(cap))
    assert back.names == cap.names and back.fisheye62.dtype == np.float64
    np.testing.assert_array_equal(back.fisheye62, cap.fisheye62)
    logs: list[Log] = cap.to_rerun(RIG_ROOT)
    assert all(log.static and log.path.startswith(f"{RIG_ROOT}/cam_") for log in logs)
    ((t_ns, read),) = CapRig.from_layer(catalog_rows([(0, log) for log in logs]), RIG_ROOT, "video_time")
    assert t_ns == STATIC and read.names == NAMES and read.rig.names == cap.rig.names
    for name in ("intrinsics", "resolution_wh"):
        np.testing.assert_array_equal(getattr(read.rig, name), getattr(cap.rig, name))
    np.testing.assert_allclose(read.rig.cam_T_world, cap.rig.cam_T_world, atol=1e-6)  # the rotation's float32 cast is not orthonormal
    np.testing.assert_array_equal(read.fisheye62, cap.fisheye62)


@settings(max_examples=5, deadline=None)
@given(cap=cap_rigs())
def test_cap_rig_refuses_a_lens_that_is_not_kannala_brandt_and_unknown_fields(cap: CapRig) -> None:
    rows: pa.Table = catalog_rows([(0, log) for log in cap.to_rerun(RIG_ROOT)])
    model: str = f"{RIG_ROOT}/cam_03/pinhole:simplecv.components.DistortionModel"
    rows = rows.set_column(rows.column_names.index(model), model, pa.array([["brown_conrady"]]))
    with pytest.raises(ValueError, match="cam_03"):
        list(CapRig.from_layer(rows, RIG_ROOT, "video_time"))
    with pytest.raises(SerdeError):
        from_dict(CapRig, to_dict(cap) | {"device": "cap_b"})


@settings(max_examples=40, deadline=None)
@given(
    times=st.lists(st.integers(0, 10**12), min_size=1, max_size=30, unique=True),
    seed=st.integers(0, 2**16),
    broken=st.lists(st.integers(0, 29), max_size=5),
)
def test_poses_read_the_slam_layers_rows_whole_sorted_and_finite(times: list[int], seed: int, broken: list[int]) -> None:
    rng = np.random.default_rng(seed)
    quaternion: Float64[ndarray, "n 4"] = Rotation.random(len(times), random_state=seed).as_quat().astype(np.float32).astype(np.float64)
    translation: Float64[ndarray, "n 3"] = rng.uniform(-5.0, 5.0, (len(times), 3)).astype(np.float32).astype(np.float64)
    bad: set[int] = {b for b in broken if b < len(times)}
    logs: list[tuple[int, Log]] = [
        (
            t,
            Log(
                RIG_ROOT, rr.Transform3D(translation=[np.nan, 0.0, 0.0] if i in bad else translation[i], quaternion=rr.Quaternion(xyzw=quaternion[i]))
            ),
        )
        for i, t in enumerate(times)
    ]
    keep: list[int] = sorted((i for i in range(len(times)) if i not in bad), key=lambda i: times[i])
    read: list[tuple[int, CatalogPoses]] = list(CatalogPoses.from_layer(catalog_rows(logs), RIG_ROOT, "video_time"))
    if not keep:
        assert read == []
        return
    ((t_ns, poses),) = read
    assert t_ns == STATIC and poses.t_ns.tolist() == [times[i] for i in keep]
    np.testing.assert_allclose(poses.world_from_rig[:, :3, 3], translation[keep])
    np.testing.assert_allclose(poses.world_from_rig[:, :3, :3], Rotation.from_quat(quaternion[keep]).as_matrix(), atol=1e-12)
    back: CatalogPoses = from_dict(CatalogPoses, to_dict(poses))
    np.testing.assert_array_equal(back.world_from_rig, poses.world_from_rig)
    assert back.t_ns.dtype == np.int64


def test_poses_are_read_never_written() -> None:
    with pytest.raises(NotImplementedError, match="slam_rs"):
        CatalogPoses(np.zeros(0, np.int64), np.zeros((0, 4, 4))).to_rerun(RIG_ROOT)


@settings(max_examples=40, deadline=None)
@given(
    counts=st.lists(st.integers(0, 10**6), min_size=7, max_size=7),
    seconds=st.lists(st.floats(0.0, 1e4), min_size=7, max_size=7),
    final=st.booleans(),
)
def test_layer_run_is_static_numbers_that_read_back(counts: list[int], seconds: list[float], final: bool) -> None:
    run = LayerRun(
        *counts, scale=1.0 - 1e-3 * counts[0] % 1, scale_final=final, **dict(zip([f.name for f in fields(LayerRun)][9:], seconds, strict=True))
    )
    assert from_dict(LayerRun, to_dict(run)) == run
    (log,) = run.to_rerun("/verify/hands/run")
    assert log.static and log.path == "/verify/hands/run"
    ((t_ns, back),) = LayerRun.from_layer(catalog_rows([(0, log)]), "/verify/hands/run", "video_time")
    assert t_ns == STATIC and back == run
