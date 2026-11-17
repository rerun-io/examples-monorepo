"""The persistent keyframe map through the Python boundary: off (Offline when on) by default, either mode on request, off on request, and invisible to the poses."""

from typing import Literal

import numpy as np
import pytest
from fixture_types import RigFactory, TextureFactory
from jaxtyping import Float64, UInt8
from numpy import ndarray

from slam_rs import _core


def _run(rig: RigFactory, texture: TextureFactory, mode: Literal["offline", "realtime"] | None) -> tuple[list[Float64[ndarray, "7"]], _core.Vio]:
    """Track a still stereo pair for twelve framesets with the back end off (``None``) or in ``mode``; return the poses and the flushed Vio."""
    config: _core.VioConfig = _core.VioConfig().with_backend(mode or "off")
    vio: _core.Vio = _core.Vio(rig(2), config, backend="cpu")
    if mode is None:
        assert vio.map_snapshot() is None
    else:
        empty: _core.MapSnapshot | None = vio.map_snapshot()
        assert empty is not None
        assert empty.keyframe_t_ns.shape == (0,)
        assert empty.points_world.shape == (0, 3)
        assert empty.covisibility_edges.shape == (0, 2)
    for t_ns in range(0, 261_000_000, 1_000_000):
        vio.push_imu(t_ns, [0.0, 0.0, 0.0], [0.0, 0.0, 9.81])
    images: list[UInt8[ndarray, "h w"]] = [texture(0, 0), texture(-2, 0)]
    poses: list[Float64[ndarray, "7"]] = []
    for index in range(12):
        result: _core.VioResult = vio.track(index * 20_000_000, images)
        assert result.status == _core.VioStatus.Tracking
        assert result.world_from_rig is not None
        # No loop closes on a still rig: the online corrected pose is the raw one, bit for bit, in every mode.
        corrected: Float64[ndarray, "7"] | None = result.world_from_rig_corrected
        assert corrected is not None
        np.testing.assert_array_equal(corrected, result.world_from_rig)
        poses.append(result.world_from_rig)
    vio.flush()
    return poses, vio


def test_the_back_end_is_off_and_offline_by_default() -> None:
    """Opt in until loop closing (b2) gives the map a use: default runs are what they were before a2."""
    config: _core.VioConfig = _core.VioConfig()
    assert config.backend_enabled is False
    assert config.backend_mode == "offline"


@pytest.mark.parametrize("mode", ["offline", "realtime"])
def test_the_map_snapshot_has_typed_copied_arrays_and_never_moves_a_pose(
    rig: RigFactory, texture: TextureFactory, mode: Literal["offline", "realtime"]
) -> None:
    on_poses, on = _run(rig, texture, mode)
    off_poses, _off = _run(rig, texture, None)
    for with_map, without in zip(on_poses, off_poses, strict=True):
        np.testing.assert_array_equal(with_map, without)

    snapshot: _core.MapSnapshot | None = on.map_snapshot()
    assert snapshot is not None
    assert snapshot.mode == mode
    k: int = snapshot.keyframe_t_ns.shape[0]
    p: int = snapshot.point_ids.shape[0]
    e: int = snapshot.covisibility_edges.shape[0]
    r: int = snapshot.pgo_epoch
    assert k >= 1
    assert snapshot.keyframes == k
    for array, dtype, shape in (
        (snapshot.keyframe_t_ns, np.int64, (k,)),
        (snapshot.keyframe_poses, np.float64, (k, 7)),
        (snapshot.keyframe_active, np.bool_, (k,)),
        (snapshot.point_ids, np.int64, (p,)),
        (snapshot.point_hosts, np.int64, (p,)),
        (snapshot.points_world, np.float32, (p, 3)),
        (snapshot.covisibility_edges, np.int64, (e, 2)),
        (snapshot.covisibility_weights, np.uint32, (e,)),
        (snapshot.keyframe_raw_poses, np.float64, (k, 7)),
        (snapshot.pgo_runs, np.int64, (r, 5)),
    ):
        assert array.dtype == dtype
        assert array.shape == shape
    assert np.all(np.diff(snapshot.keyframe_t_ns) > 0)
    assert np.all(np.isfinite(snapshot.points_world))
    assert np.isin(snapshot.point_hosts, snapshot.keyframe_t_ns).all()
    assert snapshot.landmarks >= p
    assert snapshot.observations >= 2 * snapshot.landmarks
    np.testing.assert_allclose(np.linalg.norm(snapshot.keyframe_poses[:, 3:], axis=1), 1.0, rtol=1e-12)
    # Copied: writing to one snapshot's array leaves the next one alone.
    snapshot.keyframe_t_ns[:] = -1
    again: _core.MapSnapshot | None = on.map_snapshot()
    assert again is not None
    assert (again.keyframe_t_ns >= 0).all()
    # Every stamp was taken (flush waits for Realtime), and nothing was behind.
    assert snapshot.stamps >= 1
    assert snapshot.step_count == snapshot.stamps
    assert snapshot.queries_dropped == 0
    assert snapshot.queries_run == k + snapshot.culled_redundant + snapshot.culled_over_budget
    # Realtime readers get views the thread posted (the flush posts the last); Offline takes them on demand.
    assert (snapshot.publish_count >= 1) if mode == "realtime" else (snapshot.publish_count == 0)
