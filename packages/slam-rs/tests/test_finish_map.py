"""``Vio.finish_map`` through the Python boundary (PR d2): the map's keyframes before and after the global BA, typed copies, and the GBA's scalars."""

from typing import Literal

import numpy as np
import pytest
from fixture_types import RigFactory, TextureFactory
from jaxtyping import Float64, UInt8
from numpy import ndarray

from slam_rs import _core


def _tracked(rig: RigFactory, texture: TextureFactory, mode: Literal["offline", "realtime"] | None) -> tuple[list[Float64[ndarray, "7"]], _core.Vio]:
    """Track a still stereo pair for twelve framesets with the back end off (``None``) or in ``mode``; return the poses and the flushed Vio."""
    vio: _core.Vio = _core.Vio(rig(2), _core.VioConfig().with_backend(mode or "off"), backend="cpu")
    for t_ns in range(0, 261_000_000, 1_000_000):
        vio.push_imu(t_ns, [0.0, 0.0, 0.0], [0.0, 0.0, 9.81])
    images: list[UInt8[ndarray, "h w"]] = [texture(0, 0), texture(-2, 0)]
    poses: list[Float64[ndarray, "7"]] = []
    for index in range(12):
        result: _core.VioResult = vio.track(index * 20_000_000, images)
        assert result.world_from_rig is not None
        poses.append(result.world_from_rig)
    vio.flush()
    return poses, vio


def test_finish_map_is_none_with_the_back_end_off(rig: RigFactory, texture: TextureFactory) -> None:
    _poses, vio = _tracked(rig, texture, None)
    assert vio.finish_map(run_gba=True) is None


@pytest.mark.parametrize("mode", ["offline", "realtime"])
def test_finish_map_returns_typed_keyframes_and_the_gba_scalars(
    rig: RigFactory, texture: TextureFactory, mode: Literal["offline", "realtime"]
) -> None:
    poses, vio = _tracked(rig, texture, mode)
    snapshot: _core.MapSnapshot | None = vio.map_snapshot()
    assert snapshot is not None

    plain: _core.MapResult | None = vio.finish_map(run_gba=False)
    assert plain is not None
    assert plain.gba is None
    np.testing.assert_array_equal(plain.keyframe_t_ns, snapshot.keyframe_t_ns)
    np.testing.assert_array_equal(plain.keyframe_poses, snapshot.keyframe_poses)
    np.testing.assert_array_equal(plain.keyframe_poses_vio, snapshot.keyframe_poses)

    result: _core.MapResult | None = vio.finish_map(run_gba=True)
    assert result is not None
    k: int = result.keyframe_t_ns.shape[0]
    assert k == snapshot.keyframe_t_ns.shape[0] >= 1
    for array, dtype, shape in (
        (result.keyframe_t_ns, np.int64, (k,)),
        (result.keyframe_poses_vio, np.float64, (k, 7)),
        (result.keyframe_poses, np.float64, (k, 7)),
    ):
        assert array.dtype == dtype
        assert array.shape == shape
    np.testing.assert_array_equal(result.keyframe_poses_vio, snapshot.keyframe_poses)
    np.testing.assert_allclose(np.linalg.norm(result.keyframe_poses[:, 3:], axis=1), 1.0, rtol=1e-9)

    gba: _core.GbaRun | None = result.gba
    assert gba is not None
    n: int = gba.iteration_lambda.shape[0]
    assert 1 <= n <= 20
    for array, dtype, shape in (
        (gba.iteration_lambda, np.float64, (n,)),
        (gba.iteration_cost, np.float64, (n,)),
        (gba.iteration_accepted, np.bool_, (n,)),
        (gba.iteration_cg, np.int64, (n,)),
        (gba.iteration_stage_s, np.float64, (n, 5)),
    ):
        assert array.dtype == dtype
        assert array.shape == shape
    assert gba.final_cost <= gba.initial_cost
    assert gba.termination in ("converged", "max_iterations", "lambda_overflow")
    assert gba.keyframes >= 1
    assert gba.threads == 2
    assert gba.use_double is False
    assert min(gba.build_s, gba.solve_s, gba.apply_s) >= 0.0

    # The map now holds the GBA's poses; the poses track returned are untouched.
    after: _core.MapSnapshot | None = vio.map_snapshot()
    assert after is not None
    np.testing.assert_array_equal(after.keyframe_poses, result.keyframe_poses)
    _off_poses, _off = _tracked(rig, texture, None)
    for with_map, without in zip(poses, _off_poses, strict=True):
        np.testing.assert_array_equal(with_map, without)
