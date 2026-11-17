"""The persistent keyframe map through the Python boundary: on by default, off on request, and invisible to the poses."""

import json

import numpy as np
from fixture_types import RigFactory, TextureFactory
from jaxtyping import Float64, UInt8
from numpy import ndarray

from slam_rs import _core


def _run(rig: RigFactory, texture: TextureFactory, backend_enabled: bool) -> tuple[list[Float64[ndarray, "7"]], dict[str, int] | None]:
    """Track a still stereo pair for twelve framesets; return the poses and the map counts."""
    config: _core.VioConfig = _core.VioConfig.from_json(json.dumps({"value0": {"backend.enabled": backend_enabled}}))
    vio: _core.Vio = _core.Vio(rig(2), config, backend="cpu")
    empty: dict[str, int] | None = {"keyframes": 0, "landmarks": 0, "observations": 0} if backend_enabled else None
    assert vio.map_stats == empty
    for t_ns in range(0, 261_000_000, 1_000_000):
        vio.push_imu(t_ns, [0.0, 0.0, 0.0], [0.0, 0.0, 9.81])
    images: list[UInt8[ndarray, "h w"]] = [texture(0, 0), texture(-2, 0)]
    poses: list[Float64[ndarray, "7"]] = []
    for index in range(12):
        result: _core.VioResult = vio.track(index * 20_000_000, images)
        assert result.status == _core.VioStatus.Tracking
        assert result.world_from_rig is not None
        poses.append(result.world_from_rig)
    vio.flush()
    return poses, vio.map_stats


def test_the_map_fills_by_default_and_never_moves_a_pose(rig: RigFactory, texture: TextureFactory) -> None:
    on_poses, on_stats = _run(rig, texture, backend_enabled=True)
    off_poses, off_stats = _run(rig, texture, backend_enabled=False)
    assert off_stats is None
    assert on_stats is not None
    assert on_stats["keyframes"] >= 1
    assert on_stats["observations"] >= 2 * on_stats["landmarks"]
    for on, off in zip(on_poses, off_poses, strict=True):
        np.testing.assert_array_equal(on, off)
