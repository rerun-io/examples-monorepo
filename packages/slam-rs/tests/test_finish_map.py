"""``Vio.finish_map`` through the Python boundary (PR d2): the map's keyframes before and after the global BA, typed copies, and the GBA's scalars."""

import json
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
    assert gba.lane == "cpu"
    assert gba.use_double is False
    assert min(gba.build_s, gba.solve_s, gba.apply_s) >= 0.0
    assert 0.0 <= gba.gauge_tilt_deg < 1.0

    # The map now holds the GBA's poses; the poses track returned are untouched.
    after: _core.MapSnapshot | None = vio.map_snapshot()
    assert after is not None
    np.testing.assert_array_equal(after.keyframe_poses, result.keyframe_poses)
    _off_poses, _off = _tracked(rig, texture, None)
    for with_map, without in zip(poses, _off_poses, strict=True):
        np.testing.assert_array_equal(with_map, without)


@pytest.mark.parametrize(("model", "extra"), [("ds", {"xi": 0.0, "alpha": 0.5}), ("eucm", {"alpha": 0.5, "beta": 1.0})])
def test_a_rig_the_gba_cannot_project_is_refused_before_any_frame(rig: RigFactory, model: str, extra: dict[str, float]) -> None:
    """The GBA projects every model the frontend does, so a replay that tracked never meets ``finish_map``'s ValueError: ds and eucm stop at ``Vio(...)``."""
    document: dict[str, dict[str, list[dict[str, object]]]] = json.loads(rig(2).to_json())
    for camera in document["value0"]["intrinsics"]:
        camera["camera_type"] = model
        camera["intrinsics"] = {"fx": 100.0, "fy": 100.0, "cx": 100.0, "cy": 100.0, **extra}
    calibration: _core.Calibration = _core.Calibration.from_json(json.dumps(document))
    with pytest.raises(ValueError, match=f"camera model {model} is parsed but its projection is not implemented"):
        _core.Vio(calibration, _core.VioConfig().with_backend("offline"), backend="cpu")


def test_ba_backend_is_parsed_and_a_cpu_frontend_never_probes_for_it(rig: RigFactory) -> None:
    """``ba_backend`` takes ``follow``, ``cpu`` or ``gpu``; on a CPU frontend ``follow`` keeps the GBA on the CPU without the back-end probe."""
    probes: int = _core.device_probes_run()
    follow: _core.Vio = _core.Vio(rig(2), _core.VioConfig().with_backend("offline"), backend="cpu", ba_backend="follow")
    _core.Vio(rig(2), _core.VioConfig().with_backend("offline"), backend="cpu", ba_backend="cpu")
    assert _core.device_probes_run() == probes
    assert (follow.backend, follow.gba_lane) == ("cpu", "cpu")
    with pytest.raises(ValueError, match="expected follow, cpu or gpu"):
        _core.Vio(rig(2), _core.VioConfig(), backend="cpu", ba_backend="auto")  # type: ignore[arg-type]
