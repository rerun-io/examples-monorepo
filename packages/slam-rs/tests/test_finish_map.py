"""``Vio.finish_map`` through the Python boundary (PR d2): the map's keyframes before and after the global BA, typed copies, and the GBA's scalars."""

import json
from typing import Literal

import numpy as np
import pytest
from fixture_types import RigFactory, TextureFactory
from jaxtyping import Float64, Int64, UInt8
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
    np.testing.assert_array_equal(plain.keyframe_poses_before_gba, snapshot.keyframe_poses)

    result: _core.MapResult | None = vio.finish_map(run_gba=True)
    assert result is not None
    k: int = result.keyframe_t_ns.shape[0]
    assert k == snapshot.keyframe_t_ns.shape[0] >= 1
    for array, dtype, shape in (
        (result.keyframe_t_ns, np.int64, (k,)),
        (result.keyframe_poses_before_gba, np.float64, (k, 7)),
        (result.keyframe_poses, np.float64, (k, 7)),
    ):
        assert array.dtype == dtype
        assert array.shape == shape
    np.testing.assert_array_equal(result.keyframe_poses_before_gba, snapshot.keyframe_poses)
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


@pytest.mark.parametrize("run_gba", [False, True])
def test_the_final_trajectory_is_the_raw_vio_reanchored_on_the_finished_map(rig: RigFactory, texture: TextureFactory, run_gba: bool) -> None:
    """``final_trajectory`` (PR k0): the GBA on the map after loops and PGO, then the raw VIO re-anchored on its keyframes; without it, f1's +LC final."""
    from slam_rs.tracking import final_trajectory, reanchored
    from slam_rs.trajectory import Trajectory

    poses, vio = _tracked(rig, texture, "offline")
    stacked: Float64[ndarray, "n 7"] = np.stack(poses)
    estimate: Trajectory = Trajectory(
        t_ns=np.arange(len(poses), dtype=np.int64) * 20_000_000, position_m=stacked[:, 0:3].copy(), quaternion_wxyz=np.roll(stacked[:, 3:7], 1, axis=1)
    )
    snapshot: _core.MapSnapshot | None = vio.map_snapshot()
    assert snapshot is not None
    lc_final: Trajectory = reanchored(estimate, snapshot)

    run = final_trajectory(vio, estimate, run_gba=run_gba)
    final, result = run.final, run.map_result
    assert result is not None
    assert (result.gba is not None) is run_gba
    assert run.gba_failure is None
    np.testing.assert_array_equal(final.t_ns, estimate.t_ns)
    rows: Int64[ndarray, " k"] = np.searchsorted(estimate.t_ns, result.keyframe_t_ns)
    np.testing.assert_array_equal(estimate.t_ns[rows], result.keyframe_t_ns)
    np.testing.assert_allclose(final.position_m[rows], result.keyframe_poses[:, 0:3], rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(final.position_m, reanchored(estimate, result).position_m)
    if not run_gba:
        np.testing.assert_array_equal(final.position_m, lc_final.position_m)
        np.testing.assert_array_equal(final.quaternion_wxyz, lc_final.quaternion_wxyz)
    # The raw VIO is not touched.
    np.testing.assert_array_equal(estimate.position_m, stacked[:, 0:3])


def test_the_final_trajectory_is_the_raw_vio_with_the_back_end_off(rig: RigFactory, texture: TextureFactory) -> None:
    from slam_rs.tracking import final_trajectory
    from slam_rs.trajectory import Trajectory

    poses, vio = _tracked(rig, texture, None)
    stacked: Float64[ndarray, "n 7"] = np.stack(poses)
    estimate: Trajectory = Trajectory(
        t_ns=np.arange(len(poses), dtype=np.int64) * 20_000_000, position_m=stacked[:, 0:3].copy(), quaternion_wxyz=np.roll(stacked[:, 3:7], 1, axis=1)
    )
    run = final_trajectory(vio, estimate, run_gba=True)
    assert run.map_result is None
    assert run.final is estimate


class _GbaRefused:
    """A flushed Vio whose global BA fails, as the core's ``finish_map(run_gba=True)`` does on a refused solve or a GPU lane error."""

    def __init__(self, vio: _core.Vio) -> None:
        self.vio: _core.Vio = vio
        self.asked: list[bool] = []

    def finish_map(self, run_gba: bool) -> _core.MapResult | None:
        self.asked.append(run_gba)
        if run_gba:
            raise ValueError("global BA: the solver refused the problem")
        return self.vio.finish_map(run_gba=False)


def test_a_failed_gba_falls_back_to_the_map_after_loops_and_pgo(rig: RigFactory, texture: TextureFactory) -> None:
    """A GBA error never loses the run (k0 review): one warning naming it, and the final run is f1's +LC final."""
    from slam_rs.tracking import final_trajectory, reanchored
    from slam_rs.trajectory import Trajectory

    poses, vio = _tracked(rig, texture, "offline")
    stacked: Float64[ndarray, "n 7"] = np.stack(poses)
    estimate: Trajectory = Trajectory(
        t_ns=np.arange(len(poses), dtype=np.int64) * 20_000_000, position_m=stacked[:, 0:3].copy(), quaternion_wxyz=np.roll(stacked[:, 3:7], 1, axis=1)
    )
    snapshot: _core.MapSnapshot | None = vio.map_snapshot()
    assert snapshot is not None
    refused: _GbaRefused = _GbaRefused(vio)
    with pytest.warns(RuntimeWarning, match="the solver refused the problem"):
        run = final_trajectory(refused, estimate, run_gba=True)
    assert refused.asked == [True, False]
    assert run.gba_failure == "global BA: the solver refused the problem"
    assert run.map_result is not None
    assert run.map_result.gba is None
    lc_final: Trajectory = reanchored(estimate, snapshot)
    np.testing.assert_array_equal(run.final.position_m, lc_final.position_m)
    np.testing.assert_array_equal(run.final.quaternion_wxyz, lc_final.quaternion_wxyz)
