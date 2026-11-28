"""The Python door to the Mojo solver: one extension call per solve.

The host arrays go in as contiguous float32 / int32 / uint8 copies, passed to `_caspar_mojo` as
`(address, *shape)` tuples; the extension copies them to the GPU, runs CASPAR's LM + reduced-camera-system PCG
there and writes the accepted state back. Quaternions are normalised here, at write-back, as COLMAP's adapter does.
"""

from __future__ import annotations

import time
from typing import Final, NamedTuple, TypeAlias

import numpy as np
from jaxtyping import Float32, Float64
from numpy import ndarray

from caspar_mojo.problem import INTRINSICS_COUNT, BaOptions, BaProblem, BaResult, CameraModel

ArraySpec: TypeAlias = tuple[int, ...]
"""`(address, *shape)` of a contiguous numpy array, the extension's array argument."""

EXIT_REASONS: Final[tuple[str, ...]] = ("MAX_ITERATIONS", "CONVERGED_DIAG_EXIT", "CONVERGED_SCORE_THRESHOLD")
"""CASPAR's `ExitReason` names, indexed by the extension's exit code."""

CALIB_SIZE: Final[dict[CameraModel, int]] = {
    CameraModel.PINHOLE: 2,
    CameraModel.SIMPLE_RADIAL: 2,
    CameraModel.OPENCV_FISHEYE: 6,
    CameraModel.FULL_OPENCV: 10,
}
"""Focal-and-extra parameters per model: the width of a refined intrinsics block."""


class Linearization(NamedTuple):
    """One linearisation at a state, as the device computed it; unpacks as (cost, gradient, blocks)."""

    cost: float
    """Half the sum of squared residuals."""
    gradient: Float64[ndarray, " n_tangent"]
    """g = -J^T e over the flat tangent vector [free poses 6 | free sensors 6 | free calib K | free points 3]."""
    block_diagonal: list[Float64[ndarray, "n_free d d"]]
    """Dense J^T J blocks per non-empty group, same group order as the gradient."""


def _spec(array: ndarray) -> ArraySpec:
    return (array.ctypes.data, *array.shape)


def _model(problem: BaProblem) -> CameraModel:
    models: list[int] = np.unique(problem.camera_model).tolist()
    if len(models) != 1:
        raise ValueError(f"one camera model per solve, got {models}")
    return CameraModel(models[0])


def _inputs(problem: BaProblem) -> tuple[tuple[object, ...], list[ndarray]]:
    """The extension's input tuple and the arrays it points into (kept alive by the caller)."""
    arrays: list[ndarray] = [
        np.ascontiguousarray(problem.intrinsics, dtype=np.float32),
        np.ascontiguousarray(problem.intrinsics_free, dtype=np.uint8),
        np.ascontiguousarray(problem.sensor_from_rig, dtype=np.float32),
        np.ascontiguousarray(problem.sensor_free, dtype=np.uint8),
        np.ascontiguousarray(problem.rig_from_world, dtype=np.float32),
        np.ascontiguousarray(problem.pose_free, dtype=np.uint8),
        np.ascontiguousarray(problem.points, dtype=np.float32),
        np.ascontiguousarray(problem.point_free, dtype=np.uint8),
        np.ascontiguousarray(problem.pixels, dtype=np.float32),
        np.ascontiguousarray(problem.observation_frame, dtype=np.int32),
        np.ascontiguousarray(problem.observation_camera, dtype=np.int32),
        np.ascontiguousarray(problem.observation_point, dtype=np.int32),
    ]
    return (int(_model(problem)), *(_spec(a) for a in arrays)), arrays


def _extension():
    from caspar_mojo import _caspar_mojo

    return _caspar_mojo


def _normalized(poses: Float32[ndarray, "n 7"]) -> Float64[ndarray, "n 7"]:
    out: Float64[ndarray, "n 7"] = poses.astype(np.float64)
    out[:, :4] /= np.linalg.norm(out[:, :4], axis=1, keepdims=True)
    return out


def solve(problem: BaProblem, options: BaOptions = BaOptions()) -> BaResult:
    """CASPAR's `GraphSolver::solve` with the reduced camera system, on the GPU."""
    start: float = time.perf_counter()
    if not options.reduced_camera_system:
        raise NotImplementedError("only the reduced camera system is ported (colsfm's mode)")
    inputs, keep_alive = _inputs(problem)
    outputs: list[Float32[ndarray, "n w"]] = [
        np.array(problem.intrinsics, dtype=np.float32),
        np.array(problem.sensor_from_rig, dtype=np.float32),
        np.array(problem.rig_from_world, dtype=np.float32),
        np.array(problem.points, dtype=np.float32),
    ]
    scalars: tuple[object, ...] = (
        options.solver_iter_max,
        options.diag_init,
        options.diag_min,
        options.diag_scaling_up,
        options.diag_scaling_down,
        options.diag_exit_value,
        options.score_exit_value,
        options.solver_rel_decrease_min,
        options.reduced_pcg_iter_max,
        options.reduced_pcg_eta,
    )
    stats: tuple[float, float, int, int, int, float, int] = _extension().solve(inputs, scalars, tuple(_spec(a) for a in outputs))
    del keep_alive
    model: CameraModel = _model(problem)
    intrinsics: Float64[ndarray, "n_cameras 12"] = outputs[0].astype(np.float64)
    intrinsics[:, INTRINSICS_COUNT[model] :] = 0.0
    return BaResult(
        intrinsics=intrinsics,
        sensor_from_rig=_normalized(outputs[1]),
        rig_from_world=_normalized(outputs[2]),
        points=outputs[3].astype(np.float64),
        initial_cost=float(stats[0]),
        final_cost=float(stats[1]),
        iterations=int(stats[2]),
        pcg_iterations=int(stats[3]),
        exit_reason=EXIT_REASONS[int(stats[4])],
        solve_seconds=float(stats[5]),
        total_seconds=time.perf_counter() - start,
    )


def linearize(problem: BaProblem) -> Linearization:
    """Test hook: CASPAR's `res_jac_first` at the problem's state, computed on the GPU."""
    model: CameraModel = _model(problem)
    k: int = CALIB_SIZE[model]
    groups: list[tuple[int, int]] = [
        (int(problem.pose_free.sum()), 6),
        (int(problem.sensor_free.sum()), 6),
        (int(problem.intrinsics_free.sum()), k),
        (int(problem.point_free.sum()), 3),
    ]
    n_tangent: int = sum(n * d for n, d in groups)
    n_packed: int = sum(n * d * (d + 1) // 2 for n, d in groups)
    inputs, keep_alive = _inputs(problem)
    gradient: Float32[ndarray, " n_tangent"] = np.zeros(max(n_tangent, 1), dtype=np.float32)
    packed: Float32[ndarray, " n_packed"] = np.zeros(max(n_packed, 1), dtype=np.float32)
    cost: float = _extension().linearize(inputs, _spec(gradient), _spec(packed))
    del keep_alive
    blocks: list[Float64[ndarray, "n_free d d"]] = []
    offset: int = 0
    for n, d in groups:
        rows, cols = np.triu_indices(d)
        triangle: Float64[ndarray, "n_free t"] = packed[offset : offset + n * len(rows)].reshape(n, len(rows)).astype(np.float64)
        offset += n * len(rows)
        if n == 0:
            continue
        dense: Float64[ndarray, "n_free d d"] = np.zeros((n, d, d))
        dense[:, rows, cols] = triangle
        dense[:, cols, rows] = triangle
        blocks.append(dense)
    return Linearization(float(cost), gradient[:n_tangent].astype(np.float64), blocks)
