"""The Python door to the Mojo solver: one extension call per solve.

The door checks what the native side trusts and per-array annotations cannot express (lengths that must agree,
indices in range), then hands `_caspar_mojo` a `_Packet`: the problem by name, as C-contiguous float32 / int32 /
0-or-1 uint8 arrays. The extension takes each array's address by name, copies it to the GPU, runs CASPAR's LM +
reduced-camera-system PCG there and writes the accepted state, row-major, into a `_State`. Only refined values come
back through float32: fixed rows, and the principal point of a refined camera, stay the caller's float64. Quaternions
of refined poses are normalised here, as COLMAP's adapter does at write-back.
"""

import time
from dataclasses import dataclass
from typing import Final, NamedTuple, TypedDict

import numpy as np
from jaxtyping import Bool, Float32, Float64, Int32, UInt8
from numpy import ndarray

from caspar_mojo.problem import FOCAL_AND_EXTRA_COLUMNS, MAX_INTRINSICS, BaIteration, BaOptions, BaProblem, BaResult, CameraModel

EXIT_REASONS: Final[tuple[str, ...]] = ("MAX_ITERATIONS", "CONVERGED_DIAG_EXIT", "CONVERGED_SCORE_THRESHOLD")
"""CASPAR's `ExitReason` names, indexed by the extension's exit code."""

DEFAULT_OPTIONS: Final[BaOptions] = BaOptions()
"""colsfm's CASPAR settings (frozen, shared)."""

SHAPES: Final[dict[str, tuple[str, ...]]] = {
    "intrinsics": ("n_cameras", "12"),
    "intrinsics_free": ("n_cameras",),
    "sensor_from_rig": ("n_cameras", "7"),
    "sensor_free": ("n_cameras",),
    "rig_from_world": ("n_frames", "7"),
    "pose_free": ("n_frames",),
    "points": ("n_points", "3"),
    "point_free": ("n_points",),
    "pixels": ("n_observations", "2"),
    "observation_frame": ("n_observations",),
    "observation_camera": ("n_observations",),
    "observation_point": ("n_observations",),
}
"""Shape of each `BaProblem` array in named sizes; the sizes come from `camera_model`, `rig_from_world`, `points` and
`pixels`."""

INDEXED: Final[dict[str, str]] = {"observation_frame": "n_frames", "observation_camera": "n_cameras", "observation_point": "n_points"}
"""Each observation index array and the size it indexes."""


class Linearization(NamedTuple):
    """One linearisation at a state, as the device computed it; unpacks as (cost, gradient, blocks)."""

    cost: float
    """Half the sum of squared residuals."""
    gradient: Float64[ndarray, " n_tangent"]
    """g = -J^T e over the flat tangent vector [free poses 6 | free sensors 6 | free calib K | free points 3]."""
    block_diagonal: list[Float64[ndarray, "n_free d d"]]
    """Dense J^T J blocks per non-empty group, same group order as the gradient."""


class NativeStats(TypedDict):
    """What `_caspar_mojo.solve` returns."""

    initial_cost: float
    final_cost: float
    iterations: int
    pcg_iterations: int
    exit_reason: int
    solve_seconds: float
    launches: int
    trace: list[tuple[float, float, float, int, bool]]


@dataclass(frozen=True, slots=True)
class _Packet:
    """The problem as `_caspar_mojo` reads it, each array by name."""

    camera_model: int
    intrinsics: Float32[ndarray, "n_cameras 12"]
    intrinsics_free: UInt8[ndarray, " n_cameras"]
    sensor_from_rig: Float32[ndarray, "n_cameras 7"]
    sensor_free: UInt8[ndarray, " n_cameras"]
    rig_from_world: Float32[ndarray, "n_frames 7"]
    pose_free: UInt8[ndarray, " n_frames"]
    points: Float32[ndarray, "n_points 3"]
    point_free: UInt8[ndarray, " n_points"]
    pixels: Float32[ndarray, "n_observations 2"]
    observation_frame: Int32[ndarray, " n_observations"]
    observation_camera: Int32[ndarray, " n_observations"]
    observation_point: Int32[ndarray, " n_observations"]


@dataclass(frozen=True, slots=True)
class _State:
    """The state `_caspar_mojo.solve` writes, by name."""

    intrinsics: Float32[ndarray, "n_cameras 12"]
    sensor_from_rig: Float32[ndarray, "n_cameras 7"]
    rig_from_world: Float32[ndarray, "n_frames 7"]
    points: Float32[ndarray, "n_points 3"]


def _model(problem: BaProblem) -> CameraModel:
    models: list[int] = np.unique(problem.camera_model).tolist()
    if len(models) != 1:
        raise ValueError(f"one camera model per solve, got {models}")
    return CameraModel(models[0])


def _packet(problem: BaProblem) -> _Packet:
    """The checked problem in the native layout; raises `ValueError` on lengths or indices the kernels would trust."""
    model: CameraModel = _model(problem)
    sizes: dict[str, int] = {
        "n_cameras": len(problem.camera_model),
        "n_frames": len(problem.rig_from_world),
        "n_points": len(problem.points),
        "n_observations": len(problem.pixels),
        "12": MAX_INTRINSICS,
        "7": 7,
        "3": 3,
        "2": 2,
    }
    for name, dims in SHAPES.items():
        shape: tuple[int, ...] = np.shape(getattr(problem, name))
        expected: tuple[int, ...] = tuple(sizes[d] for d in dims)
        if shape != expected:
            raise ValueError(f"{name} has shape {shape}, expected {expected} ({', '.join(f'{d}={sizes[d]}' for d in dims)})")
    for name, size in INDEXED.items():
        index: Int32[ndarray, " n_observations"] = getattr(problem, name)
        if len(index) and (int(index.min()) < 0 or int(index.max()) >= sizes[size]):
            raise ValueError(f"{name} holds rows outside [0, {size}={sizes[size]})")
    return _Packet(
        camera_model=int(model),
        intrinsics=np.ascontiguousarray(problem.intrinsics, dtype=np.float32),
        intrinsics_free=np.ascontiguousarray(problem.intrinsics_free != 0, dtype=np.uint8),
        sensor_from_rig=np.ascontiguousarray(problem.sensor_from_rig, dtype=np.float32),
        sensor_free=np.ascontiguousarray(problem.sensor_free != 0, dtype=np.uint8),
        rig_from_world=np.ascontiguousarray(problem.rig_from_world, dtype=np.float32),
        pose_free=np.ascontiguousarray(problem.pose_free != 0, dtype=np.uint8),
        points=np.ascontiguousarray(problem.points, dtype=np.float32),
        point_free=np.ascontiguousarray(problem.point_free != 0, dtype=np.uint8),
        pixels=np.ascontiguousarray(problem.pixels, dtype=np.float32),
        observation_frame=np.ascontiguousarray(problem.observation_frame, dtype=np.int32),
        observation_camera=np.ascontiguousarray(problem.observation_camera, dtype=np.int32),
        observation_point=np.ascontiguousarray(problem.observation_point, dtype=np.int32),
    )


def _extension():
    from caspar_mojo import _caspar_mojo

    return _caspar_mojo


def _merged(
    original: Float64[ndarray, "n w"], solved: Float32[ndarray, "n w"], free: Bool[ndarray, " n"], columns: list[int]
) -> Float64[ndarray, "n w"]:
    """`original` with the solved values in its free rows' `columns`: fixed values never pass through float32."""
    out: Float64[ndarray, "n w"] = np.array(original, dtype=np.float64, order="C")
    in_columns: Bool[ndarray, " w"] = np.zeros(out.shape[1], dtype=bool)
    in_columns[columns] = True
    # A broadcast mask, not an index grid: np.ix_ over 273 k points cost 10 ms of a 50 ms call on clip 007.
    np.copyto(out, solved, where=free[:, None] & in_columns)
    return out


def _merged_poses(original: Float64[ndarray, "n 7"], solved: Float32[ndarray, "n 7"], free: Bool[ndarray, " n"]) -> Float64[ndarray, "n 7"]:
    out: Float64[ndarray, "n 7"] = _merged(original, solved, free, list(range(7)))
    out[free, :4] /= np.linalg.norm(out[free, :4], axis=1, keepdims=True)
    return out


def solve(problem: BaProblem, options: BaOptions = DEFAULT_OPTIONS) -> BaResult:
    """CASPAR's `GraphSolver::solve` with the reduced camera system, on the GPU."""
    start: float = time.perf_counter()
    if not options.reduced_camera_system:
        raise NotImplementedError("only the reduced camera system is ported (colsfm's mode)")
    packet: _Packet = _packet(problem)
    state = _State(
        intrinsics=np.empty(packet.intrinsics.shape, dtype=np.float32),
        sensor_from_rig=np.empty(packet.sensor_from_rig.shape, dtype=np.float32),
        rig_from_world=np.empty(packet.rig_from_world.shape, dtype=np.float32),
        points=np.empty(packet.points.shape, dtype=np.float32),
    )
    stats: NativeStats = _extension().solve(packet, options, state)
    calib_columns: list[int] = list(FOCAL_AND_EXTRA_COLUMNS[CameraModel(packet.camera_model)])
    return BaResult(
        intrinsics=_merged(problem.intrinsics, state.intrinsics, packet.intrinsics_free != 0, calib_columns),
        sensor_from_rig=_merged_poses(problem.sensor_from_rig, state.sensor_from_rig, packet.sensor_free != 0),
        rig_from_world=_merged_poses(problem.rig_from_world, state.rig_from_world, packet.pose_free != 0),
        points=_merged(problem.points, state.points, packet.point_free != 0, list(range(3))),
        initial_cost=stats["initial_cost"],
        final_cost=stats["final_cost"],
        iterations=stats["iterations"],
        pcg_iterations=stats["pcg_iterations"],
        exit_reason=EXIT_REASONS[stats["exit_reason"]],
        solve_seconds=stats["solve_seconds"],
        total_seconds=time.perf_counter() - start,
        trace=tuple(BaIteration(*trial) for trial in stats["trace"]),
    )


def linearize(problem: BaProblem) -> Linearization:
    """Test hook: CASPAR's `res_jac_first` at the problem's state, computed on the GPU."""
    packet: _Packet = _packet(problem)
    k: int = len(FOCAL_AND_EXTRA_COLUMNS[CameraModel(packet.camera_model)])
    groups: list[tuple[int, int]] = [
        (int(packet.pose_free.sum()), 6),
        (int(packet.sensor_free.sum()), 6),
        (int(packet.intrinsics_free.sum()), k),
        (int(packet.point_free.sum()), 3),
    ]
    n_tangent: int = sum(n * d for n, d in groups)
    n_packed: int = sum(n * d * (d + 1) // 2 for n, d in groups)
    gradient: Float32[ndarray, " n_tangent"] = np.zeros(max(n_tangent, 1), dtype=np.float32)
    packed: Float32[ndarray, " n_packed"] = np.zeros(max(n_packed, 1), dtype=np.float32)
    cost: float = _extension().linearize(packet, gradient, packed)
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
