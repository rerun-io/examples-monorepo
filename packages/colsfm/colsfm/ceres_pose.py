"""The Ceres plumbing the two pyceres solves share, and the one skew-symmetric matrix.

`colsfm.pose_graph.solve_pose_graph` and
`colsfm.extrinsic_refinement.solve_extrinsics` build very different problems out of
the same six pieces, and the two copies had already drifted apart:

1. priming a `pycolmap.Rigid3d` into the `(quaternion xyzw, translation)` parameter
   arrays Ceres owns,
2. attaching `EigenQuaternionManifold` to every quaternion block that reached the
   problem,
3. naming a linear solver — through a `Final` lookup dict rather than `getattr`,
4. filling a `pyceres.SolverOptions` from a documented options dataclass,
5. rebuilding a `Rigid3d` out of a solved parameter pair,
6. unpacking a `pyceres.SolverSummary` into plain numbers.

**The alignment changes one thing.** `solve_extrinsics` used to hard-code
`SPARSE_NORMAL_CHOLESKY` and set none of the three convergence tolerances, i.e. it
took whatever Ceres defaulted to; it now takes `CeresSolveOptions`' explicit values,
which are the pose graph's. Those values *are* pyceres' own defaults — measured on
this build, `pyceres.SolverOptions()` reports `function_tolerance = 1e-6`,
`gradient_tolerance = 1e-10` and `parameter_tolerance = 1e-8` — so the extrinsic
refinement solves exactly the problem it solved before, and the Galileo
`--optimize-extrinsics` numbers are expected to be unchanged. The values are written
down here so that a future pyceres release cannot move either solve silently.

Two ceres bindings are in play and they are **different pybind11 modules**:
`pyceres` (standalone) and `pycolmap.pyceres` (built into pycolmap). Their
`LinearSolverType` enums are unrelated types, so each gets its own lookup dict.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import numpy as np
import pyceres
import pycolmap
import pycolmap.pyceres as colmap_ceres
from jaxtyping import Float64
from numpy import ndarray

from colsfm.geometry import Matrix3, QuaternionXYZW, Vector3
from colsfm.solver_report import CeresTermination, ceres_termination

LinearSolver: TypeAlias = Literal["SPARSE_SCHUR", "SPARSE_NORMAL_CHOLESKY"]
"""The two linear solvers cuSFM uses: SPARSE_SCHUR on the CPU, the other on the cuDSS path."""

PYCERES_LINEAR_SOLVER: Final[dict[LinearSolver, pyceres.LinearSolverType]] = {
    "SPARSE_SCHUR": pyceres.LinearSolverType.SPARSE_SCHUR,
    "SPARSE_NORMAL_CHOLESKY": pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY,
}
"""Name to standalone-pyceres enum value; explicit, so a typo is a KeyError not an AttributeError."""

COLMAP_LINEAR_SOLVER: Final[dict[LinearSolver, colmap_ceres.LinearSolverType]] = {
    "SPARSE_SCHUR": colmap_ceres.LinearSolverType.SPARSE_SCHUR,
    "SPARSE_NORMAL_CHOLESKY": colmap_ceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY,
}
"""The same names for pycolmap's own built-in ceres binding, which is a distinct type."""


@dataclass(frozen=True, slots=True)
class CeresSolveOptions:
    """Ceres settings, mirroring cuSFM's `BundleAdjustmentSolver::Solve` (pose_graph_main.md §7)."""

    linear_solver: LinearSolver = "SPARSE_NORMAL_CHOLESKY"
    """The blob hard-codes SPARSE_SCHUR, but neither of the two problems here has Schur
    structure to exploit (every parameter block is a pose), so the normal-equation Cholesky
    is the honest choice and measured slightly faster. SPARSE_SCHUR is accepted and gives
    the same optimum."""
    max_num_iterations: int = 200
    """`optimization_config.max_num_iterations` in the shipped isaac/av configs."""
    num_threads: int = 1
    """The blob uses 8. Both residuals here are evaluated in Python, so Ceres' worker threads
    serialise on the GIL and 8 threads measured ~30% SLOWER than 1 on a 1000-node graph.
    Raise it only if the cost functions ever move to C++."""
    function_tolerance: float = 1e-6
    """Ceres' own default, which the blob leaves untouched."""
    gradient_tolerance: float = 1e-10
    """Ceres' own default, which the blob leaves untouched."""
    parameter_tolerance: float = 1e-8
    """Ceres' own default, which the blob leaves untouched."""
    minimizer_progress_to_stdout: bool = False
    """The isaac config sets this true; off by default here to keep library output quiet."""


@dataclass(frozen=True, slots=True)
class PoseBlocks:
    """The two parameter arrays one `Rigid3d` becomes inside a Ceres problem.

    A dataclass, not a `NamedTuple`: beartype 0.23 cannot resolve the stringified array aliases of a `NamedTuple`
    field under PEP 563. Frozen only binds the attributes; Ceres writes into the arrays in place.
    """

    quat_xyzw: QuaternionXYZW
    """Unit quaternion in Eigen `(x, y, z, w)` order, the layout `EigenQuaternionManifold` wants."""
    translation: Vector3
    """Translation in metres, owned by the problem and written in place by the solve."""


@dataclass(frozen=True, slots=True)
class SolverStats:
    """The four numbers every caller reads back out of a `pyceres.SolverSummary`.

    A dataclass, not a `NamedTuple`, for two reasons that agree: no caller unpacks it
    positionally or hashes it — every one reads `solved.iterations` and friends — and
    beartype cannot check a `Literal` type alias on a `NamedTuple` field under PEP 563
    deferred annotations, which is what `termination` is.
    """

    iterations: int
    """Minimizer iterations, successful and unsuccessful."""
    initial_cost: float
    """Ceres cost before the solve."""
    final_cost: float
    """Ceres cost after the solve."""
    termination: CeresTermination
    """How the solve ended."""


def pose_parameter_blocks(pose: pycolmap.Rigid3d, *, canonical_sign: bool = False) -> PoseBlocks:
    """Prime the two owned parameter arrays Ceres optimises one pose through.

    The arrays are fresh copies: Ceres writes into them, and neither `Rigid3d` nor the
    reconstruction it came from may move as a side effect.

    Args:
        pose: The pose to start from.
        canonical_sign: Flip the quaternion so that `w >= 0`. The blob does this on its
            pose-graph nodes; it is harmless (the two signs are the same rotation) and is
            kept for parity, but the extrinsic solve does not ask for it.

    Returns:
        A normalised quaternion in `(x, y, z, w)` order and the translation, both float64.
    """
    quat_xyzw: QuaternionXYZW = np.asarray(pose.rotation.quat, dtype=np.float64).copy()
    quat_xyzw /= np.linalg.norm(quat_xyzw)
    if canonical_sign and quat_xyzw[3] < 0.0:
        quat_xyzw = -quat_xyzw
    return PoseBlocks(quat_xyzw=quat_xyzw, translation=np.asarray(pose.translation, dtype=np.float64).copy())


def rigid3d_from_parameter_blocks(quat_xyzw: QuaternionXYZW, translation: Vector3) -> pycolmap.Rigid3d:
    """Rebuild a pose from the parameter arrays a solve left behind.

    The quaternion is renormalised: `EigenQuaternionManifold` keeps it on the sphere to
    within its own tolerance, not to the last bit.

    Args:
        quat_xyzw: Float64 quaternion with shape `[4]` in `(x, y, z, w)` order.
        translation: Float64 translation in metres with shape `[3]`.

    Returns:
        The pose those two blocks encode.
    """
    unit_quat_xyzw: QuaternionXYZW = quat_xyzw / np.linalg.norm(quat_xyzw)
    return pycolmap.Rigid3d(pycolmap.Rotation3d(unit_quat_xyzw), np.asarray(translation, dtype=np.float64).copy())


def attach_quaternion_manifolds(
    problem: pyceres.Problem,
    quaternions: Mapping[int, QuaternionXYZW],
    manifold: pyceres.EigenQuaternionManifold,
) -> None:
    """Put every quaternion block that reached the problem on the quaternion manifold.

    A block nothing referenced is not in the problem at all, and `set_manifold` on it
    would raise, so the membership test is not optional.

    Args:
        problem: The problem the residual blocks were added to.
        quaternions: The quaternion arrays, keyed by whatever identifies their pose.
        manifold: One shared `EigenQuaternionManifold`; the caller keeps it alive, because
            pyceres holds a raw pointer to it.
    """
    for quat_xyzw in quaternions.values():
        if problem.has_parameter_block(quat_xyzw):
            problem.set_manifold(quat_xyzw, manifold)


def solver_options(options: CeresSolveOptions) -> pyceres.SolverOptions:
    """Fill a `pyceres.SolverOptions` from the documented settings.

    Args:
        options: The settings.

    Returns:
        The Ceres options object, ready for `pyceres.solve`.

    Raises:
        KeyError: When the linear solver name is not one of the two supported ones.
    """
    ceres_options: pyceres.SolverOptions = pyceres.SolverOptions()
    ceres_options.linear_solver_type = PYCERES_LINEAR_SOLVER[options.linear_solver]
    ceres_options.max_num_iterations = options.max_num_iterations
    ceres_options.num_threads = options.num_threads
    ceres_options.function_tolerance = options.function_tolerance
    ceres_options.gradient_tolerance = options.gradient_tolerance
    ceres_options.parameter_tolerance = options.parameter_tolerance
    ceres_options.minimizer_progress_to_stdout = options.minimizer_progress_to_stdout
    return ceres_options


def solver_stats(summary: pyceres.SolverSummary) -> SolverStats:
    """Unpack the summary fields both solves report.

    Ceres counts successful and unsuccessful steps separately; the blob's own logs report
    their sum, so that is what is reported here.

    Args:
        summary: The summary `pyceres.solve` filled in.

    Returns:
        Iterations, the two costs and the termination type's bare name.
    """
    return SolverStats(
        iterations=int(summary.num_successful_steps) + int(summary.num_unsuccessful_steps),
        initial_cost=float(summary.initial_cost),
        final_cost=float(summary.final_cost),
        termination=ceres_termination(str(summary.termination_type)),
    )


def skew_matrix(vector: Vector3, out: Matrix3) -> Matrix3:
    """Write the skew-symmetric matrix of one 3-vector into `out`.

    The in-place form exists for the Ceres inner loops, which evaluate it per residual
    block per iteration and must not allocate. `batched_skew` is the same maths over many
    vectors at once; the two live together so there is one place to read it.

    Args:
        vector: Float64 3-vector.
        out: Float64 `[3, 3]` buffer, overwritten.

    Returns:
        `out`, holding `skew(vector)`, so that `skew(v) @ w == np.cross(v, w)`.
    """
    out[0, 0], out[0, 1], out[0, 2] = 0.0, -vector[2], vector[1]
    out[1, 0], out[1, 1], out[1, 2] = vector[2], 0.0, -vector[0]
    out[2, 0], out[2, 1], out[2, 2] = -vector[1], vector[0], 0.0
    return out


def batched_skew(vectors: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 3 3"]:
    """Skew-symmetric matrix of every row of `vectors`.

    Args:
        vectors: Float64 array of one 3-vector per row.

    Returns:
        `skew(v)` per row, so that `skew(v) @ w == np.cross(v, w)`.
    """
    skew: Float64[ndarray, "n 3 3"] = np.zeros((vectors.shape[0], 3, 3), dtype=np.float64)
    skew[:, 0, 1] = -vectors[:, 2]
    skew[:, 0, 2] = vectors[:, 1]
    skew[:, 1, 0] = vectors[:, 2]
    skew[:, 1, 2] = -vectors[:, 0]
    skew[:, 2, 0] = -vectors[:, 1]
    skew[:, 2, 1] = vectors[:, 0]
    return skew
