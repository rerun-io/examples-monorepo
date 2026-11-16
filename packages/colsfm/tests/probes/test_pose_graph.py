"""Probe (e): the pose-graph optimisation backend — pyceres against scipy.

pycolmap has no pose-graph optimiser.  It has `PoseGraph`/`PoseGraphEdge`, but
those are a *container* used by the incremental mapper's view graph, with no
`optimize` entry point.  It also has `pycolmap.cost_functions.RelativePosePriorCost`,
which looks like the missing piece — and is unusable from Python in this build
(see `test_cost_functions_cannot_reach_pyceres`).

So the pose graph must be written here, and the question is only which solver
drives it.  This module builds one synthetic loop (odometry edges plus a single
loop closure) and runs it through both candidates at the same problem size.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pyceres
import pycolmap
import pycolmap.cost_functions as colmap_costs
import pytest
from jaxtyping import Float, Int
from numpy import ndarray
from scipy.optimize import OptimizeResult, least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

NUM_NODES: int = 1000
"""Pose-graph size the plan asks to be measured at."""

ROTATION_NOISE_RAD: float = 0.002
"""Per-edge rotation noise standard deviation."""

TRANSLATION_NOISE_M: float = 0.01
"""Per-edge translation noise standard deviation."""


@dataclass(slots=True)
class LoopGraph:
    """A synthetic SE(3) pose graph: a closed circular walk with one loop closure."""

    truth_R: Float[ndarray, "n 3 3"]
    """Ground-truth `i_from_world` rotations."""
    truth_t: Float[ndarray, "n 3"]
    """Ground-truth `i_from_world` translations, in metres."""
    initial_R: Float[ndarray, "n 3 3"]
    """Drifted initial guess from integrating the noisy odometry."""
    initial_t: Float[ndarray, "n 3"]
    """Drifted initial guess translations, in metres."""
    edge_from: Int[ndarray, " m"]
    """First node of each edge."""
    edge_to: Int[ndarray, " m"]
    """Second node of each edge."""
    measured_R: Float[ndarray, "m 3 3"]
    """Measured relative rotation `i_from_j` per edge."""
    measured_t: Float[ndarray, "m 3"]
    """Measured relative translation `i_from_j` per edge, in metres."""


def build_loop_graph(num_nodes: int = NUM_NODES, seed: int = 7) -> LoopGraph:
    """Build the closed-loop pose graph.

    Args:
        num_nodes: Number of poses on the loop.
        seed: PRNG seed for the edge noise.

    Returns:
        The populated `LoopGraph`.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    angles: Float[ndarray, " n"] = np.linspace(0.0, 2.0 * np.pi, num_nodes, endpoint=False)
    world_t: Float[ndarray, "n 3"] = np.stack([10.0 * np.cos(angles), 10.0 * np.sin(angles), np.zeros(num_nodes)], axis=1)
    world_R: Float[ndarray, "n 3 3"] = Rotation.from_euler("z", angles[:, None]).as_matrix()
    truth_R: Float[ndarray, "n 3 3"] = np.transpose(world_R, (0, 2, 1))
    truth_t: Float[ndarray, "n 3"] = -np.einsum("nij,nj->ni", truth_R, world_t)

    edge_from: Int[ndarray, " m"] = np.array(list(range(num_nodes - 1)) + [num_nodes - 1], dtype=np.int64)
    edge_to: Int[ndarray, " m"] = np.array(list(range(1, num_nodes)) + [0], dtype=np.int64)

    measured_R: list[Float[ndarray, "3 3"]] = []
    measured_t: list[Float[ndarray, "3"]] = []
    for node_a, node_b in zip(edge_from, edge_to, strict=True):
        relative_R: Float[ndarray, "3 3"] = truth_R[node_a] @ truth_R[node_b].T
        relative_t: Float[ndarray, "3"] = truth_t[node_a] - relative_R @ truth_t[node_b]
        noise: Float[ndarray, "3 3"] = Rotation.from_rotvec(rng.normal(0.0, ROTATION_NOISE_RAD, 3)).as_matrix()
        measured_R.append(noise @ relative_R)
        measured_t.append(relative_t + rng.normal(0.0, TRANSLATION_NOISE_M, 3))

    initial_R: list[Float[ndarray, "3 3"]] = [truth_R[0].copy()]
    initial_t: list[Float[ndarray, "3"]] = [truth_t[0].copy()]
    for edge_index in range(num_nodes - 1):
        initial_R.append(measured_R[edge_index] @ initial_R[-1])
        initial_t.append(measured_R[edge_index] @ initial_t[-1] + measured_t[edge_index])

    return LoopGraph(
        truth_R=truth_R,
        truth_t=truth_t,
        initial_R=np.asarray(initial_R),
        initial_t=np.asarray(initial_t),
        edge_from=edge_from,
        edge_to=edge_to,
        measured_R=np.asarray(measured_R),
        measured_t=np.asarray(measured_t),
    )


def _loop_seam_m(graph: LoopGraph, poses_t: Float[ndarray, "n 3"]) -> float:
    """Distance the loop-closure edge is violated by, in metres.

    Args:
        graph: The pose graph.
        poses_t: Current `i_from_world` translations.

    Returns:
        The norm of the loop-closure translation residual.
    """
    predicted: Float[ndarray, "3"] = graph.measured_R[-1] @ poses_t[-1] + graph.measured_t[-1]
    return float(np.linalg.norm(predicted - poses_t[0]))


class _RelativePoseCost(pyceres.CostFunction):
    """A 6-DoF relative-pose residual with numerically differentiated Jacobians.

    Parameter blocks are `[quat_i (4), trans_i (3), quat_j (4), trans_j (3)]`,
    matching pycolmap's `Rigid3d` layout (`rotation.quat` is xyzw), so the same
    arrays can be handed straight to `pycolmap.Rigid3d`.  The quaternion blocks
    carry a `pyceres.EigenQuaternionManifold`.
    """

    def __init__(self, measured_R: Float[ndarray, "3 3"], measured_t: Float[ndarray, "3"]) -> None:
        super().__init__()
        self.set_num_residuals(6)
        self.set_parameter_block_sizes([4, 3, 4, 3])
        self._measured_R_transpose: Float[ndarray, "3 3"] = measured_R.T
        self._measured_t: Float[ndarray, "3"] = measured_t

    def _residual(
        self,
        quat_i: Float[ndarray, "4"],
        trans_i: Float[ndarray, "3"],
        quat_j: Float[ndarray, "4"],
        trans_j: Float[ndarray, "3"],
    ) -> Float[ndarray, "6"]:
        """Log of the measured-versus-predicted relative pose."""
        rotation_i: Float[ndarray, "3 3"] = Rotation.from_quat(quat_i / np.linalg.norm(quat_i)).as_matrix()
        rotation_j: Float[ndarray, "3 3"] = Rotation.from_quat(quat_j / np.linalg.norm(quat_j)).as_matrix()
        predicted_R: Float[ndarray, "3 3"] = rotation_i @ rotation_j.T
        rotation_error: Float[ndarray, "3"] = Rotation.from_matrix(self._measured_R_transpose @ predicted_R).as_rotvec()
        translation_error: Float[ndarray, "3"] = self._measured_R_transpose @ (trans_i - predicted_R @ trans_j - self._measured_t)
        return np.concatenate([rotation_error, translation_error])

    def Evaluate(self, parameters: list, residuals: ndarray, jacobians: list | None) -> bool:
        """Ceres entry point.

        The override must be named `Evaluate` (capital E) even though the bound
        read-only method on the class is `evaluate`; a lowercase override is
        never found and ceres aborts with "Tried to call pure virtual function".
        """
        blocks: list[Float[ndarray, " k"]] = [np.asarray(block, dtype=np.float64) for block in parameters]
        base: Float[ndarray, "6"] = self._residual(*blocks)
        residuals[:] = base
        if jacobians is None:
            return True
        step: float = 1e-7
        for block_index, jacobian in enumerate(jacobians):
            if jacobian is None:
                continue
            columns: Float[ndarray, "6 k"] = np.empty((6, blocks[block_index].size))
            for column in range(blocks[block_index].size):
                perturbed: list[Float[ndarray, " k"]] = [block.copy() for block in blocks]
                perturbed[block_index][column] += step
                columns[:, column] = (self._residual(*perturbed) - base) / step
            jacobian[:] = columns.ravel()
        return True


def solve_with_pyceres(graph: LoopGraph) -> tuple[Float[ndarray, "n 3"], float, pyceres.SolverSummary]:
    """Optimise the pose graph with pyceres.

    Args:
        graph: The pose graph.

    Returns:
        `(translations, wall seconds, summary)`.
    """
    quaternions: list[Float[ndarray, "4"]] = [np.ascontiguousarray(quat) for quat in Rotation.from_matrix(graph.initial_R).as_quat()]
    translations: list[Float[ndarray, "3"]] = [np.ascontiguousarray(vector.copy()) for vector in graph.initial_t]

    problem: pyceres.Problem = pyceres.Problem()
    for edge_index, (node_a, node_b) in enumerate(zip(graph.edge_from, graph.edge_to, strict=True)):
        problem.add_residual_block(
            _RelativePoseCost(graph.measured_R[edge_index], graph.measured_t[edge_index]),
            pyceres.TrivialLoss(),
            [quaternions[node_a], translations[node_a], quaternions[node_b], translations[node_b]],
        )
    for node in range(len(quaternions)):
        problem.set_manifold(quaternions[node], pyceres.EigenQuaternionManifold())
    problem.set_parameter_block_constant(quaternions[0])
    problem.set_parameter_block_constant(translations[0])

    options: pyceres.SolverOptions = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    options.max_num_iterations = 50
    options.num_threads = 1
    summary: pyceres.SolverSummary = pyceres.SolverSummary()
    started: float = time.perf_counter()
    pyceres.solve(options, problem, summary)
    elapsed: float = time.perf_counter() - started
    return np.asarray(translations), elapsed, summary


def solve_with_scipy(graph: LoopGraph, max_nfev: int = 40) -> tuple[Float[ndarray, "n 3"], float, OptimizeResult]:
    """Optimise the same pose graph with `scipy.optimize.least_squares`.

    Node 0 is dropped from the parameter vector to fix the gauge, and the
    Jacobian is finite-differenced under a supplied sparsity pattern.

    Args:
        graph: The pose graph.
        max_nfev: Cap on residual evaluations, so the probe terminates.

    Returns:
        `(translations, wall seconds, OptimizeResult)`.
    """
    num_nodes: int = len(graph.truth_t)
    num_edges: int = len(graph.edge_from)
    measured_R_transpose: Float[ndarray, "m 3 3"] = np.transpose(graph.measured_R, (0, 2, 1))
    fixed_rotvec: Float[ndarray, "3"] = Rotation.from_matrix(graph.truth_R[0]).as_rotvec()

    def unpack(state: Float[ndarray, " p"]) -> tuple[Float[ndarray, "n 3 3"], Float[ndarray, "n 3"]]:
        rotvecs: Float[ndarray, "n 3"] = np.zeros((num_nodes, 3))
        translations: Float[ndarray, "n 3"] = np.zeros((num_nodes, 3))
        rotvecs[0] = fixed_rotvec
        translations[0] = graph.truth_t[0]
        rotvecs[1:] = state[: 3 * (num_nodes - 1)].reshape(num_nodes - 1, 3)
        translations[1:] = state[3 * (num_nodes - 1) :].reshape(num_nodes - 1, 3)
        return Rotation.from_rotvec(rotvecs).as_matrix(), translations

    def residual(state: Float[ndarray, " p"]) -> Float[ndarray, " r"]:
        rotations, translations = unpack(state)
        predicted_R: Float[ndarray, "m 3 3"] = rotations[graph.edge_from] @ np.transpose(rotations[graph.edge_to], (0, 2, 1))
        predicted_t: Float[ndarray, "m 3"] = translations[graph.edge_from] - np.einsum("nij,nj->ni", predicted_R, translations[graph.edge_to])
        rotation_error: Float[ndarray, "m 3"] = Rotation.from_matrix(measured_R_transpose @ predicted_R).as_rotvec()
        translation_error: Float[ndarray, "m 3"] = np.einsum("nij,nj->ni", measured_R_transpose, predicted_t - graph.measured_t)
        return np.concatenate([rotation_error.ravel(), translation_error.ravel()])

    num_params: int = 6 * (num_nodes - 1)
    sparsity: lil_matrix = lil_matrix((6 * num_edges, num_params), dtype=np.int8)

    def columns_for(node: int) -> list[int]:
        if node == 0:
            return []
        offset: int = node - 1
        rotation_block: list[int] = list(range(3 * offset, 3 * offset + 3))
        translation_start: int = 3 * (num_nodes - 1) + 3 * offset
        return rotation_block + list(range(translation_start, translation_start + 3))

    for edge_index, (node_a, node_b) in enumerate(zip(graph.edge_from, graph.edge_to, strict=True)):
        touched: list[int] = columns_for(int(node_a)) + columns_for(int(node_b))
        if not touched:
            continue
        for block_offset in (0, 3 * num_edges):
            for axis in range(3):
                sparsity[block_offset + 3 * edge_index + axis, touched] = 1
    pattern = sparsity.tocsr()

    initial_state: Float[ndarray, " p"] = np.concatenate(
        [
            Rotation.from_matrix(graph.initial_R[1:]).as_rotvec().ravel(),
            graph.initial_t[1:].ravel(),
        ]
    )
    started: float = time.perf_counter()
    result = least_squares(
        residual,
        initial_state,
        jac_sparsity=pattern,
        method="trf",
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
        max_nfev=max_nfev,
    )
    elapsed: float = time.perf_counter() - started
    _, translations = unpack(result.x)
    return translations, elapsed, result


def test_pycolmap_has_no_pose_graph_optimiser() -> None:
    """`PoseGraph` is a container, not an optimiser: nothing in pycolmap solves it."""
    graph: pycolmap.PoseGraph = pycolmap.PoseGraph()
    methods: set[str] = {name for name in dir(graph) if not name.startswith("_")}
    assert "add_edge" in methods and "get_edge" in methods
    assert not methods & {"optimize", "solve", "run", "optimise"}
    edge: pycolmap.PoseGraphEdge = pycolmap.PoseGraphEdge()
    # An edge holds only a relative pose, a match count and a validity flag —
    # there is no information matrix, so an edge cannot be weighted.
    fields: set[str] = {name for name in dir(edge) if not name.startswith("_")}
    assert {"cam2_from_cam1", "num_matches", "valid"} <= fields
    assert not fields & {"covariance", "information", "weight"}
    assert not any("pose_graph" in name and ("optim" in name or "solve" in name) for name in dir(pycolmap))


def test_cost_functions_cannot_reach_pyceres() -> None:
    """`pycolmap.cost_functions` is unusable with the standalone `pyceres` build.

    Every factory returns a `ceres::CostFunction*` whose concrete type
    (`ceres::AutoDiffCostFunction<...>`) is registered in neither pybind11
    registry: pycolmap's embedded `pyceres` omits `CostFunction`, and the conda
    `pyceres` extension has its own separate registry.  Constructing one raises
    `TypeError: Unregistered type`, whatever the import order.

    That rules out `RelativePosePriorCost` / `AbsolutePosePriorCost` as the
    pose-graph residuals and is why this module implements its own.
    """
    for factory in (
        lambda: colmap_costs.RelativePosePriorCost(pycolmap.Rigid3d()),
        lambda: colmap_costs.AbsolutePosePriorCost(pycolmap.Rigid3d()),
        lambda: colmap_costs.ReprojErrorCost(pycolmap.CameraModelId.PINHOLE, np.zeros(2)),
    ):
        with pytest.raises(TypeError, match="Unable to convert function return value") as caught:
            factory()
        assert "Unregistered type" in str(caught.value.__cause__)


LOOP_PROBE_NODES: int = 200
"""Nodes in the unit-lane loop probe: the same graph shape, a fifth of the solve (the 1000-node comparison is `perf`)."""


def test_pyceres_pose_graph_closes_the_loop() -> None:
    """pyceres converges on a noisy loop and pulls the seam shut."""
    graph: LoopGraph = build_loop_graph(num_nodes=LOOP_PROBE_NODES)
    seam_before: float = _loop_seam_m(graph, graph.initial_t)
    error_before: float = float(np.linalg.norm(graph.initial_t - graph.truth_t, axis=1).mean())

    translations, elapsed, summary = solve_with_pyceres(graph)
    seam_after: float = _loop_seam_m(graph, translations)
    error_after: float = float(np.linalg.norm(translations - graph.truth_t, axis=1).mean())

    print(
        f"[probe] pyceres PGO {LOOP_PROBE_NODES} nodes: {elapsed:.2f} s wall, "
        f"cost {summary.initial_cost:.3e} -> {summary.final_cost:.3e}, "
        f"{summary.termination_type}, "
        f"jacobian {summary.jacobian_evaluation_time_in_seconds:.2f} s / "
        f"linear solver {summary.linear_solver_time_in_seconds:.2f} s"
    )
    print(f"[probe] pyceres PGO: loop seam {seam_before:.4f} -> {seam_after:.2e} m, mean node error {error_before:.4f} -> {error_after:.4f} m")
    assert summary.termination_type == pyceres.TerminationType.CONVERGENCE
    assert summary.final_cost < 1e-4 * summary.initial_cost
    # The loop-closure measurement is itself noisy, so the maximum-likelihood
    # solution spreads the seam around the loop rather than driving it to zero.
    assert seam_after < 0.25 * seam_before
    assert error_after < error_before


@pytest.mark.perf
def test_scipy_pose_graph_is_slower_per_unit_of_progress() -> None:
    """scipy solves the same graph but does not converge inside a comparable budget.

    This is the measurement behind the backend choice.  Both solvers get the
    same 1000-node graph; pyceres converges, scipy is stopped at `max_nfev`.
    """
    graph: LoopGraph = build_loop_graph()
    seam_before: float = _loop_seam_m(graph, graph.initial_t)
    translations, elapsed, result = solve_with_scipy(graph)
    seam_after: float = _loop_seam_m(graph, translations)
    print(
        f"[probe] scipy PGO {NUM_NODES} nodes: {elapsed:.2f} s wall for "
        f"nfev={result.nfev} njev={result.njev}, cost {result.cost:.3e}, status={result.status}, "
        f"loop seam {seam_before:.4f} -> {seam_after:.4f} m"
    )
    # status == 0 means `max_nfev` was reached, i.e. it had not converged.
    assert result.status == 0
    assert seam_after < seam_before


def test_relative_pose_blocks_match_pycolmap_rigid3d_layout() -> None:
    """The optimised parameter blocks map straight back onto `pycolmap.Rigid3d`.

    `Rigid3d.rotation.quat` is a writable float64 xyzw view and
    `Rigid3d.translation` a writable float64 3-vector, so a `Rigid3d` can be
    handed to `pyceres.Problem` directly rather than round-tripping through
    numpy.  Both are the same buffer on every access.
    """
    pose: pycolmap.Rigid3d = pycolmap.Rigid3d()
    quaternion: Float[ndarray, "4"] = pose.rotation.quat
    assert quaternion.dtype == np.float64 and quaternion.shape == (4,)
    assert quaternion.flags["WRITEABLE"]
    assert np.allclose(quaternion, [0.0, 0.0, 0.0, 1.0]), "layout is xyzw, not wxyz"
    assert pose.rotation.quat.__array_interface__["data"][0] == pose.rotation.quat.__array_interface__["data"][0]
    translation: Float[ndarray, "3"] = pose.translation
    assert translation.dtype == np.float64 and translation.flags["WRITEABLE"]
    # Sim3d exists too, with an extra scale, for similarity-transform problems.
    similarity: pycolmap.Sim3d = pycolmap.Sim3d()
    assert similarity.params.shape == (8,)
