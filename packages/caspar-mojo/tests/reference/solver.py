"""A numpy port of CASPAR's `GraphSolver::solve` (generated/f32/solver.cc:5284-5577, Rcs* at 11859-11921).

The tangent is split into blocks: one per free pose, free rig extrinsic, free camera calibration (focal and
extra, grouped by camera model because the block size depends on it) and free point. Every vector of the PCG is a
list with one (n_free, D) array per group. The arithmetic follows the dtype argument, so a float32 run shows what
float32 alone does to the trajectory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int64
from numpy import ndarray

from caspar_mojo.problem import BaOptions, BaProblem, BaResult, CameraModel
from reference.cameras import focal_and_extra_columns, project
from reference.poses import camera_point, normalize_pose, retract_pose

Vector = list[ndarray]
"""One (n_free, D) array per group; the camera groups come first, the point group last."""


@dataclass(frozen=True, slots=True)
class Group:
    """One node type's free blocks."""

    name: str
    """`pose`, `sensor`, `calib_<MODEL>` or `point`."""
    size: int
    """Tangent dimension D of one block."""
    nodes: Int64[ndarray, " n_free"]
    """Node row (frame, camera or point) of each free block."""
    observation_slot: Int64[ndarray, " n_observations"]
    """Free block of each observation's node, -1 when that node is fixed or of another model."""


@dataclass(frozen=True, slots=True)
class State:
    """The solver state in the working dtype."""

    rig_from_world: Float[ndarray, "n_frames 7"]
    sensor_from_rig: Float[ndarray, "n_cameras 7"]
    intrinsics: Float[ndarray, "n_cameras 12"]
    points: Float[ndarray, "n_points 3"]


@dataclass(frozen=True, slots=True)
class Linearization:
    """What `res_jac` leaves on the device: the cost, -J^T e, the block diagonal of J^T J and the Jacobians."""

    cost: float
    """Half the sum of squared residuals."""
    gradient: Vector
    """g = -J^T e per group."""
    block_diagonal: list[ndarray]
    """B per group, (n_free, D, D)."""
    jacobians: list[ndarray]
    """J per group, (n_observations, 2, D), zero where the observation's node is not in the group."""


@dataclass(frozen=True, slots=True)
class IterationRecord:
    """One Levenberg-Marquardt trial, as CASPAR's verbose log has it."""

    cost_tried: float
    """Cost at the retracted candidate."""
    cost_best: float
    """Best cost after the accept or reject."""
    damping: float
    """Lambda the trial was solved with."""
    pcg_iterations: int
    """PCG iterations entered (CASPAR's pcg_iter_ + 1)."""
    accepted: bool
    """Whether the candidate became the current state."""


def make_groups(problem: BaProblem) -> list[Group]:
    """The free blocks of `problem`, in the flat order [poses | sensors | calibrations by model | points]."""

    def group(name: str, size: int, free: ndarray, observation_node: ndarray) -> Group:
        nodes = np.flatnonzero(free).astype(np.int64)
        slot_of_node = np.full(len(free), -1, dtype=np.int64)
        slot_of_node[nodes] = np.arange(len(nodes))
        return Group(name, size, nodes, slot_of_node[observation_node])

    groups = [
        group("pose", 6, problem.pose_free, problem.observation_frame),
        group("sensor", 6, problem.sensor_free, problem.observation_camera),
    ]
    for model in CameraModel:
        free = problem.intrinsics_free & (problem.camera_model == model)
        if free.any():
            groups.append(group(f"calib_{model.name}", len(focal_and_extra_columns(model)), free, problem.observation_camera))
    groups.append(group("point", 3, problem.point_free, problem.observation_point))
    return groups


def residuals_and_jacobians(problem: BaProblem, state: State, groups: list[Group], with_jacobians: bool) -> tuple[ndarray, list[ndarray]]:
    """e = project(cam_from_world * point) - pixel per observation, and J per group when asked."""
    dtype = state.points.dtype
    n = len(problem.pixels)
    frame, camera, point = problem.observation_frame, problem.observation_camera, problem.observation_point
    in_camera = camera_point(state.rig_from_world[frame], state.sensor_from_rig[camera], state.points[point])
    residual = np.empty((n, 2), dtype=dtype)
    d_point_cam = np.empty((n, 2, 3), dtype=dtype)
    jacobians = [np.zeros((n, 2, g.size), dtype=dtype) for g in groups] if with_jacobians else []
    models = problem.camera_model[camera]
    for model in CameraModel:
        rows = np.flatnonzero(models == model)
        if len(rows) == 0:
            continue
        projection = project(model, in_camera.point_cam[rows], state.intrinsics[camera[rows]])
        residual[rows] = projection.pixel - problem.pixels[rows].astype(dtype)
        d_point_cam[rows] = projection.d_point
        if with_jacobians:
            for g, jacobian in zip(groups, jacobians, strict=True):
                if g.name == f"calib_{model.name}":
                    jacobian[rows] = projection.d_focal_and_extra
    if with_jacobians:
        for g, jacobian in zip(groups, jacobians, strict=True):
            node_jacobian = {"pose": in_camera.d_rig_from_world, "sensor": in_camera.d_sensor_from_rig, "point": in_camera.d_point}.get(g.name)
            if node_jacobian is not None:
                jacobian[:] = d_point_cam @ node_jacobian
            jacobian[g.observation_slot < 0] = 0
    return residual, jacobians


def _scatter(slots: ndarray, values: ndarray, n_slots: int) -> ndarray:
    out = np.zeros((n_slots, *values.shape[1:]), dtype=values.dtype)
    keep = slots >= 0
    np.add.at(out, slots[keep], values[keep])
    return out


def _gather(slots: ndarray, values: ndarray) -> ndarray:
    """values[slots] with zero rows for slot -1."""
    padded = np.concatenate([values, np.zeros((1, *values.shape[1:]), dtype=values.dtype)])
    return padded[slots]


def linearize(problem: BaProblem, state: State, groups: list[Group]) -> Linearization:
    """CASPAR's res_jac: cost, g = -J^T e and the block diagonal B of J^T J."""
    residual, jacobians = residuals_and_jacobians(problem, state, groups, with_jacobians=True)
    gradient = [
        _scatter(g.observation_slot, -np.einsum("nri,nr->ni", jacobian, residual), len(g.nodes))
        for g, jacobian in zip(groups, jacobians, strict=True)
    ]
    block_diagonal = [
        _scatter(g.observation_slot, np.einsum("nri,nrj->nij", jacobian, jacobian), len(g.nodes))
        for g, jacobian in zip(groups, jacobians, strict=True)
    ]
    return Linearization(float(0.5 * np.sum(residual * residual)), gradient, block_diagonal, jacobians)


def cost(problem: BaProblem, state: State, groups: list[Group]) -> float:
    """CASPAR's score: half the sum of squared residuals."""
    residual, _ = residuals_and_jacobians(problem, state, groups, with_jacobians=False)
    return float(0.5 * np.sum(residual * residual))


def damped_solve(block_diagonal: ndarray, damping: ndarray, rhs: ndarray) -> ndarray:
    """M^-1 rhs per block, M = B + lambda diag(B) + lambda 1e-6 I (kernel_*_normalize.cu), off-diagonals kept."""
    size = block_diagonal.shape[-1]
    if len(rhs) == 0:
        return rhs.copy()
    diagonal = np.einsum("nii->ni", block_diagonal)
    damped = block_diagonal.copy()
    index = np.arange(size)
    damped[:, index, index] = diagonal * (1 + damping) + damping * block_diagonal.dtype.type(1e-6)
    return np.linalg.solve(damped, rhs[:, :, None])[:, :, 0]


def jtj_cross(groups: list[Group], linearization: Linearization, p: Vector, w: Vector) -> None:
    """CASPAR's jtjnjtr_direct: w_a += J_a^T J_b p_b over the pairs of DIFFERENT blocks of each factor."""
    per_block = [
        np.einsum("nri,ni->nr", jacobian, _gather(g.observation_slot, block))
        for g, jacobian, block in zip(groups, linearization.jacobians, p, strict=True)
    ]
    total = sum(per_block)
    for index, (g, jacobian) in enumerate(zip(groups, linearization.jacobians, strict=True)):
        w[index] += _scatter(g.observation_slot, np.einsum("nri,nr->ni", jacobian, total - per_block[index]), len(g.nodes))


def dot(a: Vector, b: Vector) -> ndarray:
    """a . b over every group, in the working dtype."""
    return sum((np.sum(x * y) for x, y in zip(a, b, strict=True)), start=a[0].dtype.type(0))


def retract(problem: BaProblem, state: State, groups: list[Group], step: Vector) -> State:
    """CASPAR's retract: Pose3 retraction for poses and rig extrinsics, additive for calibrations and points."""
    rig, sensor, intrinsics, points = state.rig_from_world.copy(), state.sensor_from_rig.copy(), state.intrinsics.copy(), state.points.copy()
    for g, delta in zip(groups, step, strict=True):
        if g.name == "pose":
            rig[g.nodes] = retract_pose(rig[g.nodes], delta)
        elif g.name == "sensor":
            sensor[g.nodes] = retract_pose(sensor[g.nodes], delta)
        elif g.name == "point":
            points[g.nodes] += delta
        else:
            columns = focal_and_extra_columns(CameraModel[g.name.removeprefix("calib_")])
            intrinsics[np.ix_(g.nodes, columns)] += delta
    return State(rig, sensor, intrinsics, points)


def reference_solve(problem: BaProblem, options: BaOptions, dtype: type = np.float64) -> tuple[BaResult, list[IterationRecord]]:
    """Levenberg-Marquardt with CASPAR's PCG, reduced camera system, damping rule and exits.

    Args:
        problem: The bundle adjustment.
        options: CASPAR's solver parameters.
        dtype: np.float64 for the oracle, np.float32 to see what float32 alone changes.

    Returns:
        The result (iterations = CASPAR's iteration_count, pcg_iterations = the sum of `IterationRecord.pcg_iterations`)
        and one record per trial.
    """
    t_start = time.perf_counter()
    real = np.dtype(dtype).type
    groups = make_groups(problem)
    point_group = len(groups) - 1
    state = State(
        problem.rig_from_world.astype(dtype), problem.sensor_from_rig.astype(dtype), problem.intrinsics.astype(dtype), problem.points.astype(dtype)
    )
    reduced = options.reduced_camera_system
    eliminate_points = reduced and len(groups[point_group].nodes) > 0
    pcg_iter_cap = options.reduced_pcg_iter_max if reduced else options.pcg_iter_max

    diag = real(options.diag_init)
    up_scale = real(options.diag_scaling_up)
    exit_reason = "MAX_ITERATIONS"
    trace: list[IterationRecord] = []

    def zeros() -> Vector:
        return [np.zeros((len(g.nodes), g.size), dtype=dtype) for g in groups]

    def normalize(r: Vector) -> Vector:
        return [damped_solve(b, diag, block) for b, block in zip(lin.block_diagonal, r, strict=True)]

    def point_solve(rhs: ndarray) -> ndarray:
        return damped_solve(lin.block_diagonal[point_group], diag, rhs)

    def rcs_matvec(p: Vector, w: Vector) -> None:
        # w enters holding M p on the cameras and 0 on the points; p has a zero point part.
        jtj_cross(groups, lin, p, w)
        correction = [np.zeros_like(x) for x in p]
        correction[point_group] = -point_solve(w[point_group])
        jtj_cross(groups, lin, correction, w)
        w[point_group][:] = 0

    def matvec(p: Vector, w: Vector) -> None:
        if eliminate_points:
            rcs_matvec(p, w)
        else:
            jtj_cross(groups, lin, p, w)

    t_solve = time.perf_counter()
    lin = linearize(problem, state, groups)
    score_best = real(lin.cost)
    initial_cost = float(score_best)
    r, mp = [x.copy() for x in lin.gradient], [x.copy() for x in lin.gradient]
    p, w, step = zeros(), zeros(), zeros()
    solver_iter = 0
    while solver_iter < options.solver_iter_max:
        if solver_iter != 0 and solver_iter < options.solver_iter_max - 1:
            lin = linearize(problem, state, groups)
            r, mp = [x.copy() for x in lin.gradient], [x.copy() for x in lin.gradient]
        pcg_q = 0.0
        if eliminate_points:
            # RcsReduceRhs: r = g_c - H_cp H_pp^-1 g_p on the cameras, 0 on the points; Mp = r.
            r = [x.copy() for x in lin.gradient]
            w = [x.copy() for x in r]
            correction = [np.zeros_like(x) for x in r]
            correction[point_group] = -point_solve(r[point_group])
            jtj_cross(groups, lin, correction, w)
            r = w
            r[point_group][:] = 0
            mp = [x.copy() for x in r]
        rho_previous = real(0)
        pcg_iter = 0
        while pcg_iter < pcg_iter_cap:
            if pcg_iter == 0:
                p = normalize(r)
                w = [x.copy() for x in r]
                matvec(p, w)
                rho = dot(p, r)
                alpha = rho / dot(p, w)
                if reduced and not (_usable(rho) and _usable(alpha)):
                    step = zeros()
                    break
                step = [alpha * x for x in p]
                r_0_norm2 = dot(r, r)
                r = [x - alpha * y for x, y in zip(r, w, strict=True)]
                r_norm2 = dot(r, r)
                rho_previous = rho
            else:
                z = normalize(r)
                rho = dot(r, z)
                # The beta fix: beta = rho_k / rho_{k-1}, and rho_k is kept for the next iteration.
                beta = rho / rho_previous
                rho_previous = rho
                if reduced and not _usable(rho):
                    break
                p = [x + beta * y for x, y in zip(z, p, strict=True)]
                mp = [x + beta * y for x, y in zip(r, mp, strict=True)]
                w = [x.copy() for x in mp]
                matvec(p, w)
                alpha = rho / dot(p, w)
                if reduced and not _usable(alpha):
                    break
                step = [x + alpha * y for x, y in zip(step, p, strict=True)]
                r = [x - alpha * y for x, y in zip(r, w, strict=True)]
                r_norm2 = dot(r, r)
            if reduced:
                decrease = 0.5 * float(alpha) * float(rho)
                pcg_q -= decrease
                if pcg_q < 0.0 and (pcg_iter + 1) * decrease < -pcg_q * options.reduced_pcg_eta:
                    break
            elif r_norm2 < r_0_norm2 * real(options.pcg_rel_error_exit):
                break
            pcg_iter += 1
        pcg_iter = min(pcg_iter, pcg_iter_cap - 1)
        if eliminate_points:
            # RcsBackSubstitute: step_p = H_pp^-1 (g_p - H_pc step_c).
            camera_step = [x.copy() for x in step]
            camera_step[point_group][:] = 0
            coupling = zeros()
            jtj_cross(groups, lin, camera_step, coupling)
            step[point_group] = point_solve(lin.gradient[point_group] - coupling[point_group])

        candidate = retract(problem, state, groups, step)
        score_tried = real(cost(problem, candidate, groups))
        diag_current = diag
        accepted = bool(score_tried < score_best * real(options.solver_rel_decrease_min))
        if accepted:
            predicted = real(0.5) * dot(
                step, [g + diag * np.einsum("nii->ni", b) * s for g, b, s in zip(lin.gradient, lin.block_diagonal, step, strict=True)]
            )
            quality = (score_best - score_tried) / predicted
            quality_tmp = 2 * quality - 1
            scale = max(real(options.diag_scaling_down), real(1) - quality_tmp * quality_tmp * quality_tmp)
            diag = max(real(options.diag_min), diag * scale)
            up_scale = real(options.diag_scaling_up)
            score_best = score_tried
            state = candidate
        else:
            diag = diag * up_scale
            up_scale = up_scale * 2
        trace.append(IterationRecord(float(score_tried), float(score_best), float(diag_current), pcg_iter + 1, accepted))
        if not accepted and diag > options.diag_exit_value:
            exit_reason = "CONVERGED_DIAG_EXIT"
            break
        if score_best <= options.score_exit_value:
            exit_reason = "CONVERGED_SCORE_THRESHOLD"
            break
        solver_iter += 1
    solve_seconds = time.perf_counter() - t_solve

    rig = problem.rig_from_world.copy()
    rig[problem.pose_free] = normalize_pose(state.rig_from_world[problem.pose_free].astype(np.float64))
    sensor = problem.sensor_from_rig.copy()
    sensor[problem.sensor_free] = normalize_pose(state.sensor_from_rig[problem.sensor_free].astype(np.float64))
    result = BaResult(
        intrinsics=state.intrinsics.astype(np.float64),
        sensor_from_rig=sensor,
        rig_from_world=rig,
        points=state.points.astype(np.float64),
        initial_cost=initial_cost,
        final_cost=float(score_best),
        iterations=solver_iter,
        pcg_iterations=sum(record.pcg_iterations for record in trace),
        exit_reason=exit_reason,
        solve_seconds=solve_seconds,
        total_seconds=time.perf_counter() - t_start,
    )
    return result, trace


def _usable(x: ndarray | float) -> bool:
    return bool(np.isfinite(x) and x > 0)
