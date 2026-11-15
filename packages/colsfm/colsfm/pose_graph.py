"""Rig-level pose graph optimisation — the Python replacement for cuSFM `pose_graph_main`.

Specified by `docs/spec/pose_graph_main.md` (algorithm §6, Ceres problem §7, happy path
§8) and `docs/spec/generate_association_main.md` §7 (loop-edge weighting). What this
module implements is the "fast happy path": one global solve over the rig (vehicle)
poses, sequential edges taken from the input odometry, loop edges supplied by the
caller, one fully-fixed anchor, no robust loss.

Conventions, all from the spec:

* A node is one `synced_sample_id` (a synchronised capture of every camera), and its
  pose is `world_T_rig`. Camera poses are *derived* on export as
  `world_T_rig * rig_T_cam`; optimising per-camera poses would let the rig deform.
* An edge measurement is `source_T_target = world_T_source^-1 * world_T_target`.
  Angles are degrees on disk and radians everywhere in here.
* `pose_covariance` in the cuSFM protos is consumed as an INFORMATION matrix despite
  its name.
* The residual is `[log(R_err); t_err]` with `T_err = meas^-1 * (world_T_a^-1 *
  world_T_b)`, whitened by `chol(information).T`. There is no robust loss on pose-graph
  constraints — robustness comes from the acceptance gates on loop candidates.

Deliberately not implemented (spec §8 "Can drop"): switchable constraints, image-based
sequential edge re-estimation, the blob's mid-stream incremental re-solves (which
silently duplicate earlier edges), and the camera-frame `pose_graph.pb.txt`.

Quaternion storage order: parameter blocks hold `(x, y, z, w)`, the order used by
`pycolmap.Rotation3d.quat`, `scipy.spatial.transform.Rotation.as_quat()` and Eigen — so
the manifold is `pyceres.EigenQuaternionManifold`, not `pyceres.QuaternionManifold`
(which is w-first). The blob's choice between the two was left open by the spec (§11
question 1) and does not affect the optimum.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
import pyceres
import pycolmap
from jaxtyping import Float

from colsfm.ceres_pose import (
    CeresSolveOptions,
    PoseBlocks,
    SolverStats,
    attach_quaternion_manifolds,
    pose_parameter_blocks,
    rigid3d_from_parameter_blocks,
    skew_matrix,
    solver_options,
    solver_stats,
)
from colsfm.solver_report import CeresTermination

EdgeKind: TypeAlias = Literal["consecutive", "loop"]
Information6: TypeAlias = Float[np.ndarray, "6 6"]
Vector6: TypeAlias = Float[np.ndarray, "6"]
Matrix3: TypeAlias = Float[np.ndarray, "3 3"]
Vector3: TypeAlias = Float[np.ndarray, "3"]
QuaternionXYZW: TypeAlias = Float[np.ndarray, "4"]

_IDENTITY_3: Matrix3 = np.eye(3, dtype=np.float64)
_SMALL_ANGLE_RAD: float = 1e-8


# ======================================================================================
# graph
# ======================================================================================


@dataclass(frozen=True, slots=True)
class RigNode:
    """One optimisable node of the pose graph: a synchronised capture of the whole rig."""

    rig_id: int
    """The `synced_sample_id` shared by every camera keyframe of this capture."""
    world_T_rig: pycolmap.Rigid3d
    """Pose of the rig (vehicle) frame in world coordinates."""
    timestamp_us: int
    """Capture time in microseconds; used to order sequential edges."""


@dataclass(frozen=True, slots=True)
class PoseGraphEdge:
    """A relative-pose constraint between two rig nodes."""

    source: int
    """`rig_id` of the source node."""
    target: int
    """`rig_id` of the target node."""
    source_T_target: pycolmap.Rigid3d
    """The measurement, `world_T_source^-1 * world_T_target` (spec §6/§8)."""
    information: Information6
    """Symmetric positive-definite 6x6 weight, rotation block first then translation."""
    kind: EdgeKind
    """`"consecutive"` for odometry links, `"loop"` for revisit constraints."""


def default_information(relative_rotation_sigma_deg: float, relative_translation_sigma_m: float) -> Information6:
    """Build the blob's `DefaultPoseInformationMatrix` (spec §7 "Default information matrix").

    Args:
        relative_rotation_sigma_deg: One-sigma rotation error of a relative pose, degrees.
        relative_translation_sigma_m: One-sigma translation error of a relative pose, metres.

    Returns:
        `blockdiag(I3 / sigma_rot_rad**2, I3 / sigma_trans_m**2)` as a 6x6 float64 array.
    """
    if relative_rotation_sigma_deg <= 0.0 or relative_translation_sigma_m <= 0.0:
        raise ValueError("both sigmas must be positive to build a default information matrix")
    rotation_sigma_rad: float = np.deg2rad(relative_rotation_sigma_deg)
    information: Information6 = np.zeros((6, 6), dtype=np.float64)
    information[:3, :3] = _IDENTITY_3 / rotation_sigma_rad**2
    information[3:, 3:] = _IDENTITY_3 / relative_translation_sigma_m**2
    return information


def loop_edge_information(score: float, loop_residual_weight: float) -> Information6:
    """Weight a loop edge the way `pose_graph_main` writes it (spec §6.4 step 8).

    `Lambda = I6 * score * loop_residual_weight`, where the blob's score is `1/||t||` of
    the camera-frame two-view estimate. `generate_association_main` uses the same field
    with `score = inlier count` (its spec §7.1 item 7); either way the value that lands
    in the matrix is a single scalar on the diagonal.

    Note that the shipped isaac config leaves `relative_constraint_use_pose_covariance`
    false, so the blob's own solver OVERWRITES this matrix with the default one — see
    `solve_pose_graph` and the Galileo A/B test for the measured consequence.

    Args:
        score: The candidate score.
        loop_residual_weight: `PoseGraphConfig.loop_residual_weight` (0.25 in isaac/av).

    Returns:
        The 6x6 information matrix.
    """
    weight: float = score * loop_residual_weight
    if weight <= 0.0:
        raise ValueError(f"loop edge weight must be positive, got score={score} * weight={loop_residual_weight}")
    return np.eye(6, dtype=np.float64) * weight


def sequential_edges(
    nodes: Sequence[RigNode],
    connected_keyframe_num: int,
    relative_translation_sigma_m: float | None = None,
    relative_rotation_sigma_deg: float | None = None,
) -> list[PoseGraphEdge]:
    """Link each rig to the next `connected_keyframe_num` rigs in time order (spec §6.3).

    The measurement is the input odometry's own relative pose, `world_T_source^-1 *
    world_T_target`; the blob's image-based alternative
    (`use_sequential_edge_pose_estimation`) is off in every shipped config and measurably
    worse (spec §9.4), so it is not implemented.

    Args:
        nodes: The rig nodes; ordered internally by `timestamp_us` then `rig_id`.
        connected_keyframe_num: How many following rigs each rig links to. `RunPoseGraph`
            rejects anything below 1.
        relative_translation_sigma_m: One-sigma odometry translation error, metres. When
            this and the rotation sigma are both given the edges carry
            `default_information(...)` instead of the identity.
        relative_rotation_sigma_deg: One-sigma odometry rotation error, degrees.

    Returns:
        The consecutive edges, source-major in time order.
    """
    if connected_keyframe_num < 1:
        raise ValueError(f"connected_keyframe_num is {connected_keyframe_num}, which should be at least 1")
    if (relative_translation_sigma_m is None) != (relative_rotation_sigma_deg is None):
        raise ValueError("give both relative_translation_sigma_m and relative_rotation_sigma_deg, or neither")

    if relative_translation_sigma_m is None or relative_rotation_sigma_deg is None:
        information: Information6 = np.eye(6, dtype=np.float64)
    else:
        information = default_information(
            relative_rotation_sigma_deg=relative_rotation_sigma_deg, relative_translation_sigma_m=relative_translation_sigma_m
        )

    ordered: list[RigNode] = sorted(nodes, key=lambda node: (node.timestamp_us, node.rig_id))
    edges: list[PoseGraphEdge] = []
    for index, source in enumerate(ordered):
        for offset in range(1, connected_keyframe_num + 1):
            if index + offset >= len(ordered):
                break
            target: RigNode = ordered[index + offset]
            edges.append(
                PoseGraphEdge(
                    source=source.rig_id,
                    target=target.rig_id,
                    source_T_target=source.world_T_rig.inverse() * target.world_T_rig,
                    information=information.copy(),
                    kind="consecutive",
                )
            )
    return edges


def gate_loop_edges(edges: Sequence[PoseGraphEdge], max_translation_m: float, max_rotation_deg: float) -> list[PoseGraphEdge]:
    """Drop loop edges whose relative pose is too large (spec §6.4 step 7).

    A loop candidate survives when `||t|| <= max_translation_m` AND
    `angle_deg <= max_rotation_deg`. Consecutive edges pass through untouched.

    The shipped isaac and av configs leave `loop_edge_translation_threshold_meters` and
    `loop_edge_rotation_threshold_degrees` unset, i.e. at the proto3 default **0.0**,
    which rejects every candidate and is why `pose_graph_main` never produces a loop edge
    out of the box (spec §1, §8 "Should fix"). Pass real values (the §9.2 A/B used 5.0 m
    and 60 deg) to make the loop path live, and measure: on Galileo the loops move camera
    positions by up to 12.7 mm against a 5 mm ATE budget.

    Args:
        edges: Mixed consecutive and loop edges.
        max_translation_m: Translation magnitude gate, metres.
        max_rotation_deg: Rotation magnitude gate, degrees.

    Returns:
        The surviving edges in input order.
    """
    kept: list[PoseGraphEdge] = []
    for edge in edges:
        if edge.kind != "loop":
            kept.append(edge)
            continue
        translation_m: float = float(np.linalg.norm(edge.source_T_target.translation))
        rotation_deg: float = float(np.rad2deg(edge.source_T_target.rotation.angle()))
        if translation_m <= max_translation_m and rotation_deg <= max_rotation_deg:
            kept.append(edge)
    return kept


# ======================================================================================
# SO(3) helpers
# ======================================================================================


def rotation_matrix_from_quat_xyzw(quat_xyzw: QuaternionXYZW, out: Matrix3) -> Matrix3:
    """Write the rotation matrix of a unit quaternion `(x, y, z, w)` into `out`."""
    x, y, z, w = float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]), float(quat_xyzw[3])
    tx, ty, tz = x + x, y + y, z + z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, tyy, tzz = tx * x, ty * y, tz * z
    txy, txz, tyz = tx * y, tx * z, ty * z
    out[0, 0], out[0, 1], out[0, 2] = 1.0 - (tyy + tzz), txy - twz, txz + twy
    out[1, 0], out[1, 1], out[1, 2] = txy + twz, 1.0 - (txx + tzz), tyz - twx
    out[2, 0], out[2, 1], out[2, 2] = txz - twy, tyz + twx, 1.0 - (txx + tyy)
    return out


def _log_so3(rotation: Matrix3, out: Vector3) -> float:
    """Write the rotation vector of `rotation` into `out` and return its angle in radians.

    The blob extracts the angle as `2 * atan2(||q.vec||, |q.w|)` with a
    `2.220446049250313e-16` guard on the vector norm (spec §7); the matrix form used here
    is the same map, and the guard becomes the small-angle branch below.
    """
    cosine: float = float(np.clip((rotation[0, 0] + rotation[1, 1] + rotation[2, 2] - 1.0) * 0.5, -1.0, 1.0))
    angle_rad: float = float(np.arccos(cosine))
    scale: float = 0.5 if angle_rad < _SMALL_ANGLE_RAD else angle_rad / (2.0 * np.sin(angle_rad))
    out[0] = (rotation[2, 1] - rotation[1, 2]) * scale
    out[1] = (rotation[0, 2] - rotation[2, 0]) * scale
    out[2] = (rotation[1, 0] - rotation[0, 1]) * scale
    return angle_rad


def _inverse_right_jacobian(rotvec: Vector3, angle_rad: float, scratch: Matrix3) -> Matrix3:
    """Inverse right Jacobian of SO(3) at `rotvec`, i.e. `d log(Exp(phi) Exp(dphi)) / d dphi`."""
    skew: Matrix3 = skew_matrix(rotvec, scratch)
    if angle_rad < 1e-6:
        return _IDENTITY_3 + 0.5 * skew
    coefficient: float = 1.0 / angle_rad**2 - (1.0 + np.cos(angle_rad)) / (2.0 * angle_rad * np.sin(angle_rad))
    return _IDENTITY_3 + 0.5 * skew + coefficient * (skew @ skew)


def quaternion_plus_jacobian_transpose(quat_xyzw: QuaternionXYZW, out: Float[np.ndarray, "3 4"]) -> Float[np.ndarray, "3 4"]:
    """Write `2 * PlusJacobian(q).T` for `EigenQuaternionManifold` into `out`.

    Ceres' `EigenQuaternionManifold` uses `Plus(q, d) = q * exp(d)`, whose 4x3 Jacobian is
    `P = 0.5 * [[w*I3 + skew(v)], [-v.T]]` and satisfies `P.T @ P = 0.25 * I3`. A cost
    function must report the Jacobian in the AMBIENT 4-vector, and Ceres then forms
    `J_ambient @ P`; so `J_ambient = J_local @ (4 * P.T)` reproduces `J_local` exactly.
    This helper returns `4 * P.T = 2 * [[w*I3 + skew(v)], [-v.T]].T`.
    """
    x, y, z, w = float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]), float(quat_xyzw[3])
    out[0, 0], out[0, 1], out[0, 2], out[0, 3] = 2.0 * w, 2.0 * z, -2.0 * y, -2.0 * x
    out[1, 0], out[1, 1], out[1, 2], out[1, 3] = -2.0 * z, 2.0 * w, 2.0 * x, -2.0 * y
    out[2, 0], out[2, 1], out[2, 2], out[2, 3] = 2.0 * y, -2.0 * x, 2.0 * w, -2.0 * z
    return out


def _sqrt_information(information: Information6) -> Information6:
    """Return `chol(information).T`, so that `||S @ e||**2 == e.T @ information @ e`."""
    matrix: Information6 = np.asarray(information, dtype=np.float64)
    if matrix.shape != (6, 6):
        raise ValueError(f"information matrix must be 6x6, got {matrix.shape}")
    try:
        return np.linalg.cholesky(matrix).T
    except np.linalg.LinAlgError as error:
        raise ValueError("information matrix must be positive definite") from error


def relative_pose_error(world_T_source: pycolmap.Rigid3d, world_T_target: pycolmap.Rigid3d, source_T_target: pycolmap.Rigid3d) -> Vector6:
    """The unwhitened 6-vector residual `[log(R_err); t_err]` of one edge, radians and metres.

    Args:
        world_T_source: Current pose of the source node.
        world_T_target: Current pose of the target node.
        source_T_target: The edge measurement.

    Returns:
        `[rotation vector of E.rotation; E.translation]` with
        `E = source_T_target^-1 * (world_T_source^-1 * world_T_target)`.
    """
    error_transform: pycolmap.Rigid3d = source_T_target.inverse() * (world_T_source.inverse() * world_T_target)
    residual: Vector6 = np.empty(6, dtype=np.float64)
    _log_so3(np.ascontiguousarray(error_transform.rotation.matrix(), dtype=np.float64), residual[:3])
    residual[3:] = error_transform.translation
    return residual


# ======================================================================================
# the Ceres problem
# ======================================================================================


class RelativePoseCost(pyceres.CostFunction):
    """The `OptimizePoseRelativeConstraintCost` of spec §7, in Python with analytic Jacobians.

    Parameter blocks are `[quat_source (4), translation_source (3), quat_target (4),
    translation_target (3)]`, each pair holding `world_T_rig` with the quaternion in
    `(x, y, z, w)` order. `pycolmap` ships an equivalent C++ functor
    (`pycolmap.cost_functions.RelativePosePriorCost`) but it cannot be handed to
    `pyceres.Problem`: the two extensions are built against different pybind11 internals
    versions (v12 vs v11), so `pycolmap`'s `ceres::CostFunction` is an unregistered type in
    `pyceres`' registry and the object cannot even be constructed from Python. Its residual
    is also `2 * vec(q_err)` rather than the exact `log(R_err)` the blob uses.
    """

    def __init__(self, source_T_target: pycolmap.Rigid3d, information: Information6) -> None:
        """Set up one edge.

        Args:
            source_T_target: The measurement.
            information: Symmetric positive-definite 6x6 weight.
        """
        super().__init__()
        self.set_num_residuals(6)
        self.set_parameter_block_sizes([4, 3, 4, 3])
        measured: Matrix3 = np.ascontiguousarray(source_T_target.rotation.matrix(), dtype=np.float64)
        self._target_R_source: Matrix3 = np.ascontiguousarray(measured.T)
        self._measured_translation: Vector3 = np.ascontiguousarray(source_T_target.translation, dtype=np.float64)
        self._sqrt_information: Information6 = _sqrt_information(information)
        self._source_R_world: Matrix3 = np.empty((3, 3), dtype=np.float64)
        self._world_R_target: Matrix3 = np.empty((3, 3), dtype=np.float64)
        self._scratch: Matrix3 = np.empty((3, 3), dtype=np.float64)
        self._plus_jacobian_t: Float[np.ndarray, "3 4"] = np.empty((3, 4), dtype=np.float64)
        self._error: Vector6 = np.empty(6, dtype=np.float64)
        self._block_jacobian: Float[np.ndarray, "6 3"] = np.zeros((6, 3), dtype=np.float64)

    def Evaluate(self, parameters: list, residuals: np.ndarray, jacobians: list | None) -> bool:
        """Evaluate the whitened residual and, when asked, the ambient Jacobians."""
        quat_source, translation_source, quat_target, translation_target = parameters
        world_R_source: Matrix3 = rotation_matrix_from_quat_xyzw(quat_source, self._source_R_world)
        world_R_target: Matrix3 = rotation_matrix_from_quat_xyzw(quat_target, self._world_R_target)
        source_R_world: Matrix3 = world_R_source.T
        target_R_source: Matrix3 = self._target_R_source

        error_rotation: Matrix3 = (target_R_source @ source_R_world) @ world_R_target
        source_translation: Vector3 = source_R_world @ (translation_target - translation_source)

        error: Vector6 = self._error
        angle_rad: float = _log_so3(error_rotation, error[:3])
        error[3:] = target_R_source @ (source_translation - self._measured_translation)
        residuals[:] = self._sqrt_information @ error

        if jacobians is None:
            return True

        inverse_right_jacobian: Matrix3 = _inverse_right_jacobian(error[:3], angle_rad, self._scratch)
        whitening: Information6 = self._sqrt_information
        translation_jacobian: Float[np.ndarray, "6 3"] = whitening[:, 3:] @ (target_R_source @ source_R_world)

        if jacobians[0] is not None:
            block: Float[np.ndarray, "6 3"] = self._block_jacobian
            block[:3, :] = -(inverse_right_jacobian.T @ target_R_source)
            block[3:, :] = target_R_source @ skew_matrix(source_translation, self._scratch)
            plus_t: Float[np.ndarray, "3 4"] = quaternion_plus_jacobian_transpose(quat_source, self._plus_jacobian_t)
            jacobians[0][:] = ((whitening @ block) @ plus_t).ravel()
        if jacobians[1] is not None:
            jacobians[1][:] = (-translation_jacobian).ravel()
        if jacobians[2] is not None:
            plus_t = quaternion_plus_jacobian_transpose(quat_target, self._plus_jacobian_t)
            jacobians[2][:] = ((whitening[:, :3] @ inverse_right_jacobian) @ plus_t).ravel()
        if jacobians[3] is not None:
            jacobians[3][:] = translation_jacobian.ravel()
        return True


PoseGraphSolveOptions: TypeAlias = CeresSolveOptions
"""Ceres settings for the solve; the shared `colsfm.ceres_pose.CeresSolveOptions`, whose
defaults are this stage's (spec §7 "Solver options") and which the extrinsic refinement
now shares."""


@dataclass(frozen=True, slots=True)
class PoseGraphResult:
    """What one global solve produced."""

    world_T_rig: dict[int, pycolmap.Rigid3d]
    """Optimised rig pose per `rig_id`, including the anchor (unchanged)."""
    anchor_rig_id: int
    """The node held fully constant; both of its parameter blocks were fixed."""
    iterations: int
    """Number of Ceres minimizer iterations."""
    initial_cost: float
    """Ceres' initial cost, i.e. `0.5 * sum(residual**2)` before optimising."""
    final_cost: float
    """Ceres' final cost."""
    termination: CeresTermination
    """How the solve ended, as `colsfm.solver_report` names it."""
    is_solution_usable: bool
    """`Summary::IsSolutionUsable()`; False means the poses were left where Ceres stopped."""
    brief_report: str
    """Ceres' one-line report, kept for logs."""
    edge_residual_norms: Float[np.ndarray, "n_edges"] = field(default_factory=lambda: np.empty(0))
    """Norm of each edge's whitened residual after the solve, in edge input order."""


def solve_pose_graph(
    nodes: Sequence[RigNode],
    edges: Sequence[PoseGraphEdge],
    anchor_rig_id: int | None = None,
    options: PoseGraphSolveOptions | None = None,
) -> PoseGraphResult:
    """Run one global pose-graph solve over the rig poses (spec §6.6, §7, §8).

    Each node contributes two parameter blocks: a unit quaternion (4, `(x, y, z, w)`, on
    `pyceres.EigenQuaternionManifold`) and a translation (3), together holding
    `world_T_rig`. Each edge contributes one 6-residual block with no loss function. The
    anchor node has BOTH blocks set constant — a hard gauge fix, not a soft prior.

    The blob runs this several times mid-stream and accumulates duplicated constraints
    (spec §6.5); a single global solve is deliberate here.

    Note that the blob's own solver overwrites every stored 6x6 with the default
    information matrix unless `relative_constraint_use_pose_covariance` is set, and no
    shipped config sets it. This function always honours the `information` on each edge;
    pass `np.eye(6)` to reproduce what `pose_graph_main` actually solves.

    Args:
        nodes: The rig nodes, giving the initial `world_T_rig` of every node.
        edges: Constraints; every endpoint must name a node in `nodes`.
        anchor_rig_id: Node to hold fixed. Defaults to `edges[0].source`, which is what
            `MultiOptimizePoseGraph` picks.
        options: Ceres settings; the defaults mirror the shipped `optimization_config`.

    Returns:
        The optimised poses, the Ceres summary fields, and per-edge residual norms.
    """
    if len(edges) == 0:
        raise ValueError("pose graph is empty: no edges to optimise")
    solve_options: PoseGraphSolveOptions = options if options is not None else PoseGraphSolveOptions()

    node_by_id: dict[int, RigNode] = {node.rig_id: node for node in nodes}
    if len(node_by_id) != len(nodes):
        raise ValueError("duplicate rig_id in nodes")
    for edge in edges:
        for rig_id in (edge.source, edge.target):
            if rig_id not in node_by_id:
                raise ValueError(f"edge endpoint {rig_id} has no node")
        if edge.source == edge.target:
            raise ValueError(f"edge {edge.source} -> {edge.target} is a self loop")

    anchor: int = edges[0].source if anchor_rig_id is None else anchor_rig_id
    if anchor not in node_by_id:
        raise ValueError(f"anchor rig id {anchor} has no node")

    # The blob sign-flips every quaternion so that `w >= 0`; harmless, kept for parity.
    poses: dict[int, PoseBlocks] = {
        rig_id: pose_parameter_blocks(node.world_T_rig, canonical_sign=True)
        for rig_id, node in node_by_id.items()
    }
    quaternions: dict[int, QuaternionXYZW] = {rig_id: blocks.quat_xyzw for rig_id, blocks in poses.items()}

    problem: pyceres.Problem = pyceres.Problem()
    manifold: pyceres.EigenQuaternionManifold = pyceres.EigenQuaternionManifold()
    for edge in edges:
        cost: RelativePoseCost = RelativePoseCost(edge.source_T_target, edge.information)
        problem.add_residual_block(
            cost,
            None,
            [
                poses[edge.source].quat_xyzw,
                poses[edge.source].translation,
                poses[edge.target].quat_xyzw,
                poses[edge.target].translation,
            ],
        )
    attach_quaternion_manifolds(problem, quaternions, manifold)
    if not problem.has_parameter_block(quaternions[anchor]):
        raise ValueError(f"anchor rig id {anchor} takes part in no edge")
    problem.set_parameter_block_constant(poses[anchor].quat_xyzw)
    problem.set_parameter_block_constant(poses[anchor].translation)

    summary: pyceres.SolverSummary = pyceres.SolverSummary()
    pyceres.solve(solver_options(solve_options), problem, summary)
    if not summary.IsSolutionUsable():
        print(f"[colsfm] pose graph solve failed ({summary.termination_type}): {summary.message}")

    optimised: dict[int, pycolmap.Rigid3d] = {
        rig_id: rigid3d_from_parameter_blocks(blocks.quat_xyzw, blocks.translation)
        for rig_id, blocks in poses.items()
    }

    residual_norms: Float[np.ndarray, "n_edges"] = np.array(
        [
            float(
                np.linalg.norm(
                    _sqrt_information(edge.information)
                    @ relative_pose_error(optimised[edge.source], optimised[edge.target], edge.source_T_target)
                )
            )
            for edge in edges
        ],
        dtype=np.float64,
    )

    solved: SolverStats = solver_stats(summary)
    return PoseGraphResult(
        world_T_rig=optimised,
        anchor_rig_id=anchor,
        iterations=solved.iterations,
        initial_cost=solved.initial_cost,
        final_cost=solved.final_cost,
        termination=solved.termination,
        is_solution_usable=bool(summary.IsSolutionUsable()),
        brief_report=summary.BriefReport(),
        edge_residual_norms=residual_norms,
    )


def apply_rig_update(nodes: Sequence[RigNode], world_T_rig: dict[int, pycolmap.Rigid3d]) -> list[RigNode]:
    """Return the nodes with their poses replaced by the optimised ones.

    The camera-frame derivation (`world_T_cam = world_T_rig * rig_T_cam`) belongs to the
    export stage, not here.

    Args:
        nodes: The original rig nodes.
        world_T_rig: Optimised pose per `rig_id`, e.g. `PoseGraphResult.world_T_rig`.

    Returns:
        New `RigNode`s in the input order, keeping `rig_id` and `timestamp_us`.
    """
    missing: set[int] = {node.rig_id for node in nodes} - set(world_T_rig)
    if missing:
        raise ValueError(f"no optimised pose for rig ids {sorted(missing)}")
    return [RigNode(rig_id=node.rig_id, world_T_rig=world_T_rig[node.rig_id], timestamp_us=node.timestamp_us) for node in nodes]
