"""Construction of the extrinsics-only Ceres problem, and the solve that empties it.

Split out of `colsfm.extrinsic_refinement` because assembling a problem is its own
concern: this module decides which residual blocks exist, which parameter blocks they
share, which of them are held constant and with what weight each is whitened, while
`colsfm.extrinsic_costs` decides what a block evaluates to and the driver decides how
often the whole thing is rebuilt. `solve_extrinsics` is a pure function of a frozen model
— it reads the reconstruction and returns extrinsics, writing nothing back.

The blocks are the extrinsic half of the blob's second mapping pass: one vectorised
reprojection block per free camera, one `absolute_extrinsic` block per free camera
(Eq. 14) and one `relative_extrinsic` block per co-observed camera pair (Eq. 6) carrying
that pair's rig-frame multiplicity.

## Which sigmas, measured

`vision_mapping_config.pb.txt` offers two candidate pairs for the priors. Reproducing the
blob's own final group costs from its exported extrinsics settles it. With
`extrinsic_error_meters = 0.01`, `extrinsic_error_degrees = 2` and `CauchyLoss(1.0)`:

| group | recomputed here | blob's last logged round |
|---|---|---|
| `absolute_extrinsic` | 0.7230 | 0.722949 |
| `relative_extrinsic` | 77.80 | 87.50 |

The absolute figure matches to four significant figures. `relative_pose_*_error_*`
(0.1 m / 5 deg) gives 0.09 and 10.5, eight times too small in both groups. The residual
is `[log(R_err); t_err]` with rotation in slots 0-2, whitened by
`chol(diag(I/sigma_rot_rad^2, I/sigma_trans_m^2))`, exactly the blob's
`DefaultPoseInformationMatrix` (`docs/spec/keypoints_mapper_main.md` §6.4).

Deviations 1, 2 and 4 of the `colsfm.extrinsic_refinement` docstring — the pinned
reference camera, the un-reweighted gauge keyframe and the missing eighth absolute block
— are all decisions taken here.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import pyceres
import pycolmap

from colsfm.ceres_pose import (
    CeresSolveOptions,
    LinearSolver,
    PoseBlocks,
    SolverStats,
    attach_quaternion_manifolds,
    pose_parameter_blocks,
    rigid3d_from_parameter_blocks,
    solver_options,
    solver_stats,
)
from colsfm.extrinsic_costs import RepeatedCauchyLoss, RigReprojectionCost
from colsfm.extrinsic_observations import CameraObservations, CameraPairKey, camera_observations, co_observed_camera_pairs
from colsfm.pose_graph import Information6, RelativePoseCost, default_information
from colsfm.reconstruction import RigReference
from colsfm.solver_report import CeresTermination

EXTRINSIC_LINEAR_SOLVER: Final[LinearSolver] = "SPARSE_NORMAL_CHOLESKY"
"""The extrinsics-only problem has a handful of pose blocks and no Schur structure."""


@dataclass(frozen=True, slots=True)
class ExtrinsicRefinementOptions:
    """Knobs for the alternating refinement; the defaults are the blob's own numbers."""

    regularised: bool = True
    """Run the prior-carrying pyceres refinement. False falls back to the plain pycolmap
    path, `run_mapping(..., optimize_extrinsics=True)`, which is unregularised and on
    Galileo overfits (NOTES.md deviation 9)."""
    num_rounds: int = 20
    """Ceiling on the (A)-then-(B) rounds; the tolerances below normally stop it first.
    Block-coordinate descent converges linearly, and on Galileo the per-round motion decays
    by only ~0.88 per round: 0.47, 0.25, 0.18, 0.15, ... mm, reaching the tolerances after
    **19 rounds** and 3.2 s. Three rounds would stop at a third of the final extrinsic move,
    a worse ATE (4.38 mm against 4.28) and a worse reprojection error (0.956 px against
    0.898). A dataset whose bundle adjustment is expensive — RoboCap's is ~20 s a round —
    wants a smaller ceiling."""
    translation_tolerance_mm: float = 0.1
    """Stop once no extrinsic moved further than this in a round."""
    rotation_tolerance_deg: float = 0.005
    """Stop once no extrinsic rotated further than this in a round."""
    extrinsic_translation_sigma_m: float = 0.01
    """`BundleAdjustmentConfig.extrinsic_error_meters`; one sigma of the extrinsic priors.
    `solve_extrinsics` uses whatever this says, but `refine_extrinsics` OVERRIDES it with
    the mapping configuration's own value: the two must agree, and the config is the
    authority. The default is the shipped isaac number, kept so a standalone
    `solve_extrinsics` call is still the blob's problem."""
    extrinsic_rotation_sigma_deg: float = 2.0
    """`BundleAdjustmentConfig.extrinsic_error_degrees`; one sigma of the extrinsic priors.
    Overridden from the configuration by `refine_extrinsics`, like the field above."""
    reprojection_sigma_px: float = 4.0
    """`BundleAdjustmentConfig.reprojection_error_standard_deviation`; whitens the pixel
    residual. Overridden from the configuration by `refine_extrinsics`."""
    use_absolute_prior: bool = True
    """Add the per-camera `absolute_extrinsic` blocks. Off only for ablations."""
    use_relative_prior: bool = True
    """Add the per-camera-pair `relative_extrinsic` blocks. Off only for ablations."""
    max_num_iterations: int = 200
    """`BundleAdjustmentConfig.max_num_iterations`, applied to the extrinsics-only solve.
    Overridden from the configuration by `refine_extrinsics`."""
    num_threads: int = 1
    """Ceres threads for the extrinsics-only solve. The residuals are Python, so worker
    threads serialise on the GIL; `colsfm.pose_graph` measured 8 threads slower than 1."""
    verbose: bool = True
    """Print one line per round."""


@dataclass(frozen=True, slots=True)
class ExtrinsicSolveStats:
    """What one extrinsics-only pyceres solve contained and how it ended."""

    num_cameras: int
    """Cameras with a parameter block, the pinned reference camera included."""
    num_reprojection_blocks: int
    """One vectorised block per camera that both owns observations and is free."""
    num_reprojection_residuals: int
    """`2 * observations`, i.e. every scalar reprojection residual in the problem."""
    num_absolute_prior_blocks: int
    """`absolute_extrinsic` blocks: one per FREE camera, so seven where the blob logs eight
    (its eighth is the pinned camera's, which is constant and contributes nothing)."""
    num_relative_prior_blocks: int
    """Distinct co-observed camera pairs, each carrying its own multiplicity."""
    num_relative_prior_frames: int
    """Sum of those multiplicities: the blob's own block count, 777 on Galileo."""
    iterations: int
    """Ceres minimizer iterations, successful and unsuccessful."""
    initial_cost: float
    """Ceres cost before the solve."""
    final_cost: float
    """Ceres cost after the solve."""
    termination: CeresTermination
    """How the solve ended."""
    seconds: float
    """Wall-clock seconds, problem construction included."""


def _add_reprojection_blocks(
    problem: pyceres.Problem,
    poses: Mapping[int, PoseBlocks],
    observations: Mapping[int, CameraObservations],
    reference_camera_params_id: int,
    reprojection_sigma_px: float,
    costs: list[pyceres.CostFunction],
) -> tuple[int, int]:
    """Add one vectorised reprojection block per free camera.

    Args:
        problem: The problem being built.
        poses: The parameter arrays per `camera_params_id`.
        observations: That round's frozen points and pixels per camera.
        reference_camera_params_id: The pinned rig origin, which gets no block: its
            extrinsic never moves, so its residual would be constant and Ceres would
            drop it anyway.
        reprojection_sigma_px: Whitening sigma, in pixels.
        costs: Kept-alive list every cost is appended to; pyceres holds raw pointers
            into it and a dropped reference is a segfault, not an error.

    Returns:
        The number of blocks added and the number of scalar residuals in them.
    """
    num_blocks: int = 0
    num_residuals: int = 0
    for camera_params_id, camera_obs in observations.items():
        if camera_params_id == reference_camera_params_id:
            continue
        blocks: PoseBlocks = poses[camera_params_id]
        cost: RigReprojectionCost = RigReprojectionCost(camera_obs, reprojection_sigma_px)
        costs.append(cost)
        problem.add_residual_block(cost, None, [blocks.quat_xyzw, blocks.translation])
        num_blocks += 1
        num_residuals += 2 * int(camera_obs.points_in_vehicle.shape[0])
    return num_blocks, num_residuals


def _add_absolute_priors(
    problem: pyceres.Problem,
    poses: Mapping[int, PoseBlocks],
    calibration_vehicle_T_cam: Mapping[int, pycolmap.Rigid3d],
    reference_camera_params_id: int,
    information: Information6,
    identity: PoseBlocks,
    costs: list[pyceres.CostFunction],
    losses: list[pyceres.LossFunction],
) -> int:
    """Add the blob's `absolute_extrinsic` block for every free camera (Eq. 14).

    `RelativePoseCost` against a constant identity source IS the absolute prior: its
    error is `measurement^-1 * (I^-1 * vehicle_T_cam)`.

    Args:
        problem: The problem being built.
        poses: The parameter arrays per `camera_params_id`.
        calibration_vehicle_T_cam: The calibration each prior pulls towards.
        reference_camera_params_id: The pinned rig origin. It would give a block whose
            every parameter is constant; the blob counts such a block (8 on Galileo,
            one inert) but Ceres' `Program::RemoveFixedBlocks` evaluates it with a null
            residual pointer, which segfaults the pyceres 2.6 trampoline. It
            contributes nothing, so it is left out.
        information: The 6x6 weight both prior groups share.
        identity: The constant identity source, owned by the caller so that it outlives
            the problem; set constant here once a block references it.
        costs: Kept-alive list for the cost functions.
        losses: Kept-alive list for the loss functions.

    Returns:
        The number of blocks added.
    """
    num_blocks: int = 0
    for camera_params_id in sorted(poses):
        if camera_params_id == reference_camera_params_id:
            continue
        blocks: PoseBlocks = poses[camera_params_id]
        cost: RelativePoseCost = RelativePoseCost(calibration_vehicle_T_cam[camera_params_id], information)
        loss: pyceres.CauchyLoss = pyceres.CauchyLoss(1.0)
        costs.append(cost)
        losses.append(loss)
        problem.add_residual_block(
            cost, loss, [identity.quat_xyzw, identity.translation, blocks.quat_xyzw, blocks.translation]
        )
        num_blocks += 1
    if num_blocks:
        problem.set_parameter_block_constant(identity.quat_xyzw)
        problem.set_parameter_block_constant(identity.translation)
    return num_blocks


def _add_relative_priors(
    problem: pyceres.Problem,
    poses: Mapping[int, PoseBlocks],
    calibration_vehicle_T_cam: Mapping[int, pycolmap.Rigid3d],
    pair_counts: Mapping[CameraPairKey, int],
    information: Information6,
    costs: list[pyceres.CostFunction],
    losses: list[pyceres.LossFunction],
) -> tuple[int, int]:
    """Add one `relative_extrinsic` block per co-observed camera pair (Eq. 6).

    Each block carries its pair's rig-frame multiplicity through
    `RepeatedCauchyLoss`, which is exactly equivalent to the blob's repeated blocks
    (see the module docstring).

    Args:
        problem: The problem being built.
        poses: The parameter arrays per `camera_params_id`.
        calibration_vehicle_T_cam: The calibration the relative measurement comes from.
        pair_counts: Rig frames per co-observed camera pair.
        information: The 6x6 weight both prior groups share.
        costs: Kept-alive list for the cost functions.
        losses: Kept-alive list for the loss functions.

    Returns:
        The number of blocks added, and the number of blob blocks they stand for.
    """
    num_blocks: int = 0
    num_frames: int = 0
    for (lower, higher), count in pair_counts.items():
        if lower not in poses or higher not in poses:
            continue
        # `higher_T_lower`, the calibrated relative pose the blob preserves (Eq. 6).
        measured: pycolmap.Rigid3d = calibration_vehicle_T_cam[higher].inverse() * calibration_vehicle_T_cam[lower]
        cost: RelativePoseCost = RelativePoseCost(measured, information)
        loss: RepeatedCauchyLoss = RepeatedCauchyLoss(count)
        costs.append(cost)
        losses.append(loss)
        problem.add_residual_block(
            cost,
            loss,
            [
                poses[higher].quat_xyzw,
                poses[higher].translation,
                poses[lower].quat_xyzw,
                poses[lower].translation,
            ],
        )
        num_blocks += 1
        num_frames += count
    return num_blocks, num_frames


def solve_extrinsics(
    reconstruction: pycolmap.Reconstruction,
    rig_reference: RigReference,
    calibration_vehicle_T_cam: Mapping[int, pycolmap.Rigid3d],
    options: ExtrinsicRefinementOptions | None = None,
    observations: Mapping[int, CameraObservations] | None = None,
    pair_counts: Mapping[CameraPairKey, int] | None = None,
) -> tuple[dict[int, pycolmap.Rigid3d], ExtrinsicSolveStats]:
    """Refine every free `vehicle_T_cam` with rig poses and points held fixed.

    The problem is the extrinsic half of the blob's second mapping pass: one vectorised
    reprojection block per free camera, one `absolute_extrinsic` block per camera and one
    `relative_extrinsic` block per co-observed camera pair carrying that pair's rig-frame
    multiplicity. The reference camera's two blocks are constant, because it is the rig
    origin.

    Args:
        reconstruction: A posed, triangulated model on a camera-referenced rig; read only.
        rig_reference: How that rig was built.
        calibration_vehicle_T_cam: `vehicle_T_cam` per `camera_params_id` as the input
            calibration declared it — the mean of both priors.
        options: Sigmas and solver settings; the defaults when None. `refine_extrinsics`
            derives the sigmas from the mapping configuration before calling this.
        observations: This round's folded-in observations, from `camera_observations`;
            gathered here when None. The alternation passes them so the observation
            index behind them is built once rather than once per round.
        pair_counts: Co-observed camera pairs, from `co_observed_camera_pairs`; computed
            here when None. They never change during the alternation.

    Returns:
        The refined `vehicle_T_cam` per `camera_params_id`, and what the solve contained.

    Raises:
        ValueError: When the rig is not camera-referenced, or when a camera in the
            reconstruction has no calibration to be primed from.
    """
    resolved: ExtrinsicRefinementOptions = ExtrinsicRefinementOptions() if options is None else options
    reference_camera_params_id: int | None = rig_reference.camera_params_id
    if reference_camera_params_id is None:
        raise ValueError(
            "extrinsic refinement needs a camera-referenced rig: the vehicle-body reference "
            "owns no images, so COLMAP freezes every extrinsic (NOTES.md gotcha 13)"
        )
    started: float = time.perf_counter()

    current: dict[int, pycolmap.Rigid3d] = rig_reference.vehicle_T_cam_by_camera_params_id(reconstruction)
    missing: set[int] = set(current) - set(calibration_vehicle_T_cam)
    if missing:
        raise ValueError(f"no calibration extrinsic for camera_params_id {sorted(missing)}")

    poses: dict[int, PoseBlocks] = {
        camera_params_id: pose_parameter_blocks(vehicle_T_cam)
        for camera_params_id, vehicle_T_cam in current.items()
    }
    information: Information6 = default_information(
        relative_rotation_sigma_deg=resolved.extrinsic_rotation_sigma_deg,
        relative_translation_sigma_m=resolved.extrinsic_translation_sigma_m,
    )
    problem: pyceres.Problem = pyceres.Problem()
    manifold: pyceres.EigenQuaternionManifold = pyceres.EigenQuaternionManifold()
    # Kept alive for the lifetime of the problem: pyceres holds raw pointers into these.
    costs: list[pyceres.CostFunction] = []
    losses: list[pyceres.LossFunction] = []
    identity: PoseBlocks = pose_parameter_blocks(pycolmap.Rigid3d())

    resolved_observations: Mapping[int, CameraObservations] = (
        camera_observations(reconstruction, rig_reference) if observations is None else observations
    )
    num_reprojection_blocks, num_reprojection_residuals = _add_reprojection_blocks(
        problem,
        poses,
        resolved_observations,
        reference_camera_params_id,
        resolved.reprojection_sigma_px,
        costs,
    )
    num_absolute_blocks: int = 0
    if resolved.use_absolute_prior:
        num_absolute_blocks = _add_absolute_priors(
            problem, poses, calibration_vehicle_T_cam, reference_camera_params_id, information, identity, costs, losses
        )
    resolved_pairs: Mapping[CameraPairKey, int] = {}
    if resolved.use_relative_prior:
        resolved_pairs = co_observed_camera_pairs(reconstruction) if pair_counts is None else pair_counts
    num_relative_blocks, num_relative_frames = _add_relative_priors(
        problem, poses, calibration_vehicle_T_cam, resolved_pairs, information, costs, losses
    )

    attach_quaternion_manifolds(
        problem, {camera_params_id: blocks.quat_xyzw for camera_params_id, blocks in poses.items()}, manifold
    )
    reference_blocks: PoseBlocks = poses[reference_camera_params_id]
    if problem.has_parameter_block(reference_blocks.quat_xyzw):
        problem.set_parameter_block_constant(reference_blocks.quat_xyzw)
        problem.set_parameter_block_constant(reference_blocks.translation)

    summary: pyceres.SolverSummary = pyceres.SolverSummary()
    pyceres.solve(
        solver_options(
            CeresSolveOptions(
                linear_solver=EXTRINSIC_LINEAR_SOLVER,
                max_num_iterations=resolved.max_num_iterations,
                num_threads=resolved.num_threads,
            )
        ),
        problem,
        summary,
    )

    refined: dict[int, pycolmap.Rigid3d] = {
        camera_params_id: rigid3d_from_parameter_blocks(blocks.quat_xyzw, blocks.translation)
        for camera_params_id, blocks in poses.items()
    }
    solved: SolverStats = solver_stats(summary)
    stats: ExtrinsicSolveStats = ExtrinsicSolveStats(
        num_cameras=len(poses),
        num_reprojection_blocks=num_reprojection_blocks,
        num_reprojection_residuals=num_reprojection_residuals,
        num_absolute_prior_blocks=num_absolute_blocks,
        num_relative_prior_blocks=num_relative_blocks,
        num_relative_prior_frames=num_relative_frames,
        iterations=solved.iterations,
        initial_cost=solved.initial_cost,
        final_cost=solved.final_cost,
        termination=solved.termination,
        seconds=time.perf_counter() - started,
    )
    return refined, stats
