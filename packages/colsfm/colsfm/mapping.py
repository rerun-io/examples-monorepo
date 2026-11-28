"""Triangulation and global bundle adjustment: the `keypoints_mapper_main` replacement.

Implements `docs/spec/keypoints_mapper_main.md` §5.3-§5.6 (the triangulate /
merge / complete / filter / BA outer loop) and §6 (the bundle adjustment) on
pycolmap's `IncrementalTriangulator`, `ObservationManager` and
`create_default_bundle_adjuster`, against a reconstruction built by
`colsfm.reconstruction.build_reconstruction` and a COLMAP database written by
the feature and matching stages.

```
frames_meta.json ──> build_reconstruction ──┐
                                            ├──> run_mapping ──> MappingResult
database.db (keypoints + two-view geoms) ───┘
```

Poses are a **prior, not an initialisation**: every rig frame is registered
before the first triangulation, and only bundle adjustment moves them.

## What is deliberately not reproduced

Two cuSFM behaviours are known defects and are left out (§9.5):

1. **Unweighted constant-frame residuals.** cuSFM whitens every reprojection
   residual by `1/sigma` except the ones observing the gauge keyframe, whose
   functor has no weight member — 13 of 6276 blocks in Galileo. COLMAP applies
   one weighting uniformly and there is no way to ask for the inconsistency.
2. **The 3-D depth residual** (`VehicleCameraReprojectionCost3D`) for keypoints
   carrying depth. pycolmap's bundle adjuster is 2-D only, and Galileo sets
   `keypoint_feature_has_depth: false`.

## Extrinsic refinement is real but unregularised

`optimize_extrinsics` needs a **camera-referenced** rig; on the default
vehicle-referenced one COLMAP freezes every `sensor_from_rig` and the refinement
is a silent no-op, so `run_mapping` raises instead (see `colsfm.reconstruction`).
Given one, it does refine: a 20 mm perturbation of a Galileo camera comes back to
0.70 mm.

What it cannot reproduce is cuSFM's **regularisation**. The blob's refinement
pass adds absolute-extrinsic priors (the paper's Eq. 14, one block per camera)
and relative-extrinsic constraints between cameras (Eq. 6, 777 blocks on
Galileo), both with sigmas from `vision_mapping_config.pb.txt` (§7).
`pycolmap.BundleAdjustmentOptions` exposes neither, and the Ceres problem
`create_default_bundle_adjuster` builds cannot be extended from standalone
pyceres (the pybind11 registry split, pycolmap-capabilities.md §8). colsfm
therefore optimises reprojection alone, and on Galileo that trades trajectory
accuracy for pixels: 0.861 px against 1.334 px, but 5.70 mm ATE against 4.33 mm,
with extrinsics moving up to 234 mm where the blob moves 11 mm. Only a stereo
pair's own relative extrinsic is well determined by a 0.66 m sweep. NOTES.md
deviation 9 carries the table.

So this path is the *ablation*, reached by `--no-regularised-extrinsics`. A default
run refines its extrinsics too — `--optimize-extrinsics` is on — but through
`colsfm.extrinsic_refinement`, which carries the priors and calls this module only
with `optimize_extrinsics` False.

## Overrides forced on pycolmap, and why

| Setting | pycolmap default | Here | Reason |
|---|---|---|---|
| `IncrementalTriangulatorOptions.min_angle` | 1.5 | `min_triangulation_deg` (2.0) | config §3.1 |
| `merge_max_reproj_error`, `complete_max_reproj_error` | 4.0 | the round's gate `g_k` | cuSFM merges and completes under the same decaying gate (§5.4, §5.6) |
| `ignore_two_view_tracks` | True | `min_correspondences > 2` | `min_correspondences: 3` drops two-view tracks |
| `random_seed` | -1 | `random_seed` (1) | determinism |
| `refine_focal_length` | True | `MappingOptions.refine_focal_length` (False) | `--optimize_intrinsics=false`; `--refine-camera-model` frees it |
| `refine_extra_params` | True | `refine_focal_length` (False) | same; CASPAR takes the focal only with the distortion block, so both backends free them together |
| `refine_principal_point` | False | False | same; already off |
| `refine_sensor_from_rig` | True | `MappingOptions.optimize_extrinsics` (False) | `--optimize_extrinsics=false`; needs a camera-referenced rig, see below |
| `ceres.loss_function_type` | TRIVIAL | `loss_type` (CAUCHY) | config §3.2 |
| `ceres.loss_function_scale` | 1.0 | `sigma * loss_function_scale` (4.0) | see below |
| `ceres.auto_select_solver_type` | True | False | otherwise the explicit solver choice is ignored |
| `solver_options.linear_solver_type` | SPARSE_NORMAL_CHOLESKY | `MAPPING_LINEAR_SOLVER` (SPARSE_SCHUR) | cuSFM: SPARSE_SCHUR on CPU, SPARSE_NORMAL_CHOLESKY on the cuDSS path (§6.5) |
| `solver_options.max_num_iterations` | 100 | 200 | config §3.2 |
| `solver_options.max_linear_solver_iterations` | 200 | 100 | config §3.2 |
| `solver_options.num_threads` | -1 | `num_threads` (8) | config §3.2 |
| `solver_options.function_tolerance` | **0.0** | 1e-6 | cuSFM leaves Ceres' defaults; pycolmap overrides them |
| `solver_options.gradient_tolerance` | **1e-4** | 1e-10 | same |
| `solver_options.parameter_tolerance` | **0.0** | 1e-8 | same |

**The loss scale carries the whitening.** cuSFM divides the 2-vector
reprojection residual by `reprojection_error_standard_deviation = 4.0 px` and
then applies `CauchyLoss(1.0)`. COLMAP does not whiten. `CauchyLoss(a)` is
`a^2 log(1 + s/a^2)` in the squared residual `s`, so evaluating it at `s/sigma^2`
with `a = 1` equals evaluating it at `s` with `a = sigma`, up to the constant
factor `sigma^2` that cannot move the minimum. The port therefore sets
`loss_function_scale = sigma * loss_function_scale = 4.0` (§9.2).

## What §9.3 promises that pycolmap cannot deliver

`IncrementalTriangulatorOptions` has **no** RANSAC block and no residual-type
switch: `ransac.max_error`, `confidence`, `max_num_trials`, `min_inlier_ratio`
and `TriangulationResidualType.REPROJECTION_ERROR` live on
`EstimateTriangulationOptions`, which only the standalone
`pycolmap.estimate_triangulation` takes. Driving the triangulator therefore
means accepting COLMAP's own angular create/continue thresholds (2.0 deg) and
enforcing the pixel gate `g_k` where it is reachable: on merging, on completion
and on the per-round filter. Deriving the create threshold from the gate through
the focal length (`atan(g_k / f)`) was tried and moves nothing: 2035 points
against 2035, 1.6706 px against 1.6706.

## Measured against the blob, on the blob's own matches

| | Galileo 32 keyframes | Galileo 226 keyframes |
|---|---|---|
| images with observations | 28 vs 28 | 224 vs 224 |
| points | 2035 vs 1275 | 14 496 vs 5065 |
| mean reprojection | 1.671 vs 1.699 px | 1.415 vs 1.550 px |
| worst rig-pose difference | 1.26 mm / 0.118 deg | 2.55 mm / 0.092 deg |
| the blob's own pose correction | 11.27 mm / 0.569 deg | 12.66 mm / 0.364 deg |
| runtime | 0.9 vs 1.1 s | 8.0 vs 6.7 s |

The point count is the one figure that does not match. cuSFM grows **one**
disjoint track per connected component of the match graph and triangulates it
with plain MSAC — no local optimisation, no final refit — so it rejects roughly
half its candidate tracks. COLMAP's LO-RANSAC keeps them. The result is a
superset at a lower reprojection error: 98-99 % of the blob's points have one of
ours within 5 cm, and merging is already at a fixed point (a second
`merge_all_tracks` pass returns 0), so the difference is structural rather than
a missing merge.

## `ba_backend = "caspar"`: COLMAP's GPU bundle adjustment

`MappingOptions.ba_plan` picks the solver behind the same
`create_default_bundle_adjuster` call. `caspar` needs a `CASPAR_ENABLED` build,
which the `colsfm` environment has (recipes/colmap-src); the
stock environment answers the capability probe with `unavailable`, so the plan
already says `ceres` and says why, and the flag is safe to pass anywhere. Nothing
in this module decides a backend: `colsfm.ba_backend` owns that, and the one
decision it cannot make up front -- which camera models this run holds -- is made
once, at the top of `run_mapping`, by `BaExecutionPlan.for_camera_models`.

**CASPAR drops the robust loss.** It solves plain least squares whatever
`ceres.loss_function_type` says, so the CAUCHY term at sigma 4 that whitens
cuSFM's residual (§9.2, the table above) is simply not applied: every outlier
enters the normal equations at full weight. What still rejects them is the outer
loop — the per-round gate `g_k` filters at 25, 20, 15, 10, 5 px *before* each
solve, and `filter_degenerate_points` / `filter_projection_failures` run after it.
That is enough: an ablation on KITTI 06 — Ceres with `loss_type: TRIVIAL` — scores
0.899 m Sim(3) ATE against the robust run's 0.895 m, so the loss is not what
CASPAR's 1.299 m cost (NOTES.md "CASPAR backend"). Its PCG was: a beta recurrence bug and a full-system solve
stopped it short of the minimum, which recipes/colmap-src fixes (`patches/PROVENANCE.md`).

Three more constraints, all from COLMAP 4.2.0's source and all handled here:
observations from a camera model the build has no adapter for are *silently
skipped*, so `BaExecutionPlan.for_camera_models` falls back to Ceres rather than
solve a quietly smaller problem. Which models those are is a property of the build, not
of CASPAR — stock ships PINHOLE and SIMPLE_RADIAL, the `colsfm` environment (recipes/colmap-src) adds
OPENCV_FISHEYE (`docs/caspar-fisheye-adapter.md`) — and no pycolmap call lists
them, so `caspar_supported_camera_models` measures them with one tiny solve per
model. `refine_sensor_from_rig = True` is a hard throw on a
multi-sensor frame of a stock build, so `optimize_extrinsics` falls back too unless the
build's probe refined a rig (`CasparCapability.refines_sensor_from_rig`; the regularised
`colsfm.extrinsic_refinement` path keeps CASPAR either way: it holds the extrinsics fixed
inside the pycolmap solve by construction); and `refine_focal_length` must equal
`refine_extra_params`, which it does on both backends.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pycolmap
from jaxtyping import UInt32
from numpy import ndarray

from colsfm.ba_backend import (
    BUNDLE_ADJUSTMENT_BACKEND,
    CERES_ONLY_PLAN,
    EXTRINSICS_FALLBACK_REASON,
    BaBackend,
    BaExecutionPlan,
    CasparOptions,
    PreparedSolve,
    apply_caspar_options,
)
from colsfm.ceres_pose import COLMAP_LINEAR_SOLVER, LinearSolver
from colsfm.config import BundleAdjustmentConfig, LossFunctionType, VisionMappingConfig
from colsfm.correspondences import Correspondences, load_correspondences, registered_image_ids
from colsfm.mapping_result import MappingResult, PolishStats, RoundCallback, RoundStats
from colsfm.point_filters import (
    _filter_points,
    _FilterCounts,
    filter_degenerate_points,
    filter_past_rational_pole,
    filter_projection_failures,
    projection_sanity_bound_px,
)
from colsfm.reconstruction import (
    RIG_ID,
    PosedModel,
    camera_sensor_id,
)
from colsfm.solver_report import SolverFailure, SolverReport, ceres_termination, parse_brief_report

__all__ = [
    "BundleAdjustmentSolve",
    "BaBackend",
    "CasparOptions",
    "Correspondences",
    "MappingOptions",
    "MappingResult",
    "PolishStats",
    "RoundCallback",
    "RoundStats",
    "bundle_adjustment_options",
    "bundle_adjustment_solve",
    "ceres_polish_options",
    "default_gauge_frame_id",
    "filter_degenerate_points",
    "filter_projection_failures",
    "global_ba_config",
    "load_correspondences",
    "pixel_error_schedule",
    "projection_sanity_bound_px",
    "refinement_options",
    "registered_image_ids",
    "run_mapping",
    "select_backend",
    "solve_bundle_adjustment",
    "triangulator_options",
]
"""The mapper's public surface.

`colsfm.mapping` stays the one import path callers use, whichever of its four
modules a name is defined in: the split is about giving each file one reason to
change, not about making every caller learn the new layout. `colsfm.mapping` is
the mapping stage; `colsfm.correspondences`, `colsfm.point_filters` and
`colsfm.mapping_result` are the pieces it is made of."""

MAPPING_LINEAR_SOLVER: Final[LinearSolver] = "SPARSE_SCHUR"
"""cuSFM's CPU linear solver, and the only one this stage ever wants: a bundle adjustment
is exactly the problem the Schur complement exists for. The cuDSS path uses
SPARSE_NORMAL_CHOLESKY instead (§6.5), which colsfm does not build."""

CERES_FUNCTION_TOLERANCE: Final[float] = 1e-6
"""Ceres' own default, which cuSFM keeps and pycolmap overrides to 0.0."""

CERES_GRADIENT_TOLERANCE: Final[float] = 1e-10
"""Ceres' own default, which cuSFM keeps and pycolmap overrides to 1e-4."""

CERES_PARAMETER_TOLERANCE: Final[float] = 1e-8
"""Ceres' own default, which cuSFM keeps and pycolmap overrides to 0.0."""

LOSS_FUNCTION_BY_NAME: Final[dict[LossFunctionType, pycolmap.LossFunctionType]] = {
    "TRIVIAL": pycolmap.LossFunctionType.TRIVIAL,
    "SOFT_L1": pycolmap.LossFunctionType.SOFT_L1,
    "CAUCHY": pycolmap.LossFunctionType.CAUCHY,
    "HUBER": pycolmap.LossFunctionType.HUBER,
}
"""cuSFM's `loss_type` enum to pycolmap's; the two sets coincide exactly."""


@dataclass(frozen=True, slots=True)
class MappingOptions:
    """Knobs the mapper takes from the command line rather than from a config file."""

    optimize_extrinsics: bool = False
    """Set `refine_sensor_from_rig` on this mapper's bundle adjustments; off by default.

    Not `PipelineOptions.optimize_extrinsics`, which is on: a default run refines its
    extrinsics in `colsfm.extrinsic_refinement`, whose every call here leaves this
    False. Only `--no-regularised-extrinsics` turns it on, and that is the one thing
    that takes `ba_backend="caspar"` off the GPU."""
    refine_focal_length: bool = False
    """Free every camera's focal length (fx, fy) and distortion block in every round's bundle adjustment; cuSFM holds
    intrinsics fixed. The principal point stays fixed.

    A nominal calibration that is off pushes its error into the poses (Mission office cube faces: 1-1.4 % off
    f = 800, poses 0.25 deg from Spirula's); refined from the first round, the loose early gates keep the
    observations the nominal model would have filtered."""
    fixed_camera_params_id: int | None = None
    """Camera whose extrinsic stays fixed while the others are refined; cuSFM's `--fixed_camera_name`."""
    num_threads: int | None = None
    """Ceres threads; the config's `num_threads` when None. Use 1 for a reproducible solve.
    Triangulation is single-threaded in COLMAP either way, so it does not carry cuSFM's
    threaded-triangulation non-determinism (§5.3)."""
    use_gpu: bool = False
    """Hand the Ceres solve to the GPU; COLMAP still needs 50+ images before it switches
    (`min_num_images_gpu_solver`, pycolmap-capabilities.md §11). Off by default because the
    measured gain does not exist: 1.68 s against 1.60 s on Galileo's 226 images, and 19.2 s
    against 20.9 s on 800 RoboCap images — 5 % the wrong way, then 8 % the right way. The
    blob's own mapper is Ceres on the CPU, so that is where the default stays. Which GPU
    is left to Ceres, i.e. pycolmap's `gpu_index` default of `-1`."""
    ba_plan: BaExecutionPlan = CERES_ONLY_PLAN
    """How the bundle adjustments will be solved, as `colsfm.ba_backend` decided.

    One object, not four knobs. It carries the requested backend, the one that will
    actually run, why they differ, the validated `--caspar-option` overrides and
    whether a Ceres polish closes the run — and it is *the* decision: nothing here
    re-derives any part of it. `colsfm.run_config.resolve_run` makes it before the
    run creates anything, and `run_mapping` narrows it once with the camera models
    the reconstruction turned out to hold (`BaExecutionPlan.for_camera_models`).

    The default is Ceres with no polish, which is what a caller who says nothing
    about bundle adjustment means. See the module docstring for the robust loss
    CASPAR does not apply."""
    verbose: bool = True
    """Print one line per triangulation pass and per outer round, as cuSFM does."""
    inlier_overrides: Mapping[tuple[int, int], UInt32[ndarray, "n_inliers 2"]] = MappingProxyType({})
    """Per image pair `(min id, max id)`, the subset of its stored inliers the correspondence graph carries: the loop
    stage's pose check under `loop_pairs_in_mapper="pose-checked"` (`colsfm.loop_mapper_check`). No row is rewritten."""
    round_callback: RoundCallback | None = None
    """Called once at the end of every outer round, after the round's filter and solve.

    The hook a live viewer needs: it receives the round's `RoundStats` and the
    `pycolmap.Reconstruction` in exactly the state that round left it, so a caller
    can log the cloud and the poses per round without re-running the mapper or
    scraping `verbose` output. It runs *before* the early-exit check, so the last
    round is always reported. Nothing in the mapper reads what it returns, and a
    callback that raises aborts the mapping — it is a plain call, not a try/except."""


def pixel_error_schedule(mapping_config: VisionMappingConfig) -> tuple[float, ...]:
    """The linear per-round reprojection gate, `initial` to `final` (§5.6).

    `g_k = initial + (final - initial) * k / (N - 1)`, which is 25, 20, 15, 10, 5
    for the isaac profile and 40, 32, 24, 16, 8 for AV. A single-round
    configuration stays at `initial`.

    Args:
        mapping_config: The mapping configuration.

    Returns:
        One gate per outer round, in pixels.
    """
    num_rounds: int = max(mapping_config.num_ba_iterations, 1)
    if num_rounds == 1:
        return (mapping_config.initial_max_pixel_error,)
    span: float = mapping_config.final_max_pixel_error - mapping_config.initial_max_pixel_error
    return tuple(mapping_config.initial_max_pixel_error + span * index / (num_rounds - 1) for index in range(num_rounds))


def triangulator_options(mapping_config: VisionMappingConfig, max_pixel_error: float) -> pycolmap.IncrementalTriangulatorOptions:
    """Triangulator options for one round's gate.

    Args:
        mapping_config: The mapping configuration.
        max_pixel_error: The round's reprojection gate, in pixels.

    Returns:
        Options with the module docstring's overrides applied; tracks merge under
        `min(max_pixel_error, mapping_config.max_merge_pixel_error)` when the profile caps it.
    """
    options: pycolmap.IncrementalTriangulatorOptions = pycolmap.IncrementalTriangulatorOptions()
    options.min_angle = mapping_config.min_triangulation_deg
    merge_cap: float | None = mapping_config.max_merge_pixel_error
    options.merge_max_reproj_error = max_pixel_error if merge_cap is None else min(max_pixel_error, merge_cap)
    options.complete_max_reproj_error = max_pixel_error
    options.ignore_two_view_tracks = mapping_config.min_correspondences > 2
    options.random_seed = mapping_config.random_seed
    return options


def bundle_adjustment_options(ba_config: BundleAdjustmentConfig, options: MappingOptions) -> pycolmap.BundleAdjustmentOptions:
    """Bundle adjustment options mirroring cuSFM's `BundleAdjustmentConfig` (§9.2).

    Nothing is decided here: `options.ba_plan.backend` is the backend, already
    resolved and already narrowed by the caller. This function used to re-run the
    whole fallback policy — and print it — every time a solve needed an options
    object.

    Args:
        ba_config: The Ceres settings from `vision_mapping_config.pb.txt`.
        options: Command-line style knobs, carrying the plan to execute.

    Returns:
        Options with every override of the module docstring applied, on the plan's
        backend.

    Raises:
        KeyError: When the configuration names a loss pycolmap does not have.
        ValueError: When a CASPAR plan without the rig capability meets `optimize_extrinsics`.
    """
    ba_options: pycolmap.BundleAdjustmentOptions = refinement_options(options.refine_focal_length, options.optimize_extrinsics)
    num_threads: int = ba_config.num_threads if options.num_threads is None else options.num_threads
    ba_options.ceres.loss_function_type = LOSS_FUNCTION_BY_NAME[ba_config.loss_type]
    ba_options.ceres.loss_function_scale = ba_config.reprojection_error_standard_deviation * ba_config.loss_function_scale
    ba_options.ceres.use_gpu = options.use_gpu
    ba_options.ceres.auto_select_solver_type = False
    solver_options = ba_options.ceres.solver_options
    solver_options.linear_solver_type = COLMAP_LINEAR_SOLVER[MAPPING_LINEAR_SOLVER]
    solver_options.num_threads = num_threads
    solver_options.max_num_iterations = ba_config.max_num_iterations
    solver_options.max_num_consecutive_invalid_steps = ba_config.max_invalid_steps
    solver_options.max_linear_solver_iterations = ba_config.max_linear_solver_iterations
    solver_options.function_tolerance = CERES_FUNCTION_TOLERANCE
    solver_options.gradient_tolerance = CERES_GRADIENT_TOLERANCE
    solver_options.parameter_tolerance = CERES_PARAMETER_TOLERANCE
    solver_options.minimizer_progress_to_stdout = False

    plan: BaExecutionPlan = options.ba_plan
    # The plan keeps a solve that frees `sensor_from_rig` on the GPU only when the probe refined a rig
    # (`resolve_backend`); a plan that breaks that would hold a requested rig refinement fixed without a word.
    if plan.backend != "ceres" and options.optimize_extrinsics and not plan.capability.refines_sensor_from_rig:
        raise ValueError(f"a {plan.backend} plan cannot free sensor_from_rig on this build: {EXTRINSICS_FALLBACK_REASON}")
    select_backend(ba_options, plan.backend, use_gpu=options.use_gpu, caspar_options=plan.caspar_options)
    return ba_options


def bundle_adjustment_solve(ba_config: BundleAdjustmentConfig, options: MappingOptions) -> PreparedSolve:
    """The rounds' solve: `bundle_adjustment_options` on the plan's backend, which `mojo` needs said.

    Args:
        ba_config: The Ceres settings from `vision_mapping_config.pb.txt`.
        options: Command-line style knobs, carrying the plan to execute.

    Returns:
        The prepared solve.
    """
    return PreparedSolve(backend=options.ba_plan.backend, options=bundle_adjustment_options(ba_config, options))


def refinement_options(refine_intrinsics: bool, refine_sensor_from_rig: bool) -> pycolmap.BundleAdjustmentOptions:
    """Which parameter blocks every colsfm bundle adjustment frees, on pycolmap's defaults otherwise.

    Shared with the standalone benchmarks (`colsfm.apis.caspar_baseline`), so they time the solve the mapper runs.

    Args:
        refine_intrinsics: Free the focal length and the distortion block together.
        refine_sensor_from_rig: Free the non-reference `sensor_from_rig`.

    Returns:
        Options on pycolmap's default (Ceres) backend.
    """
    ba_options: pycolmap.BundleAdjustmentOptions = pycolmap.BundleAdjustmentOptions()
    ba_options.refine_focal_length = refine_intrinsics
    ba_options.refine_principal_point = False
    # CASPAR takes the focal only together with the distortion block; Ceres follows, so both backends refine one model.
    ba_options.refine_extra_params = refine_intrinsics
    ba_options.refine_rig_from_world = True
    ba_options.refine_sensor_from_rig = refine_sensor_from_rig
    ba_options.refine_points3D = True
    # pycolmap defaults this to True, which prints a full Ceres report per round.
    ba_options.print_summary = False
    return ba_options


def select_backend(ba_options: pycolmap.BundleAdjustmentOptions, backend: BaBackend, *, use_gpu: bool, caspar_options: CasparOptions) -> None:
    """Point the options at a backend and, on a GPU backend, set the CASPAR block both GPU solvers read.

    Args:
        ba_options: Modified in place.
        backend: The solver.
        use_gpu: GPU 0, or COLMAP's own pick.
        caspar_options: Validated `--caspar-option` overrides.
    """
    ba_options.backend = BUNDLE_ADJUSTMENT_BACKEND[backend]
    if backend == "ceres":
        return
    # `-1` is COLMAP's own pick, which is what `use_gpu` off means everywhere else
    # in this dataclass. Both were measured to solve on the 5090.
    ba_options.caspar.gpu_index = "0" if use_gpu else "-1"
    # The reduced camera system (recipes/colmap-src) is off in COLMAP's defaults; `--caspar-option` may turn it off.
    ba_options.caspar.reduced_camera_system = True
    apply_caspar_options(ba_options.caspar, caspar_options)


def ceres_polish_options(ba_config: BundleAdjustmentConfig, options: MappingOptions) -> pycolmap.BundleAdjustmentOptions:
    """The same solve as the rounds, on Ceres: what polishes a finished CASPAR model.

    One config, one builder, one difference — the backend. The polish was measured with colsfm's own settings (the
    config's Cauchy loss at scale 4, 200 iterations, SPARSE_SCHUR, extrinsics and
    intrinsics fixed), so the polish has to be that solve rather than a fresh set of
    pycolmap defaults.

    Args:
        ba_config: The Ceres settings from `vision_mapping_config.pb.txt`.
        options: The run's knobs; only the plan's backend is overridden.

    Returns:
        Options selecting Ceres, otherwise identical to `bundle_adjustment_options`.
    """
    return bundle_adjustment_options(ba_config, replace(options, ba_plan=replace(options.ba_plan, backend="ceres")))


def _triangulate(
    triangulator: pycolmap.IncrementalTriangulator,
    image_ids: Sequence[int],
    mapping_config: VisionMappingConfig,
    max_pixel_error: float,
    verbose: bool,
) -> int:
    """Run the inner triangulation loop until a pass adds nothing (§5.3).

    cuSFM repeats up to `iterative_triangulation_times` passes and exits early
    when a pass adds no point; observed 1 to 8 passes on Galileo.

    Args:
        triangulator: The triangulator, already bound to the reconstruction.
        image_ids: Registered images to triangulate, in ascending id order.
        mapping_config: The mapping configuration.
        max_pixel_error: The first round's gate, in pixels.
        verbose: Print one line per pass.

    Returns:
        Total observations triangulated.
    """
    options: pycolmap.IncrementalTriangulatorOptions = triangulator_options(mapping_config, max_pixel_error)
    total: int = 0
    for pass_index in range(max(mapping_config.iterative_triangulation_times, 1)):
        added: int = sum(triangulator.triangulate_image(options, image_id) for image_id in image_ids)
        total += added
        if verbose:
            print(f"[colsfm] triangulation pass {pass_index}: {added} observations")
        if added == 0:
            break
    return total


@dataclass(frozen=True, slots=True)
class BundleAdjustmentSolve:
    """One accepted bundle adjustment, after its post-solve guards."""

    summary: pycolmap.BundleAdjustmentSummary
    """What the solver reported."""
    seconds: float
    """Seconds inside `solve()`."""
    num_diverged: int
    """Observations the post-solve guards removed (`filter_degenerate_points` + `filter_projection_failures`)."""
    num_past_pole_before: int
    """Observations removed before the adjuster was built for lying at or past their camera's rational pole
    (`filter_past_rational_pole`): a caller's new extrinsics or poses can move an observation across it."""
    num_past_pole_after: int
    """Observations the same guard removed after the solve."""

    @property
    def num_past_pole(self) -> int:
        """Observations the rational-pole guard removed on either side of the solve.

        Returns:
            `num_past_pole_before + num_past_pole_after`.
        """
        return self.num_past_pole_before + self.num_past_pole_after

    @property
    def num_removed(self) -> int:
        """Observations any guard of this boundary removed; a caller holding an index of the model rebuilds it when nonzero.

        Returns:
            `num_diverged + num_past_pole`.
        """
        return self.num_diverged + self.num_past_pole


def default_gauge_frame_id(reconstruction: pycolmap.Reconstruction) -> int:
    """cuSFM's constant keyframe (spec §6.3): the frame owning the lowest registered image id.

    Args:
        reconstruction: A model with at least one registered image.

    Returns:
        The frame id.
    """
    return reconstruction.image(min(registered_image_ids(reconstruction))).frame_id


def global_ba_config(
    reconstruction: pycolmap.Reconstruction, gauge_frame_id: int, fixed_camera_params_id: int | None = None
) -> pycolmap.BundleAdjustmentConfig:
    """Every registered image, one constant rig frame, and optionally one camera's extrinsic held: the global solve's config.

    Args:
        reconstruction: The model.
        gauge_frame_id: Frame whose `rig_from_world` stays constant.
        fixed_camera_params_id: Camera whose `sensor_from_rig` stays constant, or None; ignored for the reference sensor.

    Returns:
        The config.
    """
    config: pycolmap.BundleAdjustmentConfig = pycolmap.BundleAdjustmentConfig()
    for image_id in registered_image_ids(reconstruction):
        config.add_image(image_id)
    config.set_constant_rig_from_world_pose(gauge_frame_id)
    if fixed_camera_params_id is not None:
        fixed_sensor: pycolmap.sensor_t = camera_sensor_id(fixed_camera_params_id)
        if not reconstruction.rig(RIG_ID).is_ref_sensor(fixed_sensor):
            config.set_constant_sensor_from_rig_pose(fixed_sensor)
    return config


def solve_bundle_adjustment(
    reconstruction: pycolmap.Reconstruction,
    prepared: PreparedSolve,
    gauge_frame_id: int,
    fixed_camera_params_id: int | None,
    observation_manager: pycolmap.ObservationManager,
    depth_threshold: float,
) -> BundleAdjustmentSolve:
    """Run one global bundle adjustment over every registered image, accept it and guard its points.

    Every solve in the package goes through here (mapping rounds, the Ceres polish, extrinsic
    refinement), so they share one acceptance rule and one post-solve sanity policy: a solve
    that is not usable raises, and a usable one still has the points it walked out of the
    world (`filter_degenerate_points`) or into a camera centre (`filter_projection_failures`)
    removed before anyone reads the model. A converged solve can do both (MSD MGO07's polish:
    2 of 89,470 points at 1e154 px). Round-specific quality gates stay with the callers.

    The gauge is one constant rig frame — cuSFM fixes a single keyframe, which
    through the shared rig fixes its whole rig pose (§6.3). COLMAP's own
    `fix_gauge` is deliberately not used: it would pick its own two frames and
    the resulting trajectory would no longer be anchored where cuSFM anchors it.

    Nothing about the backend is decided here. A pycolmap built without
    `CASPAR_ENABLED` used to be discovered by this function, which caught COLMAP's
    `ValueError`, rewrote the caller's options in place and built the adjuster again;
    `colsfm.ba_backend`'s capability probe now finds the same fact by solving, before
    the run creates anything, so a CASPAR solve only ever starts on a build that has
    already run one.

    Args:
        reconstruction: The model to adjust, in place.
        prepared: The solver and its options (`bundle_adjustment_solve`, `ceres_polish_options`).
        gauge_frame_id: Frame whose `rig_from_world` stays constant.
        fixed_camera_params_id: Camera whose `sensor_from_rig` stays constant while
            the others are refined, or None to leave every extrinsic as the options
            say. Ignored when it names the rig's reference sensor, which COLMAP
            already fixes to identity.
        observation_manager: Bookkeeping bound to the reconstruction, for the guards.
        depth_threshold: `filter_degenerate_points`' bound.

    Returns:
        The summary, the solve's seconds and what the guards removed.

    Raises:
        SolverFailure: When the solve is not usable; the model is then as the solver left it.
    """
    # Before the adjuster is built: Ceres evaluates every observation it is handed, and one at a rational pole is
    # infinite there or within a finite-difference step of it.
    num_past_pole_before: int = filter_past_rational_pole(observation_manager, reconstruction)
    config: pycolmap.BundleAdjustmentConfig = global_ba_config(reconstruction, gauge_frame_id, fixed_camera_params_id)
    summary, seconds = prepared.solve(reconstruction, config)
    if not summary.is_solution_usable():
        raise SolverFailure(f"bundle adjustment ended {summary.termination_type.name} without a usable solution")
    num_diverged: int = filter_degenerate_points(observation_manager, reconstruction, depth_threshold) + filter_projection_failures(
        observation_manager, reconstruction
    )
    num_past_pole_after: int = filter_past_rational_pole(observation_manager, reconstruction)
    reconstruction.update_point_3d_errors()
    return BundleAdjustmentSolve(
        summary=summary,
        seconds=seconds,
        num_diverged=num_diverged,
        num_past_pole_before=num_past_pole_before,
        num_past_pole_after=num_past_pole_after,
    )


def rig_camera_models(reconstruction: pycolmap.Reconstruction) -> tuple[frozenset[str], ...]:
    """The camera model names of each rig's cameras, one set per rig.

    Args:
        reconstruction: The model about to be adjusted.

    Returns:
        One set of `pycolmap.CameraModelId` names per rig; the data-less IMU reference
        contributes nothing.
    """
    return tuple(
        frozenset(
            reconstruction.camera(sensor.id).model.name for sensor in rig.sensor_ids() if sensor.type == pycolmap.SensorType.CAMERA
        )
        for rig in reconstruction.rigs.values()
    )


def shared_rig_cameras(reconstruction: pycolmap.Reconstruction) -> frozenset[int]:
    """Camera ids that are a non-reference sensor of more than one rig.

    Args:
        reconstruction: The model about to be adjusted.

    Returns:
        The ids; empty for colsfm's own one-rig models.
    """
    seen: dict[int, int] = {}
    for rig in reconstruction.rigs.values():
        for sensor in rig.sensor_ids():
            if sensor.type == pycolmap.SensorType.CAMERA and not rig.is_ref_sensor(sensor):
                seen[sensor.id] = seen.get(sensor.id, 0) + 1
    return frozenset(camera_id for camera_id, count in seen.items() if count > 1)


def _check_extrinsics_are_refinable(model: PosedModel) -> None:
    """Refuse an extrinsic refinement COLMAP would silently turn into a no-op.

    COLMAP's `ParameterizeRigsAndFrames` fixes every non-reference
    `sensor_from_rig` of a rig whose reference sensor is not itself among the
    parameterised sensors — "otherwise, the relative pose between the sensors is
    not well constrained". A vehicle-body reference owns no images, so it never
    is, and `refine_sensor_from_rig = True` then moves nothing at all.

    A `PosedModel` carries the reference it was built with, so this is the only
    condition left to test: the second half of the old guard — a caller who had a
    reconstruction but no `rig_reference` to invert the bookkeeping with — cannot
    be expressed any more.

    Args:
        model: The model about to be adjusted.

    Raises:
        ValueError: When the rig's reference sensor is not a camera.
    """
    if model.reference.camera_params_id is None:
        raise ValueError(
            "Cannot refine rig extrinsics: this rig's reference sensor is "
            f"{model.reconstruction.rig(RIG_ID).ref_sensor_id}, which owns no images. "
            "COLMAP holds every `sensor_from_rig` constant when the reference sensor is "
            "not part of the problem, so the refinement would be a silent no-op. Build "
            "the reconstruction with "
            "`build_reconstruction(..., reference_camera_params_id=...)` to put the rig "
            "origin on a camera."
        )


def _polish_with_ceres(
    reconstruction: pycolmap.Reconstruction,
    ba_config: BundleAdjustmentConfig,
    options: MappingOptions,
    gauge_frame_id: int,
    fixed_camera_params_id: int | None,
    observation_manager: pycolmap.ObservationManager,
    depth_threshold: float,
) -> PolishStats:
    """One Ceres global bundle adjustment over a finished CASPAR model.

    The whole step: the same gauge frame, the same observation set, no
    re-triangulation, no merge, no complete, no quality filter — so the reprojection
    error the round loop reported and the one this returns describe the same points,
    and every difference between them is the solve's. Only the numeric-sanity guards
    every round runs after its solve run here too: a converged polish can also walk a
    weakly constrained point behind a camera (MSD MGO07: 2 of 89,470 points at
    1e154 px, which made the mean 1.9e149 px). Off by default: CASPAR reaches its own least-squares minimum, and the
    polish only trades it for the Cauchy minimum (`--caspar-ceres-polish`).

    Args:
        reconstruction: The model the rounds left behind, adjusted in place.
        ba_config: The Ceres settings from `vision_mapping_config.pb.txt`.
        options: The run's knobs; the plan's backend is forced to Ceres.
        gauge_frame_id: The same frame the rounds held constant.
        fixed_camera_params_id: The same camera the rounds held constant, or None.
        observation_manager: Bookkeeping bound to the reconstruction, for the guards.
        depth_threshold: `filter_degenerate_points`' bound, as in the rounds.

    Returns:
        What the solve cost and how far it moved the reprojection error.
    """
    polish: PreparedSolve = PreparedSolve(backend="ceres", options=ceres_polish_options(ba_config, options))
    before_px: float = reconstruction.compute_mean_reprojection_error()
    solve: BundleAdjustmentSolve = solve_bundle_adjustment(
        reconstruction, polish, gauge_frame_id, fixed_camera_params_id, observation_manager, depth_threshold
    )
    report: SolverReport = parse_brief_report(solve.summary.brief_report())
    stats: PolishStats = PolishStats(
        num_observations=reconstruction.compute_num_observations() + solve.num_diverged + solve.num_past_pole_after,
        num_diverged=solve.num_diverged,
        num_past_pole=solve.num_past_pole,
        report=report,
        ba_termination=ceres_termination(solve.summary.termination_type.name),
        mean_reprojection_error_before_px=before_px,
        mean_reprojection_error_after_px=reconstruction.compute_mean_reprojection_error(),
        seconds=solve.seconds,
    )
    if options.verbose:
        reduction: float | None = report.cost_reduction
        costs: str = (
            "cost not reported"
            if not report.is_available
            else f"cost {report.initial_cost:.6g} -> {report.final_cost:.6g}" + ("" if reduction is None else f" ({reduction:.2%})")
        )
        iterations: str = "?" if report.num_iterations is None else str(report.num_iterations)
        print(
            f"[colsfm] ceres polish: {iterations} iterations, {costs}, "
            f"reprojection {stats.mean_reprojection_error_before_px:.4f} -> "
            f"{stats.mean_reprojection_error_after_px:.4f} px in {stats.seconds:.1f} s ({stats.ba_termination}), diverged {stats.num_diverged}, past the rational pole {stats.num_past_pole}"
        )
    return stats


def run_mapping(
    model: PosedModel,
    database_path: Path,
    mapping_config: VisionMappingConfig,
    options: MappingOptions | None = None,
    gauge_frame_id: int | None = None,
) -> MappingResult:
    """Triangulate and bundle-adjust, cuSFM's `keypoints_mapper_main` §5.3-§5.6.

    ```
    load correspondences
    triangulate every registered image, repeating until a pass adds nothing
    for k in 0 .. num_ba_iterations - 1:
        merge -> complete -> filter at gate g_k -> global bundle adjustment -> guards
        stop early when (merged + completed + filtered) / observations < max_observation_change
    one Ceres global bundle adjustment, when the rounds ran on CASPAR
    ```

    The trailing guards are the one step cuSFM has no equivalent of:
    `filter_degenerate_points` and `filter_projection_failures` between them
    remove what a converged solve can still leave behind — a point outside the
    world, or one inside a camera. Both docstrings carry the measurements.

    The early exit is the configuration's own: set
    `mapping_config.max_observation_change` to 0.0 to run every round, since the
    statistic is never negative. cuSFM logs `use num_ba_iterations to stop BA`, so
    its two stopping rules are exclusive.

    Args:
        model: A reconstruction with registered, pose-carrying frames and the rig
            reference it was built with. The reconstruction is adjusted in place.
        database_path: COLMAP database with keypoints and verified two-view geometries.
        mapping_config: `vision_mapping_config.pb.txt`; its `bundle_adjustment` block
            is what the Ceres solve is configured from.
        options: Command-line style knobs; the defaults when None.
        gauge_frame_id: Frame held constant for gauge; the frame owning the
            lowest image id when None, which is cuSFM's own choice (§6.3).

    With `--caspar-ceres-polish` a CASPAR run ends on one Ceres bundle adjustment over the final model, with no
    filtering around it (off by default; see `_polish_with_ceres`). It is skipped whenever the solves ran
    on Ceres — every fallback included — because Ceres' own answer is a fixed point
    of it (one iteration, no cost change, 25 femtometres of rig motion, measured).

    Returns:
        The adjusted reconstruction with per-round statistics and timings, the
        closing `polish` when one ran, plus `refined_extrinsics` when extrinsics
        were refined.

    Raises:
        FileNotFoundError: When the database does not exist.
        ValueError: When the database and the reconstruction disagree on image
            ids, when no frame is registered, or when extrinsic refinement is
            asked for on a rig COLMAP would silently freeze.
    """
    given_options: MappingOptions = MappingOptions() if options is None else options
    reconstruction: pycolmap.Reconstruction = model.reconstruction
    # The one thing resolution could not know: which camera models this run turned out
    # to hold. Narrowed once, here, and then executed and reported verbatim -- nothing
    # below asks `colsfm.ba_backend` anything again, and nothing rewrites the backend
    # mid-run. `for_camera_models` returns the same plan unless CASPAR would silently
    # drop a model's observations.
    plan: BaExecutionPlan = given_options.ba_plan.for_camera_models(
        (camera.model.name for camera in reconstruction.cameras.values()),
        refine_intrinsics=given_options.refine_focal_length,
        rig_camera_models=rig_camera_models(reconstruction),
        shared_rig_cameras=shared_rig_cameras(reconstruction),
    )
    if plan.fallback_reason != given_options.ba_plan.fallback_reason:
        plan.announce()
    resolved_options: MappingOptions = replace(given_options, ba_plan=plan)
    ba_config: BundleAdjustmentConfig = mapping_config.bundle_adjustment
    started: float = time.perf_counter()
    image_ids: list[int] = registered_image_ids(reconstruction)
    if not image_ids:
        raise ValueError("The reconstruction has no registered frames to map")
    resolved_gauge_frame_id: int = default_gauge_frame_id(reconstruction) if gauge_frame_id is None else gauge_frame_id
    fixed_camera_params_id: int | None = None
    if resolved_options.optimize_extrinsics:
        _check_extrinsics_are_refinable(model)
        # cuSFM's constant keyframe is the lowest id in the map (§6.3), and
        # `--fixed_camera_name` pins that keyframe's camera when extrinsics move.
        fixed_camera_params_id = (
            reconstruction.image(min(image_ids)).camera_id
            if resolved_options.fixed_camera_params_id is None
            else resolved_options.fixed_camera_params_id
        )

    correspondence_started: float = time.perf_counter()
    correspondences: Correspondences = load_correspondences(
        reconstruction,
        database_path,
        mapping_config.min_num_matches_per_pair,
        inlier_overrides=resolved_options.inlier_overrides,
    )
    correspondence_seconds: float = time.perf_counter() - correspondence_started

    triangulator: pycolmap.IncrementalTriangulator = pycolmap.IncrementalTriangulator(
        correspondences.graph, reconstruction, correspondences.observation_manager
    )
    schedule: tuple[float, ...] = pixel_error_schedule(mapping_config)
    triangulation_started: float = time.perf_counter()
    _triangulate(triangulator, image_ids, mapping_config, schedule[0], resolved_options.verbose)
    triangulation_seconds: float = time.perf_counter() - triangulation_started
    # Round 0's gate would drop these anyway; doing it here only makes the line below
    # mean something. Without it RoboCap's triangulation reports 8.9e152 px, because a
    # handful of tracks resolve to a point sitting inside one of their own cameras.
    filter_projection_failures(correspondences.observation_manager, reconstruction)
    reconstruction.update_point_3d_errors()
    if resolved_options.verbose:
        print(
            f"[colsfm] triangulation finished: {reconstruction.num_points3D()} points, "
            f"reprojection {reconstruction.compute_mean_reprojection_error():.4f} px, "
            f"mean length {reconstruction.compute_mean_track_length():.4f}"
        )

    prepared: PreparedSolve = bundle_adjustment_solve(ba_config, resolved_options)
    rounds: list[RoundStats] = []
    bundle_adjustment_seconds: float = 0.0
    for round_index, max_pixel_error in enumerate(schedule):
        round_started: float = time.perf_counter()
        round_options: pycolmap.IncrementalTriangulatorOptions = triangulator_options(mapping_config, max_pixel_error)
        num_merged: int = triangulator.merge_all_tracks(round_options)
        num_completed: int = triangulator.complete_all_tracks(round_options)
        filtered: _FilterCounts = _filter_points(correspondences.observation_manager, reconstruction, mapping_config, max_pixel_error)
        # The guards run on both sides of the solve: before it (`_filter_points`, and the rational-pole guard inside the
        # solve boundary), so Ceres never linearises a NaN; after it inside the solve boundary, so the next round
        # inherits no point the solve walked out of the world.
        solve: BundleAdjustmentSolve = solve_bundle_adjustment(
            reconstruction,
            prepared,
            resolved_gauge_frame_id,
            fixed_camera_params_id,
            correspondences.observation_manager,
            mapping_config.depth_threshold,
        )
        # What the solve was handed: the model after every pre-solve removal, the pole guard's included.
        num_observations: int = reconstruction.compute_num_observations() + solve.num_diverged + solve.num_past_pole_after
        num_changed: int = num_merged + num_completed + filtered.total() + solve.num_past_pole_before
        observation_change: float = num_changed / num_observations if num_observations else 0.0
        bundle_adjustment_seconds += solve.seconds
        report: SolverReport = parse_brief_report(solve.summary.brief_report())
        rounds.append(
            RoundStats(
                round_index=round_index,
                max_pixel_error=max_pixel_error,
                num_merged=num_merged,
                num_completed=num_completed,
                num_filtered=filtered.total(),
                num_diverged=solve.num_diverged,
                num_past_pole=solve.num_past_pole,
                num_observations=num_observations,
                observation_change=observation_change,
                num_points3D=reconstruction.num_points3D(),
                mean_reprojection_error_px=reconstruction.compute_mean_reprojection_error(),
                report=report,
                ba_termination=ceres_termination(solve.summary.termination_type.name),
                seconds=time.perf_counter() - round_started,
            )
        )
        latest: RoundStats = rounds[-1]
        if resolved_options.round_callback is not None:
            resolved_options.round_callback(latest, reconstruction)
        if resolved_options.verbose:
            print(
                f"[colsfm] round {round_index} gate {max_pixel_error:.1f} px: merged {num_merged}, "
                f"completed {num_completed}, filtered {filtered.total()}, diverged {latest.num_diverged}, past the pole {latest.num_past_pole}, "
                f"change {observation_change:.6f}, points {latest.num_points3D}, "
                f"reprojection {latest.mean_reprojection_error_px:.4f} px, BA {solve.seconds:.2f} s"
            )
        if observation_change < mapping_config.max_observation_change:
            if resolved_options.verbose:
                print(f"[colsfm] observation change {observation_change:.6f} below {mapping_config.max_observation_change}")
            break

    # The rounds are done; CASPAR's answer is not yet a stationary point of the
    # objective, so one Ceres solve finishes it. Only the post-solve guards run after it.
    polish: PolishStats | None = None
    if plan.ceres_polish:
        polish = _polish_with_ceres(
            reconstruction,
            ba_config,
            resolved_options,
            resolved_gauge_frame_id,
            fixed_camera_params_id,
            correspondences.observation_manager,
            mapping_config.depth_threshold,
        )
        bundle_adjustment_seconds += polish.seconds

    return MappingResult(
        reconstruction=reconstruction,
        reference=model.reference,
        rounds=tuple(rounds),
        correspondence_seconds=correspondence_seconds,
        triangulation_seconds=triangulation_seconds,
        bundle_adjustment_seconds=bundle_adjustment_seconds,
        polish=polish,
        total_seconds=time.perf_counter() - started,
        extrinsics_refined=resolved_options.optimize_extrinsics,
        # The plan the solves ran on, verbatim. The backend and the reason used to be
        # read back off the options object and *inferred* by comparing it against a
        # second resolution, because the backend could still change mid-run.
        ba_plan=plan,
    )
