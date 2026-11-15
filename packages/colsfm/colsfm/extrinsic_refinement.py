"""Regularised rig-extrinsic refinement — the prior terms pycolmap's bundle adjuster has not.

cuSFM's second `keypoints_mapper_main` pass (`--optimize_extrinsics=true`) does not
optimise reprojection alone. Its Ceres problem carries two extra cost groups, both
visible in the shipped Galileo log
(`data/cusfm_runs/galileo/cusfm/kpmap/keypoints_mapper_main.txt`):

```
bundle_adjustment_solver.cc:1058] Added 8 absolute pose constraints of extrinsics to problem
bundle_adjustment_solver.cc:1138] Added 777 camera rig extrinsic constraints
[BA COST] initial group 'relative_extrinsic' cost = 0 (777 blocks)
[BA COST] initial group 'absolute_extrinsic' cost = 0 (8 blocks)
```

which is the paper's §4.4: reprojection (Eq. 5) plus an absolute extrinsic prior
(Eq. 14) plus the inter-camera relative extrinsic constraint (Eq. 6). pycolmap's
`BundleAdjuster` can express neither and its Ceres problem cannot be extended from
standalone pyceres (`docs/spec/pycolmap-capabilities.md` §8, the pybind11 registry
split), so this stage rebuilds the extrinsic half of that problem in pyceres and
alternates it with pycolmap's own bundle adjustment.

## Where the parts live

This module is the alternation itself — its options, its per-round and stage results, its
write-back and its stopping test — and the stage's public surface. The problem it drives
is three modules, each with one reason to change:

* `colsfm.extrinsic_observations` — which image saw which point, and where;
* `colsfm.extrinsic_costs` — what Ceres gets when it evaluates a residual block;
* `colsfm.extrinsic_solve` — which blocks the problem holds, and the solve itself.

## What the two block counts mean, measured

**8 absolute blocks** is one per camera — all eight, including the one whose extrinsic
is held constant and which therefore vanishes from Ceres' `Reduced` column. Only the free
cameras get one here; see deviation 4 below.

**777 relative blocks** is one per unordered pair of cameras that fired in the same rig
frame, once per rig frame, and not `4 declared stereo pairs x N frames`; the arithmetic
that establishes it is in `colsfm.extrinsic_observations`, which counts the pairs, and
the multiplicities it returns are what `RepeatedCauchyLoss` turns back into 777 blocks'
worth of cost.

## Which sigmas, measured

The priors use `extrinsic_error_meters = 0.01` and `extrinsic_error_degrees = 2`, not the
`relative_pose_*_error_*` pair the same configuration also offers; `colsfm.extrinsic_solve`
carries the cost reproduction that settles it.

## Whitening convention

`colsfm.mapping` hands pycolmap an unwhitened pixel residual with
`loss_function_scale = 4.0`, which is the same minimum as the blob's whitened residual
under `CauchyLoss(1.0)` because `CauchyLoss(a)` at `s` equals `a^2` times `CauchyLoss(1)`
at `s/a^2`. That equivalence only survives when reprojection is the *whole* objective.
Here it is mixed with priors, so the common scale has to be the blob's: every residual
in this stage is whitened (reprojection by `1/4 px`, priors by their sigmas) and every
loss is `CauchyLoss(1.0)`.

## Why one residual block can hold a whole camera

A Python `pyceres` cost function costs roughly a millisecond per call, and Galileo's
refinement carries ~30 000 observations, so `colsfm.extrinsic_costs` holds **all** of one
camera's observations in a single block and applies the robust loss inside `Evaluate`.
That module states why the substitution is exact rather than an approximation.

## The block-coordinate loop

Extrinsics, rig poses and points cannot go into one pyceres problem — 30 000 Python
residual blocks again — so the refinement alternates:

* **(A)** extrinsics only, in pyceres, with rig poses and points frozen at their current
  values, carrying reprojection plus both priors;
* **(B)** pycolmap's own bundle adjustment with `refine_sensor_from_rig = False`, poses
  and points free.

It stops when a round moves no extrinsic by more than 0.1 mm / 0.005 deg, or after
`num_rounds`. `ExtrinsicRefinementOptions.regularised = False` restores the plain
pycolmap path (`run_mapping(..., optimize_extrinsics=True)`) for comparison.

`refine_extrinsics` takes an already-mapped `MappingResult` and keeps adjusting its
model: the caller maps its camera-referenced `PosedModel` once and the alternation
picks up from there, rather than the same database being mapped twice.

## Deliberate deviations

1. **The reference camera's extrinsic is pinned**, because COLMAP requires the rig's
   reference sensor to be identity (`colsfm.reconstruction`). The blob leaves all eight
   free and lets the absolute priors fix the gauge. Every extrinsic here therefore moves
   relative to the gauge camera, which is the quantity NOTES.md gotcha 13 tabulates.
2. **The gauge keyframe's observations are not un-whitened.** The blob leaves its 13
   constant-frame residuals at weight 1.0 (`docs/spec/keypoints_mapper_main.md` §6.4, an
   upstream bug); with poses frozen in solve (A) those residuals are constant anyway.
3. **The camera projection Jacobian is numeric**, six batched `Camera.img_from_cam`
   calls per evaluation. pycolmap exposes no analytic camera Jacobian and the numeric one
   is model-agnostic; the pose half of the chain rule is analytic.
4. **The pinned camera gets no absolute prior block**, so seven are built where the blob
   logs eight. A block whose every parameter is constant is evaluated by Ceres'
   `Program::RemoveFixedBlocks` with a null residual pointer, and pyceres 2.6's Python
   trampoline segfaults on it. The block is inert in the blob too, so nothing is lost.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap

from colsfm.config import BundleAdjustmentConfig, VisionMappingConfig
from colsfm.extrinsic_observations import (
    CameraPairKey,
    ObservationIndex,
    build_observation_index,
    camera_observations,
    co_observed_camera_pairs,
)
from colsfm.extrinsic_solve import ExtrinsicRefinementOptions, ExtrinsicSolveStats, solve_extrinsics
from colsfm.geometry import MILLIMETRES_PER_METRE, relative_rotation_degrees
from colsfm.mapping import (
    MappingOptions,
    MappingResult,
    bundle_adjustment_options,
    registered_image_ids,
    run_mapping,
    solve_bundle_adjustment,
)
from colsfm.reconstruction import RIG_ID, PosedModel, RigReference, camera_sensor_id

# ======================================================================================
# round and stage results
# ======================================================================================


@dataclass(frozen=True, slots=True)
class ExtrinsicRefinementRound:
    """One (A) extrinsics solve followed by one (B) bundle adjustment."""

    round_index: int
    """Zero-based round number."""
    solve: ExtrinsicSolveStats
    """The extrinsics-only solve of step (A)."""
    max_translation_change_mm: float
    """Largest extrinsic origin move this round made, in millimetres."""
    max_rotation_change_deg: float
    """Largest extrinsic rotation this round made, in degrees."""
    bundle_adjustment_seconds: float
    """Seconds inside step (B)."""
    mean_reprojection_error_px: float
    """Mean reprojection error of the whole model after step (B)."""
    seconds: float
    """Wall-clock seconds for the round."""


@dataclass(frozen=True, slots=True)
class ExtrinsicRefinementResult:
    """What `refine_extrinsics` produced.

    The extrinsics and the elapsed time are read straight off `mapping`, which is a
    live view of the adjusted model (`colsfm.mapping.MappingResult`); duplicating them
    here is what let the two disagree.
    """

    mapping: MappingResult
    """The mapping result: the round-by-round statistics of the pass that produced the
    model, with the timings extended to cover the alternation that followed."""
    rounds: tuple[ExtrinsicRefinementRound, ...]
    """Per-round statistics of the alternation; empty when `regularised` was False."""
    converged: bool
    """Whether a round fell inside both tolerances rather than exhausting `num_rounds`."""
    regularised: bool
    """Whether the prior-carrying path ran; the pipeline reports it and nothing derives
    from it."""

    @property
    def refined_extrinsics(self) -> dict[int, pycolmap.Rigid3d]:
        """`vehicle_T_cam` per `camera_params_id` after the refinement.

        Returns:
            The extrinsics read back out of the adjusted rig.

        Raises:
            ValueError: When the mapping result reports none, which cannot happen for a
                model this function accepted — it needs a camera-referenced rig — and
                would mean the model was swapped underneath it.
        """
        refined: dict[int, pycolmap.Rigid3d] | None = self.mapping.refined_extrinsics
        if refined is None:
            raise ValueError(
                "the refined model reports no extrinsics: its rig reference is not a "
                "camera, so `refine_extrinsics` cannot have produced it"
            )
        return refined

    @property
    def seconds(self) -> float:
        """Wall-clock seconds for the whole stage, the mapping pass included.

        Returns:
            `mapping.total_seconds`, which `refine_extrinsics` extends to cover the
            alternation.
        """
        return self.mapping.total_seconds


# ======================================================================================
# writing the refined extrinsics back, and measuring what moved
# ======================================================================================


def apply_extrinsics(
    reconstruction: pycolmap.Reconstruction,
    rig_reference: RigReference,
    vehicle_T_cam_by_camera_params_id: Mapping[int, pycolmap.Rigid3d],
) -> None:
    """Write refined extrinsics back into the reconstruction's rig, in place.

    `sensor_from_rig = cam_T_cam_ref = vehicle_T_cam^-1 * vehicle_T_cam_ref`, the inverse
    of what `colsfm.reconstruction.build_rig` writes. The reference camera is skipped:
    COLMAP holds its `sensor_from_rig` at identity by construction, so the rig frames'
    `rig_from_world` keeps its meaning and no pose has to move.

    Args:
        reconstruction: The model to update.
        rig_reference: How its rig was built; must be camera-referenced.
        vehicle_T_cam_by_camera_params_id: The refined extrinsics.

    Raises:
        ValueError: When the rig is not camera-referenced.
    """
    if rig_reference.camera_params_id is None:
        raise ValueError("cannot write extrinsics back into a vehicle-referenced rig")
    vehicle_T_reference: pycolmap.Rigid3d = vehicle_T_cam_by_camera_params_id[rig_reference.camera_params_id]
    rig: pycolmap.Rig = reconstruction.rig(RIG_ID)
    for camera_params_id, vehicle_T_cam in vehicle_T_cam_by_camera_params_id.items():
        if camera_params_id == rig_reference.camera_params_id:
            continue
        rig.set_sensor_from_rig(camera_sensor_id(camera_params_id), vehicle_T_cam.inverse() * vehicle_T_reference)


def extrinsic_deltas(
    before: Mapping[int, pycolmap.Rigid3d], after: Mapping[int, pycolmap.Rigid3d]
) -> tuple[float, float]:
    """Largest translation and rotation change between two sets of extrinsics.

    Args:
        before: `vehicle_T_cam` per `camera_params_id` before a step.
        after: The same cameras after it.

    Returns:
        `(largest translation change in millimetres, largest rotation change in degrees)`;
        `(0.0, 0.0)` when the two share no camera.
    """
    translation_mm: list[float] = []
    rotation_deg: list[float] = []
    for camera_params_id, previous in before.items():
        if camera_params_id not in after:
            continue
        updated: pycolmap.Rigid3d = after[camera_params_id]
        translation_mm.append(
            float(np.linalg.norm(np.asarray(updated.translation) - np.asarray(previous.translation)))
            * MILLIMETRES_PER_METRE
        )
        rotation_deg.append(relative_rotation_degrees(previous, updated))
    if not translation_mm:
        return 0.0, 0.0
    return max(translation_mm), max(rotation_deg)


# ======================================================================================
# the alternation
# ======================================================================================


def refine_extrinsics(
    mapping: MappingResult,
    database_path: Path,
    mapping_config: VisionMappingConfig,
    mapping_options: MappingOptions,
    calibration_vehicle_T_cam: Mapping[int, pycolmap.Rigid3d],
    options: ExtrinsicRefinementOptions | None = None,
) -> ExtrinsicRefinementResult:
    """Alternate extrinsics-only solves with bundle adjustment, on an already-mapped model.

    ```
    (the caller has already run run_mapping on a camera-referenced model)
    for k in 0 .. num_rounds - 1:
        (A) solve_extrinsics(...)                        # poses and points frozen
        (B) bundle adjustment with sensor_from_rig fixed # extrinsics frozen
        stop when this round moved no extrinsic past the tolerances
    ```

    The mapping pass is the caller's, not this function's: under
    `--optimize-extrinsics` the pipeline builds **one** camera-referenced `PosedModel`,
    maps it once and hands the adjusted result here, instead of mapping the same
    database twice.

    With `options.regularised` False the whole thing collapses to
    `run_mapping(..., optimize_extrinsics=True)` — the unregularised pycolmap path,
    kept for comparison — run on that same already-adjusted model.

    The three sigmas and the iteration ceiling in `options` are **overridden** from
    `mapping_config.bundle_adjustment`: the pyceres problem and the pycolmap bundle
    adjustment it alternates with have to whiten by the same numbers, and the
    configuration is the authority on what those are.

    Args:
        mapping: What `run_mapping` returned for a camera-referenced `PosedModel`; its
            reconstruction is adjusted further, in place.
        database_path: COLMAP database with keypoints and verified two-view geometries;
            read only by the unregularised path.
        mapping_config: `vision_mapping_config.pb.txt`, including the
            `BundleAdjustmentConfig` every sigma comes from.
        mapping_options: Command-line style mapper knobs; `optimize_extrinsics` is
            overridden by this function.
        calibration_vehicle_T_cam: The input `vehicle_T_cam` per `camera_params_id`, which
            is what both priors pull towards.
        options: Refinement settings; the defaults when None.

    Returns:
        The updated mapping result and the per-round statistics; the refined extrinsics
        are read off the result's own model.
    """
    ba_config: BundleAdjustmentConfig = mapping_config.bundle_adjustment
    resolved: ExtrinsicRefinementOptions = dataclasses.replace(
        ExtrinsicRefinementOptions() if options is None else options,
        extrinsic_translation_sigma_m=ba_config.extrinsic_error_meters,
        extrinsic_rotation_sigma_deg=ba_config.extrinsic_error_degrees,
        reprojection_sigma_px=ba_config.reprojection_error_standard_deviation,
        max_num_iterations=ba_config.max_num_iterations,
    )
    reconstruction: pycolmap.Reconstruction = mapping.reconstruction
    rig_reference: RigReference = mapping.reference
    started: float = time.perf_counter()

    if not resolved.regularised:
        unregularised: MappingResult = run_mapping(
            PosedModel(reconstruction=reconstruction, reference=rig_reference),
            database_path,
            mapping_config,
            dataclasses.replace(mapping_options, optimize_extrinsics=True),
        )
        return ExtrinsicRefinementResult(
            mapping=dataclasses.replace(
                unregularised,
                bundle_adjustment_seconds=mapping.bundle_adjustment_seconds
                + unregularised.bundle_adjustment_seconds,
                total_seconds=mapping.total_seconds + (time.perf_counter() - started),
            ),
            rounds=(),
            converged=True,
            regularised=False,
        )

    # `mapping.ba_plan` rather than `mapping_options.ba_plan`: the mapping pass has
    # already narrowed the plan against this very reconstruction, so the rounds solve on
    # exactly what it solved on rather than re-deciding -- and announcing -- per round.
    base_options: MappingOptions = dataclasses.replace(
        mapping_options, optimize_extrinsics=False, ba_plan=mapping.ba_plan
    )
    # The same gauge `run_mapping` picked for the mapping pass: the frame owning the
    # lowest registered image id, which is cuSFM's own choice (spec §6.3).
    gauge_frame_id: int = reconstruction.image(min(registered_image_ids(reconstruction))).frame_id
    ba_options: pycolmap.BundleAdjustmentOptions = bundle_adjustment_options(ba_config, base_options)
    # Neither of these changes across the rounds: bundle adjustment moves poses and
    # points but adds and removes no observation, so which image saw which point — and
    # therefore which cameras co-observed — is fixed. Only the geometry is re-gathered.
    observation_index: ObservationIndex = build_observation_index(reconstruction)
    pair_counts: dict[CameraPairKey, int] = co_observed_camera_pairs(reconstruction)

    rounds: list[ExtrinsicRefinementRound] = []
    converged: bool = False
    for round_index in range(resolved.num_rounds):
        round_started: float = time.perf_counter()
        before: dict[int, pycolmap.Rigid3d] = rig_reference.vehicle_T_cam_by_camera_params_id(reconstruction)
        refined, stats = solve_extrinsics(
            reconstruction,
            rig_reference,
            calibration_vehicle_T_cam,
            resolved,
            observations=camera_observations(reconstruction, rig_reference, observation_index),
            pair_counts=pair_counts,
        )
        apply_extrinsics(reconstruction, rig_reference, refined)
        _, ba_seconds = solve_bundle_adjustment(reconstruction, ba_options, gauge_frame_id, None)
        reconstruction.update_point_3d_errors()
        after: dict[int, pycolmap.Rigid3d] = rig_reference.vehicle_T_cam_by_camera_params_id(reconstruction)
        translation_mm, rotation_deg = extrinsic_deltas(before, after)
        rounds.append(
            ExtrinsicRefinementRound(
                round_index=round_index,
                solve=stats,
                max_translation_change_mm=translation_mm,
                max_rotation_change_deg=rotation_deg,
                bundle_adjustment_seconds=ba_seconds,
                mean_reprojection_error_px=reconstruction.compute_mean_reprojection_error(),
                seconds=time.perf_counter() - round_started,
            )
        )
        if resolved.verbose:
            latest: ExtrinsicRefinementRound = rounds[-1]
            print(
                f"[colsfm] extrinsic round {round_index}: {stats.num_reprojection_residuals // 2} observations "
                f"in {stats.num_reprojection_blocks} blocks, {stats.num_absolute_prior_blocks} absolute and "
                f"{stats.num_relative_prior_frames} relative priors, cost "
                f"{stats.initial_cost:.6g} -> {stats.final_cost:.6g} ({stats.termination}), "
                f"moved {translation_mm:.3f} mm / {rotation_deg:.4f} deg, "
                f"reprojection {latest.mean_reprojection_error_px:.4f} px, "
                f"{latest.seconds:.2f}s"
            )
        if translation_mm < resolved.translation_tolerance_mm and rotation_deg < resolved.rotation_tolerance_deg:
            converged = True
            break

    # Only the timings and the "extrinsics moved" flag need replacing: every count on a
    # `MappingResult` is a property over this very reconstruction, so they are already up
    # to date with the alternation that just finished. The plan is the mapping pass's own
    # -- these rounds solved on it -- so it carries through untouched.
    updated: MappingResult = dataclasses.replace(
        mapping,
        bundle_adjustment_seconds=mapping.bundle_adjustment_seconds
        + sum(round_stats.bundle_adjustment_seconds for round_stats in rounds),
        total_seconds=mapping.total_seconds + (time.perf_counter() - started),
        extrinsics_refined=True,
    )
    return ExtrinsicRefinementResult(
        mapping=updated, rounds=tuple(rounds), converged=converged, regularised=True
    )
