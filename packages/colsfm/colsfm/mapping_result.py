"""What a mapping pass reports: its rounds, its closing polish and its final model.

The reporting vocabulary of `colsfm.mapping`, kept apart from the mapper for the
reason `colsfm.run_report` is kept apart from `colsfm.pipeline`: these are the
contract with a reader -- the pipeline summary, the walkthrough's charts, a
benchmark -- rather than anything the mapping loop needs to execute.

`MappingResult` computes its counters from the live reconstruction on every
access, which is exactly why a run that wants to remember them has to snapshot
them (`colsfm.run_report.MappingStats`): a later pass adjusts the same model.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import pycolmap

from colsfm.ba_backend import (
    CERES_ONLY_PLAN,
    BaBackend,
    BaExecutionPlan,
)
from colsfm.correspondences import registered_image_ids
from colsfm.reconstruction import (
    RigReference,
)
from colsfm.reconstruction import num_registered_images as num_observing_images
from colsfm.solver_report import CeresTermination, SolverReport


@dataclass(frozen=True, slots=True)
class RoundStats:
    """One outer triangulate / filter / bundle-adjust round (§5.4)."""

    round_index: int
    """Zero-based round number, `k` in the gate schedule."""
    max_pixel_error: float
    """The round's reprojection gate `g_k`, in pixels."""
    num_merged: int
    """Observations merged by `merge_all_tracks`."""
    num_completed: int
    """Observations added by `complete_all_tracks`."""
    num_filtered: int
    """Observations dropped by the reprojection, angle, depth and track-length filters."""
    num_diverged: int
    """Observations dropped by the degeneracy guard *after* the solve, i.e. points bundle
    adjustment itself pushed out of bounds. Normally 0; see `filter_degenerate_points`."""
    num_observations: int
    """Observations left in the reconstruction after filtering; cuSFM's `num observed`."""
    observation_change: float
    """`(merged + completed + filtered) / observations`, the early-exit statistic."""
    num_points3D: int
    """Points left after this round's bundle adjustment."""
    mean_reprojection_error_px: float
    """Mean reprojection error after this round, in pixels."""
    report: SolverReport
    """What the round's bundle adjustment said about its own progress.

    The iterations and the two costs, held as the one record `colsfm.solver_report`
    owns rather than spliced into three fields here and three more on `PolishStats`.
    Every field of it is optional because every one is: the CASPAR backend writes no
    Ceres-shaped report at all, and this used to be recorded as 0 iterations at 0.0
    cost, which reads as a solve that converged instantly."""
    ba_termination: CeresTermination
    """How the solve ended; the solver reports this whatever its report says, so it is
    not optional."""
    seconds: float
    """Wall-clock seconds the round took, bundle adjustment included."""


@dataclass(frozen=True, slots=True)
class PolishStats:
    """The one Ceres bundle adjustment that finishes a CASPAR mapping run.

    Nothing is merged, completed or filtered around it, so the observation set that
    goes in is the one that comes out and every number here is the solve's alone
    (`docs/caspar-build.md` § Shipped: CASPAR + Ceres polish).
    """

    num_observations: int
    """Observations the solve parameterised; unchanged by it, since nothing filters."""
    report: SolverReport
    """What the polish said about its own progress: 22 iterations on KITTI 06 from
    CASPAR's answer, ending 3.4 % below the initial cost. Optional throughout for the
    same reason as `RoundStats.report`."""
    ba_termination: CeresTermination
    """How the polish ended."""
    mean_reprojection_error_before_px: float
    """Mean reprojection error as CASPAR left it, in pixels."""
    mean_reprojection_error_after_px: float
    """Mean reprojection error after the polish, over the same observations."""
    seconds: float
    """Wall-clock seconds inside `BundleAdjuster.solve()`; 9.8 s on KITTI 06's 2156 images."""


RoundCallback: TypeAlias = Callable[[RoundStats, pycolmap.Reconstruction], None]
"""What `MappingOptions.round_callback` takes: one round's statistics and the live model."""


@dataclass(frozen=True, slots=True)
class MappingResult:
    """What `run_mapping` produced, and how long each phase took.

    Every count is a **property** computed from `reconstruction` on demand, not a
    number copied out of it at return time. `colsfm.extrinsic_refinement` keeps
    adjusting the very same model after `run_mapping` has returned, and a stored
    count would then be quietly describing a model that no longer exists.
    """

    reconstruction: pycolmap.Reconstruction
    """The adjusted reconstruction; the same object that was passed in."""
    reference: RigReference
    """Where the rig origin sits, carried through from the `PosedModel` that was mapped."""
    rounds: tuple[RoundStats, ...]
    """Per-round statistics, in order."""
    correspondence_seconds: float
    """Time spent loading keypoints and two-view geometries from the database."""
    triangulation_seconds: float
    """Time spent in the initial triangulation passes."""
    bundle_adjustment_seconds: float
    """Time spent inside `BundleAdjuster.solve`, summed over the rounds and the polish."""
    total_seconds: float
    """Wall-clock seconds for the whole call."""
    extrinsics_refined: bool = False
    """Whether this pass refined `sensor_from_rig`; what makes `refined_extrinsics`
    something other than the input calibration read back out of the rig."""
    ba_plan: BaExecutionPlan = CERES_ONLY_PLAN
    """The plan these solves executed, narrowed by this model's camera models.

    Evidence, not a request: `run_mapping` receives the run's plan, narrows it once
    against the reconstruction it is about to adjust, and returns the object it
    actually solved on. `ba_backend` and `ba_fallback_reason` read off it, so the two
    cannot disagree with each other or with what ran."""
    polish: PolishStats | None = None
    """The closing Ceres solve of a CASPAR run, or None when none ran.

    None means either `ba_plan.backend == "ceres"` — a fallback included, since a
    Ceres run is already at its own fixed point — or `--no-caspar-ceres-polish`.
    Its seconds are part of `bundle_adjustment_seconds`."""

    @property
    def ba_backend(self) -> BaBackend:
        """Which backend the solves actually ran on, fallbacks applied.

        Returns:
            `ba_plan.backend`, which is `ceres` whenever one of the fallbacks of
            `colsfm.ba_backend` fired.
        """
        return self.ba_plan.backend

    @property
    def ba_fallback_reason(self) -> str | None:
        """Why the backend is not what was asked for, or None when it is.

        Returns:
            `ba_plan.fallback_reason`: an unsupported camera model, a solve asked to
            free `sensor_from_rig`, a pycolmap built without CASPAR_ENABLED, or a
            probe that could not run. Recorded rather than only printed, so
            `summary.json` says why a run that asked for the GPU backend solved on
            the CPU.
        """
        return self.ba_plan.fallback_reason

    @property
    def num_registered_images(self) -> int:
        """Images belonging to a registered frame, whether or not they see a point.

        Returns:
            The count `registered_image_ids` reports, which is every image of a
            `build_reconstruction` model.
        """
        return len(registered_image_ids(self.reconstruction))

    @property
    def num_images_with_observations(self) -> int:
        """Images that observe at least one point; what a COLMAP export writes out.

        Returns:
            `colsfm.reconstruction.num_registered_images` of this model — the other,
            narrower sense of "registered"; see that function.
        """
        return num_observing_images(self.reconstruction)

    @property
    def num_points3D(self) -> int:
        """Triangulated points that survived every filter.

        Returns:
            The point count.
        """
        return self.reconstruction.num_points3D()

    @property
    def num_observations(self) -> int:
        """Point-to-image observations in the final map.

        Returns:
            The observation count.
        """
        return self.reconstruction.compute_num_observations()

    @property
    def mean_reprojection_error_px(self) -> float:
        """Mean reprojection error over all observations, in pixels.

        Returns:
            The mean error; valid only after `update_point_3d_errors`, which
            `run_mapping` calls at the end of every round.
        """
        return self.reconstruction.compute_mean_reprojection_error()

    @property
    def mean_track_length(self) -> float:
        """Mean observations per point; cuSFM's `Mean length`.

        Returns:
            The mean track length.
        """
        return self.reconstruction.compute_mean_track_length()

    @property
    def refined_extrinsics(self) -> dict[int, pycolmap.Rigid3d] | None:
        """`vehicle_T_cam` per `camera_params_id` after the solve.

        Returns:
            The extrinsics read back out of the adjusted rig, or None when nothing
            refined them — either because `optimize_extrinsics` was off or because
            the rig is vehicle-referenced, which COLMAP would freeze anyway. The
            reference camera's entry is the input one exactly: it is the rig origin,
            so bundle adjustment holds no block for it.
        """
        if not self.extrinsics_refined or self.reference.camera_params_id is None:
            return None
        return self.reference.vehicle_T_cam_by_camera_params_id(self.reconstruction)


