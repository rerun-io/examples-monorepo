"""Metrics that compare two cuSFM-layout runs: the NVIDIA blob against `colsfm`.

Both runs are read from the *same* directory layout (`docs/spec/export.md`), so
nothing here knows which producer wrote which run:

| Artifact | Read for |
|---|---|
| `sparse/` COLMAP text model | registered images, points, observations, reprojection error, track length |
| `kpmap/keyframes/frames_meta.json` | rig trajectory, rig rigidity spread |
| `runtime.csv` | per-stage wall clock |

This module owns *what the report says*: reading a run (`read_run`), reducing it
to serialisable metrics (`compute_run_metrics`, `compare_runs`) and judging it
(`check_acceptance`). The pieces it is built from live next door and are
re-exported here, because `tools/audit` reaches for them through this name:

| Module | Owns |
|---|---|
| `colsfm.alignment` | rigid fits, rig trajectories, ground truth, timestamp joins |
| `colsfm.runtime` | `runtime.csv` rows folded onto the plan's eight stages |
| `colsfm.bench_report` | markdown rendering and the `number` cell formatter |

## Deliberate differences from `demo_rerun.py`'s metric definitions

1. **Reprojection error is recomputed.** `kpmap_to_colmap` hardcodes
   `ERROR = 2.0` on every point (export.md §3.5), so the column in
   `points3D.txt` is not a measurement. `pycolmap.Reconstruction.update_point_3d_errors`
   followed by `compute_mean_reprojection_error` re-derives it from the tracks.
2. **The model is read with `pycolmap.Reconstruction.read_text`,** not
   `colsfm.colmap_text_model.read_colmap_model`. The text reader exists so the
   demo need not carry pycolmap's dependency stack; where pycolmap is already
   present, its own reader is the one to use.
3. **The rig trajectory comes from `kpmap/keyframes/frames_meta.json`,** through
   `FramesMeta.rig_frames()` — see `colsfm.alignment` for why.
4. **`runtime.csv` is cut down to its last run** — see `colsfm.runtime`.
5. **An unfinished run is refused.** `read_run` reads `run_state.json` and
   raises unless it says `succeeded`, so a staging directory or a run that
   died halfway cannot be benchmarked as if it were whole
   (`colsfm.run_lifecycle`). A blob reference run has no such file and is
   read exactly as before.

## Acceptance is data-driven

`compare_runs` checks bounds when it is handed bounds and does not otherwise;
which dataset the runs reconstruct is a plain label. `check_acceptance` then
drops the checks the data cannot support: the ATE row without ground truth. A
bound whose value or whose limit is not a finite measurement — run A's ATE is
NaN when fewer than three samples joined, either run's total runtime when its
`runtime.csv` is missing — is reported as *not measurable*
(`AcceptanceCheck.passed is None`) rather than as a failure. Missing stays
missing: a run that timed nothing has `total_runtime_seconds is None`, never a
0.0 s total that would beat any runtime bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np
import pycolmap
from jaxtyping import Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from serde import serde
from serde.json import to_json

from colsfm.alignment import (
    DEGENERATE_VARIANCE,
    GROUND_TRUTH_FILE_NAME,
    GROUND_TRUTH_TOLERANCE_MICROSECONDS,
    Positions,
    RigidAlignment,
    RigTrack,
    Rotations,
    Timestamps,
    align_rigid,
    match_timestamps,
    path_length_meters,
    read_ground_truth,
    rig_rigidity_spread_millimeters,
    rig_track_from_frames_meta,
)
from colsfm.export import KEYFRAME_METADATA_SUBPATH, RUNTIME_CSV_NAME, SPARSE_DIR_NAME, RuntimeRecord
from colsfm.frames_meta import FRAMES_META_NAME, FramesMeta, read_frames_meta
from colsfm.geometry import MILLIMETRES_PER_METRE
from colsfm.reconstruction import num_registered_images, registered_image_names
from colsfm.run_lifecycle import RunState, read_run_state
from colsfm.runtime import STAGE_PATTERNS, STAGES, Stage, classify_stage, latest_run_records, read_stage_runtimes, stage_runtime_seconds
from colsfm.trajectory import MIN_ALIGNMENT_SAMPLES, TrajectoryScore, score_positions

__all__ = [
    "DEGENERATE_VARIANCE",
    "GROUND_TRUTH_FILE_NAME",
    "GROUND_TRUTH_TOLERANCE_MICROSECONDS",
    "MILLIMETRES_PER_METRE",
    "STAGES",
    "STAGE_PATTERNS",
    "AcceptanceBounds",
    "AcceptanceCheck",
    "Comparison",
    "PoseDelta",
    "Positions",
    "ReconstructionMetrics",
    "RigTrack",
    "RigidAlignment",
    "Rotations",
    "RunArtifacts",
    "RunMetrics",
    "RuntimeRecord",
    "Stage",
    "StageComparison",
    "StageRuntime",
    "Timestamps",
    "TrajectoryMetrics",
    "TrajectoryReference",
    "align_rigid",
    "check_acceptance",
    "classify_stage",
    "compare_runs",
    "compare_stage_runtimes",
    "compute_pose_delta",
    "compute_run_metrics",
    "latest_run_records",
    "match_timestamps",
    "path_length_meters",
    "read_ground_truth",
    "read_run",
    "reconstruction_metrics",
    "rig_rigidity_spread_millimeters",
    "rig_track_from_frames_meta",
    "stage_runtime_seconds",
    "write_json_report",
]
"""The facade `tools/audit` and the CLI import through; most names are defined next door."""

TrajectoryReference: TypeAlias = Literal["input", "ground_truth"]
"""What a run's trajectory was scored against: the dataset's input poses, or `ground_truth.txt`."""


@runtime_checkable
class SerdeObject(Protocol):
    """Anything pyserde can serialise: a `@serde`-decorated class stamps `__serde__` on itself."""

    __serde__: ClassVar[object]


# ─────────────────────────────────────────────────────────────────────────────
# What one run holds
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """Everything read off one cuSFM-layout run directory, arrays included.

    Kept apart from `RunMetrics` because it carries numpy arrays and a
    `pycolmap.Reconstruction`, neither of which belongs in a serialised report.
    """

    name: str
    """Short label used in the report and in Rerun metadata."""
    run_dir: Path
    """The `.../cusfm` directory the artifacts were read from."""
    reconstruction: pycolmap.Reconstruction
    """The `sparse/` model, with `update_point_3d_errors` already applied."""
    frames_meta: FramesMeta
    """`kpmap/keyframes/frames_meta.json` — the optimised poses and the calibration."""
    track: RigTrack
    """Rig trajectory over the samples the reconstruction registered."""
    runtime_seconds_by_stage: dict[Stage, float]
    """Wall clock per stage, from `runtime.csv`."""

    def points_xyz(self) -> Positions:
        """Sparse point positions in the run's world frame.

        Read off `self.reconstruction` on demand rather than cached: only the
        Rerun logger wants the arrays, and caching them would double the memory a
        RoboCap run's 168 000 points cost for every other caller.

        Returns:
            Float64 positions with shape `[n_points, 3]`, in metres.
        """
        return np.asarray([point.xyz for point in self.reconstruction.points3D.values()], dtype=np.float64).reshape(-1, 3)

    def points_rgb(self) -> Int64[ndarray, "n_points 3"]:
        """Per-point colour, all zero when the run omitted `--output_rgb` (gotcha 11).

        Returns:
            Int64 colours with shape `[n_points, 3]`, 0-255 per channel.
        """
        return np.asarray([point.color for point in self.reconstruction.points3D.values()], dtype=np.int64).reshape(-1, 3)


def read_run(run_dir: Path, name: str) -> RunArtifacts:
    """Read one cuSFM-layout run directory.

    Args:
        run_dir: A `.../cusfm` directory holding `sparse/`, `kpmap/` and `runtime.csv`.
        name: Short label for the run.

    Returns:
        The artifacts, with point errors already recomputed from the tracks.

    Raises:
        FileNotFoundError: When the sparse model or the optimised metadata is missing.
        ValueError: When the directory says it holds an unfinished run.
    """
    state: RunState | None = read_run_state(run_dir)
    # A `pycusfm` reference run has no `run_state.json` at all, and is read as it
    # always was. A colsfm run has one, and a published one always says `succeeded`
    # -- `colsfm.run_lifecycle` publishes by rename, so anything else here is a
    # staging directory being read directly.
    if state is not None and state.status != "succeeded":
        raise ValueError(
            f"{run_dir} holds a run that {state.status}"
            + (f" ({state.failure})" if state.failure else "")
            + f"; it completed {list(state.stages_completed)} of {list(state.stages_planned)}"
        )
    sparse_dir: Path = run_dir / SPARSE_DIR_NAME
    meta_path: Path = run_dir / KEYFRAME_METADATA_SUBPATH
    if not (sparse_dir / "images.txt").is_file() and not (sparse_dir / "images.bin").is_file():
        raise FileNotFoundError(f"no COLMAP model under {sparse_dir}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"no optimised metadata at {meta_path}")

    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    # kpmap_to_colmap hardcodes ERROR = 2.0 (export.md §3.5); recompute from the tracks.
    reconstruction.update_point_3d_errors()
    frames_meta: FramesMeta = read_frames_meta(meta_path)

    registered_names: set[str] = registered_image_names(reconstruction)
    return RunArtifacts(
        name=name,
        run_dir=run_dir,
        reconstruction=reconstruction,
        frames_meta=frames_meta,
        track=rig_track_from_frames_meta(frames_meta, keep_image_names=registered_names),
        runtime_seconds_by_stage=read_stage_runtimes(run_dir / RUNTIME_CSV_NAME),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Serialisable metrics
# ─────────────────────────────────────────────────────────────────────────────


@serde
@dataclass(frozen=True)
class ReconstructionMetrics:
    """What a sparse model alone says, with nothing read off the metadata."""

    registered_images: int
    """Images with at least one 3D observation."""
    num_points3D: int
    """Triangulated points in the sparse model."""
    num_observations: int
    """Total 2D-3D observations across every track."""
    mean_reprojection_error_px: float
    """Recomputed from the model, never read from the `ERROR` column."""
    track_length_mean: float
    """Mean observations per point."""
    track_length_median: float
    """Median observations per point."""


@serde
@dataclass(frozen=True)
class TrajectoryMetrics:
    """Alignment of one run's rig trajectory onto a reference trajectory."""

    reference: TrajectoryReference
    """What it was compared against."""
    num_matched: int
    """Samples that joined on timestamp."""
    rmse_millimeters: float
    """Root-mean-square position error after a rigid fit with scale held at 1.0."""
    max_millimeters: float
    """Largest single position error after the same fit."""
    would_be_scale: float
    """Scale a similarity fit would have chosen; 1.0 means metric scale survived."""
    reference_path_length_meters: float
    """Distance travelled along the reference over the matched samples."""


@serde
@dataclass(frozen=True)
class StageRuntime:
    """One stage's wall clock inside a single run."""

    stage: Stage
    """Canonical stage name."""
    seconds: float
    """Summed wall clock of every `runtime.csv` row that mapped to this stage."""


@serde
@dataclass(frozen=True)
class RunMetrics:
    """Everything the report says about one run."""

    name: str
    """Short label for the run."""
    run_dir: str
    """The directory the metrics were read from."""
    reconstruction: ReconstructionMetrics
    """What the `sparse/` model says on its own."""
    total_images: int
    """Keyframes the run was given, from `kpmap/keyframes/frames_meta.json`."""
    rig_rigidity_spread_millimeters: float
    """Worst per-sample disagreement between the cameras' rig-pose estimates."""
    num_rig_frames: int
    """Synchronised samples with at least one registered image."""
    stage_runtimes: tuple[StageRuntime, ...]
    """Per-stage wall clock, in pipeline order; missing stages are omitted."""
    total_runtime_seconds: float | None
    """Sum of every stage that reported a runtime, or None when none did.

    Deliberately not 0.0: a run whose `runtime.csv` is missing or empty was not
    measured, and summing nothing into zero made it the fastest run possible."""
    points_are_black: bool
    """True when every point colour is `(0, 0, 0)` — the run omitted `--output_rgb`."""
    vs_input: TrajectoryMetrics
    """Disagreement against the input trajectory in the dataset's `frames_meta.json`."""
    vs_ground_truth: TrajectoryMetrics | None
    """ATE against `ground_truth.txt`, or None when the dataset ships none."""


@serde
@dataclass(frozen=True)
class PoseDelta:
    """Direct disagreement between the two runs' rig poses."""

    num_matched: int
    """Samples present in both runs, joined on timestamp."""
    position_rmse_millimeters: float
    """Root-mean-square position difference after a rigid fit of B onto A."""
    position_max_millimeters: float
    """Largest single position difference after the same fit."""
    rotation_rmse_degrees: float
    """Root-mean-square orientation difference after the same fit."""
    rotation_max_degrees: float
    """Largest single orientation difference after the same fit."""


@serde
@dataclass(frozen=True)
class StageComparison:
    """One row of the per-stage runtime table."""

    stage: Stage
    """Canonical stage name; the report looks its human label up at render time."""
    seconds_a: float | None
    """Run A's wall clock, or None when it reported no row for this stage."""
    seconds_b: float | None
    """Run B's wall clock, or None when it reported no row for this stage."""
    ratio_b_over_a: float | None
    """`seconds_b / seconds_a`, or None when either side is missing or A is zero."""


@serde
@dataclass(frozen=True)
class AcceptanceBounds:
    """The Galileo bounds of `docs/open-pipeline-plan.md` §Acceptance.

    Every bound is **relative to run A as this harness measures it**, not absolute.
    The plan's original absolute figures (ATE <= 5 mm, reprojection <= 1.7 px) came
    from an older NOTES table; measured here the blob reference itself scores
    5.00 mm, so an absolute 5 mm bound asks the port to beat the thing it is
    reproducing. Relative bounds also survive a new machine or a re-run of A, and
    they carry over to any dataset — `check_acceptance` drops the rows the data
    cannot support rather than the bounds being Galileo's alone.
    """

    registered_images_allowance: int = 4
    """Run B may register up to this many fewer images than run A."""
    max_ate_ratio: float = 1.10
    """Run B's ATE against `ground_truth.txt`, over run A's."""
    max_reprojection_ratio: float = 1.10
    """Run B's recomputed mean reprojection error, over run A's."""
    max_runtime_ratio: float = 2.0
    """Run B's total runtime must stay within this multiple of run A's."""


@serde
@dataclass(frozen=True)
class AcceptanceCheck:
    """One bound, evaluated against run B."""

    name: str
    """What was checked."""
    bound: str
    """The bound as written in the plan, e.g. `">= 220"`."""
    value: float | None
    """The measured value, or **None when there was nothing to measure** — a
    candidate that reported no stage runtime has no total, and a total of zero
    would read as an infinitely fast run."""
    passed: bool | None
    """Whether the measurement satisfies the bound, or **None when it could not be
    measured** — run A's ATE is NaN when fewer than three samples joined, which
    makes the bound NaN too. A check that could not be evaluated is not a failure:
    the report prints `n/a` for it and the `n/n bounds met` tally skips it."""


@serde
@dataclass(frozen=True)
class Comparison:
    """The complete benchmark result for one dataset and one pair of runs."""

    dataset: str
    """Dataset label, e.g. `galileo`; purely descriptive."""
    run_a: RunMetrics
    """Reference run — the NVIDIA blob."""
    run_b: RunMetrics
    """Candidate run — the `colsfm` pipeline."""
    pose_delta: PoseDelta
    """Rig-pose disagreement between the two runs."""
    stages: tuple[StageComparison, ...]
    """Per-stage runtime table, in pipeline order."""
    total_runtime_ratio: float | None
    """Run B's total runtime over run A's, or None when A reported nothing."""
    acceptance: tuple[AcceptanceCheck, ...]
    """Bounds checked against B; empty when `compare_runs` was given no bounds."""


# ─────────────────────────────────────────────────────────────────────────────
# Computing the metrics
# ─────────────────────────────────────────────────────────────────────────────


def reconstruction_metrics(reconstruction: pycolmap.Reconstruction) -> ReconstructionMetrics:
    """Reduce a sparse model to the numbers that need no metadata to interpret.

    The reprojection error is recomputed from the tracks; pass a model that has
    had `update_point_3d_errors` applied, as `read_run` does, or the `ERROR`
    column's hardcoded 2.0 will be what `compute_mean_reprojection_error` returns.

    Args:
        reconstruction: A COLMAP model.

    Returns:
        The model's metrics; track lengths are 0.0 for a model with no points.
    """
    track_lengths: Int64[ndarray, "n_points"] = np.asarray(
        [point.track.length() for point in reconstruction.points3D.values()], dtype=np.int64
    )
    return ReconstructionMetrics(
        registered_images=num_registered_images(reconstruction),
        num_points3D=reconstruction.num_points3D(),
        num_observations=reconstruction.compute_num_observations(),
        mean_reprojection_error_px=float(reconstruction.compute_mean_reprojection_error()),
        track_length_mean=float(track_lengths.mean()) if len(track_lengths) else 0.0,
        track_length_median=float(np.median(track_lengths)) if len(track_lengths) else 0.0,
    )


def _trajectory_metrics(
    reference_name: TrajectoryReference, track: RigTrack, reference: RigTrack, tolerance_microseconds: int
) -> TrajectoryMetrics:
    """Join two trajectories on timestamp and align the run onto the reference.

    The join and the fit are `colsfm.trajectory`'s: `score_positions` owns the rigid
    alignment, the millimetre conversion and the too-few-samples answer, and this
    function used to carry its own copy of all three. Only the reference path length
    is local, because only the benchmark's report has a column for it.
    """
    track_indices, reference_indices = match_timestamps(
        track.timestamps_microseconds, reference.timestamps_microseconds, tolerance_microseconds
    )
    source_xyz: Positions = track.world_t_rig[track_indices]
    target_xyz: Positions = reference.world_t_rig[reference_indices]
    score: TrajectoryScore = score_positions(source_xyz, target_xyz)
    return TrajectoryMetrics(
        reference=reference_name,
        num_matched=score.num_matched,
        rmse_millimeters=score.rigid_rmse_millimeters,
        max_millimeters=score.rigid_max_millimeters,
        would_be_scale=score.would_be_scale,
        # 0.0 rather than a path along too few points: an unfitted row has no reference
        # length to report, which is what the NaN residuals above already say.
        reference_path_length_meters=(
            path_length_meters(target_xyz) if score.num_matched >= MIN_ALIGNMENT_SAMPLES else 0.0
        ),
    )


def compute_run_metrics(artifacts: RunArtifacts, input_track: RigTrack, ground_truth: RigTrack | None) -> RunMetrics:
    """Reduce one run's artifacts to the numbers the report prints.

    Args:
        artifacts: A run read by `read_run`.
        input_track: Rig trajectory of the dataset's input `frames_meta.json`.
        ground_truth: Ground-truth rig trajectory, or None.

    Returns:
        The run's metrics.
    """
    points_rgb: Int64[ndarray, "n_points 3"] = artifacts.points_rgb()
    stage_runtimes: tuple[StageRuntime, ...] = tuple(
        StageRuntime(stage=stage, seconds=artifacts.runtime_seconds_by_stage[stage])
        for stage in STAGES
        if stage in artifacts.runtime_seconds_by_stage
    )
    return RunMetrics(
        name=artifacts.name,
        run_dir=str(artifacts.run_dir),
        reconstruction=reconstruction_metrics(artifacts.reconstruction),
        total_images=len(artifacts.frames_meta.keyframes),
        rig_rigidity_spread_millimeters=rig_rigidity_spread_millimeters(artifacts.frames_meta),
        num_rig_frames=len(artifacts.track),
        stage_runtimes=stage_runtimes,
        total_runtime_seconds=(
            float(sum(artifacts.runtime_seconds_by_stage.values())) if artifacts.runtime_seconds_by_stage else None
        ),
        points_are_black=bool(len(points_rgb) > 0 and not points_rgb.any()),
        vs_input=_trajectory_metrics("input", artifacts.track, input_track, 0),
        vs_ground_truth=(
            None
            if ground_truth is None
            else _trajectory_metrics("ground_truth", artifacts.track, ground_truth, GROUND_TRUTH_TOLERANCE_MICROSECONDS)
        ),
    )


def compute_pose_delta(track_a: RigTrack, track_b: RigTrack) -> PoseDelta:
    """Compare the two runs' rig poses sample by sample.

    A rigid fit of B onto A is applied first. The two runs inherit the same input
    world frame, so the fit is usually near identity, but a gauge difference
    between two independent bundle adjustments would otherwise dominate a
    "direct" difference and hide the shape disagreement, which is what this is
    meant to measure.

    Args:
        track_a: Reference run's rig trajectory.
        track_b: Candidate run's rig trajectory.

    Returns:
        Position and orientation disagreement over the samples both runs hold.
    """
    indices_b, indices_a = match_timestamps(track_b.timestamps_microseconds, track_a.timestamps_microseconds, 0)
    if len(indices_b) < 3:
        return PoseDelta(
            num_matched=len(indices_b),
            position_rmse_millimeters=float("nan"),
            position_max_millimeters=float("nan"),
            rotation_rmse_degrees=float("nan"),
            rotation_max_degrees=float("nan"),
        )
    source_xyz: Positions = track_b.world_t_rig[indices_b]
    target_xyz: Positions = track_a.world_t_rig[indices_a]
    alignment: RigidAlignment = align_rigid(source_xyz, target_xyz)
    aligned_R: Rotations = alignment.apply_rotations(track_b.world_R_rig[indices_b])
    relative_R: Rotations = np.einsum("nij,nkj->nik", aligned_R, track_a.world_R_rig[indices_a])
    angles_degrees: Float64[ndarray, "m"] = np.rad2deg(np.linalg.norm(Rotation.from_matrix(relative_R).as_rotvec(), axis=-1))
    return PoseDelta(
        num_matched=len(indices_b),
        position_rmse_millimeters=MILLIMETRES_PER_METRE * alignment.rmse_meters,
        position_max_millimeters=MILLIMETRES_PER_METRE * alignment.max_error_meters,
        rotation_rmse_degrees=float(np.sqrt((angles_degrees**2).mean())),
        rotation_max_degrees=float(angles_degrees.max()),
    )


def compare_stage_runtimes(run_a: RunArtifacts, run_b: RunArtifacts) -> tuple[StageComparison, ...]:
    """Build the per-stage runtime table.

    Args:
        run_a: Reference run.
        run_b: Candidate run.

    Returns:
        One row per stage of `STAGES`, in pipeline order, including stages
        neither run reported (both columns None) so the table shape is stable.
    """
    rows: list[StageComparison] = []
    for stage in STAGES:
        seconds_a: float | None = run_a.runtime_seconds_by_stage.get(stage)
        seconds_b: float | None = run_b.runtime_seconds_by_stage.get(stage)
        ratio: float | None = seconds_b / seconds_a if seconds_a and seconds_b is not None else None
        rows.append(StageComparison(stage=stage, seconds_a=seconds_a, seconds_b=seconds_b, ratio_b_over_a=ratio))
    return tuple(rows)


def _verdict(value: float, limit: float, *, at_most: bool) -> bool | None:
    """Judge a measurement against a bound, or declare it unmeasurable.

    Args:
        value: The measured value.
        limit: The bound to compare it against.
        at_most: True for `value <= limit`, False for `value >= limit`.

    Returns:
        The verdict, or None when either side is not finite — a run that matched
        too few samples to have an ATE has not failed the ATE bound, it has left
        it unmeasured.
    """
    if not (np.isfinite(value) and np.isfinite(limit)):
        return None
    return bool(value <= limit) if at_most else bool(value >= limit)


def check_acceptance(
    metrics_a: RunMetrics,
    metrics_b: RunMetrics,
    total_runtime_ratio: float | None,
    bounds: AcceptanceBounds,
) -> tuple[AcceptanceCheck, ...]:
    """Evaluate the plan's bounds, every one of them relative to run A.

    Args:
        metrics_a: Reference run's metrics — the blob, measured by this harness.
        metrics_b: Candidate run's metrics.
        total_runtime_ratio: Candidate total over reference total, or None.
        bounds: The bounds to apply.

    Returns:
        One check per bound the data supports; the ATE check is omitted when no
        ground truth exists. The runtime check is always present, because a
        candidate with no timings has not met the bound — it has left it
        unmeasured, and saying so is the point. A check whose value or bound is
        missing or not finite comes back with `passed=None`.
    """
    min_registered: int = metrics_a.reconstruction.registered_images - bounds.registered_images_allowance
    max_reprojection_px: float = bounds.max_reprojection_ratio * metrics_a.reconstruction.mean_reprojection_error_px
    checks: list[AcceptanceCheck] = [
        AcceptanceCheck(
            name="registered images",
            bound=f">= A - {bounds.registered_images_allowance} = {min_registered}",
            value=float(metrics_b.reconstruction.registered_images),
            passed=_verdict(float(metrics_b.reconstruction.registered_images), float(min_registered), at_most=False),
        ),
        AcceptanceCheck(
            name="mean reprojection error (px)",
            bound=f"<= {bounds.max_reprojection_ratio:.2f} x A = {max_reprojection_px:.3f}",
            value=metrics_b.reconstruction.mean_reprojection_error_px,
            passed=_verdict(metrics_b.reconstruction.mean_reprojection_error_px, max_reprojection_px, at_most=True),
        ),
    ]
    if metrics_a.vs_ground_truth is not None and metrics_b.vs_ground_truth is not None:
        max_ate_millimeters: float = bounds.max_ate_ratio * metrics_a.vs_ground_truth.rmse_millimeters
        checks.append(
            AcceptanceCheck(
                name="ATE vs ground truth (mm)",
                bound=f"<= {bounds.max_ate_ratio:.2f} x A = {max_ate_millimeters:.3f}",
                value=metrics_b.vs_ground_truth.rmse_millimeters,
                passed=_verdict(metrics_b.vs_ground_truth.rmse_millimeters, max_ate_millimeters, at_most=True),
            )
        )
    checks.append(
        AcceptanceCheck(
            name="total runtime ratio",
            bound=f"<= {bounds.max_runtime_ratio} x A",
            value=total_runtime_ratio,
            passed=(
                None
                if total_runtime_ratio is None
                else _verdict(total_runtime_ratio, bounds.max_runtime_ratio, at_most=True)
            ),
        )
    )
    return tuple(checks)


def compare_runs(
    run_a: RunArtifacts,
    run_b: RunArtifacts,
    input_dir: Path,
    dataset: str,
    bounds: AcceptanceBounds | None = None,
) -> Comparison:
    """Compute every metric for one pair of runs.

    Args:
        run_a: Reference run, read by `read_run` (the NVIDIA blob).
        run_b: Candidate run, read by `read_run` (the `colsfm` pipeline).
        input_dir: Directory holding the dataset's input `frames_meta.json` and,
            when the dataset ships one, `ground_truth.txt`.
        dataset: Dataset label for the report heading; no behaviour depends on it.
        bounds: Acceptance bounds to check B against, or None to check none.

    Returns:
        The full comparison, ready for `colsfm.bench_report`.
    """
    input_meta: FramesMeta = read_frames_meta(input_dir / FRAMES_META_NAME)
    input_track: RigTrack = rig_track_from_frames_meta(input_meta)
    ground_truth: RigTrack | None = read_ground_truth(input_dir)

    metrics_a: RunMetrics = compute_run_metrics(run_a, input_track, ground_truth)
    metrics_b: RunMetrics = compute_run_metrics(run_b, input_track, ground_truth)
    seconds_a: float | None = metrics_a.total_runtime_seconds
    seconds_b: float | None = metrics_b.total_runtime_seconds
    ratio: float | None = seconds_b / seconds_a if seconds_a and seconds_b is not None else None
    return Comparison(
        dataset=dataset,
        run_a=metrics_a,
        run_b=metrics_b,
        pose_delta=compute_pose_delta(run_a.track, run_b.track),
        stages=compare_stage_runtimes(run_a, run_b),
        total_runtime_ratio=ratio,
        acceptance=() if bounds is None else check_acceptance(metrics_a, metrics_b, ratio, bounds),
    )


def write_json_report(path: Path, report: SerdeObject) -> Path:
    """Dump any `@serde` result as indented JSON.

    Deliberately not typed to `Comparison`: the audit tools under `tools/audit`
    each produce their own `@serde` result and all of them want the same two
    decisions — create the parent directory, and write `indent=2` with a trailing
    newline so the file survives a diff.

    It lives here, next to the dataclasses it serialises, rather than in
    `colsfm.bench_report` with the markdown renderer, because `bench_report`
    imports this module and the reverse import would close a cycle.

    Args:
        path: Destination file; parent directories are created.
        report: A `@serde`-decorated object, e.g. a `Comparison`.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(report, indent=2) + "\n")
    return path
