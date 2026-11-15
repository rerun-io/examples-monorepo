"""The cuSFM stages wired into one run — `colsfm`'s `cusfm_runner.run_all`.

```
frames_meta.json ─1─> keyframes/frames_meta.json
                 ─2─> database.db (ALIKED keypoints + descriptors)
                 ─3─> pair list (consecutive + stereo)
                 ─4─> LightGlue matches + two-view geometries in the database
                 ─5─> loop edges (optional; their pairs are matched into the same database)
                 ─6─> pose_graph/{frames_meta.json,vehicle_pose.tum}
                 ─7─> triangulation + bundle adjustment
                 ─7b> the same again with the rig extrinsics free (--optimize-extrinsics)
                 ─8─> sparse/, kpmap/keyframes/frames_meta.json, output_poses/, summary.json
```

The contract mirrors `pycusfm.cusfm_runner.CusfmRunner`: the same `--input-dir`
holding `frames_meta.json` and the images it names, the same output directory
layout (`pycusfm/constants.py`), and the same `runtime.csv` — two columns,
`command,runtime_seconds` — so a colsfm run and a blob run compare row by row.
The `command` column holds the stage name rather than a blob command line;
nothing reads it as a command, and the stage name is what a benchmark table wants.

## Where things live

| module | owns |
|---|---|
| `colsfm.pipeline` | `PipelineObserver` and `run_pipeline`, the one runner |
| `colsfm.run_config` | `PipelineOptions`, `SelectionOptions`, `ResolvedRun` |
| `colsfm.run_lifecycle` | the stage ledger, the workspace and the publication boundary |
| `colsfm.run_report` | `PipelineSummary`, `MappingStats`, `LoopEdgeRecord` |
| `colsfm.stages` | one `run_*_stage` function per stage, and their results |

## One function per stage

Each stage is a `run_<stage>_stage` function taking what it needs and returning a
small result dataclass; `run_pipeline` only composes them and times each with
`timed_stage`. The stage functions carry no timing and write no `runtime.csv`
row, so a caller that wants to step through a dataset one stage at a time — the
Rerun walkthrough, `run_stage`, a notebook — reuses exactly the code the full run
executes rather than a parallel copy of it.

Ordering notes worth knowing before reading the code:

* **Pair selection cannot see loop pairs.** Stage 3 emits the consecutive and
  stereo pairs from the metadata alone (`colsfm.pairs`); loop pairs only exist
  after retrieval and verification in stage 5, which matches them into the same
  database. `colsfm.mapping.load_correspondences` reads *every* verified
  two-view geometry, so a loop pair matched in stage 5 reaches the triangulator
  without being named in stage 3's list.
* **Loop closure needs matches to verify, and matching is a batch operation.**
  The estimator pulls matches through an injected `match_fn` one pair at a time,
  and running COLMAP's matcher per pair would reload the LightGlue graph per
  call. So the stage plans first: `colsfm.loop_closure.plan_loop_search` does the
  retrieval half of the funnel — which reads no match at all — and names the
  image pairs the measurement will ask for; one batch
  `colsfm.matching.match_pairs` fills in whichever of them the database does not
  already hold; `verify_loop_plan` then measures and gates the plan, reading the
  matches back. Presence decides what to match, not count: a pair stage 4
  matched and found nothing in is finished work.
* **`--optimize-extrinsics` maps once, not twice.** On by default. The camera-referenced
  reconstruction is built up front when the flag is set, stage 7 maps it, and
  stage 7b refines the extrinsics of the model stage 7 left behind. Both
  references give the same `cam_T_world`, so nothing about stage 7's result
  depends on which one was used (`colsfm.reconstruction`).
* **The two ONNX stages have three backends each.** `--features-backend` and
  `--matching-backend` pick between RaCo-ALIKED and LightGlue+ on their own
  engines (`colsfm.features_raco`, `colsfm.matching_raco`, the default since
  NOTES.md decision 17), COLMAP's own ALIKED and LightGlue, and the blob's graphs
  on the blob's TensorRT engines (`colsfm.features_trt`, `colsfm.matching_trt`).
  They are independent, every combination runs, and the choice is invisible past
  the database: all write the same keypoint and descriptor columns and leave the
  same matches and two-view geometries. The `pycolmap` matcher is the only one
  that cannot see a per-match score, so it is the only one that substitutes a grid
  subsample for the blob's real SSC spatial NMS.
* **`--use-gpu` governs the ONNX stages only.** ALIKED and LightGlue run on the
  GPU; the Ceres solves stay on the CPU by default, as the blob's own
  `keypoints_mapper_main` and `pose_graph_main` do (docs/open-pipeline-plan.md,
  "Facts that shape the design"). It is read by the `pycolmap` backends only —
  the two engine-backed ones run on CUDA device 0 or not at all. `--ba-use-gpu` moves the
  bundle-adjustment linear solve onto the GPU; measured it buys nothing on Galileo (1.68 s against
  1.60 s) and about 8 % on 800 RoboCap images (19.2 s against 20.9 s), so the
  default stays where the blob is.
* **Threads are not the blob's `--num_thread 1`.** That flag counts worker
  *processes*, and the runner starts one per camera; COLMAP is one process, so
  `--num-threads` defaults to -1 (every core) for extraction and matching, and
  Ceres takes the config's 8 unless `--ba-num-threads` says otherwise.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace

from serde.json import to_json

from colsfm.config import CusfmConfig
from colsfm.database import (
    ImagePair,
)
from colsfm.features import ExtractionReport
from colsfm.frames_meta import FRAMES_META_NAME, FramesMeta, RigFrame, write_frames_meta
from colsfm.mapping import MappingOptions, MappingResult
from colsfm.matching import (
    MatchingOptions,
    MatchReport,
)
from colsfm.run_config import PipelineOptions, ResolvedRun, SelectionOptions, resolve_run
from colsfm.run_lifecycle import (
    EXTRINSIC_REFINEMENT_STAGE,
    KEYFRAME_DIR_NAME,
    SUMMARY_NAME,
    CheapStageName,
    RunWorkspace,
    StageLedger,
    StageName,
    git_sha,
    open_workspace,
    stage_names,
    timed_stage,
)
from colsfm.run_report import (
    ExtrinsicChange,
    MappingStats,
    PipelineSummary,
)
from colsfm.stages import (
    ExportStageResult,
    ExtrinsicRefinementStageResult,
    KeyframeSelectionStageResult,
    LoopClosureStageResult,
    PoseGraphStageResult,
    run_export_stage,
    run_extrinsic_refinement_stage,
    run_feature_extraction_stage,
    run_keyframe_selection_stage,
    run_loop_closure_stage,
    run_matching_stage,
    run_pair_selection_stage,
    run_pose_graph_stage,
    run_reconstruction_stage,
)

# ══════════════════════════════════════════════════════════════════════════════════════
# the observer
# ══════════════════════════════════════════════════════════════════════════════════════


class PipelineObserver:
    """What a bystander is told while `run_pipeline` runs; every method a no-op.

    The one seam a viewer, a notebook or a report needs to watch a run without
    re-sequencing it. Each `on_*` method is handed the stage's own typed result,
    *after* its timing window has closed, so nothing an observer does lands in
    `runtime.csv`; `stage_scope` is the only hook inside the window, and exists
    because the stage functions print their counters rather than returning them.

    Two ordering guarantees the visualisation depends on:

    * `on_reconstruction` runs before stage 7b starts, so an observer that wants
      the un-refined model sees it before the refinement mutates it in place.
    * `on_export` runs after `run_export_stage` has written every artifact, so a
      reader may open them.

    Subclass and override what you need. The base class is what `run_pipeline`
    uses when no observer is passed.
    """

    def run_started(self, options: PipelineOptions, stages: tuple[StageName, ...]) -> None:
        """Announce the run before stage 1.

        Args:
            options: The options the stages will execute, with `output_dir` already
                rebased onto the run's staging directory — so an observer that opens
                the database or reads an artifact mid-run looks where the run is
                actually writing. `summary.options` keeps the caller's own paths.
            stages: The stages it will record, in order.
        """

    def stage_scope(self, stage: StageName) -> contextlib.AbstractContextManager[None]:
        """Wrap one stage's body, inside its timing window.

        Args:
            stage: The stage about to run.

        Returns:
            A context manager entered around the stage call; the default does
            nothing. An observer that wants a stage's console output redirects
            stdout here.
        """
        return contextlib.nullcontext()

    def on_keyframe_selection(self, result: KeyframeSelectionStageResult) -> None:
        """Stage 1 finished.

        Args:
            result: The configuration read, the input collection and the selection.
        """

    def on_feature_extraction(self, report: ExtractionReport) -> None:
        """Stage 2 finished.

        Args:
            report: Per-image keypoint counts, the device used and the wall time.
        """

    def on_pair_selection(self, pairs: Sequence[ImagePair]) -> None:
        """Stage 3 finished.

        Args:
            pairs: The consecutive and stereo pairs the matcher will be given.
        """

    def on_matching(self, report: MatchReport) -> None:
        """Stage 4 finished.

        Args:
            report: Per-pair raw and inlier counts, the device used and the wall time.
        """

    def on_loop_closure(self, result: LoopClosureStageResult) -> None:
        """Stage 5 finished.

        Args:
            result: The gated loop edges, the extra pairs matched and the diagnostics.
        """

    def on_pose_graph(self, result: PoseGraphStageResult) -> None:
        """Stage 6 finished.

        Args:
            result: The nodes, the edges, the re-posed collection and the largest move.
        """

    def on_reconstruction(self, mapping: MappingResult) -> None:
        """Stage 7 finished, before stage 7b may refine the same model in place.

        Args:
            mapping: The triangulated, bundle-adjusted model and its round statistics.
        """

    def on_extrinsic_refinement(self, result: ExtrinsicRefinementStageResult) -> None:
        """Stage 7b finished; only called when `--optimize-extrinsics` is set.

        Args:
            result: The refined model, the re-calibrated collection and the movement.
        """

    def on_export(self, result: ExportStageResult, mapping: MappingResult) -> None:
        """Stage 8 finished and every artifact but `summary.json` is on disk.

        Args:
            result: The exported collection and what was written.
            mapping: The final mapping result the export was made from.
        """

    def run_finished(self, summary: PipelineSummary) -> None:
        """The run is over and `summary.json` is written.

        Args:
            summary: What the run produced and how long each stage took.
        """


# ══════════════════════════════════════════════════════════════════════════════════════
# the run
# ══════════════════════════════════════════════════════════════════════════════════════


def run_pipeline(options: PipelineOptions, observer: PipelineObserver | None = None) -> PipelineSummary:
    """Run every stage and write cuSFM's outputs into `options.output_dir`.

    The only sequencer in the package: a caller that wants to watch a run — the
    Rerun walkthrough, a notebook — passes an observer rather than composing the
    stage functions itself.

    Args:
        options: Input, output, config and the per-stage knobs.
        observer: Told about each stage as it completes; None watches nothing.

    Returns:
        The run's summary, which is also written to `<output_dir>/summary.json`.

    Raises:
        FileNotFoundError: When the input metadata or a config file is missing.
        ValueError: When a stage cannot proceed, e.g. no keyframe survives selection.
    """
    watcher: PipelineObserver = PipelineObserver() if observer is None else observer
    started: float = time.perf_counter()
    if not options.frames_meta_path.is_file():
        raise FileNotFoundError(f"No {FRAMES_META_NAME} at {options.frames_meta_path}")
    # Resolved and validated before the workspace exists: a `--caspar-option` typo
    # used to surface at the first bundle adjustment, after feature extraction,
    # matching and loop closure had already been paid for and the previous run's
    # database had already been deleted.
    resolved: ResolvedRun = resolve_run(options)
    stages: tuple[StageName, ...] = stage_names(options.optimize_extrinsics)
    workspace: RunWorkspace = open_workspace(options.output_dir, stages)
    # Every stage writes into the staging directory, the database included, because
    # the stages take an options object whose `output_dir` is that directory. The
    # summary below reports `options`, i.e. the paths the caller asked for.
    staged: PipelineOptions = replace(options, output_dir=workspace.staging_dir)
    print(f"[colsfm] config {options.config_dir} | input {options.input_dir} | output {options.output_dir}")
    watcher.run_started(staged, stages)
    try:
        summary: PipelineSummary = _run_stages(options, staged, resolved, workspace.ledger, watcher, started)
    except BaseException as error:
        workspace.fail(f"{type(error).__name__}: {error}")
        raise
    workspace.publish()
    watcher.run_finished(summary)
    return summary


def _run_stages(
    options: PipelineOptions,
    staged: PipelineOptions,
    resolved: ResolvedRun,
    ledger: StageLedger,
    watcher: PipelineObserver,
    started: float,
) -> PipelineSummary:
    """Run every stage into the staging workspace and write its summary there.

    Split out of `run_pipeline` so that the publication boundary is one `try` around
    one call rather than a `finally` wrapped around nine stages.

    Args:
        options: The caller's own options, which the summary reports.
        staged: The same options with `output_dir` on the staging directory, which
            every stage writes into.
        resolved: The per-stage settings, resolved once.
        ledger: The run's stage ledger.
        watcher: The observer, told about each stage as it completes.
        started: `time.perf_counter()` at the start of the run.

    Returns:
        The summary, already written to the staging directory.

    Raises:
        FileNotFoundError: When a config file is missing.
        ValueError: When a stage cannot proceed.
    """

    # ── 1. keyframe selection ────────────────────────────────────────────────────────
    with timed_stage(ledger, "keyframe_selection"), watcher.stage_scope("keyframe_selection"):
        selection: KeyframeSelectionStageResult = run_keyframe_selection_stage(resolved.selection)
        write_frames_meta(staged.output_dir / KEYFRAME_DIR_NAME / FRAMES_META_NAME, selection.selected)
        print(
            f"[colsfm] selected {len(selection.selected.keyframes)}/{len(selection.frames_meta.keyframes)} keyframes "
            f"in {len(selection.rig_frames)} rig frames over {len(selection.selected.cameras)} cameras"
        )
    watcher.on_keyframe_selection(selection)
    config: CusfmConfig = selection.config

    # ── 2. feature extraction ────────────────────────────────────────────────────────
    with timed_stage(ledger, "feature_extraction"), watcher.stage_scope("feature_extraction"):
        extraction: ExtractionReport = run_feature_extraction_stage(staged, selection.selected, resolved.features)
    watcher.on_feature_extraction(extraction)

    # ── 3. pair selection ────────────────────────────────────────────────────────────
    with timed_stage(ledger, "pair_selection"), watcher.stage_scope("pair_selection"):
        pairs: list[ImagePair] = run_pair_selection_stage(selection.selected, config)
    watcher.on_pair_selection(pairs)

    # ── 4. matching and verification ─────────────────────────────────────────────────
    # One object for stage 4 and stage 5's batch match: the loop stage matching its
    # candidates with settings of its own is exactly the drift this prevents.
    matching_options: MatchingOptions = resolved.matching_options(config)
    with timed_stage(ledger, "matching"), watcher.stage_scope("matching"):
        matching: MatchReport = run_matching_stage(staged, pairs, matching_options)
    watcher.on_matching(matching)

    # ── 5. loop closure ──────────────────────────────────────────────────────────────
    with timed_stage(ledger, "loop_closure"), watcher.stage_scope("loop_closure"):
        loops: LoopClosureStageResult = run_loop_closure_stage(
            selection.selected, staged.database_path, config.pose_graph, matching_options, enabled=options.loop_closure
        )
    watcher.on_loop_closure(loops)

    # ── 6. pose graph ────────────────────────────────────────────────────────────────
    with timed_stage(ledger, "pose_graph"), watcher.stage_scope("pose_graph"):
        pose_graph: PoseGraphStageResult = run_pose_graph_stage(staged, selection.selected, config, loops.edges)
    watcher.on_pose_graph(pose_graph)

    mapping_options: MappingOptions = resolved.mapping

    # ── 7. triangulation and bundle adjustment ───────────────────────────────────────
    with timed_stage(ledger, "reconstruction"), watcher.stage_scope("reconstruction"):
        reconstruction: MappingResult = run_reconstruction_stage(
            staged, pose_graph.frames_meta, config, mapping_options
        )
    # Before 7b: the refinement adjusts this very reconstruction in place.
    watcher.on_reconstruction(reconstruction)

    # ── 7b. extrinsic refinement ─────────────────────────────────────────────────────
    mapping: MappingResult = reconstruction
    mapped_meta: FramesMeta = pose_graph.frames_meta
    changes: tuple[ExtrinsicChange, ...] = ()
    rounds_run: int = 0
    if options.optimize_extrinsics:
        with timed_stage(ledger, EXTRINSIC_REFINEMENT_STAGE), watcher.stage_scope(EXTRINSIC_REFINEMENT_STAGE):
            refinement: ExtrinsicRefinementStageResult = run_extrinsic_refinement_stage(
                staged, pose_graph.frames_meta, config, mapping_options, mapping
            )
        watcher.on_extrinsic_refinement(refinement)
        mapping = refinement.mapping
        mapped_meta = refinement.frames_meta
        changes = refinement.changes
        rounds_run = refinement.rounds_run

    # ── 8. export ────────────────────────────────────────────────────────────────────
    with timed_stage(ledger, "export"), watcher.stage_scope("export"):
        export: ExportStageResult = run_export_stage(staged, mapped_meta, mapping)
    watcher.on_export(export, mapping)

    summary: PipelineSummary = PipelineSummary(
        options=options,
        git_sha=git_sha(),
        num_input_keyframes=len(selection.frames_meta.keyframes),
        num_selected_keyframes=len(selection.selected.keyframes),
        num_rig_frames=len(selection.rig_frames),
        num_pairs=len(pairs),
        num_loop_pairs=loops.num_pairs_matched,
        num_loop_edges=len(loops.edges),
        num_pose_graph_edges=len(pose_graph.edges),
        pose_graph_translation_change_m=pose_graph.translation_change_m,
        mapping=MappingStats.of(mapping),
        stage_seconds=ledger.seconds_by_stage,
        total_seconds=time.perf_counter() - started,
        extrinsic_changes=changes,
        extrinsic_refinement_rounds_run=rounds_run,
    )
    (staged.output_dir / SUMMARY_NAME).write_text(to_json(summary) + "\n")
    print(
        f"[colsfm] done in {summary.total_seconds:.2f}s | "
        f"{summary.mapping.num_images_with_observations} images, {summary.mapping.num_points3D} points, "
        f"{summary.mapping.mean_reprojection_error_px:.4f} px | extraction on {extraction.device}, "
        f"{matching.empty_pairs} empty pairs"
    )
    return summary


@dataclass(frozen=True, slots=True)
class StageResult:
    """What one standalone `stage` invocation produced."""

    stage: CheapStageName
    """The stage that ran."""
    num_selected_keyframes: int
    """Keyframes that survived selection."""
    num_rig_frames: int
    """Rig frames among them."""
    num_pairs: int
    """Pairs the pair-selection stage would match; 0 when only selection ran."""


def run_stage(options: SelectionOptions, stage: CheapStageName) -> StageResult:
    """Run one metadata-only stage, for inspecting a dataset without touching a GPU.

    Args:
        options: The metadata-only options; neither stage needs anything else, and
            neither writes, so no output directory is asked for.
        stage: `keyframe_selection` or `pair_selection`.

    Returns:
        The counts the stage produced.

    Raises:
        FileNotFoundError: When the input metadata is missing.
        ValueError: When no keyframe survives selection.
    """
    selection: KeyframeSelectionStageResult = run_keyframe_selection_stage(options)
    pairs: list[ImagePair] = (
        run_pair_selection_stage(selection.selected, selection.config) if stage == "pair_selection" else []
    )
    rig_frames: tuple[RigFrame, ...] = selection.rig_frames
    print(
        f"[colsfm] {stage}: {len(selection.selected.keyframes)}/{len(selection.frames_meta.keyframes)} keyframes, "
        f"{len(rig_frames)} rig frames, {len(pairs)} pairs"
    )
    return StageResult(
        stage=stage,
        num_selected_keyframes=len(selection.selected.keyframes),
        num_rig_frames=len(rig_frames),
        num_pairs=len(pairs),
    )
