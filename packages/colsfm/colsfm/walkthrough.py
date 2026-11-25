"""A stage-by-stage Rerun walkthrough of the `colsfm` architecture.

```
pixi run -e colsfm python -m colsfm.walkthrough \
    --input-dir data/r2b_galileo \
    --config-dir data/cusfm_configs/loop-closure-fixed \
    --output data/bench/galileo_walkthrough.rrd
```

The module is an **observer** of `colsfm.pipeline.run_pipeline`, never a second
runner: `run_walkthrough` calls `run_pipeline` with a `WalkthroughObserver`, and
the observer is handed each stage's typed result once that stage's timing window
has closed. One sequencer therefore resolves the options, times the stages into
`runtime.csv` and writes `summary.json`; the walkthrough only draws. A run under
the walkthrough leaves exactly the artifacts `python -m colsfm run` leaves, and a
new stage or backend is added in one place.

Each stage's real intermediate data lands in one recording on a `stage` sequence
timeline. Scrubbing that timeline from 1 to N walks the architecture of Figure 2
of *CuSfM: CUDA-Accelerated Structure-from-Motion* (arXiv:2510.15271):
dictionary construction → loop closure detection → pose graph optimisation on
the first row, feature extraction → feature matching on the second, camera
projection, triangulation and mapping, and extrinsic refinement on the third.

Nothing here re-implements a stage. Everything logged is read back out of the
stage results, the COLMAP database the run wrote, or the run directory the
export stage left behind, so a claim in a panel is a claim about this run.

## Where things live

| module | owns |
|---|---|
| `colsfm.walkthrough` | the CLI, `WalkthroughObserver` and the recording |
| `colsfm.walkthrough_stages` | what a stage *is*: its prose, its index, the palette |
| `colsfm.walkthrough_draw` | frusta, arcs, imagery and the notes document |
| `colsfm.walkthrough_panels` | one `log_*` per stage: what that stage produced |
| `colsfm.walkthrough_blueprint` | the tab-per-stage layout, sent once at the end |

## Entity layout

```
/walkthrough
  /stage_01                    keyframe selection
    /notes                     TextDocument: what the stage does + this run's numbers
    /input/{frusta,centres}    every input camera pose, grey
    /selected/{frusta,centres} the poses selection kept, blue
    /selected/rig_track        the rig trajectory over the kept rig frames
  /stage_02                    feature extraction
    /samples/<sensor>_<id>/{image,keypoints}   sample keyframes + their ALIKED keypoints
    /keypoint_counts           BarChart, one bar per selected keyframe
  /stage_03                    pair selection
    /centres                   camera centres
    /pairs/{consecutive,stereo}  LineStrips3D between the paired camera centres
  /stage_04                    matching
    /pair/{left,right}/{image,keypoints}   one sample pair, side by side
    /pair/stacked/{image,keypoints,matches}  the same pair concatenated, with match lines
    /inliers_per_pair          BarChart: verified inliers, one bar per pair
    /inlier_histogram          BarChart: the distribution of those counts
  /stage_05                    loop closure
    /robocap/{nodes,trajectory,loop_edges}   RoboCap's revisit graph (see the notes)
  /stage_06                    pose graph
    /before/{nodes,edges}      rig nodes and constraints as they entered the solve
    /after/{nodes,edges}       the same after it
  /stage_07                    triangulation + bundle adjustment
    /points                    the model the mapper left behind
    /rounds/<metric>           Scalars on a `round` timeline, one point per BA round
    /round_chart/<metric>      the same numbers as a BarChart, readable on `stage`
  /stage_08                    extrinsic refinement (only with --optimize-extrinsics)
    /extrinsics/{before,after,shift}   per-camera centres and the shift arrows
  /stage_08 or /stage_09       export
    /points                    the final coloured cloud
    /rigs/rig_NN               input / colsfm / blob reference rigs (exoego layout)
    /tracks/<name>/trajectory  each run's rig trajectory
    /report                    the benchmark table
```

Two deliberate departures are worth knowing before reading the code.

1. **Everything is logged at its stage index, not statically.** A latest-at
   query on `stage` therefore shows stages 1..k at index k and nothing beyond,
   which is what makes the timeline a walkthrough rather than a switchboard.
   The one exception is the rig calibration `simplecv.log_rig_static` writes,
   which is static by construction.
2. **The rigs of the export stage carry two poses.** `log_rig_pose_stream`
   writes `world_T_rig(t)` through `rr.send_columns`, which stamps only the
   `frame_time` timeline — a chunk without a `stage` column is invisible to a
   `stage` query. The export stage therefore also logs each rig's first pose on
   the `stage` timeline, so the rig has a place to stand while the timeline
   walks the architecture, and still animates when `frame_time` is picked.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import rerun as rr
import tyro

from colsfm.ba_backend import BaBackend
from colsfm.database import ImagePair
from colsfm.features import ExtractionReport, FeatureBackend
from colsfm.frame_manifest import FrameSource
from colsfm.frames_meta import FramesMeta
from colsfm.mapping import MappingResult
from colsfm.matching import MatchingBackend, MatchReport
from colsfm.pipeline import PipelineObserver, run_pipeline
from colsfm.run_config import DEFAULT_CONFIG_DIR, DEFAULT_MIN_INTER_FRAME_DISTANCE_M, DEFAULT_MIN_INTER_FRAME_ROTATION_DEGREES, PipelineOptions
from colsfm.run_lifecycle import StageName
from colsfm.run_report import PipelineSummary
from colsfm.stages import (
    ExportStageResult,
    ExtrinsicRefinementStageResult,
    KeyframeSelectionStageResult,
    LoopClosureStageResult,
    PoseGraphStageResult,
)
from colsfm.walkthrough_blueprint import build_blueprint
from colsfm.walkthrough_panels import (
    log_export,
    log_extrinsic_refinement,
    log_feature_extraction,
    log_keyframe_selection,
    log_loop_closure,
    log_matching,
    log_pair_selection,
    log_pose_graph,
    log_reconstruction,
)
from colsfm.walkthrough_stages import (
    APPLICATION_ID,
    DEFAULT_EXTRINSIC_ARROW_SCALE,
    DEFAULT_OUTPUT,
    DEFAULT_WORK_DIR,
    SamplePane,
    WalkthroughStage,
    walkthrough_stages,
)

# ══════════════════════════════════════════════════════════════════════════════════════
# configuration
# ══════════════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class WalkthroughConfig:
    """Run one dataset through every stage and log the walkthrough to a `.rrd`."""

    input_dir: Path
    """Directory holding `frames_meta.json` and the images its `image_name`s name."""
    output: Path = DEFAULT_OUTPUT
    """Where the Rerun recording is written; parent directories are created."""
    config_dir: Path = DEFAULT_CONFIG_DIR
    """Directory of `.pb.txt` configs; the loop-closure-fixed isaac profile by default."""
    work_dir: Path | None = None
    """cuSFM-layout workspace the run writes into; a directory under `data/bench` by default."""
    optimize_extrinsics: bool = False
    """Run stage 7b, the regularised rig-extrinsic refinement, and log its shift arrows.

    Off here, on in the pipeline: the two walkthroughs — with the stage and without —
    are what the recording is *for*, so this flag chooses which one is drawn rather
    than following the pipeline's own default."""
    loop_closure: bool = True
    """Run the retrieval loop-closure search. On by default here, unlike the pipeline:
    the whole point of stage 5's panel is the rejection funnel, and a disabled stage
    prints one line and produces none."""
    min_inter_frame_distance: float = DEFAULT_MIN_INTER_FRAME_DISTANCE_M
    """Keyframe displacement gate in metres, the pipeline's default; the Galileo task passes 0.0 to keep all 226 keyframes."""
    min_inter_frame_rotation_degrees: float = DEFAULT_MIN_INTER_FRAME_ROTATION_DEGREES
    """Keyframe rotation gate in degrees, OR-combined with the distance gate."""
    use_gpu: bool = True
    """Run ALIKED and LightGlue on the GPU when one is usable."""
    num_sample_keyframes: int = 3
    """Keyframes shown with their keypoints in stage 2, spread over the sequence."""
    extrinsic_arrow_scale: float = DEFAULT_EXTRINSIC_ARROW_SCALE
    """How far stage 7b's shift arrows are exaggerated; the real shifts are sub-millimetre."""
    reference_run: Path | None = None
    """A reference run in the cuSFM layout (another colsfm run, or a cuSFM run) for the closing benchmark table."""
    robocap_loop_run: Path | None = None
    """A run whose loop closure found revisits, drawn beside Galileo's empty stage 5."""
    dataset: str = "galileo"
    """Label for this dataset; it names the recording and the benchmark report, nothing else."""
    also_at_zero: tuple[str, ...] = ()
    """Duration timelines every panel is also stamped on at 0, e.g. `video_time` for a catalog run whose overview
    lives on the segment's video timeline (`tools/overview.py`), so the stage tabs need no timeline switch."""
    features_backend: FeatureBackend = "pycolmap"
    """Which ALIKED the drawn run uses. `pycolmap` here, not the pipeline's `raco` default.

    The walkthrough is the teaching artifact: it has to draw on a bare checkout, where
    `colsfm.MODEL_DIR` holds no graph (`colsfm.model_assets`), and every panel it draws is
    the same whichever extractor filled the database. Pass `raco` to draw the default
    pipeline instead."""
    matching_backend: MatchingBackend = "pycolmap"
    """Which matcher the drawn run uses; `pycolmap` for the same reason."""
    ba_backend: BaBackend = "ceres"
    """Which bundle adjuster the drawn run uses. `ceres`, not the pipeline's `caspar`:
    only the `colsfm` environments (recipes/colmap-src) have a CASPAR build, and this recording is made
    wherever a viewer is."""

    @property
    def resolved_work_dir(self) -> Path:
        """The workspace the run will write into.

        Returns:
            `work_dir` when given, `DEFAULT_WORK_DIR` otherwise.
        """
        return DEFAULT_WORK_DIR if self.work_dir is None else self.work_dir

    def pipeline_options(self) -> PipelineOptions:
        """The options the pipeline stages are driven with.

        Returns:
            A `PipelineOptions` carrying this walkthrough's input, workspace and gates.
        """
        return PipelineOptions(
            input_dir=self.input_dir,
            output_dir=self.resolved_work_dir,
            config_dir=self.config_dir,
            min_inter_frame_distance=self.min_inter_frame_distance,
            min_inter_frame_rotation_degrees=self.min_inter_frame_rotation_degrees,
            optimize_extrinsics=self.optimize_extrinsics,
            loop_closure=self.loop_closure,
            use_gpu=self.use_gpu,
            features_backend=self.features_backend,
            matching_backend=self.matching_backend,
            ba_backend=self.ba_backend,
        )


@dataclass(frozen=True, slots=True)
class WalkthroughResult:
    """What one walkthrough produced."""

    rrd_path: Path
    """The recording written."""
    work_dir: Path
    """The cuSFM-layout workspace the stages wrote into."""
    stages: tuple[WalkthroughStage, ...]
    """The stages recorded, in timeline order."""
    seconds_by_stage: dict[str, float]
    """Wall-clock seconds per stage, the same rows as the workspace's `runtime.csv`."""


# ══════════════════════════════════════════════════════════════════════════════════════
# the run
# ══════════════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class WalkthroughObserver(PipelineObserver):
    """Draws each stage of one `colsfm.pipeline` run into the active recording.

    An observer, not a second runner: `colsfm.pipeline.run_pipeline` sequences the
    stages, resolves the options, times them into `runtime.csv` and writes
    `summary.json`; this object is handed each stage's typed result *after* its
    timing window has closed. Nothing drawn here is timed, and no drawing can
    change what the run computes — which is what keeps the walkthrough's workspace
    byte-comparable with a plain `python -m colsfm run`.

    `stage_scope` is the one hook inside the timing window, and it only redirects
    stdout: the loop-closure funnel and the Ceres termination are *printed* by the
    stage functions rather than returned, and each notes panel quotes them verbatim.
    """

    config: WalkthroughConfig
    """The parsed command line: the sample count, the arrow scale and the reference runs."""
    options: PipelineOptions
    """The options `run_pipeline` is executing, for the database and image paths.

    Replaced in `run_started` by the ones the stages actually run with, whose
    `output_dir` is the run's staging directory: mid-run, that is where the
    database and the exported artifacts are, and reading the destination would
    read the *previous* run."""
    published_options: PipelineOptions = field(init=False)
    """The same options with the caller's own `output_dir`, for the panels to *name*.

    A staging directory is where a stage wrote; it is not where the artifact ends
    up, and a finished recording that names one names a directory that no longer
    exists."""
    frames: FrameSource = field(init=False)
    """The run's frame source, handed over in `run_started`."""
    stages: tuple[WalkthroughStage, ...]
    """The timeline slots this run records, in stage order."""
    console: dict[str, str] = field(default_factory=dict)
    """Whatever each stage printed while it ran, keyed by stage name."""
    seconds_by_stage: dict[str, float] = field(default_factory=dict)
    """Wall-clock seconds per stage; filled from the run's own summary, not measured here."""
    samples: list[SamplePane] = field(default_factory=list)
    """Stage 2's sample panes. The blueprint needs them and nothing knows them up front."""
    selection: KeyframeSelectionStageResult | None = None
    """Stage 1's result; every later stage draws against the collection it selected."""
    pose_graph_meta: FramesMeta | None = None
    """Stage 6's re-posed collection, which stage 7b's before/after arrows start from."""
    export: tuple[ExportStageResult, MappingResult] | None = None
    """Stage 8's result, held until the run is published; see `on_export`."""

    def stage(self, name: StageName) -> WalkthroughStage:
        """Look up a stage's timeline slot by name.

        Args:
            name: The pipeline stage name.

        Returns:
            The walkthrough stage carrying its index and prose.

        Raises:
            KeyError: When this run does not record that stage.
        """
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(f"{name} is not one of this walkthrough's stages")

    def _selected(self) -> KeyframeSelectionStageResult:
        """Stage 1's result, which every later panel needs.

        Returns:
            What `run_keyframe_selection_stage` produced.

        Raises:
            RuntimeError: When a later stage is observed before stage 1 has run.
        """
        if self.selection is None:
            raise RuntimeError("the walkthrough observed a later stage before keyframe selection")
        return self.selection

    @contextlib.contextmanager
    def _captured(self, stage: StageName) -> Iterator[None]:
        """Run the stage with its stdout diverted into `console[stage]`.

        Args:
            stage: The stage whose output to capture.

        Yields:
            Nothing; the stage body runs inside the redirection.
        """
        buffer: io.StringIO = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                yield
        finally:
            self.console[stage] = buffer.getvalue()

    def stage_scope(self, stage: StageName) -> contextlib.AbstractContextManager[None]:
        """Capture one stage's console output.

        Args:
            stage: The stage about to run.

        Returns:
            A context manager that redirects stdout for the duration of the stage.
        """
        return self._captured(stage)

    def run_started(self, options: PipelineOptions, stages: tuple[StageName, ...], frames: FrameSource) -> None:
        """Take the staged options and set the recording's world axes before stage 1.

        Args:
            options: The options the stages will execute, rebased onto the run's
                staging directory. Everything this observer opens mid-run is there.
            stages: The stages it will record; already held as `stages`.
            frames: The run's frame source, which the feature and matching panels draw from.
        """
        self.published_options = self.options
        self.options = options
        self.frames = frames
        rr.log("/", rr.ViewCoordinates.RFU, static=True)

    def on_keyframe_selection(self, result: KeyframeSelectionStageResult) -> None:
        """Draw stage 1.

        Args:
            result: The configuration read, the input collection and the selection.
        """
        self.selection = result
        log_keyframe_selection(self.stage("keyframe_selection"), result, self.console["keyframe_selection"])

    def on_feature_extraction(self, report: ExtractionReport) -> None:
        """Draw stage 2 and remember the sample panes the blueprint will need.

        Args:
            report: Per-image keypoint counts, the device used and the wall time.
        """
        self.samples = log_feature_extraction(
            self.stage("feature_extraction"),
            self._selected().selected,
            report,
            self.options.database_path,
            self.frames,
            self.config.num_sample_keyframes,
            self.console["feature_extraction"],
            published_database_path=self.published_options.database_path,
        )

    def on_pair_selection(self, pairs: Sequence[ImagePair]) -> None:
        """Draw stage 3.

        Args:
            pairs: The consecutive and stereo pairs the matcher will be given.
        """
        selection: KeyframeSelectionStageResult = self._selected()
        log_pair_selection(self.stage("pair_selection"), selection.selected, selection.config, pairs, self.console["pair_selection"])

    def on_matching(self, report: MatchReport) -> None:
        """Draw stage 4.

        Args:
            report: Per-pair raw and inlier counts, the device used and the wall time.
        """
        log_matching(
            self.stage("matching"),
            self._selected().selected,
            report,
            self.options.database_path,
            self.frames,
            self.console["matching"],
        )

    def on_loop_closure(self, result: LoopClosureStageResult) -> None:
        """Draw stage 5.

        Args:
            result: The gated loop edges, the extra pairs matched and the diagnostics.
        """
        log_loop_closure(self.stage("loop_closure"), result, self.config.robocap_loop_run, self.console["loop_closure"])

    def on_pose_graph(self, result: PoseGraphStageResult) -> None:
        """Draw stage 6 and remember the collection it re-posed.

        Args:
            result: The nodes, the edges, the re-posed collection and the largest move.
        """
        self.pose_graph_meta = result.frames_meta
        log_pose_graph(self.stage("pose_graph"), result, self.console["pose_graph"])

    def on_reconstruction(self, mapping: MappingResult) -> None:
        """Draw stage 7, before stage 7b may adjust the same reconstruction in place.

        Args:
            mapping: The triangulated, bundle-adjusted model and its round statistics.
        """
        log_reconstruction(self.stage("reconstruction"), mapping, self.console["reconstruction"])

    def on_extrinsic_refinement(self, result: ExtrinsicRefinementStageResult) -> None:
        """Draw stage 7b.

        Args:
            result: The refined model, the re-calibrated collection and the movement.

        Raises:
            RuntimeError: When the refinement is observed before the pose graph.
        """
        if self.pose_graph_meta is None:
            raise RuntimeError("the walkthrough observed the extrinsic refinement before the pose graph")
        log_extrinsic_refinement(
            self.stage("extrinsic_refinement"),
            self.pose_graph_meta,
            result,
            self.config.extrinsic_arrow_scale,
            self.console["extrinsic_refinement"],
        )

    def on_export(self, result: ExportStageResult, mapping: MappingResult) -> None:
        """Hold stage 8's result until the run is published.

        Its notes panel closes with a benchmark table, and `colsfm.benchmark`
        refuses a run that has not finished — rightly, since mid-run the artifacts
        are still in the staging directory. So the drawing waits for
        `run_finished`; the entities still land at the export stage's own index,
        because the `stage` timeline is set by the stage, not by the call order.

        Args:
            result: The exported collection and what was written.
            mapping: The final mapping result the export was made from.
        """
        self.export = (result, mapping)

    def run_finished(self, summary: PipelineSummary) -> None:
        """Draw stage 8 against the published run, take the timings, send the blueprint.

        The blueprint is sent last, not first: stage 2's panes name the keyframes
        the sampler picked, which nothing knows before the run.

        Args:
            summary: What the run produced and how long each stage took.

        Raises:
            RuntimeError: When the run finished without an export stage.
        """
        if self.export is None:
            raise RuntimeError("the walkthrough finished a run that never exported")
        result, mapping = self.export
        log_export(
            self.stage("export"),
            # `summary.options` names the *published* directory; `self.options`
            # still names the staging one the stages wrote into.
            summary.options,
            result,
            mapping,
            self._selected().frames_meta,
            self.config.reference_run,
            self.config.dataset,
            self.console["export"],
        )
        self.seconds_by_stage = {str(stage): seconds for stage, seconds in summary.stage_seconds.items()}
        rr.send_blueprint(build_blueprint(self.stages, self.samples), make_active=True, make_default=True)


def run_walkthrough(config: WalkthroughConfig) -> WalkthroughResult:
    """Run the dataset through `colsfm.pipeline` and save the walkthrough recording.

    Args:
        config: The parsed command line.

    Returns:
        The recording's path, the workspace, the stages and their timings.

    Raises:
        FileNotFoundError: When the input metadata or a config file is missing.
        ValueError: When a stage cannot proceed.
    """
    options: PipelineOptions = config.pipeline_options()
    config.output.parent.mkdir(parents=True, exist_ok=True)
    stages: tuple[WalkthroughStage, ...] = walkthrough_stages(config.optimize_extrinsics, config.also_at_zero)
    observer: WalkthroughObserver = WalkthroughObserver(config=config, options=options, stages=stages)
    print(f"[walkthrough] input {config.input_dir} | workspace {options.output_dir} | recording {config.output}")

    # A scoped `RecordingStream` rather than `rr.init` + `rr.save`: leaving the
    # `with` block flushes *and* closes the file sink, so the `.rrd` gets its footer.
    # The extrinsics variant gets its own recording id: two `.rrd` files sharing an
    # application id *and* a recording id are one store to the viewer, so opening
    # both merges their entity trees and each shows the other's stages.
    recording_id: str = f"{config.dataset}_walkthrough" + ("_extrinsics" if config.optimize_extrinsics else "")
    recording: rr.RecordingStream = rr.RecordingStream(APPLICATION_ID, recording_id=recording_id)
    recording.save(str(config.output))
    with recording:
        run_pipeline(options, observer)
    recording.flush(timeout_sec=60.0)
    print(f"[walkthrough] wrote {config.output} ({config.output.stat().st_size / 1e6:.1f} MB)")
    return WalkthroughResult(
        rrd_path=config.output,
        work_dir=options.output_dir,
        stages=stages,
        seconds_by_stage=dict(observer.seconds_by_stage),
    )


def main() -> None:
    """Entry point for `python -m colsfm.walkthrough`."""
    run_walkthrough(tyro.cli(WalkthroughConfig))


if __name__ == "__main__":
    main()
