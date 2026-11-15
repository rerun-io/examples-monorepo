"""The command line as it is spelled, and the per-stage settings it resolves into.

`PipelineOptions` is an **input adapter**: twenty-one fields spanning unrelated
stages, kept exactly as `python -m colsfm run` spells them and exactly as
`summary.json` records them, so a run's provenance is the flags it was given.
`ResolvedRun` is what the stages actually take.

The two are separate because they answer different questions. The first is "what
did the caller ask for", which must never be rewritten -- the review found the
mapper switching backend by mutating it. The second is "what will each stage do",
which is decided once, up front, and validates as it goes: a `--caspar-option`
typo stops the run here rather than at the first bundle adjustment, after feature
extraction, matching and loop closure have been paid for.

`SelectionOptions` is the metadata-only third of the adapter. `python -m colsfm
stage` takes it, because keyframe and pair selection read `frames_meta.json` and
a config profile, touch no image, open no GPU and write nothing -- so asking them
for an output directory asked for a workspace nobody would fill.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from serde import serde

from colsfm import RESOURCES_DIR
from colsfm.ba_backend import BaBackend, BaExecutionPlan, resolve_ba_plan
from colsfm.config import CusfmConfig
from colsfm.features import DeviceChoice, FeatureBackend, FeatureOptions
from colsfm.frames_meta import FRAMES_META_NAME
from colsfm.mapping import MappingOptions
from colsfm.matching import (
    BLOB_MATCH_TOP_K,
    MatchCapMode,
    MatchingBackend,
    MatchingOptions,
    MatchLimitPolicy,
)
from colsfm.model_assets import check_raco_graphs
from colsfm.run_lifecycle import (
    DATABASE_NAME,
)

DEFAULT_CONFIG_DIR: Final[Path] = RESOURCES_DIR / "configs" / "loop-closure-fixed"
"""`isaac` with loop closure repaired; what the blob reference runs used."""


@serde
@dataclass(frozen=True, slots=True)
class SelectionOptions:
    """What keyframe and pair selection need, and nothing else.

    The metadata-only half of `PipelineOptions`, resolved out of it once at the
    start of a run. `python -m colsfm stage` takes exactly this: those two stages
    read `frames_meta.json` and a config profile, touch no image, open no GPU and
    write nothing, so requiring an output directory of them was asking for a
    workspace nobody would fill.
    """

    input_dir: Path
    """Directory holding `frames_meta.json` and the images its `image_name`s name."""
    config_dir: Path = DEFAULT_CONFIG_DIR
    """Directory of `.pb.txt` configs; the loop-closure-fixed isaac profile by default."""
    min_inter_frame_distance: float = 0.0
    """Keyframe displacement gate in metres; the demo passes 0.0 to keep every sample."""
    min_inter_frame_rotation_degrees: float = 5.0
    """Keyframe rotation gate in degrees, OR-combined with the distance gate."""
    sample_sync_threshold_microseconds: int = 100
    """Rig grouping window in microseconds; the isaac demo's value, not the gflag default."""

    @property
    def frames_meta_path(self) -> Path:
        """Where the input metadata lives.

        Returns:
            `<input_dir>/frames_meta.json`.
        """
        return self.input_dir / FRAMES_META_NAME


@serde
@dataclass(frozen=True, slots=True)
class PipelineOptions:
    """Everything one run needs, mirroring `CusfmRunner`'s constructor arguments.

    The defaults are the **fast full pipeline** (NOTES.md decision 17, 2026-09-06):
    `--features-backend raco --matching-backend raco --ba-backend caspar
    --optimize-extrinsics --loop-closure`, with the closing Ceres polish and the
    regularised refinement. Every field below says what its own ablation costs, and
    this class is also the CLI's help text, so the two cannot drift.
    """

    input_dir: Path
    """Directory holding `frames_meta.json` and the images its `image_name`s name."""
    output_dir: Path
    """Workspace root, cuSFM's `cusfm_base_dir`; created if missing."""
    config_dir: Path = DEFAULT_CONFIG_DIR
    """Directory of `.pb.txt` configs; the loop-closure-fixed isaac profile by default."""
    min_inter_frame_distance: float = 0.0
    """Keyframe displacement gate in metres; the demo passes 0.0 to keep every sample."""
    min_inter_frame_rotation_degrees: float = 5.0
    """Keyframe rotation gate in degrees, OR-combined with the distance gate."""
    sample_sync_threshold_microseconds: int = 100
    """Rig grouping window in microseconds; the isaac demo's value, not the gflag default."""
    optimize_extrinsics: bool = True
    """Refine `sensor_from_rig` after mapping; cuSFM's `--optimize_extrinsics`. On by default.

    Stage 7b, `colsfm.extrinsic_refinement`, run on the model stage 7 left behind
    (NOTES.md decision 17). It is not a second mapping pass and it does not free the
    extrinsics inside the mapper: the alternation moves them in pyceres with the
    blob's priors and holds them fixed in every bundle adjustment, which is why the
    default can ask for CASPAR and this in the same run. `--no-optimize-extrinsics`
    is the ablation; it drops the stage, which costs 2.5 s on Galileo and 118-128 s
    on RoboCap (`docs/full-pipeline-results.md`)."""
    regularised_extrinsics: bool = True
    """Carry the blob's extrinsic priors through `colsfm.extrinsic_refinement`. False falls
    back to pycolmap's own rig bundle adjustment, which has no prior terms and on Galileo
    overfits (NOTES.md deviation 9). Only read when `optimize_extrinsics` is set.

    It is also the one thing that takes `--ba-backend caspar` off the GPU: the
    unregularised path *is* `run_mapping(..., optimize_extrinsics=True)`, and CASPAR
    throws when asked to refine `sensor_from_rig` (`colsfm.ba_backend`)."""
    extrinsic_refinement_rounds: int = 20
    """Ceiling on the extrinsics-then-poses rounds the regularised refinement may take; it
    stops early on its own tolerances, after 19 rounds and 3.2 s on Galileo."""
    loop_closure: bool = True
    """Run retrieval-based loop closure. On by default, as in the blob, so every run and
    every blob comparison pays for the same stage (NOTES.md decision 10, revised). On the
    0.93 s Galileo sweep the 10 s loop gate rejects every candidate, so the stage costs time
    and changes nothing; `--no-loop-closure` is the explicit ablation."""
    use_gpu: bool = True
    """Run ALIKED and LightGlue on the GPU when one is usable. A bool rather than a
    `DeviceChoice` because `--use-gpu` is the flag the pixi tasks and the plan use; the
    `device` property below is the one place it turns into a device request."""
    ba_use_gpu: bool = False
    """Hand the bundle-adjustment linear solve to the GPU. Off by default: the measured
    difference is under 10 % and changes sign with the problem size, so the default follows
    the blob, which solves on the CPU. See `colsfm.mapping.MappingOptions.use_gpu`."""
    num_threads: int = -1
    """Threads for feature extraction and matching; -1 lets COLMAP use every core.

    The blob runs `feature_extractor_main --num_thread 1`, but that flag counts *its own*
    worker processes, not the threads inside one: the runner launches one process per
    camera. COLMAP has a single process, so 1 serialises JPEG decode against the GPU and
    costs 4.4x on Galileo (33.9 s against 7.6 s, measured)."""
    ba_backend: BaBackend = "caspar"
    """Which implementation solves the bundle adjustments — Ceres on the CPU, or COLMAP's
    CASPAR on the GPU. `caspar` by default (NOTES.md decision 17): with the closing Ceres
    polish below it is a third of the Ceres mapping time at Ceres' accuracy, and where the
    build cannot honour it the run says so and solves on Ceres anyway, so the default costs
    a machine without CASPAR nothing but a message. It needs a CASPAR-enabled pycolmap — the
    the `colsfm` environment (recipes/colmap-src) — and falls back to `ceres`, loudly, when that is
    missing, when `--no-regularised-extrinsics` asks the mapper itself to free
    `sensor_from_rig`, or when a camera model is one this build's CASPAR has no
    adapter for. Which models those are is measured at
    run time (`colsfm.mapping.caspar_supported_camera_models`): PINHOLE and SIMPLE_RADIAL
    in a stock CASPAR build, plus OPENCV_FISHEYE in
    the `colsfm` environment (recipes/colmap-src) (`docs/caspar-fisheye-adapter.md`). It also drops the robust
    loss; see `colsfm.mapping`'s module docstring."""
    ba_num_threads: int | None = None
    """Ceres threads; None takes `bundle_adjustment_config.num_threads` (8) from the config,
    which is what the blob solves with. Set 1 for a bit-reproducible solve."""
    caspar_ceres_polish: bool = True
    """Finish a `--ba-backend caspar` run with one Ceres global bundle adjustment.

    On by default: it is what makes the GPU backend shippable — KITTI 06 goes from
    1.299 m to 0.904 m Sim(3) ATE for 9.8 s on top of 31.5 s of CASPAR rounds, against
    the Ceres mapper's 0.895 m in 148.5 s (`docs/caspar-build.md` § Shipped: CASPAR +
    Ceres polish). `--no-caspar-ceres-polish` is the ablation. Inert on `ceres`,
    fallbacks included, which have already converged."""
    caspar_option: tuple[str, ...] = ()
    """CASPAR solver overrides as `name=value`, e.g. `--caspar-option solver_iter_max=1000
    pcg_iter_max=80`. Read only when `--ba-backend caspar` actually runs; see
    `colsfm.mapping.CasparOptions` for the names and `docs/caspar-build.md`
    § Convergence sweep for what moving them buys."""
    max_matches_per_pair: int | None = BLOB_MATCH_TOP_K
    """Verified matches kept per pair after the spatial subsample; None keeps every inlier.
    The blob's `match_top_k`; see `colsfm.matching.subsample_matches_by_coverage`."""
    match_cap_mode: MatchCapMode = "fixed"
    """How `max_matches_per_pair` becomes a per-pair number; see
    `colsfm.matching.MatchLimitPolicy`. `fixed` is the 500 every run before the KITTI
    follow-up used, `image_area` scales it with the frame's resolution, `off` keeps every
    verified inlier."""
    features_backend: FeatureBackend = "raco"
    """Which ALIKED runs: RaCo-ALIKED on its own engine (`colsfm.features_raco`), COLMAP's
    own ONNX one, or the blob's TensorRT engine (`colsfm.features_trt`).

    `raco` by default (NOTES.md decision 17): 2.6 s of Galileo's 20.0 s total against
    6.0 s for the blob's engine and 8.3 s for COLMAP's, at the same accuracy. It needs
    the graphs under `packages/colsfm/data/models/`, which are gitignored; a checkout without them
    is told to export them or to ask for `--features-backend pycolmap`
    (`colsfm.model_assets`), never quietly given another extractor.
    See `colsfm.features.FeatureBackend`."""
    matching_backend: MatchingBackend = "raco"
    """Which LightGlue runs: LightGlue+ (`colsfm.matching_raco`), COLMAP's own ONNX one, or
    the blob's TensorRT engine (`colsfm.matching_trt`).

    `raco` by default, as the matcher fabio-sim trained against RaCo-ALIKED: pairing the
    default extractor with COLMAP's own matcher would match descriptors it was not trained
    on. Like the `tensorrt` path it sees the per-match score the blob's SSC spatial NMS
    needs, which the `pycolmap` path cannot. See `colsfm.matching.MatchingBackend`."""

    @property
    def selection(self) -> SelectionOptions:
        """The metadata-only options, resolved out of the command line's spellings.

        Returns:
            The five gates and paths keyframe and pair selection read, so those
            stages never see the twenty-one fields they have no use for.
        """
        return SelectionOptions(
            input_dir=self.input_dir,
            config_dir=self.config_dir,
            min_inter_frame_distance=self.min_inter_frame_distance,
            min_inter_frame_rotation_degrees=self.min_inter_frame_rotation_degrees,
            sample_sync_threshold_microseconds=self.sample_sync_threshold_microseconds,
        )

    @property
    def device(self) -> DeviceChoice:
        """Device request for the two ONNX stages, derived from `use_gpu` exactly once.

        Returns:
            `auto` — which falls back to the CPU provider when CUDA or cuDNN is
            missing — or `cpu` when `use_gpu` is off.
        """
        return "auto" if self.use_gpu else "cpu"

    @property
    def database_path(self) -> Path:
        """Where the COLMAP database lives.

        Returns:
            `<output_dir>/database.db`.
        """
        return self.output_dir / DATABASE_NAME

    @property
    def frames_meta_path(self) -> Path:
        """Where the input metadata lives.

        Returns:
            `<input_dir>/frames_meta.json`, as `SelectionOptions` spells it — one
            definition, so the two can never name different files.
        """
        return self.selection.frames_meta_path


@dataclass(frozen=True, slots=True)
class ResolvedRun:
    """`PipelineOptions` turned into what each stage actually takes, once per run.

    `PipelineOptions` is an *input adapter*: twenty-one fields spanning unrelated
    stages, kept exactly as the command line spells them and as `summary.json`
    records them. It used to be carried unresolved through the whole run, so a
    stage read whichever fields it happened to need, some combinations were inert
    (`--match-cap-mode off` and a `None` budget both meant "no cap"), and the
    CASPAR strings were not parsed until feature extraction, matching, loop
    closure and the pose graph had already been paid for.

    Resolution happens once, before the workspace is created, and it validates:
    `--caspar-option solver_iter_max=2.9` stops the run here rather than being
    truncated to 2 at the first solve. What cannot be resolved this early is the
    matcher's verification block, whose thresholds live in the config profile
    stage 1 reads — `matching_options` folds those in and is the only place that
    does.
    """

    selection: SelectionOptions
    """Stages 1 and 3: the metadata gates and the config directory."""
    features: FeatureOptions
    """Stage 2: which ALIKED runs, on what device, with how many threads."""
    matching_backend: MatchingBackend
    """Stage 4: which LightGlue runs."""
    match_limit: MatchLimitPolicy
    """Stages 4 and 5: how many verified matches one pair may keep, as one policy."""
    mapping: MappingOptions
    """Stages 7 and 7b: the mapper's own knobs, with the validated CASPAR overrides."""
    ba_plan: BaExecutionPlan
    """Which backend the bundle adjustments will run on, and why, decided before any solve.

    The same object `mapping` carries, so the run reports what it executed."""
    device: DeviceChoice
    """Device request the two ONNX stages share; `--use-gpu` resolved exactly once."""
    num_threads: int
    """Threads for extraction and matching; -1 lets COLMAP use every core."""

    def matching_options(self, config: CusfmConfig) -> MatchingOptions:
        """The matcher's settings, once the config profile's verification block is known.

        Args:
            config: The profile stage 1 read.

        Returns:
            The resolved matching settings, identical for stage 4 and stage 5's
            batch match — one object, so the two cannot drift.
        """
        return MatchingOptions(
            backend=self.matching_backend,
            max_error_px=config.matching_task_worker.verification.max_pixel_error,
            confidence=config.matching_task_worker.verification.min_ransac_confidence,
            num_threads=self.num_threads,
            device=self.device,
            match_limit=self.match_limit,
        )


def resolve_run(options: PipelineOptions) -> ResolvedRun:
    """Turn one command line into the per-stage settings the run will execute.

    Called first, before the output directory exists, so everything cheap is
    validated before anything is created or replaced.

    Args:
        options: The command line, verbatim.

    Returns:
        The resolved settings for every stage.

    Raises:
        FileNotFoundError: When a `raco` backend was asked for — which is the
            default — and this checkout has none of its graphs.
        ValueError: When a `--caspar-option` item is malformed, names an option
            that is not a settable numeric CASPAR knob, or gives an integer knob a
            fractional value.
    """
    check_raco_graphs(options.features_backend, options.matching_backend)
    ba_plan: BaExecutionPlan = resolve_ba_plan(
        options.ba_backend,
        options.caspar_option,
        ceres_polish=options.caspar_ceres_polish,
        # Not `options.optimize_extrinsics`: the *regularised* refinement — the
        # default — never asks a bundle adjuster to free `sensor_from_rig`. It
        # moves the extrinsics in pyceres and holds them fixed in every pycolmap
        # solve, so the mapping pass stays on the GPU. Only the unregularised
        # ablation is `run_mapping(..., optimize_extrinsics=True)`, which is the
        # call CASPAR throws on.
        optimize_extrinsics=options.optimize_extrinsics and not options.regularised_extrinsics,
    )
    # Said once, here, where the decision was made. The only thing left to decide is
    # the camera-model check, which needs the reconstruction stage 7 builds.
    ba_plan.announce()
    return ResolvedRun(
        selection=options.selection,
        features=FeatureOptions(
            backend=options.features_backend, num_threads=options.num_threads, device=options.device
        ),
        matching_backend=options.matching_backend,
        match_limit=MatchLimitPolicy.of(options.match_cap_mode, options.max_matches_per_pair),
        # The mapper receives the plan itself, not the knobs it was made from: there is
        # one decision about bundle adjustment in a run, and this is it.
        mapping=MappingOptions(
            num_threads=options.ba_num_threads, use_gpu=options.ba_use_gpu, ba_plan=ba_plan
        ),
        ba_plan=ba_plan,
        device=options.device,
        num_threads=options.num_threads,
    )


