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
from typing import Final, Literal, TypeAlias

from serde import serde

from colsfm import RESOURCES_DIR
from colsfm.ba_backend import BaBackend, BaExecutionPlan, resolve_ba_plan
from colsfm.config import CusfmConfig
from colsfm.features import DeviceChoice, FeatureBackend, FeatureOptions
from colsfm.frame_manifest import is_catalog_input
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
from colsfm.verification import VerifyChunking

DEFAULT_CONFIG_DIR: Final[Path] = RESOURCES_DIR / "configs" / "loop-closure-fixed"
"""`isaac` with loop closure repaired; what the blob reference runs used."""

LoopPairsInMapper: TypeAlias = Literal["pose-checked", "shortlisted"]
"""`PipelineOptions.loop_pairs_in_mapper`."""

DEFAULT_EXTRINSIC_ROUNDS: Final[int] = 3
"""Extrinsic refinement rounds; see `colsfm.extrinsic_solve.ExtrinsicRefinementOptions.num_rounds` for the measurement."""

DEFAULT_MIN_INTER_FRAME_DISTANCE_M: Final[float] = 0.10
"""Keyframe displacement gate. The gates are OR-combined with `>=`, so a 0.0 distance keeps every sample and silently
disables the angle gate too: that was the default until MSD, where it kept all 8,172 frames of MGO05 (15 Hz x 4 cameras)
and made mapping about O(n^2). 0.10 m is short against 1-3 m room depth. cuSFM's isaac default is 0.5 m and its backpack
profile 0.06 m; measured on the MSD MG inputs, 0.10 m / 15 deg keeps 15-50 % of the frames (MGO05: 8.7x faster,
ATE 1.19 / 0.63 cm against 0.97 / 0.57). A capture that needs every frame (the 0.93 s Galileo demo) passes 0.0 itself."""

DEFAULT_MIN_INTER_FRAME_ROTATION_DEGREES: Final[float] = 15.0
"""Keyframe rotation gate. Headsets turn 40-90 deg/s, so at 15 Hz the 5 deg isaac gate still fires on most frames;
15 deg of a ~100 deg field of view leaves about 85 % overlap between neighbouring keyframes."""


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
    min_inter_frame_distance: float = DEFAULT_MIN_INTER_FRAME_DISTANCE_M
    """Keyframe displacement gate in metres (`DEFAULT_MIN_INTER_FRAME_DISTANCE_M`); 0.0 keeps every sample, the angle gate included."""
    min_inter_frame_rotation_degrees: float = DEFAULT_MIN_INTER_FRAME_ROTATION_DEGREES
    """Keyframe rotation gate in degrees, OR-combined with the distance gate (`DEFAULT_MIN_INTER_FRAME_ROTATION_DEGREES`)."""
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
    --optimize-extrinsics --loop-closure`, CASPAR alone (no closing Ceres solve) and the
    regularised refinement. Every field below says what its own ablation costs, and
    this class is also the CLI's help text, so the two cannot drift.
    """

    input_dir: Path
    """Directory holding `frames_meta.json` and the images its `image_name`s name."""
    output_dir: Path
    """Workspace root, cuSFM's `cusfm_base_dir`; created if missing."""
    config_dir: Path = DEFAULT_CONFIG_DIR
    """Directory of `.pb.txt` configs; the loop-closure-fixed isaac profile by default."""
    min_inter_frame_distance: float = DEFAULT_MIN_INTER_FRAME_DISTANCE_M
    """Keyframe displacement gate in metres (`DEFAULT_MIN_INTER_FRAME_DISTANCE_M`); 0.0 keeps every sample, the angle gate included."""
    min_inter_frame_rotation_degrees: float = DEFAULT_MIN_INTER_FRAME_ROTATION_DEGREES
    """Keyframe rotation gate in degrees, OR-combined with the distance gate (`DEFAULT_MIN_INTER_FRAME_ROTATION_DEGREES`)."""
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
    extrinsic_refinement_rounds: int = DEFAULT_EXTRINSIC_ROUNDS
    """Ceiling on the extrinsics-then-poses rounds the regularised refinement may take (`DEFAULT_EXTRINSIC_ROUNDS`); it
    also stops early on `ExtrinsicRefinementOptions`' tolerances."""
    refine_camera_model: bool = False
    """Free every camera's focal length (fx, fy), its distortion block and the rig (`sensor_from_rig`) in every mapping round.

    cuSFM holds the calibration fixed, so a nominal model that is off ends up in the poses: Mission office's cube faces
    are 1-1.4 % off f = 800 and 0.12-0.20 deg off the nominal rig. With Spirula's matches, colsfm's mapper ends 0.2645
    deg from Spirula's poses with the model held and 0.034 deg with it free (focals within 0.5 px). The rig is solved on
    a camera-referenced rig and replaces stage 7b; on CASPAR when the build's probe refines `sensor_from_rig`
    (recipes/colmap-src caspar-rig-refine: 46 s of mapping on Mission office), on Ceres otherwise.

    Only for a rig whose calibration is nominal: unregularised, it overfits a calibrated one (MSD MGO05 with prior-overlap
    pairs and the merge cap: ATE 14.2 mm with it, 8.2 mm without, main 10.6-12.0 mm). On Ceres the Mission office
    reconstruction takes 640 s (4 threads) against 55 s with the model held."""
    loop_pairs_in_mapper: LoopPairsInMapper = "pose-checked"
    """Which of the loop stage's image pairs and inliers the mapper triangulates from.

    `"pose-checked"` (default): every pair the stage matched, each keeping only the inliers that agree with its rig
    pair's measured pose (verified candidates) or, without one, with the prior relative pose within a drift tolerance
    (`colsfm.loop_mapper_check`); the database keeps every row. `"shortlisted"`: every pair the stage matched with all
    its inliers (fork decision 15), so cross-revisit tracks reach the triangulator even when their loop edge was
    rejected. cuSFM's way, only the image pairs of the loop edges that were kept, was removed: it cost MSD MGO05 0.23 cm
    SE(3) (1.41 against 1.18 cm) and put KITTI 06 back at cuSFM's 1.328 m, because the kitti profile's loop edge gates
    reject every edge and its revisits reach the map only through these pairs (0.845 m with `"shortlisted"`)."""
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
    verify: VerifyChunking = VerifyChunking()
    """Two-view verification's chunk size, worker threads and chunks in flight (`colsfm.verification`): memory holds
    about `verify_chunks_in_flight` chunks whatever the run's length, on every machine."""
    ba_backend: BaBackend = "caspar"
    """Which implementation solves the bundle adjustments — Ceres on the CPU, or COLMAP's
    CASPAR on the GPU. `caspar` by default (NOTES.md decision 17): a fraction of the Ceres
    mapping time at Ceres' accuracy (recipes/colmap-src `patches/PROVENANCE.md`), and where the
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
    caspar_ceres_polish: bool = False
    """Finish a `--ba-backend caspar` run with one Ceres global bundle adjustment (Cauchy loss). Off by default:
    CASPAR alone reaches its least-squares minimum (measurements: recipes/colmap-src `patches/PROVENANCE.md`).
    Inert on `ceres`, fallbacks included, which have already converged."""
    caspar_option: tuple[str, ...] = ()
    """CASPAR solver overrides as `name=value`, e.g. `--caspar-option solver_iter_max=1000 reduced_pcg_iter_max=200`.
    Read only when `--ba-backend caspar` actually runs; names in `colsfm.ba_backend.CasparOptions`. The PCG runs on the
    reduced camera system (`reduced_pcg_iter_max`, `reduced_pcg_eta`) unless `reduced_camera_system=false` or a
    full-system knob (`pcg_iter_max`, `pcg_rel_error_exit`, `pcg_rel_score_exit`, `pcg_rel_decrease_min`) selects the
    full system; mixing the two is refused before the run starts."""
    max_matches_per_pair: int | None = BLOB_MATCH_TOP_K
    """Verified matches kept per pair after the spatial subsample; None keeps every inlier.
    The blob's `match_top_k`; see `colsfm.matching.subsample_matches_by_coverage`."""
    match_cap_mode: MatchCapMode = "fixed"
    """How `max_matches_per_pair` becomes a per-pair number; see
    `colsfm.matching.MatchLimitPolicy`. `fixed` is the 500 every run before the KITTI
    follow-up used, `image_area` scales it with the frame's resolution, `off` keeps every
    verified inlier."""
    align_to_prior: bool = False
    """Before the run is published, move the whole result into the pose prior's gauge with one similarity fitted on the registered
    cameras against their input poses (`colsfm.prior_alignment`): `sparse/`, `kpmap/`, the TUM files and the catalog
    layer. For a metric prior (visual-inertial SLAM) whose frame the rest of a pipeline lives in; bundle adjustment
    itself lets scale and heading drift (scale 0.80 on an Insta360 walk-through)."""
    raco_batch_pixels: int | None = None
    """On `raco` only: pixels (batch x height x width) one shape-dynamic RaCo execution may hold, which sets each size
    group's batch ceiling and so its execution context's memory (about 1.2 GB per million pixels on the 5090). None keeps
    `colsfm.features_raco.BATCH_PIXEL_BUDGET` (2 x 1216 x 1920). Lower it on a shared GPU: 1920000 runs 800x800
    cube faces three at a time in a 2.3 GB context instead of seven in 5.3 GB."""
    features_backend: FeatureBackend = "raco"
    """Which ALIKED runs: RaCo-ALIKED on its own TensorRT engine (`colsfm.features_raco`) or COLMAP's
    own ONNX one.

    `raco` by default (NOTES.md decision 17): 2.6 s of Galileo's 20.0 s total against
    6.0 s for the blob's engine and 8.3 s for COLMAP's, at the same accuracy. It needs
    the graphs under `colsfm.MODEL_DIR`, which are not committed; a checkout without them
    is told to export them or to ask for `--features-backend pycolmap`
    (`colsfm.model_assets`), never quietly given another extractor.
    See `colsfm.features.FeatureBackend`."""
    matching_backend: MatchingBackend = "flexglue"
    """Which matcher runs: COLMAP's FlexGlue, LightGlue+ on TensorRT (`colsfm.matching_trt`), or COLMAP's own ONNX
    LightGlue. See `colsfm.matching.MatchingBackend`.

    `flexglue` by default since 2026-11-15: on RaCo-ALIKED keypoints it beat LightGlue+ on both datasets measured
    (MSD MGO07 SE(3) ATE 1.33 vs 2.08 cm with twice the points; Galileo 4.48 vs 5.17 mm, 226 vs 224 registered) at
    about 1.4x LightGlue+'s per-pair time (20.9 vs 14.6 ms on the 5090). `raco` (LightGlue+) remains the matcher
    trained on these descriptors and the one with per-match scores for SSC."""

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
    verify: VerifyChunking
    """Stages 4 and 5: how two-view verification is chunked and overlapped."""

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
            image_dir=self.selection.input_dir,
            verify=self.verify,
        )


def plan_runs_extrinsic_stage(options: PipelineOptions) -> bool:
    """Whether stage 7b runs: `--optimize-extrinsics`, unless `--refine-camera-model` already freed the rig in the mapper.

    Args:
        options: The command line.

    Returns:
        True when the run refines its extrinsics after mapping.
    """
    return options.optimize_extrinsics and not options.refine_camera_model


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
            fractional value; or when a catalog input asks for a backend that reads image files.
    """
    check_raco_graphs(options.features_backend, options.matching_backend)
    if is_catalog_input(options.input_dir):
        # COLMAP's own extractor and its ONNX Runtime FlexGlue read image files; a catalog input has none.
        file_readers: list[str] = [
            flag
            for flag, chosen in (
                ("--features-backend pycolmap", options.features_backend == "pycolmap"),
                ("--matching-backend flexglue_ort", options.matching_backend == "flexglue_ort"),
            )
            if chosen
        ]
        if file_readers:
            raise ValueError(
                f"{options.input_dir} is a catalog input (its keyframes live in the segment's video streams, not in files), and "
                f"{' and '.join(file_readers)} read image files; use --features-backend raco and --matching-backend flexglue"
            )
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
        optimize_extrinsics=(options.optimize_extrinsics and not options.regularised_extrinsics) or options.refine_camera_model,
    )
    # Said once, here, where the decision was made. The only thing left to decide is
    # the camera-model check, which needs the reconstruction stage 7 builds.
    ba_plan.announce()
    return ResolvedRun(
        selection=options.selection,
        features=FeatureOptions(
            backend=options.features_backend, num_threads=options.num_threads, device=options.device, raco_batch_pixels=options.raco_batch_pixels
        ),
        matching_backend=options.matching_backend,
        match_limit=MatchLimitPolicy.of(options.match_cap_mode, options.max_matches_per_pair),
        # The mapper receives the plan itself, not the knobs it was made from: there is
        # one decision about bundle adjustment in a run, and this is it.
        mapping=MappingOptions(
            num_threads=options.ba_num_threads,
            use_gpu=options.ba_use_gpu,
            ba_plan=ba_plan,
            optimize_extrinsics=options.refine_camera_model,
            refine_focal_length=options.refine_camera_model,
        ),
        ba_plan=ba_plan,
        device=options.device,
        num_threads=options.num_threads,
        verify=options.verify,
    )
