"""LightGlue matching plus geometric verification — the replacement for `feature_matcher_main`.

COLMAP 4.2 ships `FeatureMatcherType.ALIKED_LIGHTGLUE` and downloads the ONNX
graph itself into `~/.cache/colmap/`, so the whole stage is a
`pycolmap.match_image_pairs` call over a pair list, with verification running
inside COLMAP. Its graph is not the blob's — see `MatchingOptions.model_path`.

The blob's own pipeline is LightGlue -> score gate -> spatial NMS -> essential
matrix RANSAC (feature_matcher_main.md §5). Three of those four map straight
across; the fourth does not:

| blob | here |
|---|---|
| `mscores0 > match_threshold` (0.3) | `LightGlueONNXMatchingOptions.min_score` |
| SSC NMS down to `match_top_k: 500` | `match_limit` — a score-free grid subsample, see below |
| `cv::findEssentialMat`, threshold `0.5*(err/f0 + err/f1)` on bearings | `TwoViewGeometryOptions.ransac.max_error = 4.0` px on a real camera model |
| `n_inliers < 10` empties the pair | `TwoViewGeometryOptions.min_num_inliers = 10` |

**The SSC divergence.** The blob thins each pair's matches to a spatially uniform
top-500 using the LightGlue score as the keypoint response, which is what turns
its ~931 raw matches per Galileo pair into ~430. pycolmap exposes no per-match
score: `Database.read_matches` and `FeatureMatcher.match` both return bare
`uint32[m, 2]` index pairs, so neither the sort key nor SSC's "strongest per
cell" rule exists on this side. What the blob is *buying* with SSC — spatial
uniformity and a bounded track density — is reachable without a score, so
`subsample_matches_by_coverage` lays a fixed grid of about `match_top_k` cells
over image 0 and keeps one match per occupied cell, ties broken by keypoint
index. It runs **after** verification, so every survivor is an inlier of the same
RANSAC; only the count changes.

What binds is the number of *occupied* cells, not the cap: ALIKED keypoints
cluster on texture, so a 1920x1200 Galileo pair fills about 160 of its 476 cells
and the survivors come out well under 500. Measured over the 331 Galileo pairs:

| | mean matches per pair | mean reprojection | ATE vs ground truth |
|---|---|---|---|
| uncapped | 1300 | 1.80 px | 5.39 mm |
| `match_limit` fixed at 500 | 156 | 1.33 px | 4.32 mm |
| blob (SSC, `match_top_k: 500`) | 430 | 1.55 px | 5.00 mm |

So the cap costs nothing in accuracy — it *buys* it, by refusing to pile a
thousand near-collinear observations from one image region onto one track — and
it takes Galileo's bundle adjustment from 27.7 s to 2.8 s.

**Two phases, not one.** `match_image_pairs` can verify inline, but when a pair
fails verification COLMAP stores neither the geometry *nor the raw matches* —
measured: a pair with 400 LightGlue matches and `min_num_inliers = 1e6` leaves an
empty `matches` row. Reporting "raw matches and inliers per pair", which is what
tells a stale threshold apart from a genuinely textureless pair, therefore needs
matching and verification split: `match_image_pairs` with
`skip_geometric_verification`, then `verify_and_report`, which verifies the stored
matches in bounded chunks on worker threads (`colsfm.verification`, COLMAP's own
estimator and write rules). That is also the blob's order. A pair must not already
have a geometry row (the write would collide); at the pinned COLMAP the
skipped-verification pass writes none (older COLMAP wrote an `UNDEFINED`
placeholder per pair), and `verify_and_report` checks that none exists.

**The score threshold.** The blob's 0.3 is calibrated for its own LightGlue
engine. COLMAP's graph scores differently, and at 0.3 the low-texture
`left_stereo_*` pairs collapse: 22 of 331 Galileo pairs come back empty (6.6 %)
against the blob's 8 (2.4 %). The default here is COLMAP's own 0.1, measured at
13 empty pairs (3.9 %) with a 10th-percentile match count of 116 against the
blob's 118. `BLOB_MATCH_THRESHOLD` is kept for parity runs.

**The LightGlue+ backend.** `MatchingOptions.backend = "raco"` runs LightGlue+ (`colsfm.matching_trt`) on a TensorRT
engine (a run's default matcher is FlexGlue, below). It *does* see the per-match score, so it runs the real SSC
(`select_by_square_covering`) before verification, at the blob's own 0.3 threshold, and skips the grid subsample below
entirely. Everything above this paragraph describes the `pycolmap` path, the ablation for a machine with no engine.
Both leave the same two database tables filled, so the choice is invisible to every later stage. (The fork's third
backend, the cuSFM blob's own LightGlue engine, is not part of this package; `colsfm.matching_trt` keeps the engine
matching loop LightGlue+ runs through.)

**FlexGlue.** `MatchingOptions.backend = "flexglue"` runs FlexGlue's graphs (cvg/FlexGlue, COLMAP main #4908) as TensorRT
engines from `colsfm.flexglue_trt`; `"flexglue_ort"` runs COLMAP's own ONNX Runtime FlexGlue through the same
`match_image_pairs` call as the `pycolmap` path (the parity reference: 20 MGO07 pairs agree to within 2 matches). It reads only keypoint *locations* (RaCo-ALIKED's work as well
as any other extractor's), computes dense DINOv3-ConvNeXt features from the image pixels (so it needs `image_dir`), and
predicts each pair's covisibility: a pair below `flexglue_min_covisibility` comes back empty, which is what rejects
look-alike loop candidates. The ONNX graphs are local files from `colsfm-download-flexglue`, never fetched at run time.
"""

from __future__ import annotations

import math
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Final, Literal, TypeAlias

import numpy as np
import pycolmap
from jaxtyping import Float32, Int64
from numpy import ndarray

from colsfm import DATA_DIR
from colsfm.database import (
    ImagePair,
    Keypoints,
    KeypointsXY,
    MatchIndices,
    count_two_view_geometries,
    read_keypoints_batch,
)
from colsfm.features import DeviceChoice, ResolvedDevice, resolve_device
from colsfm.frame_manifest import FrameSource
from colsfm.pairs import write_pair_list
from colsfm.verification import (
    CapFunction,
    ChunkedVerifier,
    PairToVerify,
    VerifiedPair,
    VerifyChunking,
    write_verified,
)

FLEXGLUE_MODEL_DIR: Final[Path] = DATA_DIR / "models" / "flexglue"
"""Where `colsfm-download-flexglue` puts FlexGlue's ONNX graphs (gitignored, like the RaCo exports)."""

FLEXGLUE_MODEL_FILES: Final[dict[str, str]] = {
    "descriptor_model_path": "flexglue_descriptor.onnx",
    "matcher_model_path": "flexglue_matcher_early_stop.onnx",
    "matcher_model_path_fp16": "flexglue_matcher_early_stop_fp16.onnx",
}
"""`pycolmap.FlexGlueMatchingOptions` field to the graph it loads (cvg/FlexGlue v0.1, as COLMAP pins them)."""


@dataclass(frozen=True, slots=True)
class FlexGlueOptions:
    """FlexGlue's matcher thresholds, with COLMAP's defaults (`FlexGlueMatchingOptions`)."""

    filter_threshold: float = 0.1
    """Minimum match confidence (geometric mean of both directions)."""
    max_distance_px: float = 3.0
    """Keypoint-to-correspondence distance gate in the 512x512 frame; COLMAP passes infinity for a non-positive value."""
    min_covisibility: float = 0.5
    """A pair whose predicted covisibility is below this keeps no match."""
    early_stop_decay: float = 0.5
    """The early-stop gate after layers 2, 4, 6 and 8 (`min_covisibility * (1 - decay)^(9 - k)`); 1 disables it."""

    def tag(self) -> str:
        """A file-name-safe key for the folded graph and its engine.

        Returns:
            E.g. `ft0.1_md3_mc0.5_es0.5`.
        """
        return f"ft{self.filter_threshold:g}_md{self.max_distance_px:g}_mc{self.min_covisibility:g}_es{self.early_stop_decay:g}"

    def fp8_file(self) -> str:
        """The FP8 matcher calibrated for these thresholds (`colsfm.flexglue_fp8`); it sits beside the downloaded graphs.

        Returns:
            E.g. `flexglue_matcher_fp8_ft0.1_md3_mc0.5_es0.5.onnx`.
        """
        return f"flexglue_matcher_fp8_{self.tag()}.onnx"


FLEXGLUE_FP8_COMMAND: Final[str] = "pixi run -e colsfm colsfm-flexglue-fp8"
"""Calibrates `FlexGlueOptions.fp8_file` on the Galileo frames (`tools/flexglue_fp8.py --image-dirs` for other frames)."""

FP8_MIN_COMPUTE_CAPABILITY: Final[tuple[int, int]] = (8, 9)
"""The first CUDA compute capability with FP8 tensor cores (Ada, sm_89); TensorRT refuses FP8 Q/DQ graphs below it."""


def gpu_has_fp8() -> bool:
    """Whether the current CUDA device runs FP8 (`FP8_MIN_COMPUTE_CAPABILITY`). Without it (an Ampere RTX 3060, sm_86)
    the FlexGlue matcher runs its prepared bf16/fp32 graph, the matcher before the FP8 calibration, and needs no FP8 file.

    Returns:
        True on Ada and later; False on older GPUs and on hosts without CUDA.
    """
    import torch

    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= FP8_MIN_COMPUTE_CAPABILITY


MatcherVariant: TypeAlias = Literal["ALIKED_LIGHTGLUE", "ALIKED_BRUTEFORCE"]
"""The ALIKED matchers COLMAP 4.2 offers; the pipeline uses LightGlue."""

MatchingBackend: TypeAlias = Literal["pycolmap", "raco", "flexglue", "flexglue_ort"]
"""Which matcher runs: COLMAP's own ONNX LightGlue, LightGlue+ on a TensorRT engine, or COLMAP's FlexGlue.

`flexglue` is COLMAP's FlexGlue graphs on TensorRT (`colsfm.flexglue_trt`); `flexglue_ort` is COLMAP's own ONNX Runtime
FlexGlue, kept as its parity reference. Both ignore the descriptors.

`raco` is `colsfm.matching_trt`, the LightGlue+ that fabio-sim trained against RaCo-ALIKED. It has the per-match
score COLMAP's bindings hide and therefore runs the blob's real SSC spatial NMS rather than
`subsample_matches_by_coverage`'s score-free stand-in.

A run asks for `flexglue` by default (`colsfm.run_config.PipelineOptions`); `raco` is the matcher trained against the
default extractor and keeps per-match scores for SSC. `pycolmap` is the ablation, and the default of `MatchingOptions`
below, which is the *stage's* API rather than the run's."""

MatchCapMode: TypeAlias = Literal["fixed", "image_area", "off"]
"""`--match-cap-mode`, the command line's spelling; `MatchLimitPolicy.of` resolves it."""

MatchLimitKind: TypeAlias = Literal["unlimited", "fixed", "image_area"]
"""A resolved match limit. `off` and a `None` count both become `unlimited` exactly once."""

MatchScores: TypeAlias = Float32[ndarray, " num_matches"]
"""LightGlue's confidence per match, which only the TensorRT backend can see."""

DEFAULT_MIN_SCORE: Final[float] = 0.1
"""COLMAP's own `LightGlueONNXMatchingOptions.min_score`, measured best on Galileo."""

BLOB_MATCH_THRESHOLD: Final[float] = 0.3
"""`lightglue_params.match_threshold` in `matching_task_worker_config.pb.txt`.

Applied by the blob with a strict `>`. Calibrated for the blob's own engine; on
COLMAP's graph it empties 6.6 % of Galileo's pairs, so it is not the default.
"""

BLOB_MIN_NUM_INLIERS: Final[int] = 10
"""`SelectMatchesByEpipolarGeometry` empties a pair below this, rather than trimming it."""

BLOB_MAX_PIXEL_ERROR: Final[float] = 4.0
"""`verification_config.max_pixel_error`; COLMAP takes it in pixels directly."""

BLOB_RANSAC_CONFIDENCE: Final[float] = 0.98
"""`verification_config.min_ransac_confidence`; COLMAP's own default is 0.999."""

BLOB_MATCH_TOP_K: Final[int] = 500
"""`lightglue_params.match_top_k`; the match budget the blob's SSC NMS aims at."""

BLOB_NUM_POINTS_TOLERANCE_FRACTION: Final[float] = 0.3
"""`lightglue_params.num_points_tolerance_fraction`; SSC's acceptance band around `match_top_k`."""

CAP_REFERENCE_IMAGE_AREA_PX: Final[int] = 1920 * 1200
"""The image area `BLOB_MATCH_TOP_K` was calibrated on, i.e. one Galileo camera.

`match_cap_mode = "image_area"` keeps the *match density* of that calibration
rather than its count: one kept match per `1920*1200 / 500 = 4608` pixels of
image. Galileo therefore reproduces 500 exactly and KITTI's 1241x376 frames get
101. See `MatchLimitPolicy` and the KITTI follow-up in `docs/kitti-06-results.md`
for what that measured."""


@dataclass(frozen=True, slots=True)
class MatchLimitPolicy:
    """How many verified matches one pair may keep — one answer, not two knobs.

    `match_cap_mode` and `max_matches_per_pair` used to be separate fields, and two
    different spellings disabled the limit: `mode="off"`, and a `None` count. A
    caller reading either one alone could therefore be wrong, and both the grid
    subsample and the TensorRT matcher had to test both. Here "no limit" has
    exactly one representation, and `limit_for` is the only place the arithmetic
    lives.

    Three policies, because 500 was calibrated on one image size and one dataset
    and is not obviously the right number anywhere else:

    * **`unlimited`** — every verified inlier survives.
    * **`fixed`** — `matches_per_pair` verbatim. What every run before the KITTI
      follow-up used.
    * **`image_area`** — the same *density*, one kept match per
      `CAP_REFERENCE_IMAGE_AREA_PX / matches_per_pair` pixels, so a 1920x1200 frame
      keeps 500 and a 1241x376 KITTI frame keeps 101.
    """

    kind: MatchLimitKind
    """Which of the three policies applies."""
    matches_per_pair: int | None
    """The budget the policy is parameterised by; None exactly when `kind` is `unlimited`."""

    @staticmethod
    def of(mode: MatchCapMode, max_matches_per_pair: int | None) -> MatchLimitPolicy:
        """Fold the command line's two knobs into one policy.

        Args:
            mode: `--match-cap-mode`.
            max_matches_per_pair: `--max-matches-per-pair`; None means no limit
                whatever the mode says.

        Returns:
            The policy those two spell, with "no limit" normalised to one form.
        """
        if mode == "off" or max_matches_per_pair is None:
            return MatchLimitPolicy(kind="unlimited", matches_per_pair=None)
        return MatchLimitPolicy(kind=mode, matches_per_pair=max_matches_per_pair)

    @property
    def limits(self) -> bool:
        """Whether anything is capped at all.

        Returns:
            False for `unlimited`, True otherwise.
        """
        return self.kind != "unlimited"

    def limit_for(self, image_width: int, image_height: int) -> int | None:
        """The number of matches one pair may keep, given its first image's size.

        Args:
            image_width: Width of the pair's first image in pixels.
            image_height: Height of the pair's first image in pixels.

        Returns:
            The per-pair cap, or None when nothing is capped.
        """
        if self.matches_per_pair is None:
            return None
        if self.kind == "fixed":
            return self.matches_per_pair
        area: int = max(image_width * image_height, 1)
        return max(round(self.matches_per_pair * area / CAP_REFERENCE_IMAGE_AREA_PX), 1)


DEFAULT_MATCH_LIMIT: Final[MatchLimitPolicy] = MatchLimitPolicy(kind="fixed", matches_per_pair=BLOB_MATCH_TOP_K)
"""The blob's own budget: 500 verified matches per pair, whatever the image size."""


@dataclass(frozen=True, slots=True)
class MatchingOptions:
    """How LightGlue and the two-view verifier run over a pair list."""

    backend: MatchingBackend = "pycolmap"
    """Which matcher runs; see `MatchingBackend`.

    `pycolmap` here even though a *run* defaults to `flexglue`, for the reason
    `colsfm.features.FeatureOptions.backend` gives: this is the stage's API, and it
    must work where no engine does. `colsfm.run_config` always passes the run's
    choice explicitly.

    On `raco` the `variant`, `model_path`, `min_score`, `max_num_matches`,
    `skip_image_pairs_in_same_frame`, `device` and `gpu_index` fields are not
    read: the engine is LightGlue+, its score gate is
    `tensorrt_min_score`, its thinning is `select_by_square_covering`, and it
    runs on CUDA device 0 or not at all. Verification is the same
    `colsfm.verification` path on every backend, so `min_num_inliers`,
    `max_error_px`, `confidence` and `verify` still apply."""
    variant: MatcherVariant = "ALIKED_LIGHTGLUE"
    """Matcher COLMAP runs; LightGlue is what the blob uses."""
    min_score: float = DEFAULT_MIN_SCORE
    """LightGlue score gate; see the module docstring for why this is not the blob's 0.3."""
    model_path: Path | None = None
    """Explicit LightGlue ONNX file; None lets COLMAP download and cache its own.

    Not a slot for the blob's `pycusfm/models/aliked_lightglue/lightglue_aliked.onnx`:
    that graph takes four inputs (`kpts0/1`, `desc0/1`) and returns `matches0` as
    an int32 `[num_matches, 2]` index matrix, while COLMAP's own
    `aliked-lightglue.onnx` takes six (adding `image_size0/1`, so normalisation
    happens inside the graph) and returns an int64 `[num_keypoints0]` assignment
    vector. `CreateLightGlueONNXFeatureMatcher` rejects the mismatch by throwing
    inside a worker thread, which aborts the process.
    """
    max_num_matches: int = 32768
    """COLMAP's cap on matches per pair; never binds at 2048 keypoints per image."""
    min_num_inliers: int = BLOB_MIN_NUM_INLIERS
    """Inlier count below which the pair is emptied, matching the blob's rule."""
    max_error_px: float = BLOB_MAX_PIXEL_ERROR
    """RANSAC inlier threshold in pixels."""
    confidence: float = BLOB_RANSAC_CONFIDENCE
    """RANSAC success probability driving the trial count."""
    skip_image_pairs_in_same_frame: bool = False
    """Must stay False: the stereo pairs of one rig frame are exactly what cuSFM matches."""
    device: DeviceChoice = "auto"
    """Compute device; `auto` falls back to the CPU provider when cuDNN is missing."""
    gpu_index: str = "-1"
    """CUDA device index as COLMAP's comma-separated string."""
    num_threads: int = -1
    """COLMAP matching threads; -1 lets COLMAP choose."""
    match_limit: MatchLimitPolicy = DEFAULT_MATCH_LIMIT
    """How many verified matches one pair may keep, as one resolved policy.

    The score-free stand-in for the blob's SSC spatial NMS — see
    `subsample_matches_by_coverage` and the module docstring. On the `raco`
    backend the number is the SSC *target* instead, applied before verification
    with the real LightGlue scores, and the post-verification grid subsample does
    not run at all. `MatchLimitPolicy.of` is what turns `--match-cap-mode` and
    `--max-matches-per-pair` into it."""
    tensorrt_min_score: float = BLOB_MATCH_THRESHOLD
    """LightGlue score gate on the engine (`raco`) backend, applied with `>` as the blob does.

    A second field rather than a second default for `min_score`, because the two
    graphs are not the same graph: 0.3 is the blob's engine threshold, which the
    LightGlue+ path keeps, while `min_score` is calibrated for COLMAP's, which the
    `pycolmap` backend runs (see the module docstring's "score threshold" note)."""
    num_points_tolerance_fraction: float = BLOB_NUM_POINTS_TOLERANCE_FRACTION
    """SSC's acceptance band around the target, as a fraction of it; `raco` only."""
    image_dir: Path | None = None
    """Root the database's image names resolve against; required by `flexglue`, which reads the pixels."""
    flexglue_model_dir: Path = FLEXGLUE_MODEL_DIR
    """Where `colsfm-download-flexglue` put the three FlexGlue graphs; `flexglue` only."""
    flexglue_min_covisibility: float = 0.5
    """FlexGlue's pair rejection: a pair whose predicted covisibility is below this keeps no match (COLMAP's default)."""
    verify: VerifyChunking = VerifyChunking()
    """How two-view verification is chunked and overlapped (`colsfm.verification`)."""


DEFAULT_MATCHING_OPTIONS: Final[MatchingOptions] = MatchingOptions()
"""Shared immutable default, so the signatures below hold no constructor call."""


@dataclass(frozen=True, slots=True)
class PairMatchStats:
    """What one image pair produced."""

    raw_matches: int
    """Matches LightGlue kept before verification.

    On the `pycolmap` backend that is after the score gate only, because the grid
    subsample runs later; on `raco` it is after the score gate *and* SSC,
    because that is where the blob puts its spatial NMS too. `retention` is
    therefore comparable to the blob's published 0.985 on the TensorRT path and
    is a different quantity on the pycolmap one."""
    inlier_matches: int
    """Matches surviving two-view verification; 0 when the pair was emptied."""

    @property
    def retention(self) -> float:
        """Inliers as a fraction of raw matches.

        Returns:
            The ratio, or 0.0 when the pair had no raw matches. The blob's median
            on Galileo is 0.985 — verification is a light filter, not a gate.
        """
        return 0.0 if self.raw_matches == 0 else self.inlier_matches / self.raw_matches


@dataclass(frozen=True, slots=True)
class MatchReport:
    """What one matching run produced, per pair and in aggregate."""

    pair_stats: dict[ImagePair, PairMatchStats] = field(default_factory=dict)
    """Raw and inlier counts keyed by the pair as given."""
    device: ResolvedDevice = "cpu"
    """The device the run actually used."""
    elapsed_seconds: float = 0.0
    """Wall time of both phases: LightGlue matching and two-view verification."""

    @property
    def empty_pairs(self) -> int:
        """How many pairs came back with no verified match.

        Returns:
            The count of pairs whose inlier list is empty.
        """
        return sum(1 for stats in self.pair_stats.values() if stats.inlier_matches == 0)

    @property
    def median_inliers(self) -> float:
        """Median verified match count over all pairs.

        Returns:
            The median, or 0.0 when nothing was matched.
        """
        if not self.pair_stats:
            return 0.0
        return float(median(stats.inlier_matches for stats in self.pair_stats.values()))

    @property
    def median_retention(self) -> float:
        """Median inlier fraction over the pairs that produced any raw match.

        Returns:
            The median retention, or 0.0 when no pair produced a raw match.
        """
        retentions: list[float] = [stats.retention for stats in self.pair_stats.values() if stats.raw_matches > 0]
        return float(median(retentions)) if retentions else 0.0

    def summary(self) -> str:
        """One line naming the pair count, the inlier median and the wall time.

        Returns:
            A printable summary; `"no pairs"` when nothing was matched.
        """
        if not self.pair_stats:
            return f"matched no pairs on {self.device} in {self.elapsed_seconds:.2f}s"
        empty_fraction: float = 100.0 * self.empty_pairs / len(self.pair_stats)
        return (
            f"matched {len(self.pair_stats)} pairs on {self.device} in {self.elapsed_seconds:.2f}s, "
            f"median inliers {self.median_inliers:.0f}, retention {self.median_retention:.3f}, "
            f"{self.empty_pairs} empty ({empty_fraction:.1f}%)"
        )


def _matching_options(options: MatchingOptions, device: ResolvedDevice) -> pycolmap.FeatureMatchingOptions:
    """Build the pycolmap matcher options.

    Args:
        options: The typed settings.
        device: The already-resolved device.

    Returns:
        `FeatureMatchingOptions` with the ALIKED LightGlue sub-options filled in.
    """
    matching_options: pycolmap.FeatureMatchingOptions = pycolmap.FeatureMatchingOptions()
    if options.backend == "flexglue_ort":
        if options.image_dir is None:
            raise ValueError("the flexglue matcher reads image pixels: MatchingOptions.image_dir is required")
        matching_options.type = pycolmap.FeatureMatcherType.FLEXGLUE
        flexglue = matching_options.flexglue
        flexglue.image_path = str(options.image_dir)
        flexglue.min_covisibility = options.flexglue_min_covisibility
        for field_name, file_name in FLEXGLUE_MODEL_FILES.items():
            setattr(flexglue, field_name, str(options.flexglue_model_dir / file_name))
    else:
        matching_options.type = getattr(pycolmap.FeatureMatcherType, options.variant)
    matching_options.use_gpu = device == "cuda"
    matching_options.gpu_index = options.gpu_index
    matching_options.num_threads = options.num_threads
    matching_options.max_num_matches = options.max_num_matches
    matching_options.skip_image_pairs_in_same_frame = options.skip_image_pairs_in_same_frame
    matching_options.aliked.lightglue.min_score = options.min_score
    if options.model_path is not None:
        matching_options.aliked.lightglue.model_path = str(options.model_path)
    return matching_options


def verification_options(options: MatchingOptions) -> pycolmap.TwoViewGeometryOptions:
    """Build the two-view geometry options that mirror cuSFM's verifier.

    COLMAP works in pixels against the real camera model, so the blob's
    `MapErrorToNormalizedPlane` conversion (`0.5*(err/f0 + err/f1)` on bearing
    vectors) is not needed: `max_error` takes the 4.0 px straight from the
    config.

    Args:
        options: The typed settings.

    Returns:
        `TwoViewGeometryOptions` carrying the blob's thresholds.
    """
    geometry_options: pycolmap.TwoViewGeometryOptions = pycolmap.TwoViewGeometryOptions()
    geometry_options.min_num_inliers = options.min_num_inliers
    geometry_options.ransac.max_error = options.max_error_px
    geometry_options.ransac.confidence = options.confidence
    return geometry_options


def subsample_matches_by_coverage(
    keypoints_xy: KeypointsXY,
    matches: MatchIndices,
    image_width: int,
    image_height: int,
    max_matches: int,
) -> MatchIndices:
    """Thin a pair's matches to one per grid cell, so the survivors cover the image.

    The score-free replacement for the blob's SSC non-maximum suppression
    (feature_matcher_main.md §5.3). SSC sorts the matched keypoints by their
    LightGlue score and binary-searches a suppression radius until roughly
    `match_top_k` survive; pycolmap exposes no per-match score, so neither the
    sort key nor the "strongest per cell" rule is reachable *on this backend*.
    What *is* reachable is the property the blob is buying with it — spatial
    uniformity — so this lays a fixed grid over image 0 and keeps one match per
    occupied cell. The TensorRT backend does see the score and runs the real
    thing instead; that is `select_by_square_covering`.

    The grid is `num_columns = floor(sqrt(max_matches * width / height))` by
    `num_rows = floor(max_matches / num_columns)`, i.e. cells of roughly
    `sqrt(width * height / max_matches)` on a side, rounded so the cell count
    never exceeds `max_matches`. That makes the cap a real maximum rather than
    the blob's plus-or-minus 30 % target band.

    Ties inside a cell are broken by the image-0 keypoint index, which makes the
    result a pure function of the inputs; nothing here depends on match order or
    on a random seed.

    Args:
        keypoints_xy: Float32 `[num_keypoints, 2]` xy pixel coordinates of the
            pair's **first** image, as `colsfm.database.read_keypoints` returns them.
        matches: Int64 `[num_matches, 2]` keypoint index pairs, first image first.
        image_width: Width of the first image in pixels.
        image_height: Height of the first image in pixels.
        max_matches: Target match count, i.e. roughly the cell count.

    Returns:
        Int64 `[num_kept, 2]` subset of `matches`, in ascending image-0 keypoint
        index order. `matches` is returned unchanged when it already holds
        `max_matches` rows or fewer, when the cap is not positive, or when the
        image has no area.
    """
    if len(matches) <= max_matches or max_matches <= 0 or image_width <= 0 or image_height <= 0:
        return np.asarray(matches, dtype=np.int64)

    order: Int64[ndarray, "num_matches"] = np.lexsort((matches[:, 1], matches[:, 0]))
    ordered: MatchIndices = np.asarray(matches, dtype=np.int64)[order]
    positions_xy: KeypointsXY = keypoints_xy[ordered[:, 0]]
    num_columns: int = max(int(math.sqrt(max_matches * image_width / image_height)), 1)
    num_rows: int = max(max_matches // num_columns, 1)
    columns: Int64[ndarray, "num_matches"] = np.clip((positions_xy[:, 0] * num_columns / image_width).astype(np.int64), 0, num_columns - 1)
    rows: Int64[ndarray, "num_matches"] = np.clip((positions_xy[:, 1] * num_rows / image_height).astype(np.int64), 0, num_rows - 1)
    cells: Int64[ndarray, "num_matches"] = rows * num_columns + columns
    # `np.unique` returns the first index of each value because `cells` is scanned in
    # the sorted keypoint order above, so the winner per cell is the lowest index.
    _, first_in_cell = np.unique(cells, return_index=True)
    return ordered[np.sort(first_in_cell)]


def select_by_square_covering(
    keypoints_xy: KeypointsXY,
    scores: MatchScores,
    image_width: int,
    image_height: int,
    target: int,
    tolerance: float = BLOB_NUM_POINTS_TOLERANCE_FRACTION,
) -> Int64[ndarray, " num_kept"]:
    """The blob's spatial non-maximum suppression: SSC over scored points.

    `GlobalNonMaximumSuppressionViaSquareCovering` (feature_matcher_main.md §5.3)
    is the SSC algorithm of Bailo et al. 2018: sort the points by strength, then
    binary-search a suppression square width `w`; for each candidate width, lay a
    grid of cells `w/2` on a side, walk the points strongest first, keep a point
    whose cell is not yet covered and mark the 5x5 block of cells around it.
    Stop as soon as the surviving count lands inside
    `[target - target*tolerance, target + target*tolerance]`.

    This is the real thing, not `subsample_matches_by_coverage`'s score-free
    stand-in: it is only reachable on the TensorRT matching backend, which has
    the LightGlue score the blob uses as each point's response
    (`colsfm.matching_trt`). Unlike the stand-in it targets a *band* around
    `target` rather than a hard ceiling, so a result slightly above `target` is
    the algorithm working, not a bug.

    The search bounds are the blob's: `low = floor(sqrt(n / target))` and
    `high = (exp1 + exp3) / (2*target - 2)` with `exp1 = width + height + 2*target`
    and `exp3 = sqrt(exp2)`, `exp2` the discriminant of the same quadratic.

    Args:
        keypoints_xy: Float32 `[num_points, 2]` xy pixel coordinates, one per
            scored point — for a match list, the coordinates in the *first* image.
        scores: Float32 `[num_points]` strength per point; the LightGlue score.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        target: Points to aim for, i.e. the blob's `match_top_k`.
        tolerance: Half-width of the acceptable band as a fraction of `target`.

    Returns:
        Int64 indices into `keypoints_xy`, ascending. Every point is kept —
        `arange(num_points)` — when `target` is not positive, when there are
        already no more than `target` points, or when the image has no area,
        which is the blob's own `matches.size() < match_top_k` early return. A
        `target` of one short-circuits to the single strongest point.
    """
    num_points: int = len(scores)
    if target <= 0 or num_points <= target or image_width <= 0 or image_height <= 0:
        return np.arange(num_points, dtype=np.int64)

    order: Int64[ndarray, " num_points"] = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    if target < 2:
        # The search bound below divides by `2 * target - 2`. A target of one is not a
        # suppression problem anyway: it is "keep the strongest point".
        return np.sort(order[:target])
    ordered_xy: KeypointsXY = np.asarray(keypoints_xy, dtype=np.float32)[order]
    ordered_indices: list[int] = order.tolist()

    exp1: float = float(image_width + image_height + 2 * target)
    exp2: float = float(
        4 * image_width
        + 4 * target
        + 4 * image_height * target
        + image_height * image_height
        + image_width * image_width
        - 2 * image_height * image_width
        + 4 * image_height * image_width * target
    )
    low: float = float(math.floor(math.sqrt(num_points / target)))
    high: float = (exp1 + math.sqrt(max(exp2, 0.0))) / (2 * target - 2)
    lowest_acceptable: int = round(target - target * tolerance)
    highest_acceptable: int = round(target + target * tolerance)

    previous_width: float = -1.0
    kept: list[int] = ordered_indices[:target]
    while low <= high:
        width: float = low + (high - low) / 2.0
        if width == previous_width:
            break
        previous_width = width
        kept = _cover_squares(ordered_xy, ordered_indices, width, image_width, image_height)
        if lowest_acceptable <= len(kept) <= highest_acceptable:
            break
        if len(kept) < lowest_acceptable:
            high = width - 1.0
        else:
            low = width + 1.0
    return np.sort(np.asarray(kept, dtype=np.int64))


def _cover_squares(ordered_xy: KeypointsXY, ordered_indices: Sequence[int], width: float, image_width: int, image_height: int) -> list[int]:
    """One SSC pass at one suppression width.

    The covered set is a Python `set` of flat cell ids rather than a boolean
    grid: only the kept points write, and each writes 25 cells, so the whole pass
    is a few thousand hash operations instead of a NumPy slice assignment per
    point. That is what keeps the binary search inside the per-pair budget.

    Args:
        ordered_xy: Float32 `[num_points, 2]` coordinates, already sorted strongest first.
        ordered_indices: The original index of each row of `ordered_xy`.
        width: Suppression square width in pixels; cells are `width / 2` on a side.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        The original indices of the kept points, strongest first.
    """
    cell_size: float = max(width / 2.0, 1e-6)
    num_cell_columns: int = int(image_width / cell_size)
    num_cell_rows: int = int(image_height / cell_size)
    stride: int = num_cell_columns + 1
    span: int = int(width / cell_size)
    columns: list[int] = np.clip((ordered_xy[:, 0] / cell_size).astype(np.int64), 0, num_cell_columns).tolist()
    rows: list[int] = np.clip((ordered_xy[:, 1] / cell_size).astype(np.int64), 0, num_cell_rows).tolist()

    covered: set[int] = set()
    kept: list[int] = []
    for index, row, column in zip(ordered_indices, rows, columns, strict=True):
        if row * stride + column in covered:
            continue
        kept.append(index)
        column_low: int = max(0, column - span)
        column_high: int = min(num_cell_columns, column + span)
        for covered_row in range(max(0, row - span), min(num_cell_rows, row + span) + 1):
            covered.update(range(covered_row * stride + column_low, covered_row * stride + column_high + 1))
    return kept


def spatial_cap(options: MatchingOptions, sizes: Mapping[int, tuple[int, int]]) -> CapFunction | None:
    """The post-verification spatial cap as a per-pair function, or None when nothing is capped.

    Caps each pair's verified inliers to `match_limit`'s budget, sized from the first image
    (`subsample_matches_by_coverage`); in `fixed` mode every pair gets the same budget.

    Args:
        options: The matching settings, for `match_limit`.
        sizes: `(width, height)` per image id.

    Returns:
        The cap, or None under `--match-cap-mode off`.
    """
    if not options.match_limit.limits:
        return None

    def cap(pair: ImagePair, keypoints1: KeypointsXY, inliers: MatchIndices) -> MatchIndices:
        width, height = sizes[pair[0]]
        budget: int | None = options.match_limit.limit_for(width, height)
        if budget is None or len(inliers) <= budget:
            return inliers
        return subsample_matches_by_coverage(keypoints1, inliers, width, height, budget)

    return cap


def require_no_geometries(database_path: Path, pairs: Sequence[ImagePair]) -> None:
    """Refuse pairs that already have a two-view geometry row; read-only.

    None can exist: every run starts from a fresh database, the loop stage verifies only pairs it matched itself, and
    no matcher writes placeholder rows at the pinned COLMAP. A row would make the verifier's write fail on its unique
    pair id, so it is reported here, by count, instead.

    Args:
        database_path: The database.
        pairs: The pairs about to be verified.

    Raises:
        ValueError: When any pair already has a row.
    """
    existing: int = count_two_view_geometries(database_path, pairs)
    if existing:
        raise ValueError(f"{existing} of {len(pairs)} pairs already have a two-view geometry row; verification would collide with them")


def verifier_inputs(database: pycolmap.Database) -> tuple[dict[int, pycolmap.Camera], dict[int, tuple[int, int]]]:
    """Camera and pixel size per image id, read once.

    Args:
        database: An open database.

    Returns:
        `(camera per image id, (width, height) per image id)`.
    """
    cameras: dict[int, pycolmap.Camera] = {camera.camera_id: camera for camera in database.read_all_cameras()}
    camera_of: dict[int, pycolmap.Camera] = {image.image_id: cameras[image.camera_id] for image in database.read_all_images()}
    return camera_of, {image_id: (int(camera.width), int(camera.height)) for image_id, camera in camera_of.items()}


def write_and_count(
    database: pycolmap.Database, chunk: Sequence[VerifiedPair], stats: dict[ImagePair, PairMatchStats], *, write_matches: bool
) -> int:
    """Write one verified chunk and keep only its counts, so the chunk itself can be dropped.

    Args:
        database: An open database.
        chunk: The verified pairs.
        stats: Raw and inlier counts per pair, updated in place.
        write_matches: Also write the raw matches (`colsfm.verification.write_verified`).

    Returns:
        The inlier matches the spatial cap removed in this chunk.
    """
    write_verified(database, chunk, write_matches=write_matches)
    stats.update({item.pair: PairMatchStats(raw_matches=len(item.matches), inlier_matches=item.num_inliers) for item in chunk})
    return sum(item.num_capped for item in chunk)


def verify_and_report(
    database_path: Path, pairs: Sequence[ImagePair], options: MatchingOptions, *, device: ResolvedDevice, started: float, cap: bool
) -> tuple[MatchReport, int]:
    """Verify raw matches already in the database, in bounded chunks, and report: the tail of the backends that write
    their matches themselves (COLMAP's matcher, LightGlue+). FlexGlue streams its chunks to the verifier instead.

    Each chunk's matches and keypoints are read, verified on `options.verify.verify_threads` workers (`colsfm.verification`)
    and its geometries written in one transaction, so memory holds `options.verify.verify_chunks_in_flight` chunks at most.

    Args:
        database_path: The database holding the raw matches.
        pairs: The matched pairs.
        options: Verification, chunking and match-limit settings.
        device: What the report names as the matching device.
        started: `time.perf_counter()` at the start of matching, for the report's wall time.
        cap: Apply the post-verification spatial cap (the backends without per-match scores; LightGlue+ runs SSC before
            verification instead).

    Returns:
        The report, and the inlier matches the cap removed (0 without it).
    """
    require_no_geometries(database_path, pairs)
    stats: dict[ImagePair, PairMatchStats] = {}
    num_capped: int = 0
    with pycolmap.Database.open(database_path) as database:
        cameras, sizes = verifier_inputs(database)

        def stored_chunks() -> Iterator[list[PairToVerify]]:
            for first in range(0, len(pairs), options.verify.verify_chunk_pairs):
                chunk_pairs: Sequence[ImagePair] = pairs[first : first + options.verify.verify_chunk_pairs]
                keypoints: dict[int, Keypoints] = read_keypoints_batch(database_path, sorted({image_id for pair in chunk_pairs for image_id in pair}))
                yield [
                    PairToVerify(
                        pair=pair,
                        matches=np.asarray(database.read_matches(*pair), dtype=np.int64).reshape(-1, 2),
                        keypoints1=keypoints[pair[0]],
                        keypoints2=keypoints[pair[1]],
                    )
                    for pair in chunk_pairs
                ]

        with ChunkedVerifier(cameras, verification_options(options), spatial_cap(options, sizes) if cap else None, options.verify) as verifier:
            for finished in verifier.verified(stored_chunks()):
                num_capped += write_and_count(database, finished, stats, write_matches=False)
    return MatchReport(pair_stats=stats, device=device, elapsed_seconds=time.perf_counter() - started), num_capped


def match_pairs(
    database_path: Path,
    pairs: Sequence[ImagePair],
    options: MatchingOptions = DEFAULT_MATCHING_OPTIONS,
    frames: FrameSource | None = None,
) -> MatchReport:
    """Match and verify a list of image pairs, writing the results into the database.

    Both images of every pair must already carry keypoints and descriptors (see
    `colsfm.features.extract_features`). Matching runs first and verification after
    it (`colsfm.verification`; FlexGlue overlaps the two chunk by chunk), so the raw
    match count survives a pair that fails verification; the database ends up with
    both tables filled.

    Args:
        database_path: An existing COLMAP database.
        pairs: Image-id pairs, normalised and deduplicated by `colsfm.pairs`.
        options: Matching and verification settings.
        frames: The run's frame source, for the `flexglue` backend (it reads pixels); one is opened on
            `options.image_dir` for this call when None.

    Returns:
        Per-pair raw and inlier counts, the device used and the wall time.

    Raises:
        FileNotFoundError: When the database is missing.
        KeyError: When a pair names an image the database does not hold.
        RuntimeError: When `options.device` is `cuda` and CUDA is unusable.
        ImportError: When `options.backend` is `raco` and TensorRT or
            `cuda-python` is not importable.
    """
    if options.backend == "raco":
        # Imported here, not at module scope: TensorRT and `cuda-python` are only needed by this branch.
        from colsfm.matching_trt import LIGHTGLUE_PROFILE, match_pairs_tensorrt, normalize_keypoints_raco
        from colsfm.model_assets import RACO_LIGHTGLUE_ONNX_PATH

        # LightGlue+ on RaCo-ALIKED: the blob's graph path with its own graph and keypoint frame, and the same profile
        # so the two engines stay comparable (both are fed 2048 keypoints per image).
        return match_pairs_tensorrt(
            database_path,
            list(pairs),
            options,
            onnx_path=RACO_LIGHTGLUE_ONNX_PATH,
            profile=LIGHTGLUE_PROFILE,
            normalize=normalize_keypoints_raco,
            label="raco",
        )
    if options.backend == "flexglue":
        # Imported here for the same reason: TensorRT, torch and cv2 are only needed by this branch.
        from colsfm.flexglue_trt import match_pairs_flexglue

        return match_pairs_flexglue(database_path, list(pairs), options, frames)
    if not database_path.is_file():
        raise FileNotFoundError(f"No COLMAP database at {database_path}")
    if not pairs:
        return MatchReport(device=resolve_device(options.device))

    device: ResolvedDevice = resolve_device(options.device)
    with tempfile.TemporaryDirectory(prefix="colsfm-pairs-") as scratch:
        pair_list_path: Path = Path(scratch) / "pairs.txt"
        write_pair_list(pair_list_path, pairs, database_path)
        pairing_options: pycolmap.ImportedPairingOptions = pycolmap.ImportedPairingOptions()
        pairing_options.match_list_path = pair_list_path
        matching_options: pycolmap.FeatureMatchingOptions = _matching_options(options, device)
        matching_options.skip_geometric_verification = True
        started: float = time.perf_counter()
        pycolmap.match_image_pairs(
            database_path,
            matching_options=matching_options,
            pairing_options=pairing_options,
            verification_options=verification_options(options),
            device=pycolmap.Device.cuda if device == "cuda" else pycolmap.Device.cpu,
        )
    report, dropped = verify_and_report(database_path, pairs, options, device=device, started=started, cap=True)
    print(f"colsfm.matching: spatial cap ({options.match_limit.kind}, {options.match_limit.matches_per_pair}/pair) dropped {dropped} inlier matches")
    print(f"colsfm.matching: {report.summary()}")
    return report
