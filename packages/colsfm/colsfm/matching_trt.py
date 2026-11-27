"""LightGlue matching through the blob's own TensorRT engine.

The `tensorrt` half of `colsfm.matching`. It runs
`pycusfm/models/aliked_lightglue/lightglue_aliked.onnx` through the engine the
blob itself built (`lightglue_aliked_fp16_10_13_3_9_sm_12_0.engine`, accepted as
a trusted prebuilt engine when the TensorRT version and the GPU match), which
buys the one thing
the pycolmap backend cannot have: **the per-match score**. With it, the blob's
four steps map across exactly, in the blob's own order
(feature_matcher_main.md §5):

| blob | here |
|---|---|
| `NormalizedKeyPoints`: `(kpts - [w/2, h/2]) / (max(w, h)/2)` | `normalize_keypoints` |
| `PostProcess`: keep `mscores0 > match_threshold` (0.3, a strict `>`) | `MatchingOptions.tensorrt_min_score` |
| `ApplyNonMaximumSuppression`: SSC to `match_top_k` on the LightGlue score | `colsfm.matching.select_by_square_covering` |
| `TwoViewVerification`: essential-matrix RANSAC, `n_inliers < 10` empties the pair | `colsfm.verification` (`pycolmap.estimate_two_view_geometry`, COLMAP's write rules) with the same options the pycolmap backend uses |

So this path has no divergence 2 (NOTES.md, "Where colsfm deviates"): the real
SSC runs, on the real scores, *before* verification. The post-verification grid
subsample of `colsfm.matching.subsample_matches_by_coverage` is therefore skipped
entirely here — running both would thin the same pair twice.

**One pair at a time, not sixteen.** `lightglue_aliked_b16.onnx` exists and takes
`points [32, 2048, 2]`, i.e. 16 pairs at once, but its engine is not in the repo
and would have to be built (minutes) and re-built per GPU. Measured, the batch-1
engine executes in 2.78 ms at 2048x2048 keypoints, so Galileo's 339 pairs cost
0.94 s of GPU time against the blob's whole 2.2 s stage; the batched graph would
be optimising the smaller half of the budget. It stays unused, and this note is
the reason.

**Descriptors are read lazily.** RoboCap's 4528 images hold 4.75 GB of float32
descriptors, which is more than the process should hold at once, so an LRU cache
keeps the most recent `FEATURE_CACHE_IMAGES` images' keypoints and descriptors
and reads the rest back through the one open database handle. Pairs arrive in
image order, so the hit rate is high and the reads are the exception.

**The engine does not shrink its output.** `matches0` comes back with one row per
keypoint of image 0 and `mscores0` with one score per row, valid rows first; the
score gate is what separates them, exactly as the blob's `PostProcess` loop does.
Rows whose indices fall outside either keypoint list are dropped as well, so a
graph that pads with sentinel indices cannot write a corrupt match table.
"""

from __future__ import annotations

import itertools
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, TypeAlias

import numpy as np
import pycolmap
import torch
from jaxtyping import Bool, Float32, Int64
from numpy import ndarray
from trtkit import OnnxDynamicRuntime
from trtkit.tensorrt_runtime import TensorRtDynamicRuntime

from colsfm.database import (
    Descriptors,
    ImagePair,
    KeypointsXY,
    MatchIndices,
    database_transaction,
)
from colsfm.inference import ResolvedDevice, runtime_device
from colsfm.matching import (
    MatchingOptions,
    MatchReport,
    MatchScores,
    select_by_square_covering,
    verify_and_report,
)
from colsfm.tensorrt_runtime import ShapeProfile, resolve_engine

MATCH_WRITE_CHUNK: Final[int] = 1024
"""Pairs whose matches one database transaction writes."""

ImageFeatures: TypeAlias = tuple[KeypointsXY, Descriptors]
"""One image's keypoint pixels and its float descriptors, as the engine wants them."""

KeypointNormalizer: TypeAlias = Callable[[KeypointsXY, int, int], Float32[ndarray, "1 num_keypoints 2"]]
"""Pixel keypoints plus the image's width and height, into a graph's `kpts` frame.

The blob's graph and the RaCo one disagree about that frame — see
`normalize_keypoints` and `normalize_keypoints_raco`, both here —
and it is the only thing that differs between the two matching backends."""


LIGHTGLUE_PROFILE: Final[dict[str, ShapeProfile]] = {
    "kpts0": ((1, 1, 2), (1, 3200, 2), (1, 5500, 2)),
    "kpts1": ((1, 1, 2), (1, 3200, 2), (1, 5500, 2)),
    "desc0": ((1, 1, 128), (1, 3200, 128), (1, 5500, 128)),
    "desc1": ((1, 1, 128), (1, 3200, 128), (1, 5500, 128)),
}
"""The blob's own optimisation profile, `matching_task_worker_config.pb.txt:35-78`.

Minimum 1, optimal 3200, maximum 5500 keypoints per image; ALIKED emits 2048, so
the shipped engine runs below its optimal point and above its minimum."""

FEATURE_CACHE_IMAGES: Final[int] = 512
"""Images whose keypoints and descriptors stay resident.

512 x 2048 x 128 float32 is 537 MB, which is the largest working set worth
holding; RoboCap's whole 4528 images would be 4.75 GB."""


def normalize_keypoints(keypoints_xy: KeypointsXY, image_width: int, image_height: int) -> Float32[ndarray, "1 num_keypoints 2"]:
    """Put pixel keypoints into LightGlue's normalised frame.

    `LightGlueTensorRT::NormalizedKeyPoints` (feature_matcher_main.md §5.2):
    shift by half the image size, then divide by `max(width, height) / 2`. The
    standard LightGlue `normalize_keypoints`, and not the same convention as
    ALIKED's own output, which is why the coordinates make the round trip through
    pixels rather than staying normalised.

    Args:
        keypoints_xy: Float32 `[num_keypoints, 2]` xy pixel coordinates.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        Float32 `[1, num_keypoints, 2]`, the engine's binding shape.
    """
    centre_xy: Float32[ndarray, " 2"] = np.asarray([image_width / 2.0, image_height / 2.0], dtype=np.float32)
    scale: np.float32 = np.float32(max(image_width, image_height) / 2.0)
    normalized: Float32[ndarray, "num_keypoints 2"] = (np.asarray(keypoints_xy, dtype=np.float32) - centre_xy) / scale
    return np.ascontiguousarray(normalized[None])


class FeatureCache:
    """An LRU over one database's keypoints and descriptors.

    One open `pycolmap.Database` handle for the whole matching pass, so a cache
    miss is a single SQLite read rather than an open/close cycle.
    """

    def __init__(self, database: pycolmap.Database, capacity: int = FEATURE_CACHE_IMAGES) -> None:
        """Wrap an already-open database.

        Args:
            database: The open database to read through.
            capacity: Images to keep resident.
        """
        self.database: pycolmap.Database = database
        self.capacity: int = max(capacity, 1)
        self.entries: OrderedDict[int, ImageFeatures] = OrderedDict()

    def __getitem__(self, image_id: int) -> ImageFeatures:
        """Read one image's features, from the cache when they are resident.

        Args:
            image_id: The image to read.

        Returns:
            Its Float32 `[num_keypoints, 2]` keypoint pixels and its Float32
            `[num_keypoints, 128]` descriptors.

        Raises:
            KeyError: When the image has no keypoint or descriptor row.
        """
        if image_id in self.entries:
            self.entries.move_to_end(image_id)
            return self.entries[image_id]
        if not self.database.exists_keypoints(image_id) or not self.database.exists_descriptors(image_id):
            raise KeyError(f"Image {image_id} carries no keypoints or no descriptors; run feature extraction first")
        stored_keypoints: Float32[ndarray, "num_keypoints num_columns"] = self.database.read_keypoints(image_id)
        keypoints_xy: KeypointsXY = np.ascontiguousarray(stored_keypoints[:, :2], dtype=np.float32)
        descriptors: Descriptors = np.ascontiguousarray(self.database.read_descriptors(image_id).to_float().data, dtype=np.float32)
        self.entries[image_id] = (keypoints_xy, descriptors)
        if len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return self.entries[image_id]


def normalize_keypoints_raco(keypoints_xy: KeypointsXY, image_width: int, image_height: int) -> Float32[ndarray, "1 num_keypoints 2"]:
    """Put pixel keypoints back into the frame the RaCo extractor emitted them in, for LightGlue+.

    LightGlue+ (fabio-sim's matcher trained against RaCo-ALIKED, the `raco` backend) takes cuSFM's
    `align_corners=False` normalisation over the network input and applies `[1, H_proc / W]` inside the graph, which
    composes to the same LightGlue frame as `normalize_keypoints`. This is the exact inverse of
    `colsfm.features_native.normalized_to_pixels`, `(2 * p + 1) / size - 1`. Passing `normalize_keypoints` here would
    scale x correctly by luck and y by `H / W`, which silently degrades matching rather than failing.

    Args:
        keypoints_xy: Float32 `[num_keypoints, 2]` xy pixel coordinates.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        Float32 `[1, num_keypoints, 2]`, the engine's binding shape.
    """
    size_xy: Float32[ndarray, " 2"] = np.asarray([image_width, image_height], dtype=np.float32)
    normalized: Float32[ndarray, "num_keypoints 2"] = (np.float32(2.0) * np.asarray(keypoints_xy, dtype=np.float32) + np.float32(1.0)) / size_xy - np.float32(1.0)
    return np.ascontiguousarray(normalized[None])


def _image_sizes(database: pycolmap.Database) -> dict[int, tuple[int, int]]:
    """Look up every image's pixel size through its camera.

    Args:
        database: An open database.

    Returns:
        `(width, height)` per `image_id`.
    """
    cameras: dict[int, pycolmap.Camera] = {camera.camera_id: camera for camera in database.read_all_cameras()}
    return {image.image_id: (int(cameras[image.camera_id].width), int(cameras[image.camera_id].height)) for image in database.read_all_images()}


def match_one_pair(
    runtime: TensorRtDynamicRuntime | OnnxDynamicRuntime,
    features: FeatureCache,
    sizes: dict[int, tuple[int, int]],
    pair: ImagePair,
    options: MatchingOptions,
    normalize: KeypointNormalizer = normalize_keypoints,
    *,
    device: ResolvedDevice,
) -> MatchIndices:
    """Run LightGlue over one pair and return the matches the blob would keep.

    Score gate then SSC, in the blob's order: the suppression sees only the
    matches that survived `tensorrt_min_score`, and it uses their LightGlue
    scores as the keypoint response.

    Args:
        runtime: The loaded LightGlue engine (direct binding), or the graph on ONNX Runtime.
        features: The keypoint and descriptor source.
        sizes: `(width, height)` per `image_id`, for the normalisation.
        pair: The image pair, first image first.
        options: Matching settings; `tensorrt_min_score`, `match_limit` and
            `num_points_tolerance_fraction` are read.
        normalize: Which normalised keypoint frame the graph's `kpts0`/`kpts1`
            bindings expect; see `KeypointNormalizer`.
        device: The runtime's device, where the inputs are staged.

    Returns:
        Int64 `[num_matches, 2]` keypoint index pairs, first image first.
    """
    image_id1, image_id2 = pair
    keypoints0, descriptors0 = features[image_id1]
    keypoints1, descriptors1 = features[image_id2]
    if len(keypoints0) == 0 or len(keypoints1) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    width0, height0 = sizes[image_id1]
    width1, height1 = sizes[image_id2]
    host_inputs: dict[str, ndarray] = {
        "kpts0": normalize(keypoints0, width0, height0),
        "kpts1": normalize(keypoints1, width1, height1),
        "desc0": np.ascontiguousarray(descriptors0[None], dtype=np.float32),
        "desc1": np.ascontiguousarray(descriptors1[None], dtype=np.float32),
    }
    # Keypoints and descriptors come from the database on the host; the matches go back to it on the host.
    outputs: dict[str, ndarray] = {
        name: value.cpu().numpy()
        for name, value in runtime({name: torch.from_numpy(array).to(device) for name, array in host_inputs.items()}).items()
    }
    candidates: MatchIndices = np.asarray(outputs["matches0"], dtype=np.int64).reshape(-1, 2)
    scores: MatchScores = np.asarray(outputs["mscores0"], dtype=np.float32).reshape(-1)
    accepted: Bool[ndarray, " num_candidates"] = (
        (scores > np.float32(options.tensorrt_min_score))
        & (candidates[:, 0] >= 0)
        & (candidates[:, 1] >= 0)
        & (candidates[:, 0] < len(keypoints0))
        & (candidates[:, 1] < len(keypoints1))
    )
    matches: MatchIndices = candidates[accepted]
    if not options.match_limit.limits or len(matches) == 0:
        return matches
    # The same policy as the grid subsample, so `--match-cap-mode` means one thing
    # across both backends; here the number is SSC's target rather than a ceiling.
    target: int | None = options.match_limit.limit_for(width0, height0)
    if target is None:
        return matches
    kept: Int64[ndarray, " num_kept"] = select_by_square_covering(
        keypoints0[matches[:, 0]],
        scores[accepted],
        width0,
        height0,
        target,
        options.num_points_tolerance_fraction,
    )
    return matches[kept]


def match_pairs_tensorrt(
    database_path: Path,
    pairs: Sequence[ImagePair],
    options: MatchingOptions,
    *,
    onnx_path: Path,
    normalize: KeypointNormalizer,
    profile: dict[str, ShapeProfile] | None = None,
    label: str = "engine",
) -> MatchReport:
    """Match and verify a pair list with a LightGlue TensorRT engine.

    Both images of every pair must already carry keypoints and float descriptors
    — from either feature backend, since both write the same two columns.
    Matching happens here and writes the raw matches; `verify_and_report` then
    verifies them in bounded chunks (`colsfm.verification`), which is the blob's
    order and the same path the pycolmap backend takes.

    Args:
        database_path: An existing COLMAP database.
        pairs: Image-id pairs, normalised and deduplicated by `colsfm.pairs`.
        options: Matching and verification settings.
        onnx_path: Which LightGlue graph to run. The `raco` backend passes
            the RaCo-trained one; everything else about the pass is identical.
        profile: Optimisation profile for that graph's dynamic keypoint axis;
            `None` uses the blob's own `LIGHTGLUE_PROFILE`.
        normalize: Which normalised keypoint frame the graph expects.
        label: What the one-line summary calls this backend.

    Returns:
        Per-pair raw and inlier counts, the device used and the wall time.

    Raises:
        FileNotFoundError: When the database is missing.
        KeyError: When a pair names an image with no features.
    """
    if not database_path.is_file():
        raise FileNotFoundError(f"No COLMAP database at {database_path}")
    device: ResolvedDevice = runtime_device(options.inference_runtime)
    if not pairs:
        return MatchReport(device=device)

    runtime: TensorRtDynamicRuntime | OnnxDynamicRuntime
    if options.inference_runtime == "onnxruntime":
        started: float = time.perf_counter()
        runtime = OnnxDynamicRuntime(onnx_path)
    else:
        # LightGlue has no convolutional trunk; its attention stays fp32 (TF32) under the strongly typed build.
        engine_path: Path = resolve_engine(onnx_path, profile if profile is not None else LIGHTGLUE_PROFILE, precision="fp32")
        started = time.perf_counter()
        runtime = TensorRtDynamicRuntime(engine_path, direct_binding=True)
    with pycolmap.Database.open(database_path) as database:
        sizes: dict[int, tuple[int, int]] = _image_sizes(database)
        features: FeatureCache = FeatureCache(database)
        # One commit per chunk, not per pair: on a network file system every commit is a round trip.
        for chunk in itertools.batched(pairs, MATCH_WRITE_CHUNK):
            with database_transaction(database):
                for pair in chunk:
                    matches: MatchIndices = match_one_pair(runtime, features, sizes, pair, options, normalize, device=device)
                    database.write_matches(pair[0], pair[1], matches.astype(np.uint32))

    # No spatial cap here: LightGlue+ already ran SSC on its scores before verification (`match_one_pair`).
    report, _ = verify_and_report(database_path, pairs, options, device=device, started=started, cap=False)
    print(f"colsfm.matching[{label}]: {report.summary()}")
    return report
