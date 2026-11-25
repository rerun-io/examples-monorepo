"""RaCo-ALIKED feature extraction through a batched, dynamic-shape TensorRT engine.

The `raco` half of `colsfm.features`, and the third backend after `pycolmap`
(COLMAP's own ONNX ALIKED) and `tensorrt` (the blob's static batch-1 engine).

**What the model is.** RaCo-ALIKED is fabio-sim's rotation-and-scale-consistent
retraining of ALIKED, published with LightGlue-ONNX and the subject of the
"Discovering TensorRT Optimizations for RaCo-ALIKED-LightGlue+" write-up. The
graph this module runs is `MODEL_DIR/raco-aliked-b1-16.onnx`, exported
from that package by `tools/export_cusfm_raco.py --batched-extractor-path` in
the repo's `raco` Pixi environment (`lightglue-onnx` at `d12b4ba`). Nothing here
imports `onnx` or `torch`: the `colsfm` environment only parses the graph with
TensorRT's own `OnnxParser`, which is the same split `colsfm.tensorrt_runtime`
documents.

**The engines.** The graph has a real batch axis, `image [batch, 3, h, w]`, so each engine is built with an
optimisation profile and `DEFAULT_BATCH_SIZE` images run together. Everything either side of the engine runs through
`colsfm.features_native`: GPU frames from the run's `colsfm.frames.FrameSource`, the `(k + 1) * 0.5 * size - 0.5`
mapping back onto the original image, and the two database columns written for image rows somebody else created.
What this module owns is what is RaCo's alone —
which of two graphs serves a size, what its optimisation profiles are, how wide
a batch each admits, and that its `scores` output is a rank rather than a
detector response.

**Native resolution is the default, and it is a deliberate deviation.** The
blob's feature extractor resizes every input to its network size
(`docs/spec/feature_extractor_main.md` §8), and the fixed-shape graph above does
the same: a 1226x370 KITTI frame is inferred over 1920x1200, five times the
pixels it has, for 19.6 ms against pycolmap's 7.8 ms at the native size
(`docs/kitti-06-results.md` §9). `native_resolution=True` instead runs each image at its own size rounded up to
`INPUT_DIM_DIVISOR` — so KITTI at 1248x384 and Galileo at 1920x1216. Images are
grouped by that rounded size so a batch is still one shape, and the mapping back
to pixels needs nothing new: `normalized_to_pixels` already targets the
*original* size, and rounding up (a rescale) rather than padding (a translation)
keeps it exact. `native_resolution=False` restores the stretch, on the old graph
and the old engine, so the two are measurable against each other.

**Native resolution does not mean the shape-dynamic engine.** Two engines can
serve a size, and the shape-dynamic one
(`MODEL_DIR/raco-aliked-dyn.onnx`, spatial axes symbolic) is only faster
below the top of its profile. Galileo *is* the top — 1920x1200, the profile
maximum — and there the dynamic engine costs 4.96 s against the fixed engine's
2.70 s over 226 images: one profile's tactics have to cover 256x256 to 1216x1920,
and the host pays a `cv2.resize` from 1200 to 1216 the fixed path skips. So
`select_raco_engine` decides per size group, `raco_engine="auto"`: a group the
fixed engine already accepts goes to the fixed engine, everything else to the
dynamic one, and KITTI keeps its 11.86 to 4.87 ms an image. `raco_engine="fixed"`
and `"dynamic"` force one, which is how the two are measured against each other.

**Scores are a selection order, not a detector response.** Upstream's boundary
ranker returns its top 2048 points already ordered but never materialises a
dense score map, so `tools/export_cusfm_raco.py` writes a strictly decreasing
`arange(2048, 0, -1) / 2048` in the `scores` output and stamps
`cusfm_scores=monotonic_upstream_selection_priority` into the ONNX metadata. The
smallest of those is `1/2048 = 0.00049`, which is *below* the blob's
`detector_threshold` of 0.005, so applying that gate here would throw away ten
points per image for no reason and for no meaning. `min_score` is therefore read
but compared against a rank, and `DEFAULT_MIN_SCORE` is 0: the honest reading is
that this graph has no score gate, and the matcher's SSC (which runs on
*LightGlue* scores, `colsfm.matching_trt`) is where selection actually happens.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias

from colsfm.database import keypoint_counts
from colsfm.features import RacoEngineChoice, TensorRTExtraction
from colsfm.features_native import (
    ALIKED_IMAGE_BINDING,
    DESCRIPTOR_TYPE,
    ExtractionGroup,
    ImageTask,
    NativeModel,
    image_tasks,
    run_native_extraction,
)
from colsfm.frames import FrameSource, ImageFolderFrames
from colsfm.keypoint_masks import MaskSource
from colsfm.model_assets import RACO_DYNAMIC_ONNX_PATH, RACO_ONNX_PATH
from colsfm.tensorrt_runtime import ShapeProfile, resolve_engine

NETWORK_HEIGHT: Final[int] = 1200
"""The fixed-shape graph's input height (the cuSFM blob's own ALIKED input, which the export keeps)."""

NETWORK_WIDTH: Final[int] = 1920
"""The fixed-shape graph's input width."""

MINIMUM_BATCH_SIZE: Final[int] = 1
"""The optimisation profile's `min`, so a one-image run still executes."""

DEFAULT_BATCH_SIZE: Final[int] = 8
"""Images per execution, and the profile's `opt`; `tools/raco_extract.py`'s own default."""

MAXIMUM_BATCH_SIZE: Final[int] = DEFAULT_BATCH_SIZE
"""The fixed-shape profile's `max`: the batch runs use, not the graph's export ceiling of 16.

TensorRT sizes an execution context for the profile maximum: batch 16 at 1200x1920 asked for 19.1 GB, and with a
per-size dynamic context alive beside it Galileo ran the 5090 out of memory. Batch 8 halves that."""

DYNAMIC_MAXIMUM_BATCH_SIZE: Final[int] = 8
"""The shape-dynamic profile's `max` batch.

16 is inside the fixed-shape graph's export bounds but its execution already
failed on a 32 GB 5090 (NOTES.md "RaCo backend"), and with the spatial axes
dynamic the builder has to size its tactic workspace for the *maximum* shape as
well, which pushes the build itself over. 8 is `DEFAULT_BATCH_SIZE`, so nothing
in the extraction path wants more."""

DYNAMIC_BATCH_PIXEL_BUDGET: Final[int] = 2 * 1216 * 1920
"""Pixels one shape-dynamic execution may hold at its profile maximum (batch x height x width).

TensorRT sizes an execution context's activation memory for the profile maximum: batch 8 at 1216x1920 asked for
11.3 GB on the RTX 5090 and 17.1 GB on the GB10. Two full frames' worth keeps a large-image group to a few GB and still
lets 480x640 images run eight at a time."""

INPUT_DIM_DIVISOR: Final[int] = 32
"""RaCo's four-level pyramid pools by 32, so both network sides must be multiples of it.

`tools/export_cusfm_raco.INPUT_DIM_DIVISOR` is the same constant read off
upstream's `Extractor.raco_aliked.input_dim_divisor`; the dynamic graph's input
is literally declared as `32 * height_factor` by `32 * width_factor`."""

MINIMUM_NETWORK_SIDE: Final[int] = 256
"""The profile's `min` height and width.

Below this the detector's top-2048 head starts competing with the number of
candidates the boundary ranker can find, and no camera this pipeline sees is
smaller."""

OPTIMAL_NETWORK_HEIGHT: Final[int] = 768
"""The profile's `opt` height; a midpoint between KITTI's 384 and Galileo's 1216."""

OPTIMAL_NETWORK_WIDTH: Final[int] = 1024
"""The profile's `opt` width; the same midpoint between KITTI's 1248 and Galileo's 1920."""

MAXIMUM_NETWORK_HEIGHT: Final[int] = 1216
"""The profile's `max` height: Galileo's 1200 rounded up to `INPUT_DIM_DIVISOR`.

1216 is also the height the *fixed*-shape graph resamples to internally, so a
Galileo frame reaches the same convolutions either way."""

MAXIMUM_NETWORK_WIDTH: Final[int] = 1920
"""The profile's `max` width, which Galileo already is."""

RACO_MODEL: Final[NativeModel] = NativeModel(
    name="raco",
    image_binding=ALIKED_IMAGE_BINDING,
    descriptor_type=DESCRIPTOR_TYPE,
    score_meaning="selection_rank",
)
"""RaCo's graphs as `colsfm.features_native` sees them.

`selection_rank` is the load-bearing half, and the module docstring above is its
justification: this graph's `scores` never touched a score map, so the blob's
`detector_threshold` means nothing against it and `colsfm.features` passes
`RACO_MIN_SCORE` instead. Both graphs bind their image as `image` and store
`ALIKED_N16ROT` descriptors, which is why one runner drives both."""


def raco_profile(network_height: int = NETWORK_HEIGHT, network_width: int = NETWORK_WIDTH) -> dict[str, ShapeProfile]:
    """The optimisation profile the batch-dynamic, fixed-shape graph is built with.

    Args:
        network_height: The graph's input height.
        network_width: The graph's input width.

    Returns:
        One `(minimum, optimal, maximum)` shape for the `image` binding.
    """
    return {
        ALIKED_IMAGE_BINDING: (
            (MINIMUM_BATCH_SIZE, 3, network_height, network_width),
            (DEFAULT_BATCH_SIZE, 3, network_height, network_width),
            (MAXIMUM_BATCH_SIZE, 3, network_height, network_width),
        )
    }


def raco_dynamic_profile(network_size: tuple[int, int] | None = None, batch_ceiling: int | None = None) -> dict[str, ShapeProfile]:
    """The optimisation profile the shape-dynamic graph is built with.

    Without a size, one profile spanning 256x256 to 1216x1920. With one, the profile tops out at that size and is
    tuned for it: TensorRT sizes an execution context's activation memory for the profile maximum (11.3 GB at batch 8
    and 1216x1920), so an engine per size group keeps a 480x640 run at a small fraction of that on a shared GPU.
    trtkit keys the cached engine by profile, so each size builds once per machine.

    Args:
        network_size: `(network_height, network_width)` of the group the engine serves, or `None` for the full range.
        batch_ceiling: The profile's max batch for a sized engine; `dynamic_batch_ceiling(network_size)` when None.

    Returns:
        One `(minimum, optimal, maximum)` shape for the `image` binding, at batch 1 to `DYNAMIC_MAXIMUM_BATCH_SIZE`.
    """
    if network_size is None:
        optimal: tuple[int, int] = (OPTIMAL_NETWORK_HEIGHT, OPTIMAL_NETWORK_WIDTH)
        maximum: tuple[int, int] = (MAXIMUM_NETWORK_HEIGHT, MAXIMUM_NETWORK_WIDTH)
        ceiling: int = DYNAMIC_MAXIMUM_BATCH_SIZE
    else:
        optimal = maximum = network_size
        ceiling = dynamic_batch_ceiling(network_size) if batch_ceiling is None else batch_ceiling
    return {
        ALIKED_IMAGE_BINDING: (
            (MINIMUM_BATCH_SIZE, 3, min(MINIMUM_NETWORK_SIDE, maximum[0]), min(MINIMUM_NETWORK_SIDE, maximum[1])),
            (min(DEFAULT_BATCH_SIZE, ceiling), 3, *optimal),
            (ceiling, 3, *maximum),
        )
    }


def network_size_for(image_height: int, image_width: int) -> tuple[int, int]:
    """The network size one image runs at when the engine is shape-dynamic.

    The image's own size, rounded **up** to `INPUT_DIM_DIVISOR` and clamped into
    the profile. Rounding up rather than padding keeps the mapping back to pixels
    a plain linear rescale, which is exactly what `normalized_to_pixels` already
    is — a padded border would put keypoints in a frame that mapping does not
    describe.

    Args:
        image_height: The image's own height in pixels.
        image_width: The image's own width in pixels.

    Returns:
        `(network_height, network_width)`, both multiples of `INPUT_DIM_DIVISOR`
        and inside the profile's bounds.
    """

    def rounded(side: int, ceiling: int) -> int:
        grid: int = -(-max(side, MINIMUM_NETWORK_SIDE) // INPUT_DIM_DIVISOR) * INPUT_DIM_DIVISOR
        return min(grid, ceiling)

    return rounded(image_height, MAXIMUM_NETWORK_HEIGHT), rounded(image_width, MAXIMUM_NETWORK_WIDTH)


RacoEngineKind: TypeAlias = Literal["fixed", "dynamic"]
"""Which of the two RaCo graphs an engine was built from."""


@dataclass(frozen=True, slots=True)
class RacoEngine:
    """The engine one size group runs through, and the input size it is handed."""

    kind: RacoEngineKind
    """`fixed` is the batch-dynamic graph at its declared size, `dynamic` the shape-dynamic one."""
    onnx_path: Path
    """The graph the engine is built from, and the cache name it is stored under."""
    network_height: int
    """The height the engine takes, which is what preprocessing resizes to."""
    network_width: int
    """The width the engine takes."""
    maximum_batch_size: int
    """The images its optimisation profile admits in one execution."""


def fixed_engine_size_group() -> tuple[int, int]:
    """The `network_size_for` group whose images the fixed-shape engine also serves.

    The two graphs do not take the same numbers for the same picture: the fixed
    one is declared at 1200x1920 and interpolates to 1216x1920 *inside* itself,
    while the dynamic export rounds 1200 up to 1216 on the host. So the group is
    the fixed engine's own input put through the same rounding — 1216x1920 —
    and not the input itself.

    Returns:
        `(network_height, network_width)` of the group the fixed engine covers.
    """
    return network_size_for(NETWORK_HEIGHT, NETWORK_WIDTH)


def select_raco_engine(network_size: tuple[int, int], choice: RacoEngineChoice = "auto", batch_pixels: int | None = None) -> RacoEngine:
    """Pick the engine one size group runs on.

    `auto` exists because the dynamic engine is not free at the top of its
    profile: Galileo, which *is* the profile maximum, extracts in 4.96 s against
    the fixed engine's 2.70 s (`docs/gpu-preprocessing.md`), part single-profile
    tactic selection and part a host `cv2.resize` from 1200 to 1216 that the
    fixed path skips. Below the maximum the dynamic engine is the whole win —
    KITTI goes 11.86 to 4.87 ms an image — so the rule is size-directed, not
    global.

    Args:
        network_size: `(network_height, network_width)` from `network_size_for`.
        choice: `auto` decides by size; `fixed` and `dynamic` force one engine.
        batch_pixels: The dynamic engine's pixel budget per execution; `DYNAMIC_BATCH_PIXEL_BUDGET` when None.

    Returns:
        The graph, the input size to preprocess to, and the batch ceiling.
    """
    fixed: bool = choice == "fixed" or (choice == "auto" and network_size == fixed_engine_size_group())
    if fixed:
        return RacoEngine(
            kind="fixed",
            onnx_path=RACO_ONNX_PATH,
            network_height=NETWORK_HEIGHT,
            network_width=NETWORK_WIDTH,
            maximum_batch_size=MAXIMUM_BATCH_SIZE,
        )
    return RacoEngine(
        kind="dynamic",
        onnx_path=RACO_DYNAMIC_ONNX_PATH,
        network_height=network_size[0],
        network_width=network_size[1],
        maximum_batch_size=dynamic_batch_ceiling(network_size, batch_pixels),
    )


def dynamic_batch_ceiling(network_size: tuple[int, int], batch_pixels: int | None = None) -> int:
    """The batch ceiling of a shape-dynamic engine for one size: as many images as `DYNAMIC_BATCH_PIXEL_BUDGET` holds.

    Args:
        network_size: `(network_height, network_width)`.
        batch_pixels: The pixel budget; `DYNAMIC_BATCH_PIXEL_BUDGET` when None.

    Returns:
        Between `MINIMUM_BATCH_SIZE` and `DYNAMIC_MAXIMUM_BATCH_SIZE`.
    """
    budget: int = DYNAMIC_BATCH_PIXEL_BUDGET if batch_pixels is None else batch_pixels
    fits: int = budget // max(1, network_size[0] * network_size[1])
    return max(MINIMUM_BATCH_SIZE, min(DYNAMIC_MAXIMUM_BATCH_SIZE, fits))


def engine_file(engine: RacoEngine) -> Path:
    """Build or find the cached engine file for one selection.

    Args:
        engine: What `select_raco_engine` returned.

    Returns:
        The serialised engine, built on first use.
    """
    if engine.kind == "fixed":
        return resolve_engine(engine.onnx_path, raco_profile())
    return resolve_engine(engine.onnx_path, raco_dynamic_profile((engine.network_height, engine.network_width), engine.maximum_batch_size))


def _size_groups(tasks: Sequence[ImageTask]) -> dict[tuple[int, int], list[ImageTask]]:
    """Group images by the network size they will run at, keeping their order.

    A batch is one TensorRT execution at one input shape, so images of different
    sizes cannot share one. Grouping first means a mixed-camera set still fills
    its batches instead of flushing a short one at every size change.

    Args:
        tasks: The images, in the order their results are wanted.

    Returns:
        Tasks per `(network_height, network_width)`, first-seen size first.
    """
    groups: dict[tuple[int, int], list[ImageTask]] = {}
    for task in tasks:
        groups.setdefault(network_size_for(task.image_height, task.image_width), []).append(task)
    return groups


def extraction_groups(
    tasks: Sequence[ImageTask], *, native_resolution: bool, choice: RacoEngineChoice, batch_pixels: int | None = None
) -> list[ExtractionGroup]:
    """Decide which engine runs which images, and at what size.

    The whole of RaCo's engine policy, and the only part of extraction this
    backend still owns: group by the size each image will run at, pick an engine
    per group, and resolve — building on first use — the engine file. Resolving
    every group before any of them executes means a build that fails leaves no
    half-extracted database behind.

    Args:
        tasks: The images, in the order their results are wanted.
        native_resolution: Run each image at its own size rounded up to
            `INPUT_DIM_DIVISOR`. `False` puts every image in one group at the
            fixed graph's declared 1920x1200.
        choice: Which engine serves a group; see `select_raco_engine`.
        batch_pixels: The dynamic engines' pixel budget per execution; see `select_raco_engine`.

    Returns:
        One group per network size, first-seen size first; empty when there is
        nothing to extract.
    """
    sizes: dict[tuple[int, int], list[ImageTask]] = _size_groups(tasks) if native_resolution else {fixed_engine_size_group(): list(tasks)}
    groups: list[ExtractionGroup] = []
    for network_size, group in sizes.items():
        if not group:
            continue
        engine: RacoEngine = select_raco_engine(network_size, choice, batch_pixels)
        groups.append(
            ExtractionGroup(
                engine_path=engine_file(engine),
                tasks=group,
                network_height=engine.network_height,
                network_width=engine.network_width,
                maximum_batch_size=engine.maximum_batch_size,
            )
        )
    return groups


def extract_raco(
    database_path: Path,
    image_root: Path,
    image_names: Sequence[str],
    *,
    min_score: float,
    max_num_features: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    native_resolution: bool = True,
    raco_engine: RacoEngineChoice = "auto",
    frames: FrameSource | None = None,
    masks: MaskSource | None = None,
    batch_pixels: int | None = None,
) -> TensorRTExtraction:
    """Run the batched RaCo-ALIKED engine over the named images and store the features.

    Args:
        database_path: An existing COLMAP database whose image rows are written.
        image_root: Directory the `image_names` are relative to.
        image_names: Images to extract, as stored in the database.
        min_score: Gate on the graph's `scores` output. See the module docstring:
            that output is a selection rank, so anything above 0 truncates by
            rank rather than by detector response.
        max_num_features: Keypoint ceiling per image; the graph's own top-2048
            head normally binds first, so this only ever truncates.
        batch_size: Images executed together, 1 to the graph's batch ceiling.
            `native_resolution` bounds it at `DYNAMIC_MAXIMUM_BATCH_SIZE` unless
            `raco_engine` forces the fixed engine, because `auto` may reach the
            dynamic one for any size group.
        native_resolution: Run each image at its own size, rounded up to
            `INPUT_DIM_DIVISOR`, on whichever engine `raco_engine` picks for that
            size. `False` is the legacy behaviour: the fixed-shape graph, and
            every frame stretched to 1920x1200 whatever it was.
        raco_engine: Which engine serves a size group; see `select_raco_engine`.
            Read only when `native_resolution` is on — the stretch is the fixed
            engine by definition.
        frames: Where the pixels come from as GPU tensors (`colsfm.frames`); None reads
            the image files under `image_root` through an `ImageFolderFrames`.
        masks: Per-image keypoint masks (`colsfm.keypoint_masks`); None = unmasked.
        batch_pixels: The dynamic engines' pixel budget per execution; see `select_raco_engine`.

    Returns:
        Keypoints stored per `image_id`, and the wall time of the whole pass
        including the engine load.

    Raises:
        FileNotFoundError: When the database, the image root or the ONNX graph is missing.
        ValueError: When `batch_size` is outside the graph's profile.
    """
    if not database_path.is_file():
        raise FileNotFoundError(f"No COLMAP database at {database_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"No image directory at {image_root}")
    engine_choice: RacoEngineChoice = raco_engine if native_resolution else "fixed"
    maximum_batch_size: int = MAXIMUM_BATCH_SIZE if engine_choice == "fixed" else DYNAMIC_MAXIMUM_BATCH_SIZE
    if not MINIMUM_BATCH_SIZE <= batch_size <= maximum_batch_size:
        raise ValueError(f"batch_size must be in [{MINIMUM_BATCH_SIZE}, {maximum_batch_size}], got {batch_size}")

    started: float = time.perf_counter()
    tasks: list[ImageTask] = image_tasks(database_path, image_names)
    run_native_extraction(
        database_path,
        extraction_groups(tasks, native_resolution=native_resolution, choice=engine_choice, batch_pixels=batch_pixels),
        RACO_MODEL,
        frames if frames is not None else ImageFolderFrames(image_root),
        min_score=min_score,
        max_num_features=max_num_features,
        batch_size=batch_size,
        masks=masks,
    )
    elapsed_seconds: float = time.perf_counter() - started
    return keypoint_counts(database_path, [task.image_id for task in tasks]), elapsed_seconds
