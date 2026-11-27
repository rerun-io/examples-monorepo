"""The `raco` feature and matching backends: batched RaCo-ALIKED and LightGlue+.

**Two halves, gated differently.** The first half is pure contracts — the
normalisation round trip between extractor and matcher, the size grid, the
profile bounds, the engine-selection rule, the batch-size guard. None of them
loads an engine or reads a weight, so none of them is gated on one. They used to
be, by a module-level `pytestmark`, which meant a fresh clone — where
`data/cusfm_models/` is gitignored and therefore empty — ran none of this and
reported it as skipped.

The second half really does need the graphs and the GPU, and carries
`requires_raco_models`. Rebuild the graphs with `pixi run -e colsfm colsfm-raco-export`
and `pixi run -e colsfm colsfm-raco-export --batched-extractor-path
data/cusfm_models/raco-aliked-b1-16.onnx`.

The bands are the blob's own Galileo statistics again (feature_matcher_main.md
§6): 2048 keypoints per keyframe, median 430 verified matches per pair. RaCo is
a *different* extractor, so this asserts that it lands in the same neighbourhood
rather than on the same number — that is the "near-parity" claim, and a band is
what near-parity means.

The one timing claim — that the fixed engine beats the shape-dynamic one at the
top of its profile, which is why `select_raco_engine` has an `auto` rule at all —
is marked `perf` and skipped unless asked for; see `tests/colsfm/conftest.py`.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from jaxtyping import Float32, UInt8
from numpy import ndarray

from colsfm import PACKAGE_DIR as REPO_ROOT
from colsfm.database import ImagePair, KeypointsXY, create_database, read_descriptors_from_database, read_keypoints
from colsfm.features import BLOB_MAX_KEYPOINTS, ExtractionReport, FeatureOptions, RacoEngineChoice, extract_features
from colsfm.frames_meta import FRAMES_META_NAME, CameraParams, FramesMeta, KeyframeMeta, read_frames_meta
from colsfm.matching import MatchingOptions, MatchReport, match_pairs
from colsfm.pairs import select_pairs
from colsfm.pipeline import run_pipeline
from colsfm.run_config import PipelineOptions
from colsfm.run_report import PipelineSummary

pytest.importorskip("tensorrt", reason="the raco backend needs the tensorrt package")
pytest.importorskip("cuda.bindings.runtime", reason="the raco backend allocates through cuda-python")

BLOB_DESCRIPTOR_DIM: int = 128
"""RaCo-ALIKED's descriptor width, and LightGlue+'s `desc0`/`desc1` binding width."""

MIN_MEDIAN_INLIERS: int = 250
"""Lower band on the median verified matches per pair; the blob's Galileo median is 430."""

MAX_MEDIAN_INLIERS: int = 650
"""Upper band on the same, i.e. SSC's target plus its 30 % tolerance."""

SMOKE_MIN_INTER_FRAME_DISTANCE_M: float = 0.5
"""`feature_extractor_main`'s own default gate; keeps 34 of Galileo's 226 keyframes.

The same subset `tests/colsfm/test_pipeline.py` and the TensorRT smoke run on."""

SMOKE_MIN_REGISTERED_IMAGES: int = 30
"""Of the 34 selected keyframes, at least this many must end up registered."""

SMOKE_MIN_MEAN_MATCHES: int = 40
"""Lower band on the mean verified matches per pair over the 34-frame selection."""

SMOKE_MAX_MEAN_MATCHES: int = 650
"""Upper band on the same: SSC's target plus its tolerance."""

FP16_EDGE_TOLERANCE_PX: float = 1.0
"""How far outside the image an FP16 engine's edge keypoint may land.

The graph emits normalised coordinates and the half-precision arithmetic can put
a point on the border at -1.0008 rather than -1, which `normalized_to_pixels`
turns into -0.76 px on a 640 px image (the edge itself is -0.5). It is rounding, not a frame error: a frame error moves every
point by a fraction of the image, not the border ones by half a pixel."""


def _raco_models_available() -> bool:
    """Whether both RaCo graphs and a usable CUDA device are present.

    Returns:
        True when the extractor and matcher graphs exist and
        a CUDA device answers, so an engine can be loaded or built.
    """
    import torch

    from colsfm.features_raco import RACO_DYNAMIC_ONNX_PATH, RACO_ONNX_PATH
    from colsfm.model_assets import RACO_LIGHTGLUE_ONNX_PATH

    if not RACO_ONNX_PATH.is_file() or not RACO_DYNAMIC_ONNX_PATH.is_file() or not RACO_LIGHTGLUE_ONNX_PATH.is_file():
        return False
    return torch.cuda.is_available()


requires_raco_models = pytest.mark.skipif(not _raco_models_available(), reason="the RaCo ONNX graphs or a CUDA device are missing")
"""Applied to the tests that load an engine, and to nothing else.

The gate used to be a module-level `pytestmark`, which took the pure contracts
below down with it."""


# ── Pure contracts: no engine, no weights, no device ─────────────────────────


def test_the_matcher_frame_is_the_exact_inverse_of_the_extractor_mapping() -> None:
    """`normalize_keypoints_raco` undoes `normalized_to_pixels` to float32 precision.

    This is the whole contract between the two RaCo modules: the extractor emits
    normalised coordinates, writes pixels, and the matcher has to hand the graph
    back exactly what the extractor produced. A mismatch here does not raise
    anywhere — it just moves every keypoint by a fraction of the image and quietly
    costs matches — so it is asserted directly rather than through match counts.
    """
    from colsfm.features_native import normalized_to_pixels
    from colsfm.matching_trt import normalize_keypoints_raco

    generator: np.random.Generator = np.random.default_rng(0)
    normalized_xy: Float32[ndarray, "num_keypoints 2"] = generator.uniform(-1.0, 1.0, size=(512, 2)).astype(np.float32)
    pixels_xy: KeypointsXY = normalized_to_pixels(normalized_xy, 1920, 1200)
    recovered: Float32[ndarray, "1 num_keypoints 2"] = normalize_keypoints_raco(pixels_xy, 1920, 1200)
    assert recovered.shape == (1, 512, 2)
    assert np.allclose(recovered[0], normalized_xy, atol=1e-6)


def test_the_blob_normalisation_would_not_have_been_the_inverse() -> None:
    """The two graphs really do want different frames, so the split is not cosmetic.

    Guards the reason `KeypointNormalizer` exists: on a 1920x1200 image the blob's
    normalisation agrees with RaCo's in x and disagrees in y by `H / W`. If a
    refactor ever collapsed the two, this is what would fail.
    """
    from colsfm.matching_trt import normalize_keypoints, normalize_keypoints_raco

    keypoints_xy: KeypointsXY = np.asarray([[0.0, 0.0], [1919.0, 1199.0], [960.0, 600.0]], dtype=np.float32)
    blob: Float32[ndarray, "1 3 2"] = normalize_keypoints(keypoints_xy, 1920, 1200)
    raco: Float32[ndarray, "1 3 2"] = normalize_keypoints_raco(keypoints_xy, 1920, 1200)
    assert np.allclose(blob[0, :, 0], raco[0, :, 0], atol=2e-3)
    assert not np.allclose(blob[0, :, 1], raco[0, :, 1], atol=1e-2)


def test_the_batch_size_is_checked_against_the_graphs_profile(tmp_path: Path, galileo_input: FramesMeta) -> None:
    """A batch outside 1..16 is refused before an engine is touched.

    The optimisation profile is built for that range; TensorRT would otherwise
    reject the shape several seconds later, after an engine load.
    """
    from colsfm.features_raco import extract_raco

    database_path: Path = tmp_path / "batch.db"
    create_database(database_path, galileo_input.filtered(list(galileo_input.rig_frames()[0].keyframe_ids)))
    with pytest.raises(ValueError, match="batch_size"):
        extract_raco(database_path, tmp_path, [], min_score=0.0, max_num_features=BLOB_MAX_KEYPOINTS, batch_size=32)


def test_the_dynamic_profile_covers_both_datasets_on_the_networks_grid() -> None:
    """The one profile has to admit every size `network_size_for` can return.

    An engine built with bounds that do not contain a dataset fails at the first
    `set_input_shape` of a run, minutes after the build; the profile and the
    rounding are written in two places, so this checks they agree.
    """
    from colsfm.features_raco import (
        ALIKED_IMAGE_BINDING,
        INPUT_DIM_DIVISOR,
        network_size_for,
        raco_dynamic_profile,
    )

    minimum, optimal, maximum = raco_dynamic_profile()[ALIKED_IMAGE_BINDING]
    assert minimum == (1, 3, 256, 256)
    assert maximum == (8, 3, 1216, 1920)
    for shape in (minimum, optimal, maximum):
        assert shape[2] % INPUT_DIM_DIVISOR == 0 and shape[3] % INPUT_DIM_DIVISOR == 0
    for index in range(4):
        assert minimum[index] <= optimal[index] <= maximum[index]

    kitti_size: tuple[int, int] = network_size_for(370, 1226)
    galileo_size: tuple[int, int] = network_size_for(1200, 1920)
    assert kitti_size == (384, 1248)
    assert galileo_size == (1216, 1920)
    for height, width in (kitti_size, galileo_size, network_size_for(480, 640), network_size_for(17, 17)):
        assert minimum[2] <= height <= maximum[2]
        assert minimum[3] <= width <= maximum[3]


def test_the_engine_is_picked_from_the_size_a_group_runs_at() -> None:
    """Native resolution is a size, not an engine: Galileo keeps the fixed one.

    The shape-dynamic engine wins below the top of its profile and loses at the
    top, so `auto` sends the one size group the fixed engine already accepts
    back to it. The size compared is the group, 1216x1920 — the fixed graph
    interpolates its declared 1200x1920 input to 1216 itself — while the size
    the engine is *handed* stays the 1200x1920 it declares.
    """
    from colsfm.features_raco import (
        RACO_DYNAMIC_ONNX_PATH,
        RACO_ONNX_PATH,
        RacoEngine,
        network_size_for,
        select_raco_engine,
    )

    galileo: RacoEngine = select_raco_engine(network_size_for(1200, 1920))
    assert galileo.kind == "fixed"
    assert galileo.onnx_path == RACO_ONNX_PATH
    assert (galileo.network_height, galileo.network_width) == (1200, 1920)

    kitti: RacoEngine = select_raco_engine(network_size_for(370, 1226))
    assert kitti.kind == "dynamic"
    assert kitti.onnx_path == RACO_DYNAMIC_ONNX_PATH
    assert (kitti.network_height, kitti.network_width) == (384, 1248)

    crop: RacoEngine = select_raco_engine(network_size_for(480, 640))
    assert crop.kind == "dynamic"
    assert (crop.network_height, crop.network_width) == (480, 640)


def test_either_engine_can_be_forced_whatever_the_size() -> None:
    """`fixed` and `dynamic` override the size rule, which is how the two are timed."""
    from colsfm.features_raco import MAXIMUM_BATCH_SIZE, RacoEngine, network_size_for, select_raco_engine

    forced_dynamic: RacoEngine = select_raco_engine(network_size_for(1200, 1920), "dynamic")
    assert forced_dynamic.kind == "dynamic"
    assert (forced_dynamic.network_height, forced_dynamic.network_width) == (1216, 1920)
    # Two full frames: the pixel budget (DYNAMIC_BATCH_PIXEL_BUDGET) caps a 1216x1920 dynamic engine at batch 2.
    assert forced_dynamic.maximum_batch_size == 2

    forced_fixed: RacoEngine = select_raco_engine(network_size_for(370, 1226), "fixed")
    assert forced_fixed.kind == "fixed"
    assert (forced_fixed.network_height, forced_fixed.network_width) == (1200, 1920)
    # The fixed profile's own max (8), not the graph's export ceiling of 16, whose context ran Galileo out of memory.
    assert forced_fixed.maximum_batch_size == MAXIMUM_BATCH_SIZE


# ── Engine-backed: needs the RaCo graphs and a CUDA device ───────────────────


@pytest.fixture(scope="module")
def raco_stereo_frame(
    galileo_input: FramesMeta, galileo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, list[ImagePair], ExtractionReport, MatchReport]:
    """Extract and match one Galileo rig frame through both RaCo backends.

    Args:
        galileo_input: The 226-keyframe Galileo input collection.
        galileo_input_dir: Where its images live.
        tmp_path_factory: pytest's per-module temporary directory factory.

    Returns:
        The database, the pairs matched, and both reports.
    """
    keyframe_ids: list[int] = list(galileo_input.rig_frames()[0].keyframe_ids)
    subset: FramesMeta = galileo_input.filtered(keyframe_ids)
    database_path: Path = tmp_path_factory.mktemp("raco") / "galileo.db"
    create_database(database_path, subset)
    extraction: ExtractionReport = extract_features(
        database_path,
        galileo_input_dir,
        [keyframe.image_name for keyframe in subset.keyframes],
        FeatureOptions(backend="raco"),
    )
    pairs: list[ImagePair] = select_pairs(subset, connected_keyframe_num=1)
    matching: MatchReport = match_pairs(database_path, pairs, MatchingOptions(backend="raco"))
    return database_path, pairs, extraction, matching


@requires_raco_models
def test_the_batched_graph_stores_2048_keypoints_per_image(
    raco_stereo_frame: tuple[Path, list[ImagePair], ExtractionReport, MatchReport],
) -> None:
    """RaCo's top-k head is fixed at 2048 and the batching does not lose the tail.

    A batch that mis-slices its outputs would show up as a short last image or as
    every image holding the first image's points; the count is the cheap half of
    that check and `test_the_keypoints_land_inside_the_original_image` the rest.
    """
    _database_path, _pairs, extraction, _matching = raco_stereo_frame
    counts: list[int] = list(extraction.keypoint_counts.values())
    assert counts
    assert all(count == BLOB_MAX_KEYPOINTS for count in counts), counts
    assert extraction.device == "cuda"
    assert extraction.elapsed_seconds > 0.0


@requires_raco_models
def test_the_keypoints_land_inside_the_original_image(
    raco_stereo_frame: tuple[Path, list[ImagePair], ExtractionReport, MatchReport],
    galileo_input: FramesMeta,
) -> None:
    """The stretch to 1920x1200 is undone against each camera's own size."""
    database_path, _pairs, extraction, _matching = raco_stereo_frame
    keyframe_by_id: dict[int, KeyframeMeta] = galileo_input.keyframe_by_id()
    for image_id in extraction.keypoint_counts:
        keypoints_xy: KeypointsXY = read_keypoints(database_path, image_id)
        camera: CameraParams = galileo_input.cameras[keyframe_by_id[image_id].camera_params_id]
        assert keypoints_xy.dtype == np.float32
        assert keypoints_xy.shape[1] == 2
        assert keypoints_xy.min() >= 0.0
        assert keypoints_xy[:, 0].max() <= camera.image_width
        assert keypoints_xy[:, 1].max() <= camera.image_height


@requires_raco_models
def test_the_descriptors_round_trip_as_l2_normalised_floats(
    raco_stereo_frame: tuple[Path, list[ImagePair], ExtractionReport, MatchReport],
) -> None:
    """RaCo's descriptors are 128 L2-normalised float32 columns, unchanged by the database."""
    database_path, _pairs, extraction, _matching = raco_stereo_frame
    descriptors = read_descriptors_from_database(database_path, list(extraction.keypoint_counts))
    assert set(descriptors) == set(extraction.keypoint_counts)
    for image_id, block in descriptors.items():
        assert block.dtype == np.float32
        assert block.shape == (extraction.keypoint_counts[image_id], BLOB_DESCRIPTOR_DIM)
        norms = np.linalg.norm(block, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-2), f"image {image_id}: norms {norms.min()}-{norms.max()}"


@requires_raco_models
def test_lightglue_plus_matches_in_the_blobs_band(
    raco_stereo_frame: tuple[Path, list[ImagePair], ExtractionReport, MatchReport],
) -> None:
    """Median verified matches land near the blob's 430, on a different extractor.

    This is the near-parity claim: RaCo-ALIKED plus LightGlue+ is not the blob's
    ALIKED plus the blob's LightGlue, so the number is allowed to move — but if
    the descriptors, the normalised frame or the score gate were wrong it would
    not move, it would collapse.
    """
    _database_path, pairs, _extraction, matching = raco_stereo_frame
    assert set(matching.pair_stats) == set(pairs)
    assert matching.device == "cuda"
    assert matching.empty_pairs == 0
    assert MIN_MEDIAN_INLIERS <= matching.median_inliers <= MAX_MEDIAN_INLIERS
    assert matching.median_retention >= 0.9


@requires_raco_models
def test_the_whole_pipeline_runs_on_the_raco_backend(galileo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    """The 34-frame Galileo selection registers and matches in the same band as the others.

    Deliberately the same assertions and the same bands as
    `test_tensorrt_backends.test_the_whole_pipeline_runs_on_either_backend`, so
    the three backends are compared rather than each graded on its own curve.
    """
    options: PipelineOptions = PipelineOptions(
        input_dir=galileo_input_dir,
        output_dir=tmp_path_factory.mktemp("galileo_raco"),
        min_inter_frame_distance=SMOKE_MIN_INTER_FRAME_DISTANCE_M,
        features_backend="raco",
        matching_backend="raco",
        # The backend is what this module measures; the rest of the default
        # configuration is not, so the mapper stays on Ceres with the rig fixed
        # (`tests/colsfm/test_pipeline.py` PYCOLMAP_ABLATION).
        ba_backend="ceres",
        optimize_extrinsics=False,
    )
    summary: PipelineSummary = run_pipeline(options)
    mean_matches: float = summary.mapping.num_observations / max(summary.num_pairs, 1)
    print(
        f"[colsfm] raco end to end | registered {summary.mapping.num_registered_images} "
        f"| points {summary.mapping.num_points3D} | {mean_matches:.0f} observations per pair "
        f"| {summary.mapping.mean_reprojection_error_px:.3f} px"
    )
    assert summary.mapping.num_registered_images >= SMOKE_MIN_REGISTERED_IMAGES
    assert summary.mapping.num_points3D > 0
    assert SMOKE_MIN_MEAN_MATCHES <= mean_matches <= SMOKE_MAX_MEAN_MATCHES


def _cropped_galileo_meta(galileo_input: FramesMeta, galileo_input_dir: Path, image_root: Path, *, height: int, width: int) -> FramesMeta:
    """Write a top-left crop of one Galileo keyframe and describe it as its own camera.

    Args:
        galileo_input: The 226-keyframe Galileo input collection.
        galileo_input_dir: Where its images live.
        image_root: Directory the crop is written under, at the same relative name.
        height: Crop height in pixels.
        width: Crop width in pixels.

    Returns:
        A one-keyframe collection whose camera declares the crop's size.
    """
    keyframe: KeyframeMeta = galileo_input.keyframes[0]
    read = cv2.imread(str(galileo_input_dir / keyframe.image_name), cv2.IMREAD_COLOR)
    assert read is not None
    source: UInt8[ndarray, "full_height full_width 3"] = np.asarray(read)
    destination: Path = image_root / keyframe.image_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), source[:height, :width])
    subset: FramesMeta = galileo_input.filtered([keyframe.keyframe_id])
    camera: CameraParams = subset.cameras[keyframe.camera_params_id]
    return replace(subset, cameras={keyframe.camera_params_id: replace(camera, image_width=width, image_height=height)})


@requires_raco_models
def test_a_640x480_crop_runs_at_its_own_size_and_stays_inside_it(galileo_input: FramesMeta, galileo_input_dir: Path, tmp_path: Path) -> None:
    """A camera far below the profile's maximum still fills its 2048-point head.

    The fixed-shape graph would have stretched this crop back up to 1920x1200 and
    its keypoints would have landed inside the crop by construction. Here the
    engine really is asked for 640x480, so a wrong input shape, a wrong batch
    slice or a mapping that used the network size rather than the camera's would
    all show as points outside the crop.
    """
    from colsfm.features_raco import extract_raco

    subset: FramesMeta = _cropped_galileo_meta(galileo_input, galileo_input_dir, tmp_path / "images", height=480, width=640)
    database_path: Path = tmp_path / "crop.db"
    create_database(database_path, subset)
    counts, elapsed_seconds = extract_raco(
        database_path,
        tmp_path / "images",
        [keyframe.image_name for keyframe in subset.keyframes],
        min_score=0.0,
        max_num_features=BLOB_MAX_KEYPOINTS,
    )
    assert elapsed_seconds > 0.0
    assert list(counts.values()) == [BLOB_MAX_KEYPOINTS]
    keypoints_xy: KeypointsXY = read_keypoints(database_path, subset.keyframes[0].keyframe_id)
    assert keypoints_xy.min() >= -FP16_EDGE_TOLERANCE_PX
    assert keypoints_xy[:, 0].max() <= 640.0 + FP16_EDGE_TOLERANCE_PX
    assert keypoints_xy[:, 1].max() <= 480.0 + FP16_EDGE_TOLERANCE_PX


@requires_raco_models
def test_galileo_at_the_profile_maximum_reproduces_the_legacy_stretch(galileo_input: FramesMeta, galileo_input_dir: Path, tmp_path: Path) -> None:
    """Galileo is the profile's maximum, so all three ways of running it agree there.

    Two claims in one run. First, `auto` at native resolution reproduces the
    legacy stretch *exactly*, because at this size it chooses the same fixed
    engine and hands it the same 1920x1200 — that is the selection rule observed
    from outside, and it is what removes the 1.8x the dynamic engine costs here.
    Second, the forced dynamic engine still agrees to under a pixel: it is handed
    1920x1216 by OpenCV where the fixed graph resamples 1200 to 1216 inside
    itself, both bilinear over the same pixels, so the detector returns the same
    points and not the same floats.
    """
    from colsfm.features_raco import extract_raco

    keyframe_ids: list[int] = list(galileo_input.rig_frames()[0].keyframe_ids)
    subset: FramesMeta = galileo_input.filtered(keyframe_ids)
    image_names: list[str] = [keyframe.image_name for keyframe in subset.keyframes]
    runs: tuple[tuple[str, bool, RacoEngineChoice], ...] = (
        ("auto", True, "auto"),
        ("dynamic", True, "dynamic"),
        ("legacy", False, "auto"),
    )
    stored: dict[str, KeypointsXY] = {}
    for label, native_resolution, raco_engine in runs:
        database_path: Path = tmp_path / f"galileo_{label}.db"
        create_database(database_path, subset)
        counts, _elapsed = extract_raco(
            database_path,
            galileo_input_dir,
            image_names,
            min_score=0.0,
            max_num_features=BLOB_MAX_KEYPOINTS,
            native_resolution=native_resolution,
            raco_engine=raco_engine,
        )
        assert all(count == BLOB_MAX_KEYPOINTS for count in counts.values())
        stored[label] = read_keypoints(database_path, keyframe_ids[0])
    assert np.array_equal(stored["auto"], stored["legacy"])
    separations: Float32[ndarray, "num_native num_legacy"] = np.linalg.norm(stored["dynamic"][:, None, :] - stored["legacy"][None, :, :], axis=2)
    nearest: Float32[ndarray, " num_native"] = separations.min(axis=1)
    assert float(np.median(nearest)) < 1.0, float(np.median(nearest))
    assert float(np.mean(nearest < 1.0)) > 0.75, float(np.mean(nearest < 1.0))


KITTI_INPUT_DIR: Path = REPO_ROOT / "data" / "kitti" / "06_colsfm_input_slam"
"""KITTI 06 as `colsfm` input; not committed, so the test below skips without it."""

KITTI_FRAME_COUNT: int = 20
"""Frames extracted on both paths; enough to fill batches of 8 and stay quick."""

KITTI_PAIR_COUNT: int = 10
"""Consecutive pairs matched on both paths."""

MIN_KITTI_MEDIAN_INLIERS: int = 300
"""Lower band on the median verified matches over those pairs, on either path.

The measured medians are 543 stretched and 574 native; the band is there to catch
a collapse, not to pin a number that a re-export is allowed to move."""


@requires_raco_models
@pytest.mark.skipif(not KITTI_INPUT_DIR.is_dir(), reason="KITTI 06 input is not present")
def test_kitti_survives_running_at_its_own_1226x370(tmp_path: Path) -> None:
    """A frame a fifth the profile's area keeps its keypoints and its matches.

    KITTI is the case native resolution exists for — the fixed-shape graph
    inferred these frames over five times the pixels they have — so this is the
    one place the change could silently cost quality rather than buy speed. Both
    paths run here and the native one has to be at least as good.
    """
    from colsfm.features_raco import extract_raco

    meta: FramesMeta = read_frames_meta(KITTI_INPUT_DIR / FRAMES_META_NAME)
    keyframe_ids: list[int] = [keyframe.keyframe_id for keyframe in meta.keyframes[:KITTI_FRAME_COUNT]]
    subset: FramesMeta = meta.filtered(keyframe_ids)
    image_names: list[str] = [keyframe.image_name for keyframe in subset.keyframes]
    pairs: list[ImagePair] = [(keyframe_ids[index], keyframe_ids[index + 1]) for index in range(KITTI_PAIR_COUNT)]
    medians: dict[bool, float] = {}
    for native_resolution in (True, False):
        database_path: Path = tmp_path / f"kitti_{native_resolution}.db"
        create_database(database_path, subset)
        counts, _elapsed = extract_raco(
            database_path,
            KITTI_INPUT_DIR,
            image_names,
            min_score=0.0,
            max_num_features=BLOB_MAX_KEYPOINTS,
            native_resolution=native_resolution,
        )
        assert list(counts.values()) == [BLOB_MAX_KEYPOINTS] * KITTI_FRAME_COUNT
        matching: MatchReport = match_pairs(database_path, pairs, MatchingOptions(backend="raco"))
        assert matching.empty_pairs == 0
        medians[native_resolution] = matching.median_inliers
    assert min(medians.values()) >= MIN_KITTI_MEDIAN_INLIERS, medians
    assert medians[True] >= 0.9 * medians[False], medians


# ── Timing: the measurement `select_raco_engine`'s `auto` rule rests on ───────


@requires_raco_models
@pytest.mark.perf
def test_the_fixed_engine_beats_the_dynamic_one_at_the_profile_maximum(galileo_input: FramesMeta, galileo_input_dir: Path, tmp_path: Path) -> None:
    """Galileo is the shape-dynamic profile's maximum, and there the fixed engine wins.

    This is the whole reason `select_raco_engine` decides per size group instead
    of always taking the shape-dynamic engine: one profile's tactics have to
    cover 256x256 to 1216x1920, and the host also pays a `cv2.resize` from 1200
    to 1216 that the fixed path skips. Measured over 226 frames: 2.70 s fixed
    against 4.96 s dynamic (`docs/gpu-preprocessing.md`).

    Marked `perf` because the assertion is a ratio of wall clocks. It is a real
    claim and worth checking on demand; it is not evidence of correctness and has
    no business failing somebody's ordinary run.
    """
    from colsfm.features_raco import extract_raco

    image_names: list[str] = [keyframe.image_name for keyframe in galileo_input.keyframes]
    seconds: dict[RacoEngineChoice, float] = {}
    for choice in ("fixed", "dynamic"):
        database_path: Path = tmp_path / f"timing_{choice}.db"
        create_database(database_path, galileo_input)
        _counts, elapsed_seconds = extract_raco(
            database_path,
            galileo_input_dir,
            image_names,
            min_score=0.0,
            max_num_features=BLOB_MAX_KEYPOINTS,
            raco_engine=choice,  # type: ignore[arg-type]
        )
        seconds[choice] = elapsed_seconds
    print(
        f"[colsfm] raco {len(image_names)} frames | fixed {seconds['fixed']:.2f}s "
        f"| dynamic {seconds['dynamic']:.2f}s | ratio {seconds['dynamic'] / seconds['fixed']:.2f}x"
    )
    assert seconds["dynamic"] > seconds["fixed"], seconds
