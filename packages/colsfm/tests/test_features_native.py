"""The extraction machinery both TensorRT feature backends share.

These are pure contracts: the score gate, the keypoint budget, the coordinate
mapping, the shape the tensor bindings are read at, and the two database columns
that get written. None of them needs a GPU, an ONNX graph or
a built engine, so none of them is gated on one — which is the point.
`tests/colsfm/test_raco_backend.py` used to gate exactly this kind of check on
both, and a fresh clone therefore ran none of it.

The one gate that stays is `pytest.importorskip("tensorrt")`:
`colsfm.features_native` imports `colsfm.tensorrt_runtime`, which imports
TensorRT and `cuda-python` at module scope. That is a *package* gate, not a
device or a weights gate; the `colsfm` environment always satisfies it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Bool, Float32
from numpy import ndarray

from colsfm.database import Descriptors, KeypointsXY, create_database, read_descriptors_from_database, read_keypoints
from colsfm.frames_meta import FramesMeta

pytest.importorskip("tensorrt", reason="colsfm.features_native imports tensorrt at module scope")
pytest.importorskip("cuda.bindings.runtime", reason="colsfm.features_native allocates through cuda-python")

from colsfm.features_native import (
    DESCRIPTOR_TYPE,
    FeatureBatch,
    ImageTask,
    NativeModel,
    image_tasks,
    normalized_to_pixels,
    select_keypoints,
    store_feature_batch,
)

DESCRIPTOR_DIM: int = 128
"""Both ALIKED graphs emit 128 float32 columns per keypoint."""


# ── The score gate and the keypoint budget ───────────────────────────────────


def test_the_gate_keeps_a_point_sitting_exactly_on_the_threshold() -> None:
    """`>=`, not `>`, which is what the blob's own post-processing does.

    A `>` here loses one point per Galileo keyframe — the 2047 the blob reports
    against its 2048 head — and nothing downstream would ever say so.
    """
    scores: Float32[ndarray, " 4"] = np.asarray([0.004, 0.005, 0.006, 0.0], dtype=np.float32)

    kept: Bool[ndarray, " 4"] = select_keypoints(scores, min_score=0.005, max_num_features=0)

    assert kept.tolist() == [False, True, True, False]


def test_the_budget_keeps_the_strongest_of_what_the_gate_left() -> None:
    """Gate first, rank second: a point the gate removed cannot come back.

    The order is the intentional one. Ranking first and gating afterwards would
    return fewer than `max_num_features` points whenever weak points outrank the
    threshold, which is a different and quieter contract.
    """
    scores: Float32[ndarray, " 5"] = np.asarray([0.9, 0.1, 0.8, 0.2, 0.7], dtype=np.float32)

    kept: Bool[ndarray, " 5"] = select_keypoints(scores, min_score=0.5, max_num_features=2)

    assert kept.tolist() == [True, False, True, False, False]


def test_a_budget_of_zero_or_less_keeps_everything_the_gate_left() -> None:
    """`max_num_features <= 0` is "no ceiling", which is how the graphs' own top-k binds."""
    scores: Float32[ndarray, " 3"] = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)

    for budget in (0, -1):
        assert select_keypoints(scores, min_score=0.0, max_num_features=budget).tolist() == [True, True, True]


def test_a_budget_above_the_survivor_count_changes_nothing() -> None:
    """The truncation only fires when the gate left more than the budget."""
    scores: Float32[ndarray, " 4"] = np.asarray([0.9, 0.1, 0.8, 0.2], dtype=np.float32)

    kept: Bool[ndarray, " 4"] = select_keypoints(scores, min_score=0.5, max_num_features=2048)

    assert kept.tolist() == [True, False, True, False]


def test_a_strictly_decreasing_rank_truncates_from_the_front() -> None:
    """RaCo's `scores` is a selection rank, so the budget takes its leading points.

    The graph writes `arange(2048, 0, -1) / 2048`; running the same gate over it
    has to keep the first `max_num_features` in the graph's own order, because
    that order *is* the score.
    """
    scores: Float32[ndarray, " 8"] = (np.arange(8, 0, -1) / 8).astype(np.float32)

    kept: Bool[ndarray, " 8"] = select_keypoints(scores, min_score=0.0, max_num_features=3)

    assert kept.tolist() == [True, True, True, False, False, False, False, False]


# ── Coordinates and the host conversion ──────────────────────────────────────


def test_normalized_coordinates_map_onto_the_original_image_not_the_network() -> None:
    """`(k + 1) * 0.5 * size - 0.5` against the camera's own size: the `align_corners=False` inverse.

    The stretch to the engine's input is undone here and nowhere else, so the
    extrema have to land on the original image's outer edges, half a pixel past the corner pixel centres.
    """
    normalized_xy: Float32[ndarray, "3 2"] = np.asarray([[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float32)

    pixels_xy: KeypointsXY = normalized_to_pixels(normalized_xy, 1920, 1080)

    np.testing.assert_array_equal(pixels_xy, np.asarray([[-0.5, -0.5], [1919.5, 1079.5], [959.5, 539.5]], dtype=np.float32))


def test_a_pixel_centre_the_graph_emits_decodes_to_that_pixel() -> None:
    """The RaCo graph normalises pixel centres with `2 * (i + 0.5) / size - 1` (`raco_export`, align_corners=False).

    Decoding with the `align_corners=True` inverse instead scaled every keypoint by `1 - 1 / size` about the centre:
    half a pixel at the edges, a 0.5 px focal error on a 1600 px cube face (sfm-root H2).
    """
    columns: Float32[ndarray, " 4"] = np.asarray([0.0, 1.0, 800.0, 1599.0], dtype=np.float32)
    normalized_xy: Float32[ndarray, "4 2"] = np.stack([2.0 * (columns + 0.5) / 1600.0 - 1.0] * 2, axis=1).astype(np.float32)

    pixels_xy: KeypointsXY = normalized_to_pixels(normalized_xy, 1600, 1600)

    np.testing.assert_allclose(pixels_xy, np.stack([columns, columns], axis=1), atol=1e-3)
    assert pixels_xy.dtype == np.float32
    assert pixels_xy.flags.c_contiguous


# ── The typed boundary at the tensor bindings ────────────────────────────────


def test_a_flat_batch_one_output_and_a_batched_one_arrive_in_the_same_shape() -> None:
    """`FeatureBatch` is where binding ranks stop mattering.

    The blob's static graph hands back `[1, n, ...]` and a batched graph hands
    back `[b, n, ...]`; both become `[batch, ...]` here, which is what let the
    two backends' storage loops merge into one.
    """
    generator: np.random.Generator = np.random.default_rng(0)
    keypoints: Float32[ndarray, "4 2"] = generator.uniform(-1.0, 1.0, size=(4, 2)).astype(np.float32)
    descriptors: Float32[ndarray, "4 8"] = generator.normal(size=(4, 8)).astype(np.float32)
    scores: Float32[ndarray, " 4"] = generator.uniform(size=4).astype(np.float32)

    flat: FeatureBatch = FeatureBatch.from_bindings(
        {"keypoints": keypoints, "descriptors": descriptors, "scores": scores}, 1
    )
    ranked: FeatureBatch = FeatureBatch.from_bindings(
        {"keypoints": keypoints[None], "descriptors": descriptors[None], "scores": scores[None]}, 1
    )

    assert flat.keypoints_bn2.shape == (1, 4, 2)
    assert flat.descriptors_bnd.shape == (1, 4, 8)
    assert flat.scores_bn.shape == (1, 4)
    np.testing.assert_array_equal(flat.keypoints_bn2, ranked.keypoints_bn2)
    np.testing.assert_array_equal(flat.descriptors_bnd, ranked.descriptors_bnd)
    np.testing.assert_array_equal(flat.scores_bn, ranked.scores_bn)


def test_a_multi_image_batch_keeps_each_image_in_its_own_row() -> None:
    """A mis-sliced batch is the failure this typed boundary exists to prevent."""
    keypoints: Float32[ndarray, "3 4 2"] = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    descriptors: Float32[ndarray, "3 4 8"] = np.arange(96, dtype=np.float32).reshape(3, 4, 8)
    scores: Float32[ndarray, "3 4"] = np.arange(12, dtype=np.float32).reshape(3, 4)

    features: FeatureBatch = FeatureBatch.from_bindings(
        {"keypoints": keypoints, "descriptors": descriptors, "scores": scores}, 3
    )

    np.testing.assert_array_equal(features.keypoints_bn2[1], keypoints[1])
    np.testing.assert_array_equal(features.descriptors_bnd[2], descriptors[2])
    np.testing.assert_array_equal(features.scores_bn[0], scores[0])


# ── The database write ───────────────────────────────────────────────────────


def _two_keyframes(galileo_input: FramesMeta) -> FramesMeta:
    """The first two Galileo keyframes, which are two different cameras.

    Args:
        galileo_input: The 226-keyframe Galileo input collection.

    Returns:
        A two-keyframe collection, so a stored batch can be checked per image.
    """
    return galileo_input.filtered([keyframe.keyframe_id for keyframe in galileo_input.keyframes[:2]])




def test_storing_a_batch_writes_exactly_the_gated_points_per_image(
    galileo_input: FramesMeta, tmp_path: Path
) -> None:
    """Two images, two different gates, two different rows — and the right pixels.

    The mapping back to pixels uses each image's *own* camera size, so a batch
    that shared one size, or that wrote image 0's descriptors under image 1's id,
    both show up here.
    """
    import pycolmap

    subset: FramesMeta = _two_keyframes(galileo_input)
    database_path: Path = tmp_path / "store.db"
    create_database(database_path, subset)
    tasks: list[ImageTask] = image_tasks(database_path, [keyframe.image_name for keyframe in subset.keyframes])

    keypoints_bn2: Float32[ndarray, "2 3 2"] = np.asarray(
        [[[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]], [[1.0, 1.0], [0.0, 0.0], [-1.0, -1.0]]], dtype=np.float32
    )
    descriptors_bnd: Float32[ndarray, "2 3 128"] = np.tile(
        np.arange(3, dtype=np.float32).reshape(1, 3, 1), (2, 1, DESCRIPTOR_DIM)
    )
    scores_bn: Float32[ndarray, "2 3"] = np.asarray([[0.9, 0.004, 0.8], [0.1, 0.2, 0.3]], dtype=np.float32)

    with pycolmap.Database.open(database_path) as database:
        store_feature_batch(
            database,
            tasks,
            FeatureBatch(keypoints_bn2=keypoints_bn2, descriptors_bnd=descriptors_bnd, scores_bn=scores_bn),
            min_score=0.005,
            max_num_features=0,
        )

    first: KeypointsXY = read_keypoints(database_path, tasks[0].image_id)
    second: KeypointsXY = read_keypoints(database_path, tasks[1].image_id)
    assert first.shape == (2, 2)
    assert second.shape == (3, 2)
    np.testing.assert_array_equal(
        first,
        np.asarray(
            # Normalised -1 / +1 are the image's outer edges, half a pixel past the corner pixel centres (align_corners=False).
            [[-0.5, -0.5], [tasks[0].image_width - 0.5, tasks[0].image_height - 0.5]],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(second[0], np.asarray([tasks[1].image_width - 0.5, tasks[1].image_height - 0.5], dtype=np.float32))

    stored: dict[int, Descriptors] = read_descriptors_from_database(database_path, [task.image_id for task in tasks])
    first_descriptors: Descriptors = stored[tasks[0].image_id]
    assert first_descriptors.shape == (2, DESCRIPTOR_DIM)
    np.testing.assert_array_equal(first_descriptors[:, 0], np.asarray([0.0, 2.0], dtype=np.float32))
    assert stored[tasks[1].image_id].shape == (3, DESCRIPTOR_DIM)


def test_the_budget_applies_per_image_not_per_batch(
    galileo_input: FramesMeta, tmp_path: Path
) -> None:
    """Each image gets its own `max_num_features`, which a flattened batch would not."""
    import pycolmap

    subset: FramesMeta = _two_keyframes(galileo_input)
    database_path: Path = tmp_path / "budget.db"
    create_database(database_path, subset)
    tasks: list[ImageTask] = image_tasks(database_path, [keyframe.image_name for keyframe in subset.keyframes])

    with pycolmap.Database.open(database_path) as database:
        store_feature_batch(
            database,
            tasks,
            FeatureBatch(
                keypoints_bn2=np.zeros((2, 4, 2), dtype=np.float32),
                descriptors_bnd=np.zeros((2, 4, DESCRIPTOR_DIM), dtype=np.float32),
                scores_bn=np.tile(np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32), (2, 1)),
            ),
            min_score=0.0,
            max_num_features=2,
        )

    assert read_keypoints(database_path, tasks[0].image_id).shape[0] == 2
    assert read_keypoints(database_path, tasks[1].image_id).shape[0] == 2


def test_image_tasks_refuse_a_name_the_database_does_not_hold(
    galileo_input: FramesMeta, tmp_path: Path
) -> None:
    """A typo in an image name is a `KeyError` here, not a silently skipped image."""
    database_path: Path = tmp_path / "missing.db"
    create_database(database_path, _two_keyframes(galileo_input))

    with pytest.raises(KeyError, match="no image named"):
        image_tasks(database_path, ["not/a/real/image.jpeg"])


# ── The two model adapters ───────────────────────────────────────────────────


def test_raco_scores_are_a_rank_not_a_detector_response() -> None:
    """RaCo's `scores` output is a monotone selection rank, so the blob's detector threshold must not gate it."""
    from colsfm.features import BLOB_DETECTOR_THRESHOLD, RACO_MIN_SCORE
    from colsfm.features_raco import RACO_MODEL

    assert RACO_MODEL.score_meaning == "selection_rank"
    assert RACO_MIN_SCORE == 0.0
    assert BLOB_DETECTOR_THRESHOLD > RACO_MIN_SCORE


def test_the_raco_model_takes_one_image_binding_and_stores_aliked_descriptors() -> None:
    """The graph takes one `image` input and stores `ALIKED_N16ROT` descriptors, legible to every later stage."""
    from colsfm.features_raco import RACO_MODEL

    assert isinstance(RACO_MODEL, NativeModel)
    assert RACO_MODEL.image_binding == "image"
    assert RACO_MODEL.descriptor_type == DESCRIPTOR_TYPE
