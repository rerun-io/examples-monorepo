"""Tests for `colsfm.retrieval`, the Python replacement for cuSFM's BoW stage.

Seams under test (all public):

* `build_retrieval_index` — turning per-image descriptors into a similarity matrix.
* `RetrievalIndex.query` / `.score` — ranked candidates, self and temporal exclusion.
* `RetrievalIndex.search` — the same, plus what the temporal gate dropped.
* `read_descriptors_from_database` — the pycolmap read path.
* `bow_vocab` — the Rust vocabulary's TF-IDF and L1 score, against the spec's formulas.

The Galileo test reads the shipped blob keyframe protos directly rather than running the
feature extractor, so it is offline and needs no GPU. Its reader lives in this file
because the feature/database modules belong to other workers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pycolmap
import pytest
from jaxtyping import Float32, Float64, Int32
from numpy import ndarray

from colsfm.database import read_descriptors_from_database
from colsfm.frames_meta import FramesMeta, KeyframeMeta
from colsfm.retrieval import (
    ALIKED_DESCRIPTOR_DIM,
    GOOD_SCORE_THRESHOLD,
    BruteForceConfig,
    Candidate,
    Descriptors,
    RetrievalBackend,
    RetrievalConfig,
    RetrievalIndex,
    RetrievalQuery,
    VocabConfig,
    build_retrieval_index,
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
"""Repo root, so data paths resolve regardless of the working directory."""

FRAME_PERIOD_US: Final[int] = 100_000
"""Synthetic capture period, so a temporal gate can be expressed in whole frames."""


# --------------------------------------------------------------------------------------
# synthetic corpus
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SyntheticCorpus:
    """A descriptor corpus with a known revisit structure."""

    descriptors: dict[int, Descriptors]
    """Descriptors per image id; ids are `0 .. n_images - 1`."""
    timestamps_us: dict[int, int]
    """Capture time per image id, one `FRAME_PERIOD_US` apart."""
    loop_offset: int
    """Image `i` and image `i + loop_offset` share `loop_fraction` of their descriptors."""
    loop_fraction: float
    """Fraction of descriptors an image shares with its revisit partner."""
    neighbour_fraction: float
    """Fraction an image shares with its immediate temporal neighbour."""


def make_synthetic_corpus(
    n_images: int = 100,
    descriptors_per_image: int = 64,
    loop_offset: int = 50,
    loop_fraction: float = 0.6,
    neighbour_fraction: float = 0.45,
    seed: int = 0,
) -> SyntheticCorpus:
    """Build a corpus where temporal neighbours look alike and `i`/`i + 50` are a revisit.

    Every image observes `descriptors_per_image` landmarks. Image `i` inherits its leading
    45 % of slots from the trailing 45 % of image `i - 1`, so consecutive frames overlap
    the way real video does while frames two apart share nothing. Image `i + loop_offset`
    then copies the trailing 60 % of image `i`'s slots — the revisit the retrieval must
    find. The two overlaps are deliberately different so the revisit has a unique best
    score and the test cannot pass on a tie.

    Args:
        n_images: How many images to synthesise; must be at least `2 * loop_offset`.
        descriptors_per_image: Landmarks observed per image.
        loop_offset: Distance in frames between an image and its revisit partner.
        loop_fraction: Fraction of slots a revisit pair shares.
        neighbour_fraction: Fraction of slots consecutive frames share.
        seed: Seed for the landmark directions and the observation noise.

    Returns:
        The corpus, with unit-norm float32 descriptors.

    Raises:
        ValueError: When the corpus is too short to hold a revisit pair.
    """
    if n_images < 2 * loop_offset:
        raise ValueError(f"n_images {n_images} must be at least 2 * loop_offset {loop_offset}")
    generator: np.random.Generator = np.random.default_rng(seed)
    loop_slots: int = round(loop_fraction * descriptors_per_image)
    neighbour_slots: int = round(neighbour_fraction * descriptors_per_image)

    landmark_ids: Int32[ndarray, "n_images descriptors_per_image"] = np.arange(n_images * descriptors_per_image, dtype=np.int32).reshape(
        n_images, descriptors_per_image
    )
    for image_id in range(1, n_images):
        landmark_ids[image_id, :neighbour_slots] = landmark_ids[image_id - 1, -neighbour_slots:]
    for image_id in range(loop_offset, 2 * loop_offset):
        landmark_ids[image_id, -loop_slots:] = landmark_ids[image_id - loop_offset, -loop_slots:]

    landmarks: Float32[ndarray, "n_landmarks 128"] = generator.standard_normal((n_images * descriptors_per_image, ALIKED_DESCRIPTOR_DIM)).astype(
        np.float32
    )
    landmarks /= np.linalg.norm(landmarks, axis=1, keepdims=True)

    descriptors: dict[int, Descriptors] = {}
    for image_id in range(n_images):
        observed: Float32[ndarray, "descriptors_per_image 128"] = landmarks[landmark_ids[image_id]].copy()
        observed += generator.standard_normal(observed.shape).astype(np.float32) * np.float32(0.02)
        descriptors[image_id] = np.ascontiguousarray(observed / np.linalg.norm(observed, axis=1, keepdims=True))

    return SyntheticCorpus(
        descriptors=descriptors,
        timestamps_us={image_id: image_id * FRAME_PERIOD_US for image_id in range(n_images)},
        loop_offset=loop_offset,
        loop_fraction=loop_slots / descriptors_per_image,
        neighbour_fraction=neighbour_slots / descriptors_per_image,
    )


@pytest.fixture(scope="module")
def synthetic_corpus() -> SyntheticCorpus:
    """The shared synthetic corpus."""
    return make_synthetic_corpus()


@pytest.fixture(scope="module")
def synthetic_index(synthetic_corpus: SyntheticCorpus) -> RetrievalIndex:
    """An index over the synthetic corpus, built with the default settings."""
    return build_retrieval_index(synthetic_corpus.descriptors)


# --------------------------------------------------------------------------------------
# a. synthetic retrieval
# --------------------------------------------------------------------------------------


def test_revisit_partner_is_the_top_candidate_once_neighbours_are_excluded(
    synthetic_corpus: SyntheticCorpus, synthetic_index: RetrievalIndex
) -> None:
    """With the temporal gate on, `i + 50` outranks everything for every query."""
    gate_us: int = 5 * FRAME_PERIOD_US
    for query_id in range(synthetic_corpus.loop_offset):
        candidates: list[Candidate] = synthetic_index.query(query_id, top_k=20, min_time_gap_us=gate_us, timestamps=synthetic_corpus.timestamps_us)
        assert candidates, f"image {query_id} retrieved nothing"
        assert candidates[0].image_id == query_id + synthetic_corpus.loop_offset
        assert candidates[0].score > 0.1


def test_query_excludes_itself_and_its_temporal_neighbours(synthetic_corpus: SyntheticCorpus, synthetic_index: RetrievalIndex) -> None:
    """The gate removes the query and every image within `min_time_gap_us` of it."""
    gate_us: int = 5 * FRAME_PERIOD_US
    candidates: list[Candidate] = synthetic_index.query(10, top_k=20, min_time_gap_us=gate_us, timestamps=synthetic_corpus.timestamps_us)
    returned: set[int] = {candidate.image_id for candidate in candidates}
    assert returned.isdisjoint(set(range(6, 15)))
    assert 10 not in returned


def test_the_gate_is_what_removes_the_neighbours(synthetic_corpus: SyntheticCorpus, synthetic_index: RetrievalIndex) -> None:
    """Unfiltered, the query's neighbours rank second and third; the gate is what drops them.

    This is the reason `docs/spec/bow.md` §9.2 lists neighbour exclusion under "must keep"
    — the pose graph already carries those links as sequential edges, so a candidate that
    close in time adds no information.
    """
    unfiltered: list[Candidate] = synthetic_index.query(10, top_k=5)
    assert {candidate.image_id for candidate in unfiltered[1:3]} == {9, 11}
    gated: list[Candidate] = synthetic_index.query(10, top_k=5, min_time_gap_us=5 * FRAME_PERIOD_US, timestamps=synthetic_corpus.timestamps_us)
    assert {9, 11}.isdisjoint({candidate.image_id for candidate in gated})


def test_scores_are_symmetric_and_match_the_known_overlap(synthetic_corpus: SyntheticCorpus, synthetic_index: RetrievalIndex) -> None:
    """Mutual-nearest-neighbour votes recover the fraction of shared landmarks."""
    assert synthetic_index.score(10, 60) == pytest.approx(synthetic_corpus.loop_fraction, abs=0.05)
    assert synthetic_index.score(60, 10) == synthetic_index.score(10, 60)
    assert synthetic_index.score(10, 11) == pytest.approx(synthetic_corpus.neighbour_fraction, abs=0.05)
    assert synthetic_index.score(10, 10) == 1.0
    assert synthetic_index.score(0, 99) < 0.05


def test_unrelated_images_score_near_zero() -> None:
    """Random descriptor sets must not clear a `good_score_threshold`-style gate."""
    generator: np.random.Generator = np.random.default_rng(7)
    descriptors: dict[int, Descriptors] = {}
    for image_id in range(8):
        rows: Float32[ndarray, "64 128"] = generator.standard_normal((64, ALIKED_DESCRIPTOR_DIM)).astype(np.float32)
        descriptors[image_id] = np.ascontiguousarray(rows / np.linalg.norm(rows, axis=1, keepdims=True))
    index: RetrievalIndex = build_retrieval_index(descriptors)
    off_diagonal: Float32[ndarray, "n n"] = index.similarity - np.eye(8, dtype=np.float32)
    assert float(off_diagonal.max()) < 0.1


def test_query_rejects_an_unknown_image_and_a_missing_timestamp_map(synthetic_index: RetrievalIndex) -> None:
    """Bad arguments fail loudly rather than silently returning nothing."""
    with pytest.raises(KeyError):
        synthetic_index.query(10_000)
    with pytest.raises(ValueError):
        synthetic_index.query(10, min_time_gap_us=1)
    with pytest.raises(ValueError):
        synthetic_index.query(10, top_k=0)


def test_build_rejects_a_single_image() -> None:
    """An index over one image can answer nothing."""
    with pytest.raises(ValueError):
        build_retrieval_index({0: np.zeros((4, ALIKED_DESCRIPTOR_DIM), dtype=np.float32)})


@pytest.mark.parametrize("backend", ["brute_force", "vocab"])
def test_build_rejects_descriptors_of_different_widths_on_either_backend(backend: RetrievalBackend) -> None:
    """A ragged corpus is a `ValueError`, as the docstring promises, on both paths.

    The check used to live inside the brute force's own stacking step, so the vocab path
    reached `numpy.concatenate` with mismatched rows and failed there with a message about
    array dimensions instead of about the descriptors.
    """
    generator: np.random.Generator = np.random.default_rng(3)
    corpus: dict[int, Descriptors] = {
        0: generator.standard_normal((16, ALIKED_DESCRIPTOR_DIM), dtype=np.float32),
        1: generator.standard_normal((16, ALIKED_DESCRIPTOR_DIM), dtype=np.float32),
        2: generator.standard_normal((16, ALIKED_DESCRIPTOR_DIM // 2), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="descriptors, expected"):
        build_retrieval_index(corpus, RetrievalConfig(backend=backend))


def test_search_reports_what_the_temporal_gate_dropped_from_the_same_walk(synthetic_corpus: SyntheticCorpus, synthetic_index: RetrievalIndex) -> None:
    """`search` returns the hits and the gated count together, so a funnel can compose.

    The loop-closure stage used to issue a second, ungated query only to count the temporal
    rejections; the two numbers then described different candidate sets. Here the gate runs
    inside the one ranking walk, so every hit the walk looked at is either returned or
    counted.

    The gate also runs inside the walk rather than over a finished top-k, which is what
    keeps all `top_k` slots useful: with three slots, the gated query still returns three
    candidates where a gated top-3 would have returned one.
    """
    gate_us: int = 5 * FRAME_PERIOD_US
    answer: RetrievalQuery = synthetic_index.search(10, top_k=20, min_time_gap_us=gate_us, timestamps=synthetic_corpus.timestamps_us)
    assert answer.candidates
    assert answer.rejected_by_time > 0
    assert answer.candidates == synthetic_index.query(10, top_k=20, min_time_gap_us=gate_us, timestamps=synthetic_corpus.timestamps_us)
    assert synthetic_index.search(10, top_k=20).rejected_by_time == 0

    narrow: RetrievalQuery = synthetic_index.search(10, top_k=3, min_time_gap_us=gate_us, timestamps=synthetic_corpus.timestamps_us)
    ungated_top3: list[Candidate] = synthetic_index.query(10, top_k=3)
    assert len(narrow.candidates) == 3
    assert {candidate.image_id for candidate in ungated_top3} & {9, 11}
    assert {candidate.image_id for candidate in narrow.candidates}.isdisjoint({9, 11})


def _walk_in_score_order(index: RetrievalIndex, image_id: int, top_k: int, gap_us: int | None, timestamps_us: dict[int, int]) -> RetrievalQuery:
    """The ranking walk `search` promises, spelled from its definition.

    Every positive score, best first, equal scores in row order; the query itself is
    skipped, a hit closer in time than `gap_us` is counted and skipped, the walk stops at
    `top_k` candidates.
    """
    scores: Float32[ndarray, "n_images"] = index.similarity[index.image_ids.index(image_id)]
    ranked: list[int] = sorted((row for row in range(len(index.image_ids)) if scores[row] > 0.0), key=lambda row: (-float(scores[row]), row))
    candidates: list[Candidate] = []
    rejected_by_time: int = 0
    for row in ranked:
        if len(candidates) == top_k:
            break
        other_id: int = index.image_ids[row]
        if other_id == image_id:
            continue
        if gap_us is not None and abs(timestamps_us[other_id] - timestamps_us[image_id]) < gap_us:
            rejected_by_time += 1
            continue
        candidates.append(Candidate(image_id=other_id, score=float(scores[row])))
    return RetrievalQuery(candidates=candidates, rejected_by_time=rejected_by_time)


@pytest.mark.parametrize("seed", range(6))
def test_partial_search_equals_the_full_sort_with_ties_and_the_time_gate(seed: int) -> None:
    """`search` ranks exactly as a full stable sort would, ties and the time gate included.

    The scores take five levels, zero among them, so most hits tie with many others and the
    k-th score is shared across the cut; the gate is wide enough to drop dozens of hits per
    query, so the walk must reach past the first `top_k` rows.
    """
    generator: np.random.Generator = np.random.default_rng(seed)
    n_images: int = 300
    levels: Float32[ndarray, "5"] = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    upper: Float32[ndarray, "n_images n_images"] = np.triu(levels[generator.integers(0, len(levels), size=(n_images, n_images))], 1)
    similarity: Float32[ndarray, "n_images n_images"] = upper + upper.T + np.eye(n_images, dtype=np.float32)
    image_ids: tuple[int, ...] = tuple(sorted(generator.choice(10 * n_images, size=n_images, replace=False).tolist()))
    timestamps_us: dict[int, int] = {image_id: int(generator.integers(0, 60 * FRAME_PERIOD_US)) for image_id in image_ids}
    index: RetrievalIndex = RetrievalIndex(
        image_ids=image_ids, similarity=similarity, descriptors_per_image=1, config=RetrievalConfig(), build_seconds=0.0
    )
    for image_id in image_ids[::15]:
        for top_k in (1, 3, 20, n_images + 5):
            for gap_us in (None, 5 * FRAME_PERIOD_US, 30 * FRAME_PERIOD_US):
                expected: RetrievalQuery = _walk_in_score_order(index, image_id, top_k, gap_us, timestamps_us)
                assert index.search(image_id, top_k=top_k, min_time_gap_us=gap_us, timestamps=timestamps_us) == expected


def test_the_config_nests_the_two_backends_and_still_defaults_in_one_call() -> None:
    """`RetrievalConfig()` builds a whole working config; each backend owns its own knobs.

    The two backends share nothing but the seed, so a flat config made every reader work out
    which of ten fields the backend they chose actually reads. Nesting must not cost the
    zero-argument construction the pipeline relies on.
    """
    config: RetrievalConfig = RetrievalConfig()
    assert config.brute_force == BruteForceConfig()
    assert config.vocab == VocabConfig()
    assert RetrievalConfig(vocab=VocabConfig(depth=2)).vocab.depth == 2
    with pytest.raises(ValueError, match="max_descriptors_per_image"):
        BruteForceConfig(max_descriptors_per_image=0)
    with pytest.raises(ValueError, match="vocab depth"):
        VocabConfig(depth=0)


def test_the_backend_is_derived_from_the_config_and_the_corpus(synthetic_corpus: SyntheticCorpus, synthetic_index: RetrievalIndex) -> None:
    """`RetrievalIndex.backend` is a property, so it cannot disagree with what built it."""
    assert synthetic_index.backend == synthetic_index.config.resolve_backend(len(synthetic_index.image_ids))
    assert synthetic_index.backend == "brute_force"
    assert build_retrieval_index(synthetic_corpus.descriptors, RetrievalConfig(brute_force_max_images=1)).backend == "vocab"


# --------------------------------------------------------------------------------------
# a. runtime
# --------------------------------------------------------------------------------------


def _random_corpus(n_images: int, descriptors_per_image: int, seed: int) -> dict[int, Descriptors]:
    """Build `n_images` sets of unit-norm random descriptors.

    Args:
        n_images: How many images to synthesise.
        descriptors_per_image: Descriptors per image.
        seed: Seed for the generator.

    Returns:
        Float32 descriptors per image id.
    """
    generator: np.random.Generator = np.random.default_rng(seed)
    corpus: dict[int, Descriptors] = {}
    for image_id in range(n_images):
        rows: Float32[ndarray, "n 128"] = generator.standard_normal((descriptors_per_image, ALIKED_DESCRIPTOR_DIM), dtype=np.float32)
        corpus[image_id] = np.ascontiguousarray(rows / np.linalg.norm(rows, axis=1, keepdims=True))
    return corpus


@pytest.mark.perf
@pytest.mark.parametrize(("n_images", "budget_seconds"), [(226, 20.0), (900, 60.0)])
def test_brute_force_build_stays_inside_the_runtime_budget(n_images: int, budget_seconds: float) -> None:
    """226 x 2048 in a few seconds and 900 x 2048 under a minute, on CPU NumPy.

    The full brute force over all 2048 descriptors per image is 5.5e13 FLOP for Galileo
    and 8.7e14 for 900 images, i.e. ~370 s and ~5900 s at this machine's measured 147
    GFLOP/s — so `RetrievalConfig.max_total_descriptors` is what makes the budget
    reachable. See the module docstring of `colsfm.retrieval` for the recall it costs.

    The backend is named rather than left to `auto`, which would send the 900-image case to
    the vocab index; that one has its own budget test below.

    Every assertion here is a wall-clock bound, so the whole test is `perf`: what it
    measures on a shared machine is the machine. The recall these settings buy is
    asserted on the real Galileo descriptors instead, without a clock.
    """
    corpus: dict[int, Descriptors] = _random_corpus(n_images, 2048, seed=n_images)
    started: float = time.perf_counter()
    index: RetrievalIndex = build_retrieval_index(corpus, RetrievalConfig(backend="brute_force"))
    elapsed: float = time.perf_counter() - started
    index.query(0, top_k=20)
    print(
        f"retrieval build: {n_images} images x 2048 descriptors -> "
        f"{index.descriptors_per_image} kept/image, {elapsed:.1f} s (budget {budget_seconds:.0f} s)"
    )
    assert elapsed < budget_seconds


# --------------------------------------------------------------------------------------
# a. the vocabulary-tree backend
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_vocab_index(synthetic_corpus: SyntheticCorpus) -> RetrievalIndex:
    """A vocabulary-tree index over the same synthetic corpus the brute force uses."""
    return build_retrieval_index(synthetic_corpus.descriptors, RetrievalConfig(backend="vocab"))


def test_auto_takes_brute_force_for_a_short_sequence_and_the_vocabulary_for_a_long_one() -> None:
    """`auto` switches at `brute_force_max_images`, so Galileo and RoboCap take different paths.

    The switch exists because the brute force's descriptor budget is global: at 4528 images
    `max_total_descriptors` leaves 22 descriptors each and every score collapses.
    """
    config: RetrievalConfig = RetrievalConfig()
    assert config.backend == "auto"
    assert config.resolve_backend(226) == "brute_force"
    assert config.resolve_backend(config.brute_force_max_images) == "brute_force"
    assert config.resolve_backend(config.brute_force_max_images + 1) == "vocab"
    assert config.resolve_backend(4528) == "vocab"
    assert RetrievalConfig(backend="brute_force").resolve_backend(4528) == "brute_force"
    assert RetrievalConfig(backend="vocab").resolve_backend(2) == "vocab"


def test_the_vocabulary_backend_finds_the_revisit_once_neighbours_are_excluded(
    synthetic_corpus: SyntheticCorpus, synthetic_vocab_index: RetrievalIndex
) -> None:
    """The recall test the brute force passes, on the same corpus and the same gate.

    This is the seam that matters: whatever the score means, `i + 50` must outrank every
    other image once the temporal gate removes `i`'s neighbours.
    """
    assert synthetic_vocab_index.backend == "vocab"
    gate_us: int = 5 * FRAME_PERIOD_US
    for query_id in range(synthetic_corpus.loop_offset):
        candidates: list[Candidate] = synthetic_vocab_index.query(
            query_id, top_k=20, min_time_gap_us=gate_us, timestamps=synthetic_corpus.timestamps_us
        )
        assert candidates, f"image {query_id} retrieved nothing"
        assert candidates[0].image_id == query_id + synthetic_corpus.loop_offset
        assert candidates[0].score > GOOD_SCORE_THRESHOLD


def test_the_vocabulary_score_ranks_the_revisit_above_the_neighbour_above_the_stranger(
    synthetic_vocab_index: RetrievalIndex,
) -> None:
    """Scores are symmetric, 1.0 on the diagonal, and ordered by the known overlap.

    The DBoW2 L1 score is not the shared-landmark fraction — quantising descriptors into
    words loses some of the overlap — so the assertions are on the ordering and on the
    unrelated pair staying under the score gate, not on the exact value.
    """
    revisit: float = synthetic_vocab_index.score(10, 60)
    neighbour: float = synthetic_vocab_index.score(10, 11)
    stranger: float = synthetic_vocab_index.score(0, 99)
    assert synthetic_vocab_index.score(10, 10) == 1.0
    assert synthetic_vocab_index.score(60, 10) == revisit
    assert revisit > neighbour > stranger
    assert stranger < GOOD_SCORE_THRESHOLD


def test_the_vocabulary_backend_is_reproducible_and_the_seed_is_what_moves_it(
    synthetic_corpus: SyntheticCorpus, synthetic_vocab_index: RetrievalIndex
) -> None:
    """k-means seeding is drawn from `RetrievalConfig.seed`, so a rebuild is bit-identical.

    The blob's tree is explicitly *not* reproducible — it seeds libc `rand()` with nothing
    (`docs/spec/bow.md` §6.5) — so this is a property the reimplementation adds.
    """
    rebuilt: RetrievalIndex = build_retrieval_index(synthetic_corpus.descriptors, RetrievalConfig(backend="vocab"))
    np.testing.assert_array_equal(rebuilt.similarity, synthetic_vocab_index.similarity)
    reseeded: RetrievalIndex = build_retrieval_index(synthetic_corpus.descriptors, RetrievalConfig(backend="vocab", seed=7))
    assert not np.array_equal(reseeded.similarity, synthetic_vocab_index.similarity)
    assert reseeded.query(10, top_k=1)[0].score > 0.0


def l1_score(query: dict[int, float], document: dict[int, float]) -> float:
    """The DBoW2 L1 score of two L1-normalised BoW vectors, literally `docs/spec/bow.md` §9.5.

    `s = -0.5 * sum_{w in q & d} (|q_w - d_w| - |q_w| - |d_w|)`.
    """
    shared: set[int] = set(query) & set(document)
    return -0.5 * sum(abs(query[word] - document[word]) - abs(query[word]) - abs(document[word]) for word in shared)


def test_the_rust_score_agrees_with_the_spec_tf_idf_and_l1_reference() -> None:
    """`bow_vocab`'s similarity must equal the spec's TF-IDF (§7.2) and L1 score (§9.5), pair by pair.

    The words come from the Rust tree itself (`Vocabulary.words`); the weights and the score
    are then recomputed here from the spec's formulas: raw counts times `log(n_images / df)`,
    zero-weight words dropped, rows L1-normalised, and the literal L1 expression.
    """
    from bow_vocab import Vocabulary

    corpus: dict[int, Descriptors] = _random_corpus(6, 300, seed=3)
    images: list[Descriptors] = [corpus[image_id] for image_id in sorted(corpus)]
    vocabulary: Vocabulary = Vocabulary(np.concatenate(images), branching=4, depth=3, iterations=5, seed=0)
    scores: Float32[ndarray, "n_images n_images"] = vocabulary.similarity(images)

    counts: list[dict[int, int]] = []
    for image in images:
        words, tally = np.unique(vocabulary.words(image), return_counts=True)
        counts.append({int(word): int(count) for word, count in zip(words, tally, strict=True)})
    document_frequency: dict[int, int] = {}
    for image_counts in counts:
        for word in image_counts:
            document_frequency[word] = document_frequency.get(word, 0) + 1
    bows: list[dict[int, float]] = []
    for image_counts in counts:
        weights: dict[int, float] = {word: count * np.log(len(images) / document_frequency[word]) for word, count in image_counts.items()}
        weights = {word: weight for word, weight in weights.items() if weight > 0.0}
        total: float = sum(weights.values())
        bows.append({word: weight / total for word, weight in weights.items()})

    for first in range(len(images)):
        assert scores[first, first] == 1.0
        for second in range(len(images)):
            if first != second:
                assert l1_score(bows[first], bows[second]) == pytest.approx(float(scores[first, second]), abs=1e-5)


def test_the_vocab_backend_borrows_the_descriptors_and_refuses_a_layout_it_cannot() -> None:
    """The vocab backend hands the per-image arrays to Rust without a copy, so a non-C-contiguous one is refused.

    The database reader yields C-contiguous float32 arrays, which is the only production source;
    converting a whole corpus would double its memory, so nothing converts.
    """
    corpus: dict[int, Descriptors] = _random_corpus(6, 300, seed=4)
    config: RetrievalConfig = RetrievalConfig(backend="vocab", vocab=VocabConfig(branching=4, depth=2))
    assert build_retrieval_index(corpus, config).similarity.shape == (6, 6)
    corpus[3] = np.asfortranarray(corpus[3])
    with pytest.raises(ValueError, match="C-contiguous"):
        build_retrieval_index(corpus, config)


def test_the_binding_refuses_arrays_it_would_misread() -> None:
    """`bow_vocab` borrows the numpy buffer, so it must refuse what it cannot read row by row.

    A Fortran-ordered or strided array would be read with its rows mixed; a wrong width or dtype
    is refused too. A C-contiguous row slice is a valid view and still works.
    """
    from bow_vocab import Vocabulary

    generator: np.random.Generator = np.random.default_rng(5)
    data: Float32[ndarray, "n 128"] = generator.standard_normal((400, 128)).astype(np.float32)
    settings: dict[str, int] = {"branching": 4, "depth": 2, "iterations": 3, "seed": 0}
    vocabulary: Vocabulary = Vocabulary(data, **settings)
    for misread in (np.asfortranarray(data), data[:, ::2]):
        with pytest.raises(ValueError):
            Vocabulary(misread, **settings)
    narrow: Float32[ndarray, "n 96"] = np.ascontiguousarray(data[:, :96])
    assert Vocabulary(narrow, **settings).n_words == 16  # any width trains; the tree's width is then the contract
    for bad in (np.asfortranarray(data), data[:, ::2], narrow):
        with pytest.raises(ValueError):
            vocabulary.words(bad)
        with pytest.raises(ValueError):
            vocabulary.similarity([data, bad])
    wide: Float64[ndarray, "n 128"] = data.astype(np.float64)
    with pytest.raises(TypeError):
        Vocabulary(wide, **settings)
    with pytest.raises(TypeError):
        vocabulary.words(wide)
    with pytest.raises(TypeError):
        vocabulary.similarity([data, wide])
    rows: Float32[ndarray, "m 128"] = data[10:50]
    np.testing.assert_array_equal(vocabulary.words(rows), vocabulary.words(data)[10:50])
    assert vocabulary.similarity([data[:200], rows]).shape == (2, 2)


VOCAB_PROBE_IMAGES: int = 100
"""Synthetic images in the vocabulary fixture: 204 800 descriptors still fill all 10^5 leaves."""


@dataclass(frozen=True, slots=True)
class VocabBuild:
    """A vocabulary index over `VOCAB_PROBE_IMAGES` synthetic images, with what it cost to make and use."""

    index: RetrievalIndex
    """The built index."""
    build_seconds: float
    """Wall-clock seconds `build_retrieval_index` took. Only the `perf` test reads this."""
    query_seconds: float
    """Wall-clock seconds all the queries took. Only the `perf` test reads this."""


@pytest.fixture(scope="module")
def vocab_build() -> VocabBuild:
    """Build the vocabulary index once and query every image of it.

    Two tests read this — the tree's shape and the runtime budget — and the build
    is the most expensive single step in this module, so it happens once.

    Returns:
        The index, its build time and the time all the queries took.
    """
    corpus: dict[int, Descriptors] = _random_corpus(VOCAB_PROBE_IMAGES, 2048, seed=1)
    started: float = time.perf_counter()
    index: RetrievalIndex = build_retrieval_index(corpus, RetrievalConfig(backend="vocab"))
    build_seconds: float = time.perf_counter() - started
    started = time.perf_counter()
    for image_id in index.image_ids:
        index.query(image_id, top_k=20)
    query_seconds: float = time.perf_counter() - started
    print(
        f"vocab retrieval: {VOCAB_PROBE_IMAGES} x 2048 -> {index.n_words} words, build {build_seconds:.1f} s, {len(index.image_ids)} queries {query_seconds:.2f} s"
    )
    return VocabBuild(index=index, build_seconds=build_seconds, query_seconds=query_seconds)


def test_the_vocabulary_backend_builds_the_whole_tree(vocab_build: VocabBuild) -> None:
    """100 x 2048 descriptors fill every leaf: the word count is `branching ** depth`.

    The claim that does not depend on the machine, so it runs by default. What the
    build and the queries cost is the `perf` test below.
    """
    assert vocab_build.index.n_words == VocabConfig().branching ** VocabConfig().depth


@pytest.mark.perf
def test_the_vocabulary_backend_builds_and_queries_inside_the_budget(vocab_build: VocabBuild) -> None:
    """100 x 2048 descriptors: build and every query, both well under the pipeline budget.

    The build trains the vocabulary and scores every pair (0.3 s on all cores when measured),
    so the queries only read the matrix (0.02 s for 100). The bounds are generous multiples so
    the test fails on an algorithmic regression rather than on a busy machine.
    """
    assert vocab_build.build_seconds < 90.0
    assert vocab_build.query_seconds < 30.0


# --------------------------------------------------------------------------------------
# b. real Galileo descriptors
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GalileoRecall:
    """How well one index's top candidates match the geometry of the Galileo sweep."""

    n_keyframes: int
    """Keyframes queried."""
    same_or_stereo_camera: int
    """Queries whose top candidate came from their own camera or its stereo partner."""
    top1_within_two_rigs: int
    """Queries whose top candidate is within 2 rig frames."""
    top3_within_two_rigs: int
    """Queries with a top-3 candidate within 2 rig frames."""


def galileo_recall(index: RetrievalIndex, frames_meta: FramesMeta) -> GalileoRecall:
    """Tally an index's top-3 candidates against the rig order and the stereo pairing.

    Args:
        index: An index over the Galileo keyframes.
        frames_meta: The metadata of the same run.

    Returns:
        The three counts, out of the keyframes queried.
    """
    stereo_partner: dict[int, int] = {}
    for pair in frames_meta.stereo_pairs:
        stereo_partner[pair.left_camera_params_id] = pair.right_camera_params_id
        stereo_partner[pair.right_camera_params_id] = pair.left_camera_params_id
    keyframe_by_id: dict[int, KeyframeMeta] = frames_meta.keyframe_by_id()
    rig_order: dict[int, int] = {rig.synced_sample_id: position for position, rig in enumerate(frames_meta.rig_frames())}

    same_camera_family: int = 0
    top1_within_two_rigs: int = 0
    top3_within_two_rigs: int = 0
    for keyframe_id in index.image_ids:
        candidates: list[Candidate] = index.query(keyframe_id, top_k=3)
        assert candidates, f"keyframe {keyframe_id} retrieved nothing"
        query_camera: int = keyframe_by_id[keyframe_id].camera_params_id
        query_rig: int = rig_order[keyframe_by_id[keyframe_id].synced_sample_id]
        best_camera: int = keyframe_by_id[candidates[0].image_id].camera_params_id
        same_camera_family += best_camera in (query_camera, stereo_partner[query_camera])
        offsets: list[int] = [abs(rig_order[keyframe_by_id[candidate.image_id].synced_sample_id] - query_rig) for candidate in candidates]
        top1_within_two_rigs += offsets[0] <= 2
        top3_within_two_rigs += min(offsets) <= 2
    return GalileoRecall(
        n_keyframes=len(index.image_ids),
        same_or_stereo_camera=same_camera_family,
        top1_within_two_rigs=top1_within_two_rigs,
        top3_within_two_rigs=top3_within_two_rigs,
    )


# --------------------------------------------------------------------------------------
# the pycolmap read path
# --------------------------------------------------------------------------------------


def test_descriptors_round_trip_through_a_colmap_database(tmp_path: Path, synthetic_corpus: SyntheticCorpus) -> None:
    """`read_descriptors_from_database` recovers float32 ALIKED blocks bit-exactly.

    COLMAP 4.2 stores descriptors as an opaque uint8 blob, so a 128-d float32 row is a
    512-byte uint8 row on disk and only `FeatureDescriptorsFloat.to_bytes()` /
    `FeatureDescriptors.to_float()` reinterpret it correctly
    (`docs/spec/pycolmap-capabilities.md` §4).
    """
    database_path: Path = tmp_path / "database.db"
    database: pycolmap.Database = pycolmap.Database.open(str(database_path))
    camera: pycolmap.Camera = pycolmap.Camera(camera_id=1, model="PINHOLE", width=640, height=480, params=[500.0, 500.0, 320.0, 240.0])
    database.write_camera(camera, use_camera_id=True)
    wanted: list[int] = [0, 1, 2]
    for image_id in wanted:
        database.write_image(pycolmap.Image(image_id=image_id + 1, name=f"{image_id}.png", camera_id=1), use_image_id=True)
        database.write_descriptors(
            image_id + 1,
            pycolmap.FeatureDescriptorsFloat(data=synthetic_corpus.descriptors[image_id], type=pycolmap.FeatureExtractorType.ALIKED_N32).to_bytes(),
        )
    database.close()

    recovered: dict[int, Descriptors] = read_descriptors_from_database(database_path)
    assert sorted(recovered) == [1, 2, 3]
    for image_id in wanted:
        np.testing.assert_array_equal(recovered[image_id + 1], synthetic_corpus.descriptors[image_id])

    index: RetrievalIndex = build_retrieval_index(recovered, RetrievalConfig(brute_force=BruteForceConfig(max_descriptors_per_image=64)))
    assert index.image_ids == (1, 2, 3)


def test_read_descriptors_rejects_a_missing_database(tmp_path: Path) -> None:
    """A missing database is an error, not an empty result."""
    with pytest.raises(FileNotFoundError):
        read_descriptors_from_database(tmp_path / "absent.db")
