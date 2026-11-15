"""Image retrieval for loop closure — the Python replacement for cuSFM's BoW stage.

`generate_bow_vocabulary_main` + `generate_bow_index_main` exist only to answer "which
other keyframes look like this keyframe?" so that `generate_association_main` and
`pose_graph_main` can propose loop pairs without all-pairs matching
(`docs/spec/bow.md` §1, §9.1). Nothing downstream reads the word ids, the tree, the IDF
values or the file formats, so this module drops all of them and keeps only the ranked
candidate list.

Two backends, chosen by sequence length
---------------------------------------

`RetrievalConfig.backend` is `"auto"`: **brute force** at or below
`brute_force_max_images` (500) images, **vocab** above it. Both produce the same
`RetrievalIndex` — a symmetric `[0, 1]` score matrix over the indexed images — so
`query`, `score` and every consumer are backend-agnostic. What differs is the score's
definition, and therefore its calibration; see `GOOD_SCORE_THRESHOLD`.

**Brute force** (`docs/spec/bow.md` §9.4 Option A) scores a pair by the
**mutual-nearest-neighbour vote ratio**:

    score(a, b) = |{ i : nn_b(i) = j and nn_a(j) = i and sim(i, j) >= floor }| / n_per_image

Mutual NN was chosen over max-pooled similarity because it is what the downstream verifier
does anyway (LightGlue + essential matrix on mutual matches), so the score predicts the
inlier count rather than merely "these two images contain a similar patch". It lands in
`[0, 1]`, a frame retrieves itself with 1.0, and the similarity floor keeps unrelated pairs
near 0.

**Vocab** (`docs/spec/bow.md` §9.4 Option C, minus the parts §9.3 says to drop) trains a
hierarchical k-means vocabulary, encodes each image as an L1-normalised TF-IDF vector and
scores a pair with the blob's own DBoW2 L1 score (§7.4, §9.5). It lives in
`colsfm.vocab_tree`; this module owns only the backend choice and the calibration.

Why brute force does not scale, measured
----------------------------------------

The brute force is `(n_images * n_descriptors)^2 * 128` inner products, so
`BruteForceConfig.max_total_descriptors` caps the per-image count at
`max_total_descriptors // n_images`. That is the knee at Galileo's 226 images (256
descriptors each, 9.4 s) and it **breaks down** on RoboCap: 4528 images leave 22
descriptors per image, the largest off-diagonal score is 0.227, and after the loop stage's
temporal gate nothing clears `GOOD_SCORE_THRESHOLD` at all — zero loop candidates on a
124 m walk with 90 genuine revisits. Raising the cap is not a fix: giving those 4528 images
Galileo's 256 descriptors each costs 3.4e14 FLOP, about 40 minutes at this machine's
measured 147 GFLOP/s float32 GEMM, and using every descriptor — which is what the vocab
path does — would be 2.2e16 FLOP, about 41 hours. The quadratic term is why the vocab
backend exists.

Measured on Galileo (226 images x 2048 descriptors), top-1 candidate within 2 rig frames /
top-1 in the same or the stereo-paired camera / a top-3 candidate within 2 rig frames:

| descriptors per image | build | within 2 rigs | same-or-stereo camera | top-3 within 2 rigs |
|---|---|---|---|---|
| 128 | 2.8 s | 164/226 | 226/226 | 216/226 |
| 256 (default) | 9.9 s | 194/226 | 226/226 | 222/226 |
| 512 | 52.6 s | 207/226 | 226/226 | 221/226 |

Note that `docs/spec/bow.md` §9.4's "a couple of seconds on CPU" for Option A assumes a
`faiss.IndexFlatIP`. faiss is **not** in the `colsfm` environment and neither is torch, so
that figure does not transfer.

Is the brute-force backend worth keeping? Measured on Galileo, both backends
--------------------------------------------------------------------------

The reuse review proposed dropping the brute force and running one backend everywhere, so
both were measured on the same 226 Galileo keyframes and the same 2048 shipped ALIKED
descriptors each (`data/cusfm_runs/galileo/cusfm/keyframes`), on this machine:

| backend | build | descriptors per image | top-1 within 2 rigs | top-3 within 2 rigs | same-or-stereo camera | median best score |
|---|---|---|---|---|---|---|
| brute force (what `auto` picks here) | 11.0 s | 256 of 2048 | 194/226 | 222/226 | 226/226 | 0.148 |
| vocab | 7.4 s | 2048, all of them | **222/226** | **226/226** | 226/226 | 0.373 |

All 226 queries cost under 0.01 s on either index, because both score every pair during the
build.

The vocabulary is both faster to build and more accurate here, so the *quality* argument for
keeping two backends is gone: `brute_force_max_images` no longer buys recall, only a second
score definition to calibrate. What it still buys is a path whose behaviour on a short
sequence is measured rather than extrapolated, and the vocabulary's own accuracy depends on
a k-means fit that a 226-image corpus barely trains. The decision is left open deliberately:
the numbers above are what it should be made on, and dropping `brute_force` would also drop
`BruteForceConfig`, `RetrievalConfig.backend` and `brute_force_max_images` with it.

The vocabulary tree, and why it is not the blob's
-------------------------------------------------

`docs/spec/bow.md` §6 builds `k = 9`, `depth = 7` — 5.4 M reserved nodes for 65 533 to
436 358 actual words, i.e. **about one descriptor per leaf**. §6.8 measures the occupancy
at 0.014 and 0.091 against the blob's own 0.7 maturity bar, and §9.4 concludes the tree is
"degenerate ... slower and no better" than a real vocabulary. So the shape here is chosen
by measurement instead. On the 4528 RoboCap images (9 253 392 ALIKED descriptors,
everything the extractor produced, no subsampling):

| tree | words | build | doc freq (median/max) | median IDF | top-1 within 2 rig frames |
|---|---|---|---|---|---|
| k=10, depth 4 | 10 000 | 67.3 s | 650 / 2094 | 1.94 | 4518/4528 |
| k=10, depth 5 | 100 000 | 42.2 s | 36 / 1996 | 4.83 | 4517/4528 |

Depth 5 is both faster and healthier: the all-pairs scoring cost is `sum_w df_w^2`, so ten
times more words makes the words ten times rarer and the scoring nearly ten times cheaper,
while the IDF stops collapsing towards zero. The 42.2 s breaks down as 6.0 s training,
12.4 s assignment, 0.6 s TF-IDF and 23.2 s all-pairs scoring; the 4528 `query` calls that
follow cost 1.4 s together, because the scoring already happened. For comparison the blob
spends 162.4 s on the vocabulary, 81.0 s on the index and 75.7 s on the association stage
(`docs/open-pipeline-plan.md`).

**Descriptors are not subsampled on the vocab path**, and that is a measured decision, not
an oversight. Keeping 512 per image cuts the build to 20.3 s but moves the median gated
best score from 0.153 to 0.069 — the L1 score counts *shared* words, so thinning the
descriptors thins the overlap and silently rescales the score out from under
`GOOD_SCORE_THRESHOLD`. Using every descriptor is also what the blob does (§6.3: "no
sampling, no capping"), which is what keeps its 0.1 threshold meaningful here.

What is deliberately not copied from `docs/spec/bow.md`
-------------------------------------------------------

* The integer-division IDF of §6.7 (`logf(N // df)`), reproduced only for bit-comparison
  (§9.3). The plain `log(n_images / df)` is used instead. Its one visible consequence —
  every word with `df > N/2` collapsing to weight 0 — happens naturally here, since
  `log(N/df)` goes to 0 as `df` goes to `N`, and the zero rows are dropped by
  `eliminate_zeros`.
* `min_shared_word_number = 50` (§9.1, and `docs/spec/generate_association_main.md` §7.1
  item 4, which lists it as a "must keep"). The score already subsumes it: since every
  term of `s = sum_w min(q_w, d_w)` is bounded by `q_w`, a pair that shares few words can
  only score high if those few words carry most of the query's L1 mass — which is the
  strong evidence the shared-word count was a proxy for. Keeping the count as well would
  add a second threshold to tune for no measured gain.
* The `VocabularyNode` slab format, BIRCH incremental building, `apply_merge_before_split`
  and the 2 GB shard split (§9.3).
* Detector-response-weighted subsampling. `_subsample` takes an even stride, because the
  blob's `Keyframe` proto carries no per-keypoint score (`docs/spec/feature_extractor_main.md`
  §4) and no caller ever had one to offer.

Extension point for Option B (global descriptors)
-------------------------------------------------

`docs/spec/bow.md` §9.4 Option B replaces the descriptor stack with one global vector per
image (NetVLAD / DINOv2 / SALAD). It plugs in here without touching the callers: build an
index whose `similarity` matrix is `global_descriptors @ global_descriptors.T` on
L2-normalised rows, then `query` works unchanged. It is deliberately not implemented — it
needs a model download and a forward pass per keyframe, and the vocab backend already
covers all 90 of the blob's RoboCap loop pairs.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, Literal, TypeAlias

import numpy as np
from jaxtyping import Bool, Float32, Int32
from numpy import ndarray

from colsfm.vocab_tree import DEFAULT_VOCAB_CONFIG, Descriptors, SimilarityMatrix, VocabConfig, vocab_similarity

RetrievalBackend: TypeAlias = Literal["auto", "brute_force", "vocab"]
"""How `build_retrieval_index` builds the score matrix; `auto` picks by image count."""

ResolvedBackend: TypeAlias = Literal["brute_force", "vocab"]
"""A `RetrievalBackend` after `auto` has been resolved against the corpus."""

ALIKED_DESCRIPTOR_DIM: Final[int] = 128
"""Descriptor width `feature_extractor_main` writes (`docs/spec/feature_extractor_main.md` §4)."""

GOOD_SCORE_THRESHOLD: Final[float] = 0.1
"""Retrieval score a loop candidate must clear, on either backend.

The two backends produce different quantities — a mutual-nearest-neighbour vote ratio and
the blob's DBoW2 L1 score — so the value was calibrated twice, independently, and both
calibrations landed on 0.1:

* **brute_force** — 0.1 is the blob's own number reused; on Galileo the median keyframe's
  best score is 0.15 and the median pair is below 0.02, so the gate separates near views
  from far ones.
* **vocab** — the blob's own DBoW2 L1 score, so the blob's own 0.1
  (`docs/spec/generate_association_main.md` §7.1 item 4) transfers by construction, and it
  was re-measured rather than assumed. On RoboCap's 4528 images, bucketing every top-20 hit
  that survives the loop stage's 12.1 s temporal gate by the distance between the two rig
  frames:

  | score | hits | median distance | within 1 m | within 3 m |
  |---|---|---|---|---|
  | [0.00, 0.05) | 742 | 2.84 m | 12 % | 55 % |
  | [0.05, 0.10) | 31 251 | 0.95 m | 52 % | 84 % |
  | [0.10, 0.15) | 35 555 | 0.54 m | 88 % | 99 % |
  | [0.15, 0.20) | 16 622 | 0.43 m | 95 % | 100 % |
  | [0.20, 1.00) | 6 390 | 0.37 m | 99 % | 100 % |

  0.1 is where precision jumps from 52 % to 88 %, so it is the knee, not an inherited
  constant. Raise it to 0.15 to cut the pairs handed to the matcher roughly in half at
  95 % precision; the recall headroom is there, because the gated top-20 covers all 90 of
  the blob's RoboCap loop pairs at either setting.

Because both numbers are the same, `RetrievalIndex` carries no per-backend threshold and
`LoopClosureConfig.good_score_threshold` resolves to this constant; `RetrievalIndex.backend`
stays, so a run can still report which score definition produced its numbers."""


@dataclass(frozen=True, slots=True)
class Candidate:
    """One retrieval hit for a query image."""

    image_id: int
    """Database image id of the retrieved image; the cuSFM keyframe id."""
    score: float
    """Similarity in `[0, 1]`; 1.0 means identical. Its definition depends on the backend
    that built the index — a mutual-nearest-neighbour vote ratio for `brute_force`, the
    DBoW2 L1 score for `vocab`."""


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """One `RetrievalIndex.search` answer: the hits, and what the temporal gate cost it."""

    candidates: list[Candidate]
    """Hits that passed the temporal gate, best first."""
    rejected_by_time: int
    """Hits the temporal gate dropped while filling `candidates`.

    Counted inside the same ranking walk that produced the hits, so a caller's funnel
    reconciles: `len(candidates) + rejected_by_time` is everything the index looked at."""


@dataclass(frozen=True, slots=True)
class BruteForceConfig:
    """Settings of the mutual-nearest-neighbour backend."""

    max_descriptors_per_image: int = 256
    """Upper bound on descriptors kept per image; the measured knee of the recall/runtime
    curve on Galileo. The vocab backend uses every descriptor, because thinning them
    rescales the L1 score (module docstring)."""
    max_total_descriptors: int = 100_000
    """Global cap, so the quadratic pass stays inside the runtime budget. The per-image
    count becomes `min(max_descriptors_per_image, max_total_descriptors // n_images)`."""
    min_descriptor_similarity: float = 0.9
    """Cosine floor a mutual pair must clear to vote. Measured on Galileo: dropping the
    floor moves top-1-within-2-rig-frames from 194/226 down to 146/226, because unrelated
    images always produce a few mutual pairs by chance."""
    similarity_chunk_elements: int = 40_000_000
    """Roughly how many float32 similarities to hold at once (~160 MB). Only affects peak
    memory and cache behaviour, never the result."""

    def __post_init__(self) -> None:
        """Reject settings that would produce an empty or meaningless index.

        Raises:
            ValueError: When any bound is below its smallest useful value, or
                `min_descriptor_similarity` is not a cosine.
        """
        if self.max_descriptors_per_image < 1:
            raise ValueError(f"max_descriptors_per_image must be at least 1, got {self.max_descriptors_per_image}")
        if self.max_total_descriptors < 1:
            raise ValueError(f"max_total_descriptors must be at least 1, got {self.max_total_descriptors}")
        if not -1.0 <= self.min_descriptor_similarity <= 1.0:
            raise ValueError(f"min_descriptor_similarity must be a cosine in [-1, 1], got {self.min_descriptor_similarity}")
        if self.similarity_chunk_elements < 1:
            raise ValueError(f"similarity_chunk_elements must be at least 1, got {self.similarity_chunk_elements}")


DEFAULT_BRUTE_FORCE_CONFIG: BruteForceConfig = BruteForceConfig()
"""Shared immutable default, so `RetrievalConfig` holds no constructor call."""


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Which retrieval backend to build, and the settings of each.

    The two backends share nothing but the seed, so their knobs live in their own
    dataclasses: a flat config invited every reader to wonder which of ten fields the
    backend they chose actually reads.
    """

    backend: RetrievalBackend = "auto"
    """Which index to build. `auto` takes `brute_force` at or below
    `brute_force_max_images` and `vocab` above it, because brute force is quadratic in the
    image count and its descriptor budget collapses on a long sequence; see the module
    docstring."""
    brute_force_max_images: int = 500
    """Image count at which `auto` switches to the vocab backend. Galileo's 226 images stay
    on brute force, whose recall is measured there; RoboCap's 4528 do not, because at that
    length `BruteForceConfig.max_total_descriptors` leaves 22 descriptors per image and
    every score collapses."""
    seed: int = 0
    """Seed for the vocabulary's training subsample and k-means. The vocab backend is
    deterministic given this seed; the blob's is not, because it seeds libc `rand()` with
    nothing (`docs/spec/bow.md` §6.5). The brute force is deterministic regardless."""
    brute_force: BruteForceConfig = DEFAULT_BRUTE_FORCE_CONFIG
    """Settings the `brute_force` backend reads; ignored by `vocab`."""
    vocab: VocabConfig = DEFAULT_VOCAB_CONFIG
    """Settings the `vocab` backend reads; ignored by `brute_force`."""

    def __post_init__(self) -> None:
        """Reject a backend switch that can never fire.

        Raises:
            ValueError: When `brute_force_max_images` is below one image.
        """
        if self.brute_force_max_images < 1:
            raise ValueError(f"brute_force_max_images must be at least 1, got {self.brute_force_max_images}")

    def resolve_backend(self, n_images: int) -> ResolvedBackend:
        """Turn `auto` into a concrete backend for a corpus of this size.

        Args:
            n_images: How many images the index will hold.

        Returns:
            `"brute_force"` or `"vocab"`.
        """
        if self.backend != "auto":
            return self.backend
        return "brute_force" if n_images <= self.brute_force_max_images else "vocab"


@dataclass(frozen=True, slots=True)
class RetrievalIndex:
    """Precomputed image-to-image similarity over a fixed set of images.

    The index is built over the same frames that query it, so a frame always retrieves
    itself with score 1.0 (`docs/spec/bow.md` §9.2); `query` skips it.
    """

    image_ids: tuple[int, ...]
    """Indexed image ids, ascending. Row/column `i` of `similarity` is `image_ids[i]`."""
    similarity: SimilarityMatrix
    """Symmetric scores in `[0, 1]`; see `Candidate.score` for what they mean per backend."""
    descriptors_per_image: int
    """Descriptors each image contributed: the subsample size for `brute_force` (and the
    score's denominator), the mean count for `vocab`, which subsamples nothing."""
    config: RetrievalConfig
    """Settings the index was built with; `backend` is derived from it."""
    build_seconds: float
    """Wall time the build took, so pipelines can report a per-stage runtime."""
    n_words: int = 0
    """Vocabulary size for `vocab`, 0 for `brute_force`."""
    _position_by_image_id: dict[int, int] = field(init=False, repr=False, compare=False)
    """Row of each image id, built once: `query` and `score` would otherwise re-scan
    `image_ids` on every lookup (116 us per call, ~1 s per RoboCap run)."""

    def __post_init__(self) -> None:
        """Build the id-to-row lookup the accessors use."""
        object.__setattr__(self, "_position_by_image_id", {image_id: position for position, image_id in enumerate(self.image_ids)})

    @property
    def backend(self) -> ResolvedBackend:
        """Which backend produced `similarity`.

        Derived rather than stored: the corpus size and the config are what chose it, and a
        stored copy could only ever disagree with them.

        Returns:
            `"brute_force"` or `"vocab"`.
        """
        return self.config.resolve_backend(len(self.image_ids))

    def query(
        self,
        image_id: int,
        top_k: int = 20,
        min_time_gap_us: int | None = None,
        timestamps: Mapping[int, int] | None = None,
    ) -> list[Candidate]:
        """Rank the other indexed images by similarity to `image_id`.

        Args:
            image_id: Query image id; must be in `image_ids`.
            top_k: Maximum number of candidates to return.
            min_time_gap_us: Minimum `|dt|` in microseconds a candidate must be from the
                query, or None to apply no temporal gate.
            timestamps: Capture time in microseconds per image id. Required when
                `min_time_gap_us` is given.

        Returns:
            Candidates with a non-zero score, best first, at most `top_k` of them.

        Raises:
            KeyError: When `image_id` is not indexed, or a needed timestamp is missing.
            ValueError: When `min_time_gap_us` is given without `timestamps`, or
                `top_k` is not positive.
        """
        return self.search(image_id, top_k, min_time_gap_us, timestamps).candidates

    def search(
        self,
        image_id: int,
        top_k: int = 20,
        min_time_gap_us: int | None = None,
        timestamps: Mapping[int, int] | None = None,
    ) -> RetrievalQuery:
        """Rank the other indexed images, and report what the temporal gate cost.

        `query_result_number = 20` is the blob's own top-N (`docs/spec/bow.md` §9.1). The
        query image is always excluded, and so are images closer in time than
        `min_time_gap_us` — the pose graph already has sequential edges, so a loop
        candidate must be separated in time to add information
        (`docs/spec/bow.md` §9.2, `docs/spec/generate_association_main.md` §6.4).

        The gate is applied *inside* the ranking walk rather than to a finished top-k, so
        all `top_k` slots stay useful on a sequence whose nearest neighbours in score are
        its neighbours in time. That is also why the count of gated hits comes back from
        here: deriving it from a second, ungated query would count a different set.

        Args:
            image_id: Query image id; must be in `image_ids`.
            top_k: Maximum number of candidates to return.
            min_time_gap_us: Minimum `|dt|` in microseconds a candidate must be from the
                query, or None to apply no temporal gate.
            timestamps: Capture time in microseconds per image id. Required when
                `min_time_gap_us` is given.

        Returns:
            The candidates and the number of hits the temporal gate dropped.

        Raises:
            KeyError: When `image_id` is not indexed, or a needed timestamp is missing.
            ValueError: When `min_time_gap_us` is given without `timestamps`, or
                `top_k` is not positive.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")
        if min_time_gap_us is not None and timestamps is None:
            raise ValueError("min_time_gap_us needs timestamps to apply the temporal gate")
        position: int = self._position_of(image_id)

        scores: Float32[ndarray, "n_images"] = self.similarity[position]
        order: Int32[ndarray, "n_images"] = np.argsort(-scores, kind="stable").astype(np.int32)
        gated: bool = min_time_gap_us is not None and timestamps is not None
        capture_us: Mapping[int, int] = timestamps if timestamps is not None else {}
        gap_us: int = min_time_gap_us if min_time_gap_us is not None else 0
        query_timestamp_us: int = capture_us[image_id] if gated else 0

        candidates: list[Candidate] = []
        rejected_by_time: int = 0
        for rank in order:
            if len(candidates) == top_k:
                break
            score: float = float(scores[rank])
            if score <= 0.0:
                break
            other_id: int = self.image_ids[int(rank)]
            if other_id == image_id:
                continue
            if gated and abs(capture_us[other_id] - query_timestamp_us) < gap_us:
                rejected_by_time += 1
                continue
            candidates.append(Candidate(image_id=other_id, score=score))
        return RetrievalQuery(candidates=candidates, rejected_by_time=rejected_by_time)

    def score(self, image_id_a: int, image_id_b: int) -> float:
        """Look up the stored score of one pair.

        Args:
            image_id_a: First image id.
            image_id_b: Second image id.

        Returns:
            The similarity in `[0, 1]`.

        Raises:
            KeyError: When either id is not indexed.
        """
        return float(self.similarity[self._position_of(image_id_a), self._position_of(image_id_b)])

    def _position_of(self, image_id: int) -> int:
        """Row index of an image id.

        Args:
            image_id: The image id to locate.

        Returns:
            Its position in `image_ids`.

        Raises:
            KeyError: When the id is not indexed.
        """
        position: int | None = self._position_by_image_id.get(image_id)
        if position is None:
            raise KeyError(f"image id {image_id} is not in the retrieval index")
        return position


# ======================================================================================
# brute force (docs/spec/bow.md §9.4 Option A)
# ======================================================================================


def _subsample(descriptors: Descriptors, keep: int) -> Descriptors:
    """Reduce one image's descriptors to `keep` rows with an even stride.

    A stride is unbiased because the blob's `Keyframe` proto carries no per-keypoint score
    (`docs/spec/feature_extractor_main.md` §4), so there is nothing to rank by, and taking a
    prefix would bias towards whatever order the detector happened to emit.

    Args:
        descriptors: Float32 descriptors with shape `[n_descriptors, descriptor_dim]`.
        keep: How many rows to keep; must not exceed `n_descriptors`.

    Returns:
        Float32 descriptors with shape `[keep, descriptor_dim]`.
    """
    if len(descriptors) == keep:
        return descriptors
    stride: int = len(descriptors) // keep
    return descriptors[:: max(1, stride)][:keep]


def _stack_descriptors(
    descriptors_by_image: Mapping[int, Descriptors],
    image_ids: tuple[int, ...],
    keep: int,
) -> Float32[ndarray, "n_images keep descriptor_dim"]:
    """Subsample every image to `keep` descriptors and L2-normalise them.

    The blob's ALIKED descriptors come out with a norm of 1.0002 rather than exactly 1,
    so they are renormalised here: the score is a cosine and must not inherit that drift.

    Args:
        descriptors_by_image: Descriptors per image id.
        image_ids: Image ids in the index's order.
        keep: Descriptors to keep per image.

    Returns:
        Float32 array with shape `[n_images, keep, descriptor_dim]`, rows unit-norm.
    """
    rows: list[Descriptors] = [_subsample(np.asarray(descriptors_by_image[image_id], dtype=np.float32), keep) for image_id in image_ids]
    stacked: Float32[ndarray, "n_images keep descriptor_dim"] = np.stack(rows).astype(np.float32)
    norms: Float32[ndarray, "n_images keep 1"] = np.linalg.norm(stacked, axis=2, keepdims=True)
    return stacked / np.maximum(norms, np.float32(1e-12))


def _brute_force_similarity(
    descriptors_by_image: Mapping[int, Descriptors],
    image_ids: tuple[int, ...],
    keep: int,
    config: BruteForceConfig,
) -> SimilarityMatrix:
    """Mutual-nearest-neighbour vote ratios over the subsampled descriptor stack.

    One chunked `float32` GEMM gives, for every ordered pair of images, the forward and
    backward nearest neighbour of every descriptor; a pair votes when the two agree and the
    cosine clears `BruteForceConfig.min_descriptor_similarity`.

    Args:
        descriptors_by_image: Descriptors per image id.
        image_ids: Image ids in the index's order.
        keep: Descriptors kept per image.
        config: The brute-force settings.

    Returns:
        The symmetric score matrix, 1.0 on the diagonal.
    """
    n_images: int = len(image_ids)
    stacked: Float32[ndarray, "n_images keep descriptor_dim"] = _stack_descriptors(descriptors_by_image, image_ids, keep)
    flat: Float32[ndarray, "n_total descriptor_dim"] = np.ascontiguousarray(stacked.reshape(n_images * keep, -1))

    votes: SimilarityMatrix = np.zeros((n_images, n_images), dtype=np.float32)
    own_index: Int32[ndarray, "keep"] = np.arange(keep, dtype=np.int32)
    chunk: int = max(1, config.similarity_chunk_elements // (keep * n_images * keep))
    floor: np.float32 = np.float32(config.min_descriptor_similarity)
    for first in range(0, n_images, chunk):
        last: int = min(first + chunk, n_images)
        block: Float32[ndarray, "block descriptor_dim"] = flat[first * keep : last * keep]
        similarity: Float32[ndarray, "block keep n_images keep"] = (block @ flat.T).reshape(last - first, keep, n_images, keep)
        forward: Int32[ndarray, "block keep n_images"] = np.argmax(similarity, axis=3).astype(np.int32)
        backward: Int32[ndarray, "block n_images keep"] = np.argmax(similarity, axis=1).astype(np.int32)
        forward_by_image: Int32[ndarray, "block n_images keep"] = np.ascontiguousarray(forward.transpose(0, 2, 1))
        round_trip: Int32[ndarray, "block n_images keep"] = np.take_along_axis(backward, forward_by_image, axis=2)
        mutual: Bool[ndarray, "block n_images keep"] = round_trip == own_index[None, None, :]
        best_similarity: Float32[ndarray, "block n_images keep"] = np.ascontiguousarray(
            np.take_along_axis(similarity, forward[:, :, :, None], axis=3)[:, :, :, 0].transpose(0, 2, 1)
        )
        mutual &= best_similarity >= floor
        votes[first:last] = mutual.sum(axis=2, dtype=np.float32)

    scores: SimilarityMatrix = votes / np.float32(keep)
    np.fill_diagonal(scores, np.float32(1.0))
    return scores


# ======================================================================================
# the stage
# ======================================================================================


def build_retrieval_index(descriptors_by_image: Mapping[int, Descriptors], config: RetrievalConfig | None = None) -> RetrievalIndex:
    """Score every image pair, with the backend `config.backend` selects.

    Args:
        descriptors_by_image: Local descriptors per image id, `[n_descriptors, dim]` each.
            Images may carry different counts; on the brute-force path the smallest count
            caps the whole index, on the vocab path every descriptor is used.
        config: Backend choice and its settings; the defaults are the measured ones.

    Returns:
        The index, with its wall-clock build time recorded.

    Raises:
        ValueError: When there are fewer than two images, an image has no descriptors, or
            the descriptor widths disagree.
    """
    settings: RetrievalConfig = RetrievalConfig() if config is None else config
    image_ids: tuple[int, ...] = tuple(sorted(descriptors_by_image))
    n_images: int = len(image_ids)
    if n_images < 2:
        raise ValueError(f"a retrieval index needs at least two images, got {n_images}")
    counts: list[int] = [int(descriptors_by_image[image_id].shape[0]) for image_id in image_ids]
    if min(counts) < 1:
        raise ValueError("every image must carry at least one descriptor")
    # Checked here rather than per backend: the vocab path used to fail on a ragged corpus
    # with a bare `numpy.concatenate` error deep inside the assignment pass.
    descriptor_dim: int = int(descriptors_by_image[image_ids[0]].shape[1])
    for image_id in image_ids:
        width: int = int(descriptors_by_image[image_id].shape[1])
        if width != descriptor_dim:
            raise ValueError(f"image {image_id} has {width}-d descriptors, expected {descriptor_dim}")
    backend: ResolvedBackend = settings.resolve_backend(n_images)

    started: float = time.perf_counter()
    if backend == "vocab":
        similarity, n_words = vocab_similarity(descriptors_by_image, image_ids, counts, settings.vocab, settings.seed)
        descriptors_per_image: int = sum(counts) // n_images
    else:
        n_words = 0
        brute_force: BruteForceConfig = settings.brute_force
        descriptors_per_image = min(brute_force.max_descriptors_per_image, min(counts), max(1, brute_force.max_total_descriptors // n_images))
        similarity = _brute_force_similarity(descriptors_by_image, image_ids, descriptors_per_image, brute_force)
    build_seconds: float = time.perf_counter() - started

    print(
        f"colsfm.retrieval: {backend} index over {n_images} images, "
        f"{descriptors_per_image} descriptors/image, {n_words} words, {build_seconds:.1f}s"
    )
    return RetrievalIndex(
        image_ids=image_ids,
        similarity=similarity,
        descriptors_per_image=descriptors_per_image,
        config=settings,
        build_seconds=build_seconds,
        n_words=n_words,
    )
