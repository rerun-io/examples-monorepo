"""Hierarchical k-means vocabulary and the DBoW2 L1 all-pairs score.

This is the `vocab` backend of `colsfm.retrieval`, split out because it is a self-contained
library — train a tree on a descriptor sample, assign every descriptor to a word, encode
each image as an L1-normalised TF-IDF row and score every pair through the inverted index.
`colsfm.retrieval` owns the choice of backend, the `RetrievalIndex` and the calibration of
the score; nothing here reads a database, a camera or a rig. The `Descriptors` and
`SimilarityMatrix` aliases are declared here, at the bottom of the dependency, and
re-exported by `colsfm.retrieval` for its callers.

The measured shape of the tree, why it is not the blob's `k = 9, depth = 7`, and why the
descriptors are not subsampled on this path, are all recorded in `colsfm.retrieval`'s module
docstring, next to the recall tables the decisions were made from.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from jaxtyping import Bool, Float32, Int32, Int64
from numpy import ndarray
from scipy import sparse
from scipy.cluster.vq import kmeans2
from threadpoolctl import threadpool_limits

from colsfm.database import Descriptors

SimilarityMatrix: TypeAlias = Float32[ndarray, "n_images n_images"]
"""Symmetric image-to-image scores in `[0, 1]`, with 1.0 on the diagonal."""


@dataclass(frozen=True, slots=True)
class VocabConfig:
    """Settings of the vocabulary-tree backend."""

    branching: int = 10
    """Children per vocabulary-tree node. The blob uses 9 (`docs/spec/bow.md` §3); 10 is the
    same thing with round word counts."""
    depth: int = 5
    """Tree levels, so the vocabulary holds `branching ** depth` words — 100 000 here.
    Measured against depth 4 on RoboCap: same top-1 recall, 1.6x faster to build and a median
    IDF of 4.83 instead of 1.94. The blob's depth 7 is degenerate (`docs/spec/bow.md` §6.8)
    and is not copied. Lowered automatically when the training sample cannot fill the tree."""
    training_descriptors: int = 200_000
    """Descriptors drawn (seeded, evenly across images) to train the tree. 200 000 trains in
    6.0 s against 13.6 s for 500 000, with no measurable recall difference on RoboCap."""
    kmeans_iterations: int = 10
    """Lloyd iterations per node, matching `weighted_kmeans_max_number_of_iterations`
    (`docs/spec/bow.md` §6.5)."""
    assign_chunk_descriptors: int = 1_000_000
    """Descriptors held in memory per assignment pass (~500 MB at 128-d). Only affects peak
    memory, never the result."""

    def __post_init__(self) -> None:
        """Reject settings that cannot produce a usable vocabulary.

        Raises:
            ValueError: When any bound is below its smallest useful value.
        """
        if self.branching < 2:
            raise ValueError(f"vocab branching must be at least 2, got {self.branching}")
        if self.depth < 1:
            raise ValueError(f"vocab depth must be at least 1, got {self.depth}")
        if self.training_descriptors < self.branching:
            raise ValueError(f"vocab training_descriptors must be at least branching, got {self.training_descriptors}")
        if self.kmeans_iterations < 1:
            raise ValueError(f"vocab kmeans_iterations must be at least 1, got {self.kmeans_iterations}")
        if self.assign_chunk_descriptors < 1:
            raise ValueError(f"vocab assign_chunk_descriptors must be at least 1, got {self.assign_chunk_descriptors}")


DEFAULT_VOCAB_CONFIG: VocabConfig = VocabConfig()
"""Shared immutable default, so `RetrievalConfig` holds no constructor call."""


@dataclass(frozen=True, slots=True)
class VocabularyTree:
    """A trained hierarchical k-means vocabulary, one centroid array per level.

    Level `l` holds `branching ** (l + 1)` centroid rows; the children of node `n` are rows
    `n * branching` to `(n + 1) * branching`. A leaf's index in the last level is its word
    id, so word ids need no separate table — unlike the blob, which numbers words by their
    rank among childless nodes in flat-array order (`docs/spec/bow.md` §6.6).
    """

    branching: int
    """Children per node."""
    centroids: tuple[Float32[ndarray, "n_children descriptor_dim"], ...]
    """Centroid rows per level, `[branching ** (level + 1), descriptor_dim]` each."""
    biases: tuple[Float32[ndarray, "n_children"], ...]
    """`-0.5 * ||centroid||^2` per row, so the descent is one `argmax` of
    `descriptor @ centroids.T + bias`. Rows of nodes that were never populated hold
    `-inf`, which is what keeps an all-zero centroid from winning a descent."""

    @property
    def depth(self) -> int:
        """How many levels the tree has.

        Returns:
            The level count; the word ids run over `branching ** depth`.
        """
        return len(self.centroids)

    @property
    def n_words(self) -> int:
        """Size of the vocabulary.

        Returns:
            `branching ** depth`, including leaves no descriptor ever reaches.
        """
        return self.branching**self.depth


@dataclass(frozen=True, slots=True)
class NodePartition:
    """Rows of one vocabulary level, grouped by the node they sit in.

    A dataclass rather than a `NamedTuple` because beartype cannot decorate a
    `NamedTuple.__new__` whose fields carry jaxtyping annotations, and losing the runtime
    shape check is a worse trade than losing tuple unpacking.
    """

    order: Int64[ndarray, "n_rows"]
    """Row indices, sorted so that every node's rows are contiguous."""
    bounds: Int64[ndarray, "n_nodes_plus_one"]
    """Slice bounds into `order`: node `k` owns `order[bounds[k] : bounds[k + 1]]`."""


def partition_by_node(node_of_row: Int64[ndarray, "n_rows"], n_nodes: int) -> NodePartition:
    """Group rows by the tree node they currently sit in, with one sort.

    Both the training pass and the assignment pass walk a level node by node, and both need
    the rows of one node contiguously. Sorting once and slicing beats gathering per node: a
    level of 100 000 nodes over 9 M descriptors is one `argsort`, not 100 000 boolean masks.

    Args:
        node_of_row: Node index per row, in `[0, n_nodes)`.
        n_nodes: How many nodes the level has.

    Returns:
        The grouping order and its slice bounds.
    """
    order: Int64[ndarray, "n_rows"] = np.argsort(node_of_row, kind="stable")
    bounds: Int64[ndarray, "n_nodes_plus_one"] = np.searchsorted(node_of_row[order], np.arange(n_nodes + 1))
    return NodePartition(order=order, bounds=bounds)


def effective_depth(config: VocabConfig, n_training_descriptors: int) -> int:
    """Deepest tree the training sample can fill, capped at `config.depth`.

    A level whose nodes hold fewer than one training descriptor each cannot be clustered
    and produces empty leaves, which cost memory and buy nothing. The bound is therefore
    `branching ** depth <= n_training_descriptors`.

    Args:
        config: The vocabulary settings.
        n_training_descriptors: Descriptors the tree will be trained on.

    Returns:
        A depth of at least 1.
    """
    depth: int = 1
    while depth < config.depth and config.branching ** (depth + 1) <= n_training_descriptors:
        depth += 1
    return depth


def training_sample(
    descriptors_by_image: Mapping[int, Descriptors],
    image_ids: Sequence[int],
    target: int,
    seed: int,
) -> Float32[ndarray, "n_sample descriptor_dim"]:
    """Draw a seeded subsample of descriptors, spread evenly over the images.

    Drawing a per-image quota rather than a global random subset keeps the vocabulary from
    over-fitting whichever part of the sequence happens to be densest, and makes the sample
    independent of how many descriptors each image carries.

    Args:
        descriptors_by_image: Descriptors per image id.
        image_ids: Image ids in the index's order.
        target: How many descriptors to draw in total.
        seed: Seed for the draw.

    Returns:
        Float32 descriptors with shape `[n_sample, descriptor_dim]`. Every image
        contributes the same quota, so `n_sample` is `target` rounded up to a whole
        number of images; truncating back to `target` would silently drop the tail of
        the sequence from the training set.
    """
    generator: np.random.Generator = np.random.default_rng(seed)
    quota: int = max(1, -(-target // len(image_ids)))
    blocks: list[Descriptors] = []
    for image_id in image_ids:
        descriptors: Descriptors = descriptors_by_image[image_id]
        if len(descriptors) <= quota:
            blocks.append(np.asarray(descriptors, dtype=np.float32))
            continue
        chosen: Int64[ndarray, "quota"] = np.sort(generator.choice(len(descriptors), quota, replace=False))
        blocks.append(np.asarray(descriptors[chosen], dtype=np.float32))
    return np.ascontiguousarray(np.concatenate(blocks, axis=0))


def train_vocabulary(sample: Float32[ndarray, "n_sample descriptor_dim"], depth: int, config: VocabConfig, seed: int) -> VocabularyTree:
    """Train a hierarchical k-means vocabulary on a descriptor sample.

    One `scipy.cluster.vq.kmeans2` call per node, k-means++ seeded from a per-node draw of
    a seeded generator, so the whole tree is reproducible. The recursion is breadth-first
    over levels rather than depth-first over nodes, which lets the level's descriptors be
    partitioned with one sort instead of one gather per node.

    Nodes holding fewer descriptors than `branching` become their own centroids and leave
    the remaining child slots dead (`-inf` bias); the blob drops empty clusters the same
    way (`docs/spec/bow.md` §6.4).

    Args:
        sample: Float32 training descriptors with shape `[n_sample, descriptor_dim]`.
        depth: Levels to build.
        config: The vocabulary settings.
        seed: Seed for the k-means draws.

    Returns:
        The trained tree.
    """
    branching: int = config.branching
    dim: int = int(sample.shape[1])
    generator: np.random.Generator = np.random.default_rng(seed)
    centroids_per_level: list[Float32[ndarray, "n_children descriptor_dim"]] = []
    biases_per_level: list[Float32[ndarray, "n_children"]] = []

    node_of_sample: Int64[ndarray, "n_sample"] = np.zeros(len(sample), dtype=np.int64)
    n_nodes: int = 1
    for _level in range(depth):
        centroids: Float32[ndarray, "n_children descriptor_dim"] = np.zeros((n_nodes * branching, dim), dtype=np.float32)
        live: Bool[ndarray, "n_children"] = np.zeros(n_nodes * branching, dtype=bool)
        level: NodePartition = partition_by_node(node_of_sample, n_nodes)
        child_of_sample: Int64[ndarray, "n_sample"] = np.zeros(len(sample), dtype=np.int64)
        for node in range(n_nodes):
            members: Int64[ndarray, "n_members"] = level.order[level.bounds[node] : level.bounds[node + 1]]
            base: int = node * branching
            if len(members) == 0:
                continue
            block: Float32[ndarray, "n_members descriptor_dim"] = sample[members]
            if len(members) < branching:
                centroids[base : base + len(members)] = block
                live[base : base + len(members)] = True
                child_of_sample[members] = base + np.arange(len(members))
                continue
            # A node whose descriptors are all duplicates of one another makes k-means++
            # divide by a zero squared-distance sum and leaves clusters empty, which scipy
            # reports as a UserWarning and NumPy as a RuntimeWarning. Both are expected
            # here and harmless: the duplicate rows collapse onto one centroid, which is
            # what the blob's own "empty clusters are dropped" does
            # (`docs/spec/bow.md` §6.4). Silenced around this call only.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                warnings.simplefilter("ignore", category=RuntimeWarning)
                book, labels = kmeans2(
                    block,
                    branching,
                    iter=config.kmeans_iterations,
                    minit="++",
                    seed=int(generator.integers(1 << 31)),
                    check_finite=False,
                )
            centroids[base : base + branching] = np.asarray(book, dtype=np.float32)
            live[base : base + branching] = True
            child_of_sample[members] = base + np.asarray(labels, dtype=np.int64)
        biases: Float32[ndarray, "n_children"] = np.where(live, -0.5 * np.sum(centroids * centroids, axis=1), -np.inf).astype(np.float32)
        centroids_per_level.append(centroids)
        biases_per_level.append(biases)
        node_of_sample = child_of_sample
        n_nodes *= branching
    return VocabularyTree(branching=branching, centroids=tuple(centroids_per_level), biases=tuple(biases_per_level))


def assign_words(descriptors: Float32[ndarray, "n_descriptors descriptor_dim"], tree: VocabularyTree) -> Int32[ndarray, "n_descriptors"]:
    """Greedily descend every descriptor to its leaf word.

    Pure greedy descent, no beam and no backtracking, exactly as
    `VisualVocabulary::SearchFeature` does (`docs/spec/bow.md` §7.1). Nearest centroid by
    squared L2 is `argmax(descriptor @ centroids.T - 0.5 * ||centroid||^2)`, since
    `||descriptor||^2` is the same for every candidate — which is why the descriptors need
    no renormalising here.

    Args:
        descriptors: Float32 descriptors with shape `[n_descriptors, descriptor_dim]`.
        tree: The trained vocabulary.

    Returns:
        Int32 word ids with shape `[n_descriptors]`, in `[0, tree.n_words)`.
    """
    branching: int = tree.branching
    node: Int64[ndarray, "n_descriptors"] = np.zeros(len(descriptors), dtype=np.int64)
    for centroids, biases in zip(tree.centroids, tree.biases, strict=True):
        n_nodes: int = len(centroids) // branching
        level: NodePartition = partition_by_node(node, n_nodes)
        child: Int64[ndarray, "n_descriptors"] = np.zeros(len(descriptors), dtype=np.int64)
        for parent in range(n_nodes):
            members: Int64[ndarray, "n_members"] = level.order[level.bounds[parent] : level.bounds[parent + 1]]
            if len(members) == 0:
                continue
            base: int = parent * branching
            candidates: Float32[ndarray, "branching descriptor_dim"] = centroids[base : base + branching]
            affinity: Float32[ndarray, "n_members branching"] = descriptors[members] @ candidates.T + biases[base : base + branching]
            child[members] = base + np.argmax(affinity, axis=1)
        node = child
    return node.astype(np.int32)


def bow_matrix(
    word_of_descriptor: Int32[ndarray, "n_descriptors"],
    image_of_descriptor: Int32[ndarray, "n_descriptors"],
    n_images: int,
    n_words: int,
) -> sparse.csr_matrix:
    """Turn word assignments into L1-normalised TF-IDF rows.

    `docs/spec/bow.md` §7.2: the BoW vector is `v_w = tf_w * idf_w / sum_u (tf_u * idf_u)`
    with raw term counts, and words of zero weight are dropped. The IDF is
    `log(n_images / df_w)`, not the blob's integer-division variant (§6.7, §9.3): the
    quirk's only visible effect is that a word appearing in more than half the images
    collapses to weight 0, and `log(n_images / df)` already decays to 0 as `df` approaches
    `n_images`.

    Args:
        word_of_descriptor: Int32 word id per descriptor, shape `[n_descriptors]`.
        image_of_descriptor: Int32 image row per descriptor, shape `[n_descriptors]`.
        n_images: Rows of the result.
        n_words: Columns of the result.

    Returns:
        A `[n_images, n_words]` CSR matrix whose non-negative rows each sum to 1, except
        rows whose every word was dropped, which are empty.
    """
    counts: sparse.csr_matrix = sparse.coo_matrix(
        (np.ones(len(word_of_descriptor), dtype=np.float32), (image_of_descriptor, word_of_descriptor)),
        shape=(n_images, n_words),
    ).tocsr()
    counts.sum_duplicates()
    document_frequency: Int64[ndarray, "n_words"] = np.asarray((counts > 0).sum(axis=0)).ravel()
    inverse_document_frequency: Float32[ndarray, "n_words"] = np.zeros(n_words, dtype=np.float32)
    seen: Bool[ndarray, "n_words"] = document_frequency > 0
    inverse_document_frequency[seen] = np.log(n_images / document_frequency[seen]).astype(np.float32)

    weighted: sparse.csr_matrix = counts.multiply(inverse_document_frequency[None, :]).tocsr()
    weighted.eliminate_zeros()
    row_sums: Float32[ndarray, "n_images"] = np.asarray(weighted.sum(axis=1)).ravel()
    row_sums[row_sums == 0.0] = 1.0
    return (sparse.diags(1.0 / row_sums) @ weighted).tocsr().astype(np.float32)


def l1_score(query: Mapping[int, float], document: Mapping[int, float]) -> float:
    """The DBoW2 L1 score of two L1-normalised BoW vectors (`docs/spec/bow.md` §9.5).

    `s = -0.5 * sum_{w in q & d} (|q_w - d_w| - |q_w| - |d_w|)`, which lands in `[0, 1]`
    with 1.0 for identical vectors. This is the reference implementation the vectorised
    inverted index in `all_pairs_l1_score` is checked against.

    Args:
        query: Word id to weight for the query image; weights must be non-negative.
        document: Word id to weight for the candidate image.

    Returns:
        The score in `[0, 1]`.
    """
    shared: set[int] = set(query) & set(document)
    return -0.5 * sum(abs(query[word] - document[word]) - abs(query[word]) - abs(document[word]) for word in shared)


def all_pairs_l1_score(bow: sparse.csr_matrix) -> SimilarityMatrix:
    """Score every image pair through the inverted index.

    For non-negative weights `-0.5 * (|a - b| - a - b)` is exactly `min(a, b)`, so the
    DBoW2 L1 score of `docs/spec/bow.md` §7.4 is the histogram intersection
    `s(q, d) = sum_w min(q_w, d_w)`. Written that way it needs no per-query pass: walking
    the inverted index once and adding the outer minimum of each word's posting list into
    the score matrix scores every pair at the same time, which is why `query` is free
    afterwards.

    The cost is `sum_w df_w^2`, quadratic in the image count at a fixed vocabulary size —
    the reason `VocabConfig.depth` matters as much as it does. Measured on RoboCap's 4528
    images: 23.2 s at 100 000 words against 44.2 s at 10 000.

    Args:
        bow: `[n_images, n_words]` CSR matrix of L1-normalised non-negative rows.

    Returns:
        The symmetric score matrix, 1.0 on the diagonal.
    """
    n_images: int = bow.shape[0]
    scores: SimilarityMatrix = np.zeros((n_images, n_images), dtype=np.float32)
    columns: sparse.csc_matrix = bow.tocsc()
    for word in range(bow.shape[1]):
        start: int = int(columns.indptr[word])
        end: int = int(columns.indptr[word + 1])
        if end - start < 2:
            continue
        documents: Int32[ndarray, "df"] = columns.indices[start:end]
        weights: Float32[ndarray, "df"] = columns.data[start:end]
        scores[np.ix_(documents, documents)] += np.minimum(weights[:, None], weights[None, :])
    np.fill_diagonal(scores, np.float32(1.0))
    return np.clip(scores, 0.0, 1.0, out=scores)


def assignment_batches(counts: Sequence[int], max_descriptors: int) -> list[tuple[int, int]]:
    """Split the images into contiguous runs of roughly `max_descriptors` descriptors.

    Assignment is a pure per-descriptor function, so batching only bounds peak memory: a
    whole RoboCap corpus is 9.3 M descriptors, i.e. 4.7 GB at 128-d float32, and stacking
    it in one array on top of the caller's own copy is what the batches avoid.

    Args:
        counts: Descriptor count per image, in the index's order.
        max_descriptors: Soft upper bound on a batch; one image is never split.

    Returns:
        Half-open `[start, stop)` image ranges covering every image exactly once.
    """
    batches: list[tuple[int, int]] = []
    start: int = 0
    running: int = 0
    for position, count in enumerate(counts):
        running += count
        if running >= max_descriptors:
            batches.append((start, position + 1))
            start = position + 1
            running = 0
    if start < len(counts):
        batches.append((start, len(counts)))
    return batches


def vocab_similarity(
    descriptors_by_image: Mapping[int, Descriptors],
    image_ids: Sequence[int],
    counts: Sequence[int],
    config: VocabConfig,
    seed: int,
) -> tuple[SimilarityMatrix, int]:
    """Train a vocabulary, encode every image and score every pair.

    Args:
        descriptors_by_image: Descriptors per image id.
        image_ids: Image ids in the index's order.
        counts: Descriptor count per image, in the same order.
        config: The vocabulary settings.
        seed: Seed for the training subsample and the k-means.

    Returns:
        The symmetric score matrix and the vocabulary size.
    """
    n_images: int = len(image_ids)
    # One BLAS thread: training and assignment are thousands of `n x 10` products, where a threaded BLAS (MKL in this
    # environment) spends its time synchronising — 105 s against 21 s for 200 images at 8 threads against 1.
    with threadpool_limits(limits=1, user_api="blas"):
        sample: Float32[ndarray, "n_sample descriptor_dim"] = training_sample(descriptors_by_image, image_ids, config.training_descriptors, seed)
        tree: VocabularyTree = train_vocabulary(sample, effective_depth(config, len(sample)), config, seed)
        del sample

        word_batches: list[Int32[ndarray, "n_batch"]] = []
        image_batches: list[Int32[ndarray, "n_batch"]] = []
        for start, stop in assignment_batches(counts, config.assign_chunk_descriptors):
            block: Float32[ndarray, "n_batch descriptor_dim"] = np.ascontiguousarray(
                np.concatenate([np.asarray(descriptors_by_image[image_id], dtype=np.float32) for image_id in image_ids[start:stop]], axis=0)
            )
            word_batches.append(assign_words(block, tree))
            image_batches.append(np.repeat(np.arange(start, stop, dtype=np.int32), counts[start:stop]))

    bow: sparse.csr_matrix = bow_matrix(np.concatenate(word_batches), np.concatenate(image_batches), n_images, tree.n_words)
    return all_pairs_l1_score(bow), tree.n_words
