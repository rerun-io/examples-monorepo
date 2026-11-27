"""Two-view verification in memory, in bounded chunks, overlapped with whatever produces the matches.

COLMAP's batch `verify_matches` reads every pair's matches and keypoints back out of the database after matching is
done, verifies them, and writes the geometries: on MSD MGO05 it ran 7 s after the GPU had finished, and 10 s more in the
loop stage, while the GPU sat idle. Here each chunk of pairs is verified on a fixed pool of worker threads as soon as
the matcher hands it over (`pycolmap.estimate_two_view_geometry` releases the GIL: 7.7x on 8 threads, measured), and
the matcher keeps working on the next chunk meanwhile.

Memory stays bounded by construction: `ChunkedVerifier.verified` admits a chunk only while fewer than
`verify_chunks_in_flight` are verifying. When the window is full it finishes the oldest and hands it to the caller, who
writes and drops it, before it asks the producer for the next one; so a CPU that falls behind holds the matcher back
instead of letting chunks pile up. Memory holds `verify_chunks_in_flight` verifying chunks plus the one the producer is
building, whatever the length of the run, and the rule is the same on every machine.

Each chunk is written in one transaction (`colsfm.database.database_transaction`), so a failure leaves the earlier
chunks and none of the failing one.

The result is what COLMAP's verifier writes (`WriteVerifiedTwoViewGeometry` in `feature_matching_utils.cc` at the
pinned COLMAP): the same `EstimateTwoViewGeometry`, an `UNDEFINED` result is not stored, and a `DEGENERATE` result or
one with fewer than `min_num_inliers` inliers becomes an empty `DEGENERATE` row. RANSAC runs on a fixed seed
(`colsfm.matching.verification_options`), so a pair's result does not depend on which thread verifies it: given the
same options, `verify_matches` writes the same rows.
"""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Final

import numpy as np
import pycolmap

from colsfm.database import ImagePair, KeypointsXY, MatchIndices, database_transaction

DEFAULT_VERIFY_THREADS: Final[int] = max(1, min(8, os.cpu_count() or 1))
"""Worker threads per verifier; 8 saturate the per-pair estimation (7.7x measured), more add little."""


@dataclass(frozen=True, slots=True)
class VerifyChunking:
    """How verification is chunked and overlapped; the same rule on every machine, with no branch on its RAM."""

    verify_chunk_pairs: int = 256
    """Pairs verified, and written, per chunk; also FlexGlue's download chunk."""
    verify_threads: int = DEFAULT_VERIFY_THREADS
    """Worker threads that verify pairs (`pycolmap.estimate_two_view_geometry` releases the GIL)."""
    verify_chunks_in_flight: int = 2
    """Chunks verifying at once; past that the matcher waits, so memory holds about this many chunks."""

    def __post_init__(self) -> None:
        """Reject a chunk, pool or window of zero.

        Raises:
            ValueError: When a setting is below 1.
        """
        if min(self.verify_chunk_pairs, self.verify_threads, self.verify_chunks_in_flight) < 1:
            raise ValueError(f"verify_chunk_pairs, verify_threads and verify_chunks_in_flight must be at least 1, got {self}")


DEFAULT_VERIFY_CHUNKING: Final[VerifyChunking] = VerifyChunking()
"""256 pairs per chunk, up to 8 threads, 2 chunks in flight."""


CapFunction = Callable[[ImagePair, KeypointsXY, MatchIndices], MatchIndices]
"""Thins a pair's verified inlier matches (the post-verification spatial cap); returns them unchanged when not capped."""


@dataclass(frozen=True, slots=True)
class PairToVerify:
    """One matched pair and the keypoints its matches index into, taken from data the matcher already holds."""

    pair: ImagePair
    """`(image_id1, image_id2)`, the order the matches are in."""
    matches: MatchIndices
    """Raw matches, int64 `[n, 2]`."""
    keypoints1: KeypointsXY
    """Keypoint pixels of the first image."""
    keypoints2: KeypointsXY
    """Keypoint pixels of the second image."""


@dataclass(frozen=True, slots=True)
class VerifiedPair:
    """What verification decided for one pair, ready to write."""

    pair: ImagePair
    """`(image_id1, image_id2)`."""
    matches: MatchIndices
    """The raw matches, as given."""
    geometry: pycolmap.TwoViewGeometry | None
    """The row to write; None when COLMAP would store nothing (`UNDEFINED`)."""
    num_capped: int
    """Inlier matches the spatial cap removed."""

    @property
    def num_inliers(self) -> int:
        """Verified inlier matches after the cap; 0 for a degenerate or unstored pair.

        Returns:
            The count.
        """
        return 0 if self.geometry is None else len(self.geometry.inlier_matches)


def verify_pair(
    item: PairToVerify,
    camera1: pycolmap.Camera,
    camera2: pycolmap.Camera,
    options: pycolmap.TwoViewGeometryOptions,
    cap: CapFunction | None,
) -> VerifiedPair:
    """Verify one pair as COLMAP's verifier does, then apply the spatial cap.

    Args:
        item: The pair, its matches and keypoints.
        camera1: The first image's camera.
        camera2: The second image's camera.
        options: Verification settings (`colsfm.matching.verification_options`).
        cap: The spatial cap, or None.

    Returns:
        The geometry to write and what the cap removed.
    """
    geometry: pycolmap.TwoViewGeometry = pycolmap.estimate_two_view_geometry(
        camera1,
        np.asarray(item.keypoints1, dtype=np.float64),
        camera2,
        np.asarray(item.keypoints2, dtype=np.float64),
        np.ascontiguousarray(item.matches, dtype=np.uint32),
        options,
    )
    if int(geometry.config) == int(pycolmap.TwoViewGeometryConfiguration.UNDEFINED):
        return VerifiedPair(pair=item.pair, matches=item.matches, geometry=None, num_capped=0)
    if int(geometry.config) == int(pycolmap.TwoViewGeometryConfiguration.DEGENERATE) or len(geometry.inlier_matches) < options.min_num_inliers:
        degenerate: pycolmap.TwoViewGeometry = pycolmap.TwoViewGeometry()
        degenerate.config = int(pycolmap.TwoViewGeometryConfiguration.DEGENERATE)
        return VerifiedPair(pair=item.pair, matches=item.matches, geometry=degenerate, num_capped=0)
    num_capped: int = 0
    if cap is not None:
        inliers: MatchIndices = np.asarray(geometry.inlier_matches, dtype=np.int64)
        kept: MatchIndices = cap(item.pair, item.keypoints1, inliers)
        num_capped = len(inliers) - len(kept)
        if num_capped:
            geometry.inlier_matches = kept.astype(np.uint32)
    return VerifiedPair(pair=item.pair, matches=item.matches, geometry=geometry, num_capped=num_capped)


class ChunkedVerifier:
    """A pool of verifier threads; `verified` runs chunks through it in order, at most `verify_chunks_in_flight` at once."""

    def __init__(
        self,
        cameras: Mapping[int, pycolmap.Camera],
        options: pycolmap.TwoViewGeometryOptions,
        cap: CapFunction | None = None,
        chunking: VerifyChunking = DEFAULT_VERIFY_CHUNKING,
    ) -> None:
        """Start the worker pool.

        Args:
            cameras: Camera per image id.
            options: Verification settings.
            cap: The spatial cap, or None.
            chunking: The worker threads, and the chunks admitted at once.
        """
        self._cameras: Mapping[int, pycolmap.Camera] = cameras
        self._options: pycolmap.TwoViewGeometryOptions = options
        self._cap: CapFunction | None = cap
        self._chunks_in_flight: int = chunking.verify_chunks_in_flight
        self._pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=chunking.verify_threads, thread_name_prefix="colsfm-verify")

    def verified(self, chunks: Iterable[Sequence[PairToVerify]]) -> Iterator[list[VerifiedPair]]:
        """Verify `chunks` and yield each one finished, in the order they came.

        A chunk is pulled from `chunks` and admitted only while fewer than `verify_chunks_in_flight` are in flight; when
        the window is full, the oldest is finished and yielded first. A finished oldest chunk is yielded as soon as it is
        seen, so the caller can write and drop it early. A failure in a worker or in the producer is raised here; the
        chunks still in flight are cancelled (a running estimate is waited for when the pool shuts down).

        Args:
            chunks: The chunks, produced lazily (FlexGlue downloads the next one only when it is asked for).

        Yields:
            Each chunk's verified pairs, in its own order.
        """
        in_flight: deque[list[Future[VerifiedPair]]] = deque()
        try:
            for chunk in chunks:
                while in_flight and (len(in_flight) >= self._chunks_in_flight or all(future.done() for future in in_flight[0])):
                    # Results first, then the pop: a chunk whose result raises stays in `in_flight` and is cancelled below.
                    finished: list[VerifiedPair] = [future.result() for future in in_flight[0]]
                    in_flight.popleft()
                    yield finished
                in_flight.append(
                    [
                        self._pool.submit(verify_pair, item, self._cameras[item.pair[0]], self._cameras[item.pair[1]], self._options, self._cap)
                        for item in chunk
                    ]
                )
            while in_flight:
                last: list[VerifiedPair] = [future.result() for future in in_flight[0]]
                in_flight.popleft()
                yield last
        finally:
            for futures in in_flight:
                for future in futures:
                    future.cancel()

    def __enter__(self) -> ChunkedVerifier:
        return self

    def __exit__(self, *_: object) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)


def write_verified(database: pycolmap.Database, chunk: Sequence[VerifiedPair], *, write_matches: bool) -> None:
    """Write one verified chunk in one transaction: the raw matches when the matcher did not, and every stored geometry.

    All or nothing: if any write raises, the transaction rolls back and the chunk leaves no row behind.

    Args:
        database: An open database.
        chunk: The verified pairs.
        write_matches: Also write the raw matches (a matcher that streamed them here instead of writing them itself).
    """
    with database_transaction(database):
        for verified in chunk:
            image_id1, image_id2 = verified.pair
            if write_matches:
                database.write_matches(image_id1, image_id2, np.ascontiguousarray(verified.matches, dtype=np.uint32))
            if verified.geometry is not None:
                database.write_two_view_geometry(image_id1, image_id2, verified.geometry)
