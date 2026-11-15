"""Which rig pairs are worth measuring, decided from retrieval alone.

The retrieval half of the loop funnel, and the half that reads no match at all:
a query per keyframe of an allowed camera, the score gate, the time gate, the
same-rig gate, one hit per `|dt|` band per query, then one hit per unordered rig
pair. `colsfm.loop_closure.plan_loop_search` is its only caller, and the fact
that none of this needs a match is exactly why the pipeline's discovery run --
which learned the pair list by *executing* the whole search against a matcher
that returned nothing -- was unnecessary.

The blob applies the time gate AFTER retrieval, so temporal neighbours eat slots
out of its top-20 and it needs an early abort when more than half the hits are
too close (spec §6.4). `RetrievalIndex.search` applies it inside the ranking walk
instead, which keeps all `top_k` slots useful and reports what the gate dropped
from the same walk.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from colsfm.frames_meta import FramesMeta, KeyframeMeta
from colsfm.geometry import MICROSECONDS_PER_SECOND
from colsfm.loop_pose import RigPoseRejection, RigPoseRejectionReason
from colsfm.pairs import normalise_pair
from colsfm.retrieval import RetrievalIndex, RetrievalQuery


@dataclass(slots=True)
class FunnelCounters:
    """The running counts of one loop search, before they are frozen.

    One declaration of the twelve names, which used to be spelled three times: as string
    literals in a `dict.fromkeys`, as diagnostics fields, and again in a rejection-to-counter
    table with an unreachable None key.
    """

    queries: int = 0
    """Query keyframes a retrieval was issued for."""
    candidates_retrieved: int = 0
    """Retrieval hits the index examined, before any gate."""
    rejected_by_score: int = 0
    """Hits dropped by the score gate."""
    rejected_by_time: int = 0
    """Hits dropped by the temporal gate."""
    rejected_by_same_rig: int = 0
    """Hits whose rig frame is the query's own."""
    after_banding: int = 0
    """Hits left after `select_best_candidates`."""
    after_deduplication: int = 0
    """Rig pairs left after one hit per unordered rig pair."""
    rejected_no_matches: int = 0
    """Rig pairs with too few 2-D-3-D observations."""
    rejected_by_geometry: int = 0
    """Rig pairs the generalized PnP failed on."""
    rejected_by_is_good: int = 0
    """Rig pairs that failed `is_good_match`."""
    rejected_by_direction: int = 0
    """Rig pairs the direction cross-check dropped."""
    verified: int = 0
    """Rig pairs that produced a metric relative pose."""

    def count(self, rejection: RigPoseRejection) -> None:
        """Add one `colsfm.loop_pose` rejection to the counter that owns it.

        Args:
            rejection: The gate that turned the rig pair down.
        """
        reason: RigPoseRejectionReason = rejection.reason
        if reason == "no_observations":
            self.rejected_no_matches += 1
        elif reason == "no_pose":
            self.rejected_by_geometry += 1
        elif reason == "too_few_inliers":
            # The same rule `is_good_match` states, applied one stage earlier.
            self.rejected_by_is_good += 1
        else:
            self.rejected_by_direction += 1

    def freeze(self, *, duration_seconds: float, gap_seconds: float, score_threshold: float, edges: int) -> LoopClosureDiagnostics:
        """Turn the counters and the run's resolved gates into the reported diagnostics.

        Args:
            duration_seconds: The session duration the gates were derived from.
            gap_seconds: The temporal gap actually applied.
            score_threshold: The retrieval score gate actually applied.
            edges: Edges left after `gate_loop_edges`.

        Returns:
            The frozen diagnostics, carrying a copy of the counters as they stand.
        """
        return LoopClosureDiagnostics(
            counters=replace(self),
            session_duration_seconds=duration_seconds,
            min_time_gap_seconds=gap_seconds,
            good_score_threshold=score_threshold,
            edges=edges,
        )


@dataclass(frozen=True, slots=True)
class LoopClosureDiagnostics:
    """Where the candidates went, so a run can be explained without re-running it.

    The counters themselves are `FunnelCounters`, held whole rather than copied out
    field by field: the twelve names were declared here as well, and `freeze` spliced
    them across with `**asdict(...)`, so adding a gate meant editing three places and
    a reader could not tell which numbers came from the same walk. What is added here
    is what the counters cannot know — the gates the run actually resolved, and the
    edges left after `gate_loop_edges`.
    """

    counters: FunnelCounters
    """The funnel itself: queries, hits, and where every hit went."""
    session_duration_seconds: float
    """`max(timestamp) - min(timestamp)` over the keyframes, in seconds."""
    min_time_gap_seconds: float
    """The temporal gap actually applied: `max(fixed threshold, ratio * session duration)`."""
    good_score_threshold: float
    """The retrieval score gate actually applied."""
    edges: int
    """Edges left after `gate_loop_edges`, i.e. what the caller receives."""

@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """One retrieval hit that passed the score, time and rig gates, before any matching.

    The funnel bands and deduplicates these, and only the survivors are handed to the
    matcher. Nothing here needs an image pair to have been matched, which is what lets the
    stage name every pair it will need in a single dry run (`find_loop_edges`).
    """

    query_keyframe_id: int
    """Keyframe the query was issued from."""
    candidate_keyframe_id: int
    """Retrieved keyframe."""
    source_rig_id: int
    """`synced_sample_id` of the query keyframe's rig frame."""
    target_rig_id: int
    """`synced_sample_id` of the candidate keyframe's rig frame."""
    score: float
    """Retrieval score in `[0, 1]`."""
    delta_seconds: float
    """`t_query - t_candidate` in seconds; `select_best_candidates` bands on its magnitude."""


def select_best_candidates(hits: Sequence[RetrievalHit], band_seconds: float) -> list[RetrievalHit]:
    """`SelectBestCandidates`: keep one hit per band of `|dt|` (spec §6.4).

    Sorts ascending by `|dt|` and walks greedily: a hit more than `band_seconds` beyond the
    last emitted one opens a new band, otherwise it replaces the last emitted one when it
    scores higher. This is what stops a single revisit producing 20 near-duplicate loop
    edges (spec §7.1 item 5).

    **Departure from the blob, measured.** `SelectBestCandidates` ranks a band first by the
    geometric inlier count and only then by the retrieval score. Ranking by inliers needs
    every candidate matched *before* the band is chosen, which is 15 325 image pairs on
    RoboCap where the banded shortlist needs 6 000 — and the pipeline has to name every pair
    it will match in one dry run, before any match result exists (`find_loop_edges`). The
    retrieval score is the only ranking available at that point. Measured on RoboCap the two
    orders pick nearly the same pairs, because a band is a 10 s window of one revisit and its
    candidates are near-duplicates of each other.

    Args:
        hits: Retrieval hits for one query keyframe that passed the score and time gates.
        band_seconds: Band width in seconds; the blob uses
            `loop_interval_threshold_in_seconds`.

    Returns:
        The kept hits, in ascending `|dt|` order.
    """
    ordered: list[RetrievalHit] = sorted(hits, key=lambda hit: (abs(hit.delta_seconds), hit.candidate_keyframe_id))
    kept: list[RetrievalHit] = []
    last_gap_seconds: float = -np.inf
    for hit in ordered:
        gap_seconds: float = abs(hit.delta_seconds)
        if gap_seconds - last_gap_seconds > band_seconds:
            kept.append(hit)
            last_gap_seconds = gap_seconds
            continue
        if hit.score > kept[-1].score:
            kept[-1] = hit
    return kept


def query_camera_ids(frames_meta: FramesMeta) -> set[int]:
    """Camera ids allowed to issue a retrieval query.

    Both blob paths query the **left** camera of each declared stereo pair and skip the rest
    ("Skip, due to frame N is not a left camera frame"). Metadata that declares no stereo
    pair has no left camera to pick, so every camera queries.

    Args:
        frames_meta: The parsed metadata.

    Returns:
        The camera ids that may issue a query.
    """
    if not frames_meta.stereo_pairs:
        return set(frames_meta.cameras)
    return {pair.left_camera_params_id for pair in frames_meta.stereo_pairs}


def hits_for_query(
    query: KeyframeMeta,
    source_rig_id: int,
    answer: RetrievalQuery,
    keyframe_by_id: Mapping[int, KeyframeMeta],
    score_threshold: float,
    counters: FunnelCounters,
) -> list[RetrievalHit]:
    """Turn one query keyframe's retrieval answer into gated hits.

    Args:
        query: The query keyframe.
        source_rig_id: `synced_sample_id` of the query keyframe's rig frame.
        answer: What `RetrievalIndex.search` returned for it.
        keyframe_by_id: Every keyframe, indexed by id.
        score_threshold: The resolved retrieval score gate.
        counters: Diagnostic counters, mutated in place.

    Returns:
        The hits that passed the score and same-rig gates, in retrieval order.
    """
    counters.candidates_retrieved += len(answer.candidates) + answer.rejected_by_time
    counters.rejected_by_time += answer.rejected_by_time
    hits: list[RetrievalHit] = []
    for candidate in answer.candidates:
        if candidate.score < score_threshold:
            counters.rejected_by_score += 1
            continue
        target: KeyframeMeta | None = keyframe_by_id.get(candidate.image_id)
        if target is None:
            continue
        if target.synced_sample_id == source_rig_id:
            counters.rejected_by_same_rig += 1
            continue
        hits.append(
            RetrievalHit(
                query_keyframe_id=query.keyframe_id,
                candidate_keyframe_id=target.keyframe_id,
                source_rig_id=source_rig_id,
                target_rig_id=target.synced_sample_id,
                score=candidate.score,
                delta_seconds=(query.timestamp_microseconds - target.timestamp_microseconds) / MICROSECONDS_PER_SECOND,
            )
        )
    return hits


def shortlist_rig_pairs(
    frames_meta: FramesMeta,
    index: RetrievalIndex,
    keyframe_by_id: Mapping[int, KeyframeMeta],
    timestamps_us: Mapping[int, int],
    *,
    top_k: int,
    candidate_band_seconds: float,
    gap_seconds: float,
    score_threshold: float,
    counters: FunnelCounters,
) -> dict[tuple[int, int], RetrievalHit]:
    """Retrieve, gate, band and deduplicate down to one hit per unordered rig pair.

    The blob applies the time gate AFTER retrieval, so temporal neighbours eat slots out of
    its top-20 and it needs an early abort when more than half the hits are too close
    (spec §6.4). `RetrievalIndex.search` applies it inside the ranking walk instead, which
    keeps all `top_k` slots useful and reports what the gate dropped from the same walk.

    Args:
        frames_meta: The parsed metadata.
        index: The retrieval index.
        keyframe_by_id: Every keyframe, indexed by id.
        timestamps_us: Capture time per keyframe id.
        top_k: Retrieval hits to ask the index for per query.
        candidate_band_seconds: Band width `select_best_candidates` collapses within.
        gap_seconds: The temporal gate actually applied.
        score_threshold: The resolved retrieval score gate.
        counters: Diagnostic counters, mutated in place.

    Returns:
        The best-scoring hit per unordered rig pair, keyed by that pair.
    """
    allowed_camera_ids: set[int] = query_camera_ids(frames_meta)
    gap_us: int = round(gap_seconds * MICROSECONDS_PER_SECOND)
    indexed: set[int] = set(index.image_ids)

    banded: list[RetrievalHit] = []
    for rig in frames_meta.rig_frames():
        for query_id in rig.keyframe_ids:
            query: KeyframeMeta = keyframe_by_id[query_id]
            if query.camera_params_id not in allowed_camera_ids or query_id not in indexed:
                continue
            counters.queries += 1
            answer: RetrievalQuery = index.search(query_id, top_k=top_k, min_time_gap_us=gap_us, timestamps=timestamps_us)
            for_query: list[RetrievalHit] = hits_for_query(query, rig.synced_sample_id, answer, keyframe_by_id, score_threshold, counters)
            banded.extend(select_best_candidates(for_query, candidate_band_seconds))

    counters.after_banding = len(banded)
    shortlist: dict[tuple[int, int], RetrievalHit] = {}
    for hit in banded:
        key: tuple[int, int] = normalise_pair(hit.source_rig_id, hit.target_rig_id)
        incumbent: RetrievalHit | None = shortlist.get(key)
        if incumbent is None or hit.score > incumbent.score:
            shortlist[key] = hit
    counters.after_deduplication = len(shortlist)
    return shortlist


