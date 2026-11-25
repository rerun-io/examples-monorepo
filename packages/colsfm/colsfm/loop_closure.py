"""Loop-closure candidate selection and verification — cuSFM's `RetrievalLoopAssociations`.

This is stage 3 of the plan of record: turn the retrieval hits of `colsfm.retrieval` into
verified `colsfm.pose_graph.PoseGraphEdge`s of kind `"loop"`. It reimplements
`AssociationWorker::RetrievalLoopAssociationsForFrame`
(`docs/spec/generate_association_main.md` §6.4) and
`PoseGraph::FindBestLoopCandidateForFrame` (`docs/spec/pose_graph_main.md` §6.4), which
are two views of the same pipeline, and deliberately departs from the blob in four
places — every one of them measured, every one documented below. The measurement itself
lives in `colsfm.loop_pose`; this module decides *which* rig frames to measure.

Pipeline per rig frame
----------------------

1. Query keyframes: the **left** camera of each declared stereo pair, matching both blob
   paths ("Skip, due to frame N is not a left camera frame"). Metadata that declares no
   stereo pair falls back to every camera.
2. Retrieval: one `RetrievalIndex.search(top_k=20)` per query keyframe, with the temporal
   gate (below) applied inside the ranking walk, then the `good_score_threshold` gate and a
   same-rig-frame gate.
3. Candidates: each query rig frame pools its query keyframes' hits and keeps the
   `loop_candidates_per_frame` best target rig frames by retrieval score (3; 1 in the kitti
   profile), then one hit per unordered rig pair.
4. Measurement: `colsfm.loop_pose.RigPoseEstimator` on each surviving rig pair — a metric
   point cloud triangulated in the source rig frame from its own cameras and its temporal
   neighbours, then `estimate_and_refine_generalized_absolute_pose` for the whole target
   rig. Its gates are `is_good_match`'s: `inliers > min_inliers` and
   `inliers / observations > min_inlier_ratio` (30 and 0.25,
   `docs/spec/generate_association_main.md` §6.6, §7.1 item 6), plus a cross-check of the
   translation direction against the generalized essential matrix.
5. Each query rig frame keeps its single best measured pair by inliers — cuSFM's `num_votes: 1`,
   the only value colsfm implements, so the field is not read — which becomes an edge
   when it has at least `good_loop_threshold` inliers (cuSFM's isaac default, 150), then
   `colsfm.pose_graph.gate_loop_edges` with real thresholds.

Steps 1-3 read nothing but the retrieval index and the timestamps. That is deliberate: the
pipeline learns which image pairs to match by running this whole stage once with a matcher
that returns nothing, so any pair whose request depends on a match result would never be
matched in production (`colsfm.pipeline._find_loop_edges`, `colsfm.loop_pose`). It is also
why step 3 ranks by retrieval score: the inlier count only exists after the measurement, and
step 5 is where it decides.

The temporal gate
-----------------

The two blob binaries gate differently and neither number is usable as shipped:

* `generate_association_main` drops candidates with
  `|dt| < loop_interval_threshold_in_seconds` (10 s).
* `pose_graph_main` drops candidates with
  `|dt| < loop_closure_interval_ratio * session_duration_seconds` (0.08 x the session).

`minimum_time_gap_seconds` takes the **maximum** of the two, i.e. a candidate must pass
both. That is the conservative reading, and it is what a union of the two specs means for
a pipeline that runs the association stage and the pose graph back to back.

**Measured trap:** the shipped Galileo sequence lasts **0.933 s** end to end (226
keyframes, 29 rig frames, a 0.66 m sweep). The fixed 10 s gate therefore rejects every
candidate in the dataset, and only the ratio gate (0.08 x 0.933 = 0.075 s, about two rig
frames) is live. A caller that wants loops on Galileo must lower
`loop_interval_threshold_in_seconds`; `LoopClosureDiagnostics.min_time_gap_seconds`
reports what was actually applied.

What is deliberately not copied
-------------------------------

* `max_mean_point_to_epipolarline_error: 10e-6`. Measured on RoboCap it rejects 168 of 180
  candidates and the blob produces **zero** loop associations
  (`docs/spec/generate_association_main.md` §6.4). The gate here is
  `RigPoseConfig.ransac_max_error_px`, expressed in pixels.
* The four-view stereo estimator with baseline-locked scale, replaced by the local metric
  map and generalized resection of `colsfm.loop_pose` — a strictly larger version of the
  same idea, using every camera of the rig and its temporal neighbours rather than one
  stereo pair.
* The `scale = 0.47` fallback for an out-of-range scale — the candidate is rejected instead.
* `loop_edge_translation_threshold_meters` / `loop_edge_rotation_threshold_degrees` default
  to **0.0** in every shipped config, which rejects every candidate and is why
  `pose_graph_main` never emits a loop edge out of the box. The defaults here are 3.0 m and
  30 deg (`docs/spec/pose_graph_main.md` §8 "Should fix").

Information matrices
--------------------

Identity by default. The blob writes `inliers * loop_residual_weight * I6` on the
association path and `(1/||t||) * loop_residual_weight * I6` on the pose-graph path, then
**overwrites both with the identity at solve time** because
`relative_constraint_use_pose_covariance` is false in every shipped config
(`docs/spec/pose_graph_main.md` §8 item 6). Measured against the blob's archived 8-loop
Galileo result, the identity lands 0.28 mm away and the stored weights 2.17 mm away — so
identity is the default and `LoopClosureConfig.use_stored_weights` is the documented
option, weighting by inlier count (the more standard of the blob's two schemes).

Measured on RoboCap: retrieval is solved, the measurement is better, the gap is elsewhere
------------------------------------------------------------------------------------------

RoboCap is 4528 fisheye keyframes, 1132 rig frames, four cameras pointing in three
directions, a 124 m walk over 150.8 s. The blob's own run emits **90** LOOP constraints and
a PGO output; everything below is scored against that output after rigid alignment.

Retrieval and pair selection were settled first, and they are right: the vocab backend's
gated top-20 covers all 90 of the blob's LOOP pairs, and replacing only the *measurement* on
our own 488 gated rig pairs with the blob's optimised relative poses lands **39.9 mm** from
its PGO, against 460.8 mm for the input trajectory alone. So the pair set can carry the
answer; what it is fed decides everything.

Estimators, on that one fixed 488-pair set:

| measurement of `source_T_target` | edges | translation error vs the oracle, median / p90 | rotation error, median | RMSE vs the blob's PGO |
|---|---|---|---|---|
| none (the input trajectory) | 0 | — | — | 460.8 mm |
| the oracle (the blob's own PGO) | 488 | 0 / 0 mm | 0.00 deg | **39.9 mm** |
| the input odometry's own relative pose | 488 | 383 / 799 mm | 6.79 deg | 460.8 mm |
| two-view essential + prior-pose scale (what this stage used to ship) | 488 | 346 / 809 mm | 7.30 deg | 609.0 mm |
| `loop_pose`, stereo pair only (`neighbour_span=0`) | 189 | 197 / 1140 mm | 6.63 deg | 457.0 mm |
| **`loop_pose`, the default (`neighbour_span=1`)** | 425 | 219 / 776 mm | 6.90 deg | **408.1 mm** |
| `loop_pose`, `neighbour_span=2` | 448 | 231 / 773 mm | 6.92 deg | 418.5 mm |
| `loop_pose`, no direction cross-check | 430 | 228 / 776 mm | 6.91 deg | 418.4 mm |
| `loop_pose`, direction cross-check at 25 deg | 414 | 227 / 779 mm | 6.85 deg | 407.6 mm |

The new estimator is a clear improvement on the old one — 609.0 mm to 408.1 mm, from worse
than no loops to better than no loops — and `neighbour_span=1` is the knee: the temporal
neighbours more than double the usable triangulation baseline over the 86 mm stereo pair,
and a second step adds nothing. But 408 mm is not 39.9 mm, and no gate closes the rest.
Every knob was swept on the same edges and every one is flat:

| selectivity or weighting | edges | RMSE vs the blob's PGO |
|---|---|---|
| all edges | 433 | 408.0 mm |
| only the 213 within 2 rig frames of a blob LOOP pair | 213 | 389.1 mm |
| `||t|| <= 0.5 m` / `<= 1 m` | 361 / 425 | 417.4 / 407.0 mm |
| inlier ratio `>= 0.30` / `>= 0.35` / `>= 0.40` | 339 / 274 / 206 | 413.0 / 412.3 / 419.1 mm |
| within 0.3 m / 0.5 m of the prior | 209 / 408 | 439.5 / 405.9 mm |
| loop information `0.3` / `0.1` / `0.01 * I6` | 433 | 408.5 / 409.9 / 413.4 mm |

Nothing here is worth 250 mm, which says the residual is not a few bad edges.

The whole stage, end to end
---------------------------

Everything above varies the measurement on one frozen pair set. Running the stage as
`colsfm.pipeline` ran it before `loop_candidates_per_frame` replaced the 10 s `|dt|` banding (so the
`select_best_candidates` row below is that banding) — retrieval, banding, batch matching, measurement, gating, with
the `data/cusfm_configs/loop-closure-fixed` profile's own 1.0 m / 10 deg edge gates:

| stage | count |
|---|---|
| query keyframes (left camera of the one stereo pair) | 1 132 |
| retrieval hits examined (`candidates_retrieved`) | 61 550 |
| dropped by the 12.07 s temporal gate | 38 910 |
| dropped by `good_score_threshold` | 4 926 |
| after `select_best_candidates` | 1 747 |
| after one rig pair each | 1 573 |
| dropped by `is_good_match` | 574 |
| dropped by the direction cross-check | 74 |
| measured | 925 |
| **after `gate_loop_edges` (1.0 m / 10 deg)** | **73** |

Those counters come from a single `RetrievalIndex.search` per query keyframe, so they
compose: 61 550 - 38 910 - 4 926 is exactly what reaches the banding. (An earlier version
issued a second, ungated query only to count the temporal rejections, so its
`candidates_retrieved` of 22 640 counted only the hits the gate had already let through and
the funnel did not add up.)

14 443 image pairs are asked of the matcher, of which 6 057 were new — fewer than the
15 325 the old two-view funnel needed, because banding on the retrieval score shortlists rig
pairs before anything is matched. Repeated runs of the same code give 73 to 75 edges and
401.1 to 404.4 mm: both generalized estimators are RANSAC and pycolmap's `set_random_seed`
does not reach them, so a pair on the inlier gate falls either way. The poses themselves are
stable to 0.1 mm, and the 1 573 shortlisted rig pairs are identical run to run.

| edges fed to `solve_pose_graph` | RMSE vs the blob's PGO |
|---|---|
| sequential only | 460.8 mm |
| + our 73 loop edges | **401.1 mm** |
| + oracle poses on those same 73 rig pairs | 261.8 mm |
| + the blob's own 90 loop edges | 135.0 mm |

Two things to read off that table. The 74 edges are the *accurate* ones — their translation
error against the oracle is a median of 77 mm and their rotation error 3.80 deg, against
219 mm and 6.90 deg over the unfiltered 488 — so the funnel's selectivity works. But the
oracle on the same 74 pairs only reaches 261.8 mm, so this configuration is capped by its
**pair set**, not by its measurement: `loop_edge_rotation_threshold_degrees: 10` throws away
847 of the 921 measured pairs, and covers 56 of the blob's 90 loop pairs instead of all 90,
because a real RoboCap revisit rotates by a median of 11.8 deg and that gate was tuned
against an estimator whose rotations were smaller. `LoopClosureConfig` defaults to 30 deg
for exactly this reason; the config profile overrides it.

The measurement pass is where the wall time goes
------------------------------------------------

The 1 573 rig pairs are the stage's whole cost after matching, and they are independent
given their source rig frame's local map. `LoopClosureConfig.num_threads` therefore runs one
source-rig group per worker thread: `pycolmap.estimate_and_refine_generalized_absolute_pose`
releases the GIL, and a group shares nothing with another but the estimator's read-only
bearing memo. Measured on the RoboCap run above, on a 32-core machine:

| measurement pass | wall time |
|---|---|
| serial (`num_threads=1`) | 451.8 s |
| 8 threads (the default) | **67.8 s** |

The results are collected in submission order and the diagnostics are accumulated after the
join, so the thread count does not reach the edge list: the two runs above shortlisted the
same 1 573 rig pairs, verified 921 and 925 of them and produced 74 and 73 edges at 404.4 and
401.1 mm — the same spread two serial runs give, because RANSAC is what moves those numbers.

The gap is the rotation, and it is not ours
-------------------------------------------

Running the estimator on the blob's **own** 90 loop rig pairs compares two independent
measurements of the same quantity with no pair selection and no pose graph in between. We
measured 87 of them:

| quantity, median over the 87 pairs | ours | the input odometry | the blob's own LOOP edge |
|---|---|---|---|
| `||t||` of the edge | 132 mm | 355 mm | 160 mm |
| relative rotation angle | 11.8 deg | 11.8 deg | 7.3 deg |
| translation vs the blob's edge | 180 mm | 329 mm | — |
| rotation vs the blob's edge | 6.5 deg | 6.4 deg | — |

The **translation** result is the estimator working: the odometry says these revisits are
355 mm apart, the blob measured 160 mm, we measured 132 mm, and we cut the disagreement with
the blob's own number from 329 mm to 180 mm. A measurement that took its magnitude from the
prior could not do that.

The **rotation** is the whole remaining gap, and the blob is the outlier, not us:

* Two independent image-based estimators — the generalized resection of `colsfm.loop_pose`
  and `pycolmap.estimate_generalized_relative_pose`, which share their correspondences and
  nothing else — agree with each other to **0.35 deg** (p90 0.63 deg) and both sit
  **6.5 deg** from the blob's stored rotation and **1.5 deg** from the odometry's.
* The blob's PGO followed its own edges: its output's relative rotation sits 1.1 deg from
  its LOOP edges and 6.0 deg from the odometry's.
* Rebuilding our edges component by component: our rotation with the blob's translation
  scores 392.9 mm, the blob's rotation with our translation scores **265.6 mm**. All of the
  correction the blob's PGO applies is rotational.
* The convention is not the explanation. Parsed exactly as stored, the blob's 1131
  CONSECUTIVE constraints reproduce `sequential_edges` to **0.0000 mm and 0.0000 deg**;
  inverting them gives 228 mm and 7.9 deg. So `pose_source_to_target` is read correctly,
  rotation included, and the 6.5 deg is a real disagreement about the world.

A 6 deg rotation error is not something a resection with ~1900 inliers at an 8 px threshold
on a 630 px focal length can hide — the residuals would be tens of pixels. The odometry
agreeing with us is also the expected pattern for a visual-inertial front end, whose
orientation drifts far more slowly than its position. RoboCap ships **no ground truth**
(NOTES.md §"RoboCap reconstruction quality"), so this cannot be settled outright; what can
be said is that the reference this stage is scored against carries a rotation two
independent estimators here disagree with, and that scoring against it caps what any
measurement in this module can reach.

Where that leaves the stage
---------------------------

The stage is **on** since 2026-09-05 (it was off until then; the blob always runs it, and
the user wants the comparison to pay for it on both sides). Nothing in this module can turn
it off: `colsfm.stages.run_loop_closure_stage(..., enabled=...)` is the one switch, fed by
`--no-loop-closure`, and it returns before this module is reached. The
paragraph below is the reasoning that kept it off, kept for the record. The estimator is a real improvement — it beats
the old one by 200 mm and beats running no loops at all, on the frozen pair set and end to
end — but 401 mm against a 135 mm target is not a result worth switching on by default, and
the one number that would justify switching it on (agreement with something that is actually
ground truth) does not exist for this dataset. A caller who turns it on must measure the
trajectory before and after, and should raise `loop_edge_rotation_threshold_degrees` above
10 first, because that gate now costs more than it saves.

On Galileo the same switch stays off for a different reason: turning loops on moves camera
positions by 5-13 mm against a 5 mm ATE budget (`docs/spec/pose_graph_main.md` §8), because
a 0.93 s, 0.66 m sweep has no drift for a loop to remove. The pipeline decides per dataset.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Final

import numpy as np
import pycolmap
from jaxtyping import Float64
from numpy import ndarray

from colsfm.config import PoseGraphConfig
from colsfm.database import ImagePair, read_keypoints_batch
from colsfm.frames_meta import FramesMeta, KeyframeMeta
from colsfm.geometry import MICROSECONDS_PER_SECOND
from colsfm.loop_pairs import required_image_pairs
from colsfm.loop_pose import (
    BEARING_CACHE_IMAGES,
    DEFAULT_RIG_POSE_CONFIG,
    Keypoints,
    MatchFunction,
    RigPoseConfig,
    RigPoseEstimate,
    RigPoseEstimator,
    RigPoseOutcome,
    RigPoseRejection,
)
from colsfm.loop_shortlist import (
    FunnelCounters,
    LoopClosureDiagnostics,
    RetrievalHit,
    Shortlist,
    shortlist_rig_pairs,
)
from colsfm.pairs import normalise_pair
from colsfm.pose_graph import Information6, PoseGraphEdge, gate_loop_edges, loop_edge_information
from colsfm.retrieval import GOOD_SCORE_THRESHOLD, RetrievalIndex
from colsfm.rig_geometry import RigFrameIndex, RigGeometry, rig_frame_index, rig_geometry

__all__ = [
    "FunnelCounters",
    "LoopCandidate",
    "LoopClosureConfig",
    "LoopClosureDiagnostics",
    "LoopClosureResult",
    "LoopSearchPlan",
    "RetrievalHit",
    "best_per_query_rig",
    "find_loop_edges",
    "pairs_outside_accepted_edges",
    "is_good_match",
    "minimum_time_gap_seconds",
    "plan_loop_search",
    "session_duration_seconds",
    "verify_loop_plan",
]
"""The loop subsystem's public surface.

`colsfm.loop_closure` stays the one import path, whichever of its two modules a
name is defined in: `colsfm.loop_shortlist` owns the retrieval half of the funnel
-- the gates, the candidate cap, the counters and the diagnostics they freeze into --
and this module owns the plan, the measurement and the gating."""

DEFAULT_MEASUREMENT_THREADS: Final[int] = min(8, os.cpu_count() or 1)
"""Worker threads the measurement pass uses by default; see `LoopClosureConfig.num_threads`."""


@dataclass(frozen=True, slots=True)
class LoopClosureConfig:
    """Gates and thresholds for the loop-closure stage."""

    top_k: int = 20
    """`pose_graph_config.loop_closure_query_count`: retrieval hits considered per query keyframe."""
    good_score_threshold: float | None = None
    """Minimum retrieval score, or None to take `colsfm.retrieval.GOOD_SCORE_THRESHOLD`.

    Both `colsfm.retrieval` backends were calibrated separately and both landed on 0.1, so
    there is one constant rather than one per backend; `colsfm.retrieval` carries the
    measurements. Set this to override, for instance to 0.15 on the vocab backend, which
    halves the pairs handed to the matcher at 95 % rather than 88 % precision."""
    loop_interval_threshold_in_seconds: float = 10.0
    """`association_config.loop_interval_threshold_in_seconds`. Note this exceeds the whole
    0.933 s Galileo session, so it must be lowered to get any candidate there."""
    loop_closure_interval_ratio: float = 0.08
    """`pose_graph_config.loop_closure_interval_ratio`, multiplied by the session duration."""
    loop_candidates_per_frame: int = 3
    """Retrieval candidates each query rig frame verifies (best retrieval score first); it keeps the best 1 by inliers.

    colsfm's own knob, read from the profile's `colsfm.toml` (`colsfm.config.ColsfmProfileConfig`); cuSFM has no
    equivalent. It is not cuSFM's `num_votes`, which counts the verified loop edges kept per rig frame (by score across
    cameras, `docs/spec/pose_graph_main.md` LoopCandidateVoting, default 1): colsfm always keeps 1 per query rig frame
    (`best_per_query_rig`), i.e. it behaves as `num_votes: 1`, and that field is not read.

    3 by default: on MSD MGO05 1 candidate scored SE(3) 1.53 cm against 1.18 (Sim(3) 0.59 against 0.60), for 130 s
    against 217 s. The kitti profile uses 1 (0.838 m against 0.857, half the loop time).

    Replaced one candidate per 10 s `|dt|` band per query keyframe, which grew with the sequence length: on MSD MGO05
    (8,172 keyframes) it asked for 24k loop image pairs, the loop pairs grew as keyframes^1.8 and the loop stage took
    506 s of 2,408."""
    good_loop_threshold: int = 150
    """`pose_graph_config.good_loop_threshold`: inliers the best measured pair of a query rig frame needs to become an
    edge. On top of `is_good_match`, which every measured pair must pass first."""
    min_inlier_ratio: float = 0.25
    """`min_matches_ratio`: inliers over the observations offered, strictly greater."""
    rig_pose: RigPoseConfig = DEFAULT_RIG_POSE_CONFIG
    """Settings of the metric rig-to-rig estimator in `colsfm.loop_pose`, which measures
    every surviving rig pair. It also owns `min_inliers` — the blob's `min_matches_num` —
    which this config reads back as a property rather than declaring a second copy of."""
    max_translation_m: float = 3.0
    """`loop_edge_translation_threshold_meters`, applied by `gate_loop_edges`. The shipped
    isaac config leaves it at 0.0, which rejects everything; `data/cusfm_configs/
    loop-closure-fixed` repairs it to 1.0.

    Measured on the blob's own 90 RoboCap LOOP edges, the relative rig translation runs
    0.007-0.962 m with a median of 0.157 m — right up against its own 1.0 m gate. 3.0 m is
    kept as the default so a slightly worse prior pose does not silently drop a true
    revisit; the geometric verification, not this gate, is what rejects a bad pair."""
    max_rotation_deg: float = 30.0
    """`loop_edge_rotation_threshold_degrees`, applied by `gate_loop_edges`. Also 0.0 in
    the shipped isaac config, and 10.0 in `loop-closure-fixed`.

    The blob's 90 RoboCap LOOP edges rotate by 2.5-12.0 deg, median 7.2 — so its own 10 deg
    gate is already binding on its own output. 30 deg leaves headroom for the same reason
    as the translation gate."""
    max_prior_translation_m: float | None = None
    """Prior-consistency gate, applied to every measured candidate before each query rig frame keeps its best: the
    candidate survives when its measured `source_T_target` and the prior trajectory's differ by at most this translation
    (and `max_prior_rotation_deg`). None = no gate.

    For a rig that sees all around (a 360 camera's cube faces) the edge-size gates cannot tell a revisit from aliasing:
    a true revisit walked the other way rotates by ~180 deg, and look-alike ceilings match 180 deg off with over a
    thousand inliers. Measured on an Insta360 X4 Air walk-through (raw VIO prior): true loops disagree with the prior by a median
    0.53 m / 1.6 deg (revisit) and 0.6 m / 3.5 deg (start-end), false ones by 2-16 m and 60-179 deg."""
    max_prior_rotation_deg: float | None = None
    """Rotation half of the prior-consistency gate; None = no gate."""
    prior_drift_floor_m: float = 3.5
    """Constant part of the drift allowance `prior_distance_bound` grants the prior between two rig frames, metres.

    One default for every dataset, not a profile key. Measured against ground truth on the DGX Spark base runs at
    a997bc83b (every pair of rig frames, `| |prior distance| - |GT distance| |`): the worst case fits under
    1.74 m + 1 % of the prior path (KITTI 06, 1,241 m: 4.2 m at 800-1,600 m of path, and a 1.8 m prior jump over 4.5 m of
    path); EuRoC V1_01 stays under 0.16 m and MSD MGO05 under 0.08 m. The allowance is twice that worst case."""
    prior_drift_per_metre: float = 0.02
    """Part of the drift allowance that grows with the prior path travelled between the two rig frames (2 %, twice the
    1 % of the worst case above)."""
    closed_gate_reach_m: float = 30.0
    """The prior distance that replaces the edge gate when the edge gates are closed (`max_translation_m <= 0`).

    No edge is accepted then, but the loop pairs still feed the mapper (removing them cost KITTI 06 42 % ATE), so the
    bound must keep every rig pair that can see common structure. On KITTI 06 the shortlisted rig pairs reach 13.4 m
    (ground truth), and their two-view inliers still agree with ground truth (81 % within 4 px beyond 10 m); 30 m is
    over twice that."""
    use_stored_weights: bool = False
    """When True the edge information is `loop_edge_information(inliers, loop_residual_weight)`
    instead of the identity. The blob's own solver discards the stored value; measured
    against its archived Galileo result the identity is 8x closer."""
    loop_residual_weight: float = 0.25
    """`pose_graph_config.loop_residual_weight`; only read when `use_stored_weights`."""
    num_threads: int = DEFAULT_MEASUREMENT_THREADS
    """Worker threads the measurement pass runs source-rig groups on.

    The 1573 RoboCap rig pairs are 437 s of the stage's 721 s and are independent given
    their source rig frame's local map; pycolmap releases the GIL inside both generalized
    estimators, so threads are real parallelism here. The edges do not depend on this: the
    groups are collected in submission order and the counters are accumulated after the
    join. 1 runs the pass inline, with no pool at all."""

    @property
    def min_inliers(self) -> int:
        """`min_matches_num`: a good loop edge needs strictly more inliers than this.

        Declared once, on `RigPoseConfig`, because the estimator's solvability floor and
        this stage's acceptance rule are the same blob constant (spec §6.6) and two copies
        could only drift apart. Here the inliers are the 2-D-3-D observations
        `colsfm.loop_pose`'s generalized resection kept.

        Returns:
            The inlier floor the estimator ran with.
        """
        return self.rig_pose.min_inliers

    @classmethod
    def from_pose_graph(cls, config: PoseGraphConfig) -> LoopClosureConfig:
        """Build the stage's settings from the `pose_graph_config.pb.txt` a run was given.

        The eight fields the config profile owns are its loop gates; everything else stays at
        the measured default. Note that the shipped isaac and av profiles leave both edge
        gates at proto3's 0.0, which rejects every candidate — only
        `data/cusfm_configs/loop-closure-fixed` repairs them.

        Args:
            config: The parsed `pose_graph_config.pb.txt`.

        Returns:
            The loop-closure settings.
        """
        return cls(
            top_k=config.loop_closure_query_count,
            loop_candidates_per_frame=config.loop_candidates_per_frame,
            good_loop_threshold=round(config.good_loop_threshold),
            loop_closure_interval_ratio=config.loop_closure_interval_ratio,
            max_translation_m=config.loop_edge_translation_threshold_meters,
            max_rotation_deg=config.loop_edge_rotation_threshold_degrees,
            loop_residual_weight=config.loop_residual_weight,
            max_prior_translation_m=config.loop_max_prior_translation_m,
            max_prior_rotation_deg=config.loop_max_prior_rotation_deg,
        )


@dataclass(frozen=True, slots=True)
class LoopCandidate:
    """One retrieval hit that survived the metric verification.

    The two halves are kept whole rather than flattened into eleven fields: `hit` is what
    retrieval and the candidate cap decided, `estimate` is what `colsfm.loop_pose` measured, and a
    reader asking "where did this number come from" gets the answer from the field name.
    """

    hit: RetrievalHit
    """The retrieval hit this rig pair was shortlisted by."""
    estimate: RigPoseEstimate
    """The metric measurement of the rig pair and its evidence."""


@dataclass(frozen=True, slots=True)
class LoopClosureResult:
    """What `find_loop_edges` produced.

    The edges are the payload; the diagnostics exist because a loop-closure stage that
    silently returns nothing is indistinguishable from one that is switched off.
    """

    edges: list[PoseGraphEdge]
    """Verified `"loop"` edges, ready for `colsfm.pose_graph.solve_pose_graph`."""
    diagnostics: LoopClosureDiagnostics
    """Per-stage counts for the run."""
    candidates: list[LoopCandidate] = field(default_factory=list)
    """Every verified candidate, before gating. Kept for inspection."""


# ======================================================================================
# gates
# ======================================================================================


def is_good_match(num_inliers: int, num_matches: int, config: LoopClosureConfig) -> bool:
    """`DetermineValidityViaInlierMatches` for a LOOP edge (spec §6.6).

    `is_good = matched_num > min_matches_num && matched_num / matches.size() > min_matches_ratio`,
    i.e. 30 inliers and a 25 % inlier ratio, both strict.

    The blob counts two-view feature matches; this stage counts the 2-D-3-D observations of
    `colsfm.loop_pose`'s generalized resection, which is the same quantity one stage further
    on — how much of what the matcher offered the geometry actually explained.

    Args:
        num_inliers: Observations the geometry kept.
        num_matches: Observations it was offered.
        config: The loop-closure settings.

    Returns:
        True when the pair may become a loop edge.
    """
    if num_matches <= 0:
        return False
    return num_inliers > config.min_inliers and num_inliers / num_matches > config.min_inlier_ratio


def session_duration_seconds(timestamps_us: Sequence[int]) -> float:
    """`GetSessionDurationInSecond`: the span of the whole capture.

    Args:
        timestamps_us: Keyframe timestamps in microseconds.

    Returns:
        `(max - min) / 1e6`, or 0.0 for fewer than two timestamps.
    """
    if len(timestamps_us) < 2:
        return 0.0
    return (max(timestamps_us) - min(timestamps_us)) / MICROSECONDS_PER_SECOND


def minimum_time_gap_seconds(timestamps_us: Sequence[int], config: LoopClosureConfig) -> float:
    """Combine the association and pose-graph temporal gates.

    `generate_association_main` uses a fixed `loop_interval_threshold_in_seconds`;
    `pose_graph_main` uses `loop_closure_interval_ratio * session_duration`. A candidate
    must pass both, so the effective gap is the larger of the two. See the module docstring
    for why the fixed 10 s is unusable on Galileo.

    Args:
        timestamps_us: Keyframe timestamps in microseconds.
        config: The loop-closure settings.

    Returns:
        The minimum `|dt|` in seconds a candidate must have from its query.
    """
    return max(config.loop_interval_threshold_in_seconds, config.loop_closure_interval_ratio * session_duration_seconds(timestamps_us))


@dataclass(frozen=True, slots=True)
class LoopSearchPlan:
    """Everything the loop search will do, decided before a single match exists.

    The stage used to discover its own pair list by *executing* itself: it ran the
    whole search — retrieval, shortlisting, local-map triangulation, the observation
    walk — against a matcher that recorded the pairs it was asked for and returned
    nothing, threw the result away, batch-matched what it had collected, and ran the
    search a second time for real. Pair scheduling therefore depended on driving the
    geometric verification path with deliberately false input, and every future
    change to verification became a change to scheduling.

    Nothing in the first pass needed matches. Retrieval, the score and time gates,
    the candidate cap and the rig-pair deduplication read the index and the
    timestamps; the pair list follows from the rig geometry and the shortlist
    (`colsfm.loop_pairs.required_image_pairs`). So the plan is simply that work, done
    once, and `verify_loop_plan` is the only pass that matches anything.
    """

    config: LoopClosureConfig
    """The gates and thresholds the plan was made under, and will be verified under."""
    image_pairs: tuple[ImagePair, ...]
    """Every image pair the measurement will ask for, normalised, deduplicated and sorted.

    The caller batch-matches whichever of these the database does not already hold,
    which is what makes one matcher load serve the whole stage."""
    groups: tuple[tuple[int, tuple[RetrievalHit, ...]], ...]
    """The measurement's own units: one source rig frame and its shortlisted hits."""
    queried_by: Mapping[tuple[int, int], frozenset[int]]
    """Per unordered rig pair, the query rig frames that kept it among their candidates; each keeps only its best measured pair."""
    geometry: RigGeometry
    """The rig's cameras, extrinsics and stereo declarations."""
    rig_index: RigFrameIndex
    """Rig frames in capture order with their prior poses."""
    keypoints: Mapping[int, Keypoints]
    """Keypoint pixels per image id, read once — 148 MB on RoboCap."""
    counters: FunnelCounters
    """The funnel so far: retrieval's own counters, which `verify_loop_plan` finishes."""
    duration_seconds: float
    """Session length, for the diagnostics."""
    gap_seconds: float
    """The temporal gate actually applied, for the diagnostics."""
    score_threshold: float
    """The resolved retrieval score gate, for the diagnostics."""

    @property
    def rig_pairs(self) -> tuple[tuple[int, int], ...]:
        """The shortlisted `(source rig id, target rig id)` pairs, in measurement order.

        Directed as the retrieval hit that survived deduplication was directed, and
        grouped by source rig frame, because a source triangulates its local map once
        however many candidates it has. One entry per unordered rig pair. Read off
        `groups`, which is the same shortlist in the order the measurement walks it.

        Returns:
            One pair per shortlisted rig pair, in ascending source-rig order.
        """
        return tuple((source_rig_id, hit.target_rig_id) for source_rig_id, hits in self.groups for hit in hits)

    @property
    def queries(self) -> int:
        """Retrieval queries the plan issued.

        Returns:
            The count, which is final: retrieval happens once, in planning.
        """
        return self.counters.queries

    @property
    def candidates_retrieved(self) -> int:
        """Retrieval hits the index returned before any gate.

        Returns:
            The count, which is final for the same reason.
        """
        return self.counters.candidates_retrieved


def plan_loop_search(
    frames_meta: FramesMeta,
    database_path: Path,
    index: RetrievalIndex,
    config: LoopClosureConfig,
    keypoints: Mapping[int, Keypoints] | None = None,
) -> LoopSearchPlan:
    """Decide which rig pairs to measure and which image pairs that needs, matching nothing.

    The retrieval half of the funnel: hits, the score, time and same-rig gates, the
    `loop_candidates_per_frame` best target rig frames per query rig frame, then one hit per unordered rig pair. None of it reads a
    match, which is exactly why the discovery run was unnecessary.

    Args:
        frames_meta: Parsed `frames_meta.json`; supplies the rig grouping, the prior poses
            the local map is triangulated with, the rig extrinsics and the calibration.
        database_path: COLMAP database holding the keypoints. Only read when `keypoints`
            is None.
        index: Retrieval index over the same keyframe ids.
        config: Gates and thresholds.
        keypoints: Keypoint pixels per image id, already read; read here when None.

    Returns:
        The plan, which `verify_loop_plan` turns into edges.

    Raises:
        FileNotFoundError: When `database_path` does not exist and no `keypoints` were given.
    """
    keyframe_by_id: dict[int, KeyframeMeta] = frames_meta.keyframe_by_id()
    timestamps_us: dict[int, int] = {keyframe_id: keyframe.timestamp_microseconds for keyframe_id, keyframe in keyframe_by_id.items()}
    duration_seconds: float = session_duration_seconds(list(timestamps_us.values()))
    gap_seconds: float = minimum_time_gap_seconds(list(timestamps_us.values()), config)
    score_threshold: float = GOOD_SCORE_THRESHOLD if config.good_score_threshold is None else config.good_score_threshold
    counters: FunnelCounters = FunnelCounters()
    empty_geometry: RigGeometry = rig_geometry(frames_meta)

    shortlist: Shortlist = shortlist_rig_pairs(
        frames_meta,
        index,
        keyframe_by_id,
        timestamps_us,
        top_k=config.top_k,
        loop_candidates_per_frame=config.loop_candidates_per_frame,
        gap_seconds=gap_seconds,
        score_threshold=score_threshold,
        counters=counters,
    )
    pixels: Mapping[int, Keypoints] = read_keypoints_batch(database_path, index.image_ids, dtype=np.float64) if keypoints is None else keypoints
    rig_index: RigFrameIndex = rig_frame_index(frames_meta)
    shortlist = _within_prior_bound(shortlist, rig_index, config, counters)
    groups: list[tuple[int, tuple[RetrievalHit, ...]]] = _group_by_source_rig(shortlist.hits)
    rig_pairs: tuple[tuple[int, int], ...] = tuple((source_rig_id, hit.target_rig_id) for source_rig_id, hits in groups for hit in hits)
    return LoopSearchPlan(
        config=config,
        image_pairs=required_image_pairs(rig_index, empty_geometry, pixels, config.rig_pose.neighbour_span, rig_pairs),
        groups=tuple(groups),
        queried_by=shortlist.queried_by,
        geometry=empty_geometry,
        rig_index=rig_index,
        keypoints=pixels,
        counters=counters,
        duration_seconds=duration_seconds,
        gap_seconds=gap_seconds,
        score_threshold=score_threshold,
    )


def prior_distance_bound(config: LoopClosureConfig, path_m: float) -> float:
    """The largest prior distance at which two rig frames can still give a loop edge or loop pairs worth matching.

    `reach + allowance`. The reach is the edge gate T (`max_translation_m`): an accepted edge measures `|t_meas| <= T`.
    The allowance is what the prior may be off by between the two rig frames: `prior_drift_floor_m +
    prior_drift_per_metre * path_m`, from ground truth (see the fields). Where the profile sets a prior translation gate
    P, an accepted edge also needs `|t_meas - t_prior| <= P` (`agrees_with_prior`: the translation of
    `prior^-1 * meas` is `R_prior^T (t_meas - t_prior)`), so by the triangle inequality `|t_prior| <= T + P`: the
    allowance is then the smaller of the two. With closed edge gates (`max_translation_m <= 0`) no edge is accepted, the
    loop pairs feed only the mapper, and the reach is `closed_gate_reach_m`.

    On Casa Loma 007 (cube profile, T 3 m, P 1.5 m) the bound is 4.5 m and drops 6,215 of 7,210 shortlisted rig pairs
    (37,290 of 42,910 loop image pairs, 95-97 % of them empty); the largest prior distance of an accepted edge there is
    3.07 m. On the EuRoC, MSD and KITTI base runs it drops none.

    Args:
        config: The edge gate, the prior gate and the drift allowance.
        path_m: The prior path travelled between the two rig frames, metres.

    Returns:
        The bound in metres.
    """
    edge_gate_open: bool = config.max_translation_m > 0.0
    reach_m: float = config.max_translation_m if edge_gate_open else config.closed_gate_reach_m
    allowance_m: float = config.prior_drift_floor_m + config.prior_drift_per_metre * path_m
    if edge_gate_open and config.max_prior_translation_m is not None:
        allowance_m = min(allowance_m, config.max_prior_translation_m)
    return reach_m + allowance_m


def _within_prior_bound(shortlist: Shortlist, rig_index: RigFrameIndex, config: LoopClosureConfig, counters: FunnelCounters) -> Shortlist:
    """Drop the shortlisted rig pairs the prior puts beyond `prior_distance_bound`.

    Applied after the candidate cap and the deduplication, as pure subtraction: a freed slot is not refilled, so the plan
    holds a subset of the unbounded plan's rig pairs, in the same order.

    Args:
        shortlist: The deduplicated shortlist.
        rig_index: Rig frames in capture order with their prior poses.
        config: The gates and the drift allowance.
        counters: Diagnostic counters; `rejected_by_prior_bound` is added to.

    Returns:
        The shortlist without those rig pairs, `queried_by` included.
    """
    centres_xyz: Float64[ndarray, "n 3"] = np.asarray(
        [rig_index.world_T_rig[rig_id].translation for rig_id in rig_index.sequence], dtype=np.float64
    ).reshape(-1, 3)
    steps_m: Float64[ndarray, "n_steps"] = np.linalg.norm(np.diff(centres_xyz, axis=0), axis=1)
    path_from_start_m: Float64[ndarray, "n"] = np.concatenate([np.zeros(1), np.cumsum(steps_m)])
    position_of: dict[int, int] = {rig_id: position for position, rig_id in enumerate(rig_index.sequence)}
    kept: dict[tuple[int, int], RetrievalHit] = {}
    for pair, hit in shortlist.hits.items():
        source: int = position_of[hit.source_rig_id]
        target: int = position_of[hit.target_rig_id]
        distance_m: float = float(np.linalg.norm(centres_xyz[source] - centres_xyz[target]))
        path_m: float = float(abs(path_from_start_m[target] - path_from_start_m[source]))
        if distance_m <= prior_distance_bound(config, path_m):
            kept[pair] = hit
    counters.rejected_by_prior_bound += len(shortlist.hits) - len(kept)
    return Shortlist(hits=kept, queried_by={pair: voters for pair, voters in shortlist.queried_by.items() if pair in kept})


def verify_loop_plan(plan: LoopSearchPlan, match_fn: MatchFunction) -> LoopClosureResult:
    """Measure and gate a planned search; the only pass that reads a match.

    Args:
        plan: What `plan_loop_search` decided.
        match_fn: `match_fn(image_id_a, image_id_b) -> Int[ndarray, "m 2"]`, which the
            caller has by now given every pair of `plan.image_pairs` to match.

    Returns:
        The gated loop edges and the diagnostics of the whole funnel, retrieval's half
        included.
    """
    config: LoopClosureConfig = plan.config
    make_estimator: Callable[..., RigPoseEstimator] = partial(
        RigPoseEstimator, index=plan.rig_index, geometry=plan.geometry, keypoints=plan.keypoints, match_fn=match_fn, config=config.rig_pose
    )
    candidates: list[LoopCandidate] = _measure_groups(list(plan.groups), make_estimator, config, plan.counters)
    if config.max_prior_translation_m is not None or config.max_prior_rotation_deg is not None:
        consistent: list[LoopCandidate] = [candidate for candidate in candidates if agrees_with_prior(candidate, plan.rig_index.world_T_rig, config)]
        print(f"[colsfm] loop closure: {len(candidates) - len(consistent)} of {len(candidates)} verified candidates disagree with the prior pose")
        candidates = consistent

    edges: list[PoseGraphEdge] = []
    for candidate in best_per_query_rig(candidates, plan.queried_by, config.good_loop_threshold, plan.counters):
        information: Information6 = (
            loop_edge_information(float(candidate.estimate.num_inliers), config.loop_residual_weight)
            if config.use_stored_weights
            else np.eye(6, dtype=np.float64)
        )
        edges.append(
            PoseGraphEdge(
                source=candidate.hit.source_rig_id,
                target=candidate.hit.target_rig_id,
                source_T_target=candidate.estimate.source_T_target,
                information=information,
                kind="loop",
            )
        )
    gated: list[PoseGraphEdge] = gate_loop_edges(edges, max_translation_m=config.max_translation_m, max_rotation_deg=config.max_rotation_deg)
    diagnostics: LoopClosureDiagnostics = plan.counters.freeze(
        duration_seconds=plan.duration_seconds,
        gap_seconds=plan.gap_seconds,
        score_threshold=plan.score_threshold,
        edges=len(gated),
    )
    return LoopClosureResult(edges=gated, diagnostics=diagnostics, candidates=candidates)


def agrees_with_prior(candidate: LoopCandidate, world_T_rig: Mapping[int, pycolmap.Rigid3d], config: LoopClosureConfig) -> bool:
    """Whether a measured loop agrees with the prior trajectory within the prior-consistency gate.

    Args:
        candidate: A measured candidate.
        world_T_rig: Prior rig poses by rig id (`RigFrameIndex.world_T_rig`).
        config: `max_prior_translation_m` and `max_prior_rotation_deg`; a None half does not gate.

    Returns:
        True when `prior_source_T_target^-1 * measured` is inside both bounds.
    """
    prior: pycolmap.Rigid3d = world_T_rig[candidate.hit.source_rig_id].inverse() * world_T_rig[candidate.hit.target_rig_id]
    error: pycolmap.Rigid3d = prior.inverse() * candidate.estimate.source_T_target
    translation_ok: bool = config.max_prior_translation_m is None or float(np.linalg.norm(error.translation)) <= config.max_prior_translation_m
    rotation_ok: bool = config.max_prior_rotation_deg is None or float(np.rad2deg(error.rotation.angle())) <= config.max_prior_rotation_deg
    return translation_ok and rotation_ok


def find_loop_edges(
    frames_meta: FramesMeta,
    database_path: Path,
    index: RetrievalIndex,
    config: LoopClosureConfig,
    match_fn: MatchFunction,
    keypoints: Mapping[int, Keypoints] | None = None,
) -> LoopClosureResult:
    """Plan a loop search and verify it in one call.

    The convenient form for a caller whose `match_fn` can answer any pair on demand — a
    synthetic scene, a notebook. `colsfm.pipeline` uses the two halves instead, because
    COLMAP's matcher is a batch operation and the plan is what says which batch.

    Args:
        frames_meta: Parsed `frames_meta.json`.
        database_path: COLMAP database holding the keypoints `match_fn`'s indices refer to.
        index: Retrieval index over the same keyframe ids.
        config: Gates and thresholds.
        match_fn: `match_fn(image_id_a, image_id_b) -> Int[ndarray, "m 2"]`.
        keypoints: Keypoint pixels per image id, already read; read here when None.

    Returns:
        The gated loop edges and the diagnostics of the run.

    Raises:
        FileNotFoundError: When `database_path` does not exist and no `keypoints` were given.
    """
    return verify_loop_plan(plan_loop_search(frames_meta, database_path, index, config, keypoints), match_fn)


def pairs_outside_accepted_edges(plan: LoopSearchPlan, edges: Sequence[PoseGraphEdge], matched: Sequence[ImagePair]) -> list[ImagePair]:
    """The pairs of `matched` that no accepted loop edge's measurement uses: what the mapper should not see.

    Args:
        plan: The plan the edges were verified from.
        edges: The accepted loop edges, directed as their measurement was.
        matched: Image pairs the loop stage matched itself.

    Returns:
        The pairs of `matched` outside every accepted edge's local maps and observations, in `matched`'s order.
    """
    kept: set[ImagePair] = set(
        required_image_pairs(plan.rig_index, plan.geometry, plan.keypoints, plan.config.rig_pose.neighbour_span, [(edge.source, edge.target) for edge in edges])
    )
    return [pair for pair in matched if pair not in kept]


def best_per_query_rig(
    candidates: Sequence[LoopCandidate], queried_by: Mapping[tuple[int, int], frozenset[int]], good_loop_threshold: int, counters: FunnelCounters
) -> list[LoopCandidate]:
    """Each query rig frame's single best measured pair by inliers, when it has at least `good_loop_threshold` of them.

    A pair that two query rig frames both kept as candidates is kept once. The order is the candidates' own (rig-pair order), so the
    edges do not depend on the measurement's thread count.

    Args:
        candidates: The verified candidates, in rig-pair order.
        queried_by: Per unordered rig pair, the query rig frames that kept it among their candidates.
        good_loop_threshold: Inliers a query rig frame's best pair needs.
        counters: Diagnostic counters, mutated in place.

    Returns:
        The kept candidates, in the order given.
    """
    best: dict[int, LoopCandidate] = {}
    for candidate in candidates:
        for rig_id in queried_by[normalise_pair(candidate.hit.source_rig_id, candidate.hit.target_rig_id)]:
            incumbent: LoopCandidate | None = best.get(rig_id)
            if incumbent is None or candidate.estimate.num_inliers > incumbent.estimate.num_inliers:
                best[rig_id] = candidate
    kept_ids: set[int] = set()
    for candidate in best.values():
        if candidate.estimate.num_inliers >= good_loop_threshold:
            kept_ids.add(id(candidate))
        else:
            counters.below_good_loop += 1
    kept: list[LoopCandidate] = [candidate for candidate in candidates if id(candidate) in kept_ids]
    counters.not_best += len(candidates) - len({id(candidate) for candidate in best.values()})
    return kept


def _group_by_source_rig(shortlist: Mapping[tuple[int, int], RetrievalHit]) -> list[tuple[int, tuple[RetrievalHit, ...]]]:
    """Order the shortlist deterministically and gather it by source rig frame.

    The order is `(source rig id, rig pair)`, which is what the serial pass walked, so the
    candidate list is the same however many threads measure it.

    Args:
        shortlist: The best hit per unordered rig pair.

    Returns:
        One `(source rig id, hits)` group per source rig frame, in ascending source order.
    """
    groups: dict[int, list[RetrievalHit]] = {}
    for key in sorted(shortlist, key=lambda pair: (shortlist[pair].source_rig_id, pair)):
        hit: RetrievalHit = shortlist[key]
        groups.setdefault(hit.source_rig_id, []).append(hit)
    return [(source_rig_id, tuple(hits)) for source_rig_id, hits in groups.items()]


def _measure_groups(
    groups: list[tuple[int, tuple[RetrievalHit, ...]]],
    make_estimator: Callable[..., RigPoseEstimator],
    config: LoopClosureConfig,
    counters: FunnelCounters,
) -> list[LoopCandidate]:
    """Run the metric rig-to-rig estimator over the planned measurement groups.

    Pairs are grouped by source rig frame so that one source triangulates its local map once
    however many candidates it has, and the groups are measured on `config.num_threads`
    worker threads. The outcomes come back in submission order and the counters are
    accumulated here, after the join, so the result does not depend on the thread count.

    Args:
        groups: One `(source rig id, hits)` group per source rig frame, as the plan ordered
            them.
        make_estimator: Builds an estimator (already carrying the matcher that can answer
            the plan) from a `cache_images` budget. Each worker thread builds its own: an
            estimator's bearing memo is mutable, and the threads share only the immutable
            geometry, keypoints and matcher. They split `BEARING_CACHE_IMAGES` evenly.
        config: Gates and thresholds.
        counters: Diagnostic counters, mutated in place.

    Returns:
        One verified candidate per rig pair that passed every gate, in rig-pair order.
    """

    workers: int = max(1, min(config.num_threads, len(groups)))
    worker: threading.local = threading.local()

    def measure(group: tuple[int, tuple[RetrievalHit, ...]]) -> list[RigPoseOutcome]:
        """Measure one source rig frame's candidates against its own local map, on this thread's estimator."""
        estimator: RigPoseEstimator | None = getattr(worker, "estimator", None)
        if estimator is None:
            estimator = worker.estimator = make_estimator(cache_images=BEARING_CACHE_IMAGES // workers)
        source_rig_id, hits = group
        return estimator.measure_group(source_rig_id, tuple(hit.target_rig_id for hit in hits))

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outcomes_per_group: list[list[RigPoseOutcome]] = list(pool.map(measure, groups))
    else:
        outcomes_per_group = [measure(group) for group in groups]

    candidates: list[LoopCandidate] = []
    for (_, hits), outcomes in zip(groups, outcomes_per_group, strict=True):
        for hit, outcome in zip(hits, outcomes, strict=True):
            if isinstance(outcome, RigPoseRejection):
                counters.count(outcome)
                continue
            if not is_good_match(outcome.num_inliers, outcome.num_observations, config):
                counters.rejected_by_is_good += 1
                continue
            counters.verified += 1
            candidates.append(LoopCandidate(hit=hit, estimate=outcome))
    return candidates
