"""The V2 milestone gate: the reference clips through the whole pipeline (D14, D35, D36, D58).

Four clauses per clip, all on the absolute device clock every basalt CSV uses,
and every error through :func:`slam_rs.trajectory.ate`, whose first argument is
the estimate and therefore drives the association — each estimate pose takes the
nearest reference pose within 5 ms, as the manifest's own C++ numbers were
produced:

* every frameset resolved: one refused for want of IMU is held and tracked again
  once the samples arrive, and anything still held when the clip ends is a lost
  frameset (D17);
* against the basalt C++ trajectory fed the same decoded pixels, at most 2 cm —
  but only on clips shorter than :data:`~slam_rs.reference.PATH_BOUND_MAX_CLIP_S`
  seconds, because past that the C++ does not meet 2 cm against its own other
  precision either (D60);
* against the ``gt.csv`` sidecar, inside the C++'s own precision band on the same
  footage — its ``f32`` and ``f64`` runs, one flag apart — or within
  :data:`~slam_rs.reference.GT_BAND_RATIO` of the band's worst member, which is
  the same rule written once because the ratio is above one (D60);
* speed: the replay's own feed loop — decode plus ``track``, nothing logged, the
  loop the C++ reference timed — within 1.2x the C++ single-thread wall for the
  same footage (D58). A port several times slower is not a port of the thing,
  so this clause is never left off.

``no_divergence`` clips gate the frameset and speed clauses plus a finite,
bounded trajectory (D36) — neither error tolerance: basalt itself sits at 43 cm
and 78 cm there, and two legitimate decode paths of the same C++ estimator
already differ by 18 to 32 cm, so a tolerance would measure noise.

The lanes follow D59. The default is the **iteration set** — MIO10 whole plus
the first ten seconds of one two-camera and one four-camera clip, about 1,650
framesets — because finding out at the end of a ten-clip run that everything
failed is the way not to iterate. ``SLAM_RS_V2_ALL=1`` runs all ten whole.
Either way the clips run **shortest first** and each one's clauses are asserted
as soon as it is measured, so the first clip that misses stops the run with its
own row printed and is the one that gets fixed. ``SLAM_RS_V2_WINDOW_S=<seconds>``
cuts every clip to its first N seconds, with the C++'s own ground-truth error
recomputed over exactly that span rather than taken from the manifest's
whole-clip figure.

Whichever lane runs, it asserts **every** clip it names, both hold-outs included
(C56): a clip whose artifacts are not on this host fails the lane instead of
being stepped over, and only a host that holds none of them skips.

The tolerances live in :mod:`slam_rs.reference`, not here: they are the
milestone's verdict and S15 decides them from measurement.

The gate drives :class:`slam_rs._core.Vio` through :class:`slam_rs.tracking.Lockstep`
and the feed directly rather than the replay tool: what is gated is the pipeline
and the manifest, not the Rerun rung over them. Nothing here logs.

The measuring tests are ``slow`` and skip cleanly on a host without the corpus;
the two that check which clips the lane refuses to leave out are not — they
decide against a relocated manifest of empty files and need no NAS.
"""

import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from _pytest.outcomes import Skipped
from jaxtyping import Float64
from numpy import ndarray

from slam_rs import _core
from slam_rs.catalog_feed import LocalSegment, SegmentFeed, open_segment
from slam_rs.reference import (
    ATE_VS_CPP_CM,
    DIVERGENCE_FACTOR,
    GT_BAND_RATIO,
    MIN_TRACKED_POSES,
    PATH_BOUND_MAX_CLIP_S,
    SPEED_TOLERANCE,
    CppAte,
    ReferenceManifest,
    ReferenceSegment,
    flow_config,
    load_manifest,
)
from slam_rs.reference_bundle import BundleFile
from slam_rs.tracking import Lockstep
from slam_rs.trajectory import (
    MIN_ASSOCIATED_POSES,
    AteResult,
    Trajectory,
    ate,
    read_trajectory,
    shift_clock,
    write_trajectory,
)

SMOKE_SEGMENT: str = "msd-index__MIO_others__MIO10_short_2_panorama"
"""The 7.6 s, 412-frameset clip the iteration set starts from."""
ITERATION_SET: tuple[tuple[str, float | None], ...] = (
    (SMOKE_SEGMENT, None),
    ("msd-index__MIO_others__MIO07_mapping_easy", 10.0),
    ("msd-g2__MGO_others__MGO07_mapping_easy", 10.0),
)
"""D59's iteration set: MIO10 whole, then ten seconds each of a two-camera and a four-camera clip."""
ALL_SEGMENTS_VARIABLE: str = "SLAM_RS_V2_ALL"
"""Set to ``1`` to gate all ten reference segments whole instead of the iteration set."""
WINDOW_VARIABLE: str = "SLAM_RS_V2_WINDOW_S"
"""Set to a number of seconds to cut every gated clip to its first N seconds."""


@dataclass(slots=True, frozen=True)
class GatedClip:
    """One clip the gate measures: a reference segment, and how much of it."""

    segment: ReferenceSegment
    """The manifest entry, which carries the policy, the C++'s own error and its wall."""
    window_s: float | None
    """Seconds from the clip's start to replay; None replays the whole segment."""

    @property
    def name(self) -> str:
        """What the row calls this clip."""
        whole: str = "" if self.window_s is None else f" first {self.window_s:g} s"
        held_out: str = ", hold-out" if self.segment.reference.hold_out else ""
        return f"{self.segment.segment_id}{whole} [{self.segment.reference.gate_policy}{held_out}]"

    @property
    def expected_framesets(self) -> int:
        """About how many framesets this clip replays, which is what orders the run."""
        if self.window_s is None:
            return self.segment.capture.num_frames
        covered: float = min(1.0, self.window_s * 1e9 / self.segment.capture.duration_ns)
        return max(1, round(covered * self.segment.capture.num_frames))


@dataclass(slots=True, frozen=True)
class SegmentRun:
    """What one clip through the whole pipeline produced, on the absolute device clock."""

    estimate: Trajectory
    """Every pose the estimator reported, in frameset order."""
    framesets: int
    """Framesets replayed."""
    lost: int
    """Framesets that never produced a pose: the estimator was still waiting for
    inertial samples covering them when the clip ended."""
    wall_s: float
    """Wall time the feed loop took: decode plus ``track``, nothing logged."""


def run_segment(
    manifest: ReferenceManifest, segment: ReferenceSegment, window_s: float | None = None, max_framesets: int | None = None
) -> SegmentRun:
    """Drive one reference clip through :class:`slam_rs._core.Vio`.

    Args:
        manifest: The reference set, which resolves the dataset's basalt config.
        segment: Manifest entry naming the layers, the IMU model and the device's
            image safe radius.
        window_s: Stop after this many seconds of the clip; None replays it whole.
        max_framesets: Stop after this many framesets; None replays the clip.

    Returns:
        The estimated trajectory, the two counts the gate reads, and the wall
        time of the feed loop — which starts once the segment is open, because
        that is the loop the C++ reference timed.
    """
    source: LocalSegment = LocalSegment(base_rrd=segment.base_path, gt_rrd=segment.gt_path)
    window_ns: int | None = None if window_s is None else int(window_s * 1e9)
    t_ns: list[int] = []
    positions: list[Float64[ndarray, " 3"]] = []
    quaternions: list[Float64[ndarray, " 4"]] = []
    replayed: int = 0
    feed: SegmentFeed
    with open_segment(source, segment.imu) as feed:
        # The hold-and-retry rule is the pipeline's contract, not the tool's, so
        # the gate drives the same :class:`slam_rs.tracking.Lockstep` the replay
        # tool does (D17); what the gate does not import is the Rerun rung.
        lockstep: Lockstep = Lockstep(vio=_core.Vio(_core.Calibration.from_catalog(feed.cameras, feed.imu), flow_config(manifest, segment)))
        started: float = time.monotonic()
        for frameset in feed.framesets():
            if max_framesets is not None and replayed >= max_framesets:
                break
            if window_ns is not None and frameset.t_ns > window_ns:
                break
            replayed += 1
            for _tracked, result in lockstep.push(frameset):
                pose: Float64[ndarray, " 7"] = result.world_from_rig
                t_ns.append(result.t_ns)
                positions.append(pose[0:3].copy())
                quaternions.append(np.roll(pose[3:7], 1).copy())
        wall_s: float = time.monotonic() - started
        offset_ns: int = feed.capture_start_time_ns
    # Whatever is still held never got samples covering it, so it produced no
    # pose: that, and only that, is a lost frameset.
    lost: int = len(lockstep.pending)
    # Exports and both references are on the absolute device clock.
    estimate: Trajectory = Trajectory(
        t_ns=np.array(t_ns, dtype=np.int64),
        position_m=np.array(positions, dtype=np.float64).reshape(-1, 3),
        quaternion_wxyz=np.array(quaternions, dtype=np.float64).reshape(-1, 4),
    )
    return SegmentRun(estimate=shift_clock(estimate, offset_ns), framesets=replayed, lost=lost, wall_s=wall_s)


def extent_m(trajectory: Trajectory) -> float:
    """The diagonal of a trajectory's bounding box, metres; ``0.0`` when it has no pose."""
    if len(trajectory) == 0:
        return 0.0
    return float(np.linalg.norm(trajectory.position_m.max(axis=0) - trajectory.position_m.min(axis=0)))


def between(trajectory: Trajectory, first_ns: int, last_ns: int) -> Trajectory:
    """The poses inside a closed time span, on the trajectory's own clock."""
    keep: slice = slice(int(np.searchsorted(trajectory.t_ns, first_ns, side="left")), int(np.searchsorted(trajectory.t_ns, last_ns, side="right")))
    return Trajectory(t_ns=trajectory.t_ns[keep], position_m=trajectory.position_m[keep], quaternion_wxyz=trajectory.quaternion_wxyz[keep])


def gate_clips(manifest: ReferenceManifest) -> list[GatedClip]:
    """The clips this run gates, shortest first (D59).

    Args:
        manifest: The frozen reference set.

    Returns:
        The iteration set, or all ten segments under
        :data:`ALL_SEGMENTS_VARIABLE`, each cut to :data:`WINDOW_VARIABLE`
        seconds when that is set, in increasing frameset count.
    """
    window_s: float | None = float(os.environ[WINDOW_VARIABLE]) if WINDOW_VARIABLE in os.environ else None
    if os.environ.get(ALL_SEGMENTS_VARIABLE) == "1":
        clips: list[GatedClip] = [GatedClip(segment=segment, window_s=window_s) for segment in manifest.segments]
    else:
        clips = [
            GatedClip(segment=manifest.by_id(segment_id), window_s=window_s if window_s is not None else default_window_s)
            for segment_id, default_window_s in ITERATION_SET
        ]
    return sorted(clips, key=lambda clip: clip.expected_framesets)


@pytest.fixture(scope="module")
def manifest() -> ReferenceManifest:
    """The frozen reference set."""
    return load_manifest()


@dataclass(slots=True, frozen=True)
class References:
    """One segment's two reference trajectories, or why it cannot be gated here."""

    cpp: Trajectory
    """The basalt C++ trajectory, on the absolute device clock."""
    truth: Trajectory
    """The ``gt.csv`` sidecar, on the same clock."""


def missing_reference(manifest: ReferenceManifest, segment: ReferenceSegment) -> str | None:
    """Why this segment cannot be gated on this host, or None when it can be.

    A reason is not a licence to leave the segment out: every clip the lane names
    is measured and asserted, both hold-outs included (C56), so one absent
    artifact fails the lane. Only a host that holds none of them — a checkout
    without the NAS mount — skips, and says so.

    Args:
        manifest: The reference set the segment came from.
        segment: The segment about to be gated.

    Returns:
        The first artifact that is not on this host, or None when all three are.
    """
    if not segment.base_path.is_file():
        return f"{segment.base_path} is not mounted"
    if not segment.gt_csv.is_file():
        return f"{segment.gt_csv} is not mounted"
    resolved: BundleFile = manifest.cpp_trajectory(segment)
    return None if resolved.available else str(resolved.reason)


def references(manifest: ReferenceManifest, segment: ReferenceSegment) -> References:
    """One segment's C++ trajectory and ground-truth sidecar, on the absolute device clock.

    Args:
        manifest: The reference set the segment came from.
        segment: A segment :func:`missing_reference` has passed.

    Returns:
        Both reference trajectories, read from this host.
    """
    return References(cpp=read_trajectory(manifest.cpp_trajectory(segment).path), truth=read_trajectory(segment.gt_csv))


def cpp_gt_band_cm(clip: GatedClip, run: SegmentRun, available: References) -> tuple[float, float]:
    """The C++'s own ground-truth error on this footage, in both of its precisions (D60).

    The band is the same C++ code on the same pixels with one flag changed, and
    it is what "as accurate as basalt" can mean at all: 0.0015 cm wide on
    ``MGO14`` and 2.3 cm wide on the 410-second ``MIO14``.

    A windowed run recomputes the ``f32`` member over exactly the span it
    replayed, or a clip whose error grows late would be gated against a budget it
    never had (D59). The ``f64`` member is the manifest's whole-clip figure,
    **unscaled**: only the ``f32`` trajectory is a reference artifact, so there is
    nothing to recompute a window from. That makes a windowed band as wide as the
    whole clip's rather than tighter, which is why a window is an iteration lane
    and the milestone is read off the whole clips.

    Args:
        clip: The clip, which says whether it was cut.
        run: What the pipeline produced for it, which says what span it covers.
        available: The two reference trajectories.

    Returns:
        The C++'s ``f32`` and ``f64`` ground-truth RMSE in centimetres.
    """
    expected: CppAte = clip.segment.reference.expected_cpp_ate
    if clip.window_s is None:
        return expected.rmse_cm, expected.rmse_cm_f64
    span: Trajectory = between(available.cpp, int(run.estimate.t_ns[0]), int(run.estimate.t_ns[-1]))
    return 100.0 * ate(span, available.truth).rmse_m, expected.rmse_cm_f64


def replayed_s(run: SegmentRun) -> float:
    """Sensor seconds the run's own trajectory spans, whole clip or window."""
    return float(run.estimate.t_ns[-1] - run.estimate.t_ns[0]) * 1e-9


def cpp_wall_s(clip: GatedClip, run: SegmentRun) -> float:
    """The C++ wall for the footage this run replayed, scaled by the frameset fraction.

    Args:
        clip: The clip, which carries the whole segment's C++ wall.
        run: What the pipeline produced for it, which says how much of it ran.

    Returns:
        Seconds the C++ feed loop took over the same framesets.
    """
    return clip.segment.reference.expected_cpp_wall_s * run.framesets / clip.segment.capture.num_frames


def clip_failures(clip: GatedClip, run: SegmentRun, available: References, against_cpp: AteResult, against_gt: AteResult) -> list[str]:
    """Every V2 clause one clip fails, in the order they are stated.

    Args:
        clip: The clip, which carries the policy and the C++'s own numbers.
        run: What the pipeline produced for it.
        available: The two reference trajectories, for the divergence bound and
            the C++'s own error over a windowed span.
        against_cpp: The run's error against the basalt C++ trajectory.
        against_gt: The run's error against the ground truth.

    Returns:
        One line per failed clause; empty when the clip passes.
    """
    failures: list[str] = []
    if run.lost:
        failures.append(f"{run.lost} of {run.framesets} framesets never got the inertial samples that cover them")
    if against_cpp.n_associated < MIN_ASSOCIATED_POSES:
        failures.append(f"only {against_cpp.n_associated} poses associated with the C++ run")
    if clip.segment.reference.gate_policy == "no_divergence":
        # basalt itself is near failure here, so only a bounded run is asserted.
        if not np.isfinite(run.estimate.position_m).all():
            failures.append("a pose is not finite")
        if extent_m(run.estimate) > DIVERGENCE_FACTOR * extent_m(available.truth):
            failures.append(f"spans {extent_m(run.estimate):.1f} m against the truth's {extent_m(available.truth):.1f} m")
    else:
        # The path bound only where the C++ meets it itself: past about a hundred
        # seconds its own two precisions are 4.24 cm apart, so 2 cm there would
        # gate the clip's length (D60).
        if replayed_s(run) < PATH_BOUND_MAX_CLIP_S and against_cpp.rmse_m * 100.0 > ATE_VS_CPP_CM:
            failures.append(f"{against_cpp.rmse_m * 100:.2f} cm from the C++ trajectory, gate is {ATE_VS_CPP_CM:.0f} cm")
        # Inside the C++'s own band, or within GT_BAND_RATIO of its worst member,
        # whichever is looser — which is the second alone, because the ratio is
        # above one and the band's worst member is its upper end.
        band: tuple[float, float] = cpp_gt_band_cm(clip, run, available)
        allowed_cm: float = GT_BAND_RATIO * max(band)
        if against_gt.rmse_m * 100.0 > allowed_cm:
            failures.append(
                f"{against_gt.rmse_m * 100:.2f} cm from ground truth, gate is {GT_BAND_RATIO}x the worst of the "
                f"C++'s own band [{band[0]:.2f}, {band[1]:.2f}] cm = {allowed_cm:.2f} cm"
            )
    # Speed is a clause of every policy: a run that does not diverge but takes
    # three times as long has not matched the thing it is a port of (D58, D59).
    allowed_s: float = SPEED_TOLERANCE * cpp_wall_s(clip, run)
    if run.wall_s > allowed_s:
        failures.append(
            f"{run.wall_s:.2f} s against the C++'s {cpp_wall_s(clip, run):.2f} s, "
            f"gate is {SPEED_TOLERANCE}x = {allowed_s:.2f} s ({run.wall_s / cpp_wall_s(clip, run):.2f}x)"
        )
    return failures


@pytest.mark.slow
def test_every_gated_clip_meets_the_v2_numbers(manifest: ReferenceManifest) -> None:
    """The V2 milestone, clip by clip, shortest first, stopping at the first miss (D59).

    Each clip prints its row as soon as it is measured and is asserted
    immediately after, so a red run names the shortest clip that misses instead
    of a table produced an hour later. A green run prints the whole table, which
    is the milestone's own record.
    """
    clips: list[GatedClip] = gate_clips(manifest)
    missing: dict[str, str] = {clip.name: reason for clip in clips if (reason := missing_reference(manifest, clip.segment)) is not None}
    if len(missing) == len(clips):
        pytest.skip("no clip of this lane is on this host: " + "; ".join(missing.values()))
    # Anything short of the whole lane is a failure, not a quiet row: a run that
    # skipped a segment has not gated the milestone the lane claims (C56).
    assert not missing, "the lane asserts every clip it names, hold-outs included (C56); these are not on this host:\n" + "\n".join(
        f"  {name}: {reason}" for name, reason in missing.items()
    )
    for clip in clips:
        available: References = references(manifest, clip.segment)
        run: SegmentRun = run_segment(manifest, clip.segment, window_s=clip.window_s)
        if len(run.estimate) < MIN_TRACKED_POSES:
            pytest.fail(f"{clip.name}: {len(run.estimate)} poses over {run.framesets} framesets is not a trajectory")
        against_cpp: AteResult = ate(run.estimate, available.cpp)
        against_gt: AteResult = ate(run.estimate, available.truth)
        expected_s: float = cpp_wall_s(clip, run)
        band = cpp_gt_band_cm(clip, run, available)
        bound: str = (
            f"bound {ATE_VS_CPP_CM:.0f} cm" if replayed_s(run) < PATH_BOUND_MAX_CLIP_S else f"no bound, {replayed_s(run):.0f} s clip"
        )
        print(
            f"{clip.name}: {len(run.estimate)}/{run.framesets} tracked, "
            f"vs C++ {against_cpp.rmse_m * 100:.2f} cm ({bound}), "
            f"vs GT {against_gt.rmse_m * 100:.2f} cm (band [{band[0]:.2f}, {band[1]:.2f}], "
            f"allowed {GT_BAND_RATIO * max(band):.2f}), "
            f"wall {run.wall_s:.2f} s, C++ {expected_s:.2f} s, ratio {run.wall_s / expected_s:.2f}"
        )
        failures: list[str] = clip_failures(clip, run, available, against_cpp, against_gt)
        assert not failures, f"{clip.name}:\n" + "\n".join(failures)


def relocated(manifest: ReferenceManifest, root: Path, absent: frozenset[str] = frozenset()) -> ReferenceManifest:
    """The reference set with every artifact moved under ``root``, and the named segments' left out.

    Empty files, because what is under test is which clips the lane refuses to
    leave out — a decision it makes before it opens anything. That is also why
    this needs no NAS: a host that holds the corpus and one that does not must
    take the same decision.

    Args:
        manifest: The real reference set, whose ten segments are kept whole apart
            from where their three artifacts sit.
        root: Directory the artifacts are written into, one subdirectory each.
        absent: Segment ids to leave without any artifact at all.

    Returns:
        The same manifest against the relocated corpus.
    """
    segments: list[ReferenceSegment] = []
    for segment in manifest.segments:
        directory: Path = root / segment.segment_id
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("base.rrd", "gt.rrd", "gt.csv", "basalt_traj.csv"):
            if segment.segment_id not in absent:
                (directory / name).write_text("")
        segments.append(
            replace(
                segment,
                base_url=f"file://{directory / 'base.rrd'}",
                gt_url=f"file://{directory / 'gt.rrd'}",
                gt_csv=directory / "gt.csv",
                # Absolute, so the manifest's own package root drops out of the
                # join, and committed, so the bundle is not consulted either.
                reference=replace(segment.reference, bundle_only=False, trajectory_csv=directory / "basalt_traj.csv"),
            )
        )
    return replace(manifest, segments=tuple(segments))


def test_a_clip_whose_reference_is_missing_fails_the_lane_rather_than_leaving_it_out(
    manifest: ReferenceManifest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C56: the all-ten lane asserts all ten, so one absent artifact is a red run.

    The bug this guards passed the lane on one to nine segments: a segment whose
    trajectory was not on the host printed a line and was stepped over, and the
    run reported success without it — including for a hold-out, which is the
    whole point of having one.
    """
    monkeypatch.setenv(ALL_SEGMENTS_VARIABLE, "1")
    corpus: ReferenceManifest = relocated(manifest, tmp_path, absent=frozenset({SMOKE_SEGMENT}))
    assert len(gate_clips(corpus)) == len(manifest.segments)
    try:
        test_every_gated_clip_meets_the_v2_numbers(corpus)
    except AssertionError as refusal:
        assert SMOKE_SEGMENT in str(refusal), f"the refusal does not name the clip it could not measure: {refusal}"
        assert "C56" in str(refusal)
    except Skipped as skipped:
        # A skip here is the bug in its other shape: a red run reported green.
        pytest.fail(f"one absent artifact skipped the lane instead of failing it: {skipped}")
    else:
        pytest.fail(f"the lane passed without ever measuring {SMOKE_SEGMENT}")


def test_a_host_that_holds_no_clip_of_the_lane_skips_it(
    manifest: ReferenceManifest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Total absence — a checkout without the NAS mount — is the one skip left, and it says so."""
    monkeypatch.setenv(ALL_SEGMENTS_VARIABLE, "1")
    corpus: ReferenceManifest = relocated(manifest, tmp_path, absent=frozenset(segment.segment_id for segment in manifest.segments))
    with pytest.raises(Skipped, match="no clip of this lane is on this host"):
        test_every_gated_clip_meets_the_v2_numbers(corpus)


@pytest.mark.slow
def test_offline_mode_is_bit_reproducible(manifest: ReferenceManifest, tmp_path: Path) -> None:
    """Two runs over the same input write byte-identical CSVs (D17).

    Offline mode has no threads and no queues, so nothing about arrival order can
    reach an estimator decision; this is the assertion that says so. A hundred
    framesets of the smoke segment reach the first marginalizations, which is
    where a scheduling dependency would first show.
    """
    segment: ReferenceSegment = manifest.by_id(SMOKE_SEGMENT)
    if not segment.base_path.is_file():
        pytest.skip(f"{segment.base_path} is not mounted on this host")
    written: list[Path] = []
    for run in range(2):
        path: Path = tmp_path / f"run_{run}.csv"
        write_trajectory(path, run_segment(manifest, segment, max_framesets=100).estimate)
        written.append(path)
    assert written[0].read_bytes() == written[1].read_bytes(), "two Offline-mode runs over the same input disagreed"
    assert len(read_trajectory(written[0])) > MIN_ASSOCIATED_POSES
