"""The V2 milestone gate: the reference segments through the whole pipeline (D14, D35, D36).

Three numbers per segment, all on the absolute device clock every basalt CSV
uses, and all with the association driven by the **estimate** — each of its poses
takes the nearest reference pose within 5 ms — because that is how the manifest's
own C++ numbers were produced:

* against the basalt C++ trajectory fed the same decoded pixels, which is the
  parity claim and starts at 2 cm (D14; the ladder tightens toward 1 cm);
* against the ``gt.csv`` sidecar, which must be within 1.2x what the C++ itself
  scored on that segment;
* every frameset resolved: one refused for want of IMU is held and tracked again
  once the samples arrive, and anything still held when the segment ends is a
  lost frameset (D17).

``no_divergence`` segments gate only the frameset clause plus a finite, bounded
trajectory (D36) — neither error tolerance: basalt itself sits at 43 cm and 78 cm
there, and two legitimate decode paths of the same C++ estimator already differ
by 18 to 32 cm, so a tolerance would measure noise.

The gate drives :class:`slam_rs._core.Vio` and the feed directly rather than the
replay tool: what is gated is the pipeline and the manifest, not the Rerun rung
over them. Nothing here logs.

Everything is ``slow`` and skips cleanly when the NAS is not mounted. The default
run is the smoke segment; ``SLAM_RS_V2_ALL=1`` runs all ten.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Float64
from numpy import ndarray

from slam_rs import _core
from slam_rs.catalog_feed import Frameset, LocalSegment, SegmentFeed, open_segment
from slam_rs.reference import ReferenceManifest, ReferenceSegment, load_manifest
from slam_rs.reference_bundle import BundleFile
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
"""The segment the gate runs by default: 7.6 s, 412 framesets."""
ALL_SEGMENTS_VARIABLE: str = "SLAM_RS_V2_ALL"
"""Set to ``1`` to run every segment of the reference set instead of the smoke one."""
CPP_TOLERANCE_M: float = 0.02
"""Largest ATE RMSE against the C++ trajectory the gate accepts (D14's starting rung)."""
GT_FACTOR: float = 1.2
"""How much worse than the C++'s own ground-truth error the port may be (D14)."""
DIVERGENCE_FACTOR: float = 10.0
"""How much larger than the ground truth's extent a ``no_divergence`` run's may be."""


@dataclass(slots=True, frozen=True)
class SegmentRun:
    """What one segment through the whole pipeline produced, on the absolute device clock."""

    estimate: Trajectory
    """Every pose the estimator reported, in frameset order."""
    framesets: int
    """Framesets replayed."""
    lost: int
    """Framesets that never produced a pose: the estimator was still waiting for
    inertial samples covering them when the segment ended."""


def run_segment(segment: ReferenceSegment, max_framesets: int | None = None) -> SegmentRun:
    """Drive one reference segment through :class:`slam_rs._core.Vio`.

    Args:
        segment: Manifest entry naming the layers, the IMU model and the device's
            image safe radius.
        max_framesets: Stop after this many framesets; None replays the segment.

    Returns:
        The estimated trajectory and the two counts the gate reads.
    """
    config: _core.VioConfig = _core.VioConfig()
    config.optical_flow_image_safe_radius = segment.reference.optical_flow_image_safe_radius
    source: LocalSegment = LocalSegment(base_rrd=segment.base_path, gt_rrd=segment.gt_path)
    t_ns: list[int] = []
    positions: list[Float64[ndarray, " 3"]] = []
    quaternions: list[Float64[ndarray, " 4"]] = []
    replayed: int = 0
    pending: list[Frameset] = []
    feed: SegmentFeed
    with open_segment(source, segment.imu) as feed:
        vio: _core.Vio = _core.Vio(_core.Calibration.from_catalog(feed.cameras, feed.imu), config)
        frameset: Frameset
        for frameset in feed.framesets():
            if max_framesets is not None and replayed >= max_framesets:
                break
            replayed += 1
            if len(frameset.imu):
                vio.push_imu_batch(
                    frameset.imu.t_ns,
                    np.ascontiguousarray(frameset.imu.gyro_rad_s),
                    np.ascontiguousarray(frameset.imu.accel_m_s2),
                )
            # A frameset refused for want of IMU is held and tracked again once
            # the next batch arrives, in its own time order, as the replay tool
            # does (D17): dropping it would lose the frame and count it as a
            # tracking loss it is not.
            pending.append(frameset)
            while pending:
                result: _core.VioResult = vio.track(pending[0].t_ns, pending[0].images)
                if result.status != _core.VioStatus.Tracking:
                    break
                pending.pop(0)
                pose: Float64[ndarray, " 7"] = result.world_from_rig
                t_ns.append(result.t_ns)
                positions.append(pose[0:3].copy())
                quaternions.append(np.roll(pose[3:7], 1).copy())
        offset_ns: int = feed.capture_start_time_ns
    # Whatever is still held never got samples covering it, so it produced no
    # pose: that, and only that, is a lost frameset.
    lost: int = len(pending)
    # Exports and both references are on the absolute device clock.
    estimate: Trajectory = Trajectory(
        t_ns=np.array(t_ns, dtype=np.int64),
        position_m=np.array(positions, dtype=np.float64).reshape(-1, 3),
        quaternion_wxyz=np.array(quaternions, dtype=np.float64).reshape(-1, 4),
    )
    return SegmentRun(estimate=shift_clock(estimate, offset_ns), framesets=replayed, lost=lost)


def extent_m(trajectory: Trajectory) -> float:
    """The diagonal of a trajectory's bounding box, metres; ``0.0`` when it has no pose."""
    if len(trajectory) == 0:
        return 0.0
    return float(np.linalg.norm(trajectory.position_m.max(axis=0) - trajectory.position_m.min(axis=0)))


def gate_segments(manifest: ReferenceManifest) -> list[ReferenceSegment]:
    """The segments this run gates: all ten under :data:`ALL_SEGMENTS_VARIABLE`, else the smoke one."""
    if os.environ.get(ALL_SEGMENTS_VARIABLE) == "1":
        return list(manifest.segments)
    return [manifest.by_id(SMOKE_SEGMENT)]


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


def references(manifest: ReferenceManifest, segment: ReferenceSegment) -> References | None:
    """One segment's C++ trajectory and ground-truth sidecar, or None with a printed reason.

    None rather than a skip, because the segments are gated in one loop: a
    machine that holds nine of the ten should gate the nine rather than report
    the whole set as skipped.

    Args:
        manifest: The reference set the segment came from.
        segment: The segment about to be gated.

    Returns:
        Both references, or None when either is not on this host.
    """
    missing: str | None = None
    if not segment.base_path.is_file():
        missing = f"{segment.base_path} is not mounted"
    elif not segment.gt_csv.is_file():
        missing = f"{segment.gt_csv} is not mounted"
    else:
        resolved: BundleFile = manifest.cpp_trajectory(segment)
        if not resolved.available:
            missing = str(resolved.reason)
        else:
            return References(cpp=read_trajectory(resolved.path), truth=read_trajectory(segment.gt_csv))
    print(f"{segment.segment_id}: not gated, {missing}")
    return None


def gate_failures(
    segment: ReferenceSegment, run: SegmentRun, truth: Trajectory, against_cpp: AteResult, against_gt: AteResult
) -> list[str]:
    """Every V2 clause one segment fails, in the order they are stated.

    Args:
        segment: The manifest entry, which carries the policy and the C++'s own error.
        run: What the pipeline produced for it.
        truth: The ground-truth sidecar, for the divergence bound.
        against_cpp: The run's error against the basalt C++ trajectory.
        against_gt: The run's error against the ground truth.

    Returns:
        One line per failed clause; empty when the segment passes.
    """
    failures: list[str] = []
    if run.lost:
        failures.append(f"{run.lost} of {run.framesets} framesets never got the inertial samples that cover them")
    if against_cpp.n_associated < MIN_ASSOCIATED_POSES:
        failures.append(f"only {against_cpp.n_associated} poses associated with the C++ run")
    if segment.reference.gate_policy == "no_divergence":
        # basalt itself is near failure here, so only a bounded run is asserted.
        if not np.isfinite(run.estimate.position_m).all():
            failures.append("a pose is not finite")
        if extent_m(run.estimate) > DIVERGENCE_FACTOR * extent_m(truth):
            failures.append(f"spans {extent_m(run.estimate):.1f} m against the truth's {extent_m(truth):.1f} m")
        return failures
    if against_cpp.rmse_m > CPP_TOLERANCE_M:
        failures.append(f"{against_cpp.rmse_m * 100:.2f} cm from the C++ trajectory, gate is {CPP_TOLERANCE_M * 100:.0f} cm")
    allowed_gt_m: float = GT_FACTOR * segment.reference.expected_cpp_ate.rmse_cm / 100.0
    if against_gt.rmse_m > allowed_gt_m:
        failures.append(
            f"{against_gt.rmse_m * 100:.2f} cm from ground truth, gate is {GT_FACTOR}x "
            f"the C++'s own {segment.reference.expected_cpp_ate.rmse_cm:.2f} cm = {allowed_gt_m * 100:.2f} cm"
        )
    return failures


@pytest.mark.slow
def test_every_gated_segment_meets_the_v2_numbers(manifest: ReferenceManifest) -> None:
    """The V2 milestone, per segment and per gate policy.

    Every gated segment is measured and reported before anything is asserted, so
    one failure still produces the whole table: this is the milestone's own
    record, not only a pass or a fail.
    """
    measured: int = 0
    failures: list[str] = []
    for segment in gate_segments(manifest):
        available: References | None = references(manifest, segment)
        if available is None:
            continue
        measured += 1
        started: float = time.monotonic()
        run: SegmentRun = run_segment(segment)
        elapsed_s: float = time.monotonic() - started
        if len(run.estimate) < MIN_ASSOCIATED_POSES:
            failures.append(f"{segment.segment_id}: {len(run.estimate)} poses over {run.framesets} framesets is not a trajectory")
            print(failures[-1])
            continue
        against_cpp: AteResult = ate(run.estimate, available.cpp)
        against_gt: AteResult = ate(run.estimate, available.truth)
        print(
            f"{segment.segment_id} [{segment.reference.gate_policy}]: "
            f"vs C++ {against_cpp.rmse_m * 100:.2f} cm, vs GT {against_gt.rmse_m * 100:.2f} cm "
            f"(C++ scored {segment.reference.expected_cpp_ate.rmse_cm:.2f} cm), "
            f"{len(run.estimate)}/{run.framesets} tracked, {elapsed_s:.1f} s"
        )
        failures.extend(
            f"{segment.segment_id}: {clause}" for clause in gate_failures(segment, run, available.truth, against_cpp, against_gt)
        )
    if measured == 0:
        pytest.skip("no reference segment is on this host")
    assert not failures, "\n".join(failures)


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
        write_trajectory(path, run_segment(segment, max_framesets=100).estimate)
        written.append(path)
    assert written[0].read_bytes() == written[1].read_bytes(), "two Offline-mode runs over the same input disagreed"
    assert len(read_trajectory(written[0])) > MIN_ASSOCIATED_POSES
