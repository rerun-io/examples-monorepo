"""Driving the estimator over a frameset feed in lockstep, with the D17 hold.

Offline mode consumes one frameset at a time in the calling thread, and a
frameset the estimator refuses for want of inertial samples is **held, not
dropped** (D17). That rule is the contract, not a detail of one caller: the
replay tool draws a Rerun rung from what tracked and the V2 gate asserts numbers
on it, and both have to hold and retry identically or the gate stops measuring
the tool. It therefore lives here once, and :class:`Lockstep` is what both
drive.

:func:`_drive` is the loop over that hold with nothing logged — the one the C++
reference timed — and it is here for the same reason: the V2 gate reads its
numbers and :mod:`slam_rs.apis.fleet_check` reports them from another machine, so
the two must feed the estimator identically or they are measuring different runs.
:func:`run_segment` drives it over an MSD clip and :func:`run_robocap` over a
RoboCap session, and the second exists because the fleet has to replay a
four-camera fisheye rig on a device with no viewer and no repository.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Float64
from numpy import ndarray

from slam_rs import _core
from slam_rs.catalog_feed import Frameset, LocalSegment, SegmentFeed, open_segment
from slam_rs.reference import ReferenceManifest, ReferenceSegment, flow_config
from slam_rs.trajectory import Trajectory, shift_clock

MAX_HELD_FRAMESETS: int = 2
"""Most framesets the hold may ever carry: one refused, plus the one whose samples unblock it."""


@dataclass(slots=True)
class Lockstep:
    """One estimator, driven frameset by frameset, holding a refused frameset.

    Everything a driver needs to report afterwards is on this value: what was
    pushed, what tracked and how long it took, how many retries it cost, and
    what is still held. The tracked count is ``len(elapsed_ms)``: the same branch
    that times a ``track`` call is the one that accepts its pose.
    """

    vio: _core.Vio
    """The estimator every frameset and every inertial sample goes through."""
    pending: list[Frameset] = field(default_factory=list)
    """Framesets refused for want of IMU, oldest first, waiting for the samples that cover them."""
    imu_samples: int = 0
    """Inertial samples pushed so far."""
    retries: int = 0
    """How many framesets were refused for want of IMU, and so had to be tracked a second time."""
    elapsed_ms: list[float] = field(default_factory=list)
    """Wall time each ``track`` call that tracked took, in the order they tracked."""

    def push(self, frameset: Frameset) -> Iterator[tuple[Frameset, _core.VioResult]]:
        """Push the frameset's inertial samples, then yield everything they now cover.

        A frameset the estimator refuses is **held, not dropped** (D17): the next
        frameset's batch runs one sample past its own frame time and therefore
        past this one's, so the held frameset tracks then — and before the
        frameset whose samples unblocked it, because time order is the
        trajectory. Nothing moved on the refusal, so the retry produces the pose
        a run that had the samples all along would have produced.

        Args:
            frameset: The frameset to track, with the samples since the previous one.

        Yields:
            Each frameset that tracked and what ``track`` returned for it, in
            trajectory order. A refused frameset yields nothing until its samples
            arrive.

        Raises:
            ValueError: If the hold ever carries more than
                :data:`MAX_HELD_FRAMESETS` framesets, which the feed's one-sample
                lead makes impossible.
        """
        if len(frameset.imu):
            # The feed already hands over the samples since the previous frameset,
            # running one past this frame time: a backend that integrates up to the
            # frame and blocks until it can deadlocks on the first frameset otherwise.
            self.vio.push_imu_batch(
                frameset.imu.t_ns,
                np.ascontiguousarray(frameset.imu.gyro_rad_s),
                np.ascontiguousarray(frameset.imu.accel_m_s2),
            )
            self.imu_samples += len(frameset.imu)
        self.pending.append(frameset)
        if len(self.pending) > MAX_HELD_FRAMESETS:
            # Every batch runs one sample past its own frame time and therefore
            # past the previous frameset's, so a second refusal in a row cannot
            # happen; a deeper hold would silently retain whole decoded framesets
            # (~1.8 MB each) until the end of the segment.
            raise ValueError(
                f"frameset {frameset.t_ns} takes the hold to {len(self.pending)} framesets, past the {MAX_HELD_FRAMESETS} "
                f"the feed's one-sample lead allows; held {[held.t_ns for held in self.pending]}"
            )
        while self.pending:
            held: Frameset = self.pending[0]
            started: float = time.monotonic()
            result: _core.VioResult = self.vio.track(held.t_ns, held.images)
            elapsed_ms: float = 1e3 * (time.monotonic() - started)
            if result.status != _core.VioStatus.Tracking:
                self.retries += 1
                return
            self.pending.pop(0)
            self.elapsed_ms.append(elapsed_ms)
            yield held, result


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


def _drive(feed: SegmentFeed, lockstep: Lockstep, stop_ns: int | None = None, max_framesets: int | None = None) -> SegmentRun:
    """Feed one open segment through the estimator with nothing logged, and time it.

    This is the loop the C++ reference timed, so the wall starts with the first
    frameset and not with opening the segment, and nothing between the two calls
    logs, encodes or draws. The poses come back on the absolute device clock the
    **feed** names: ``video_time`` plus ``capture_start_time_ns`` on MSD, and
    ``video_time`` unchanged on a rig that records the device clock itself.

    Args:
        feed: An open segment, already configured for its rig.
        lockstep: The estimator to drive, already built from that rig's calibration and config.
        stop_ns: Stop before a frameset past this feed timestamp; None replays the segment.
        max_framesets: Stop after this many framesets; None replays the segment.

    Returns:
        The estimated trajectory, the two counts the gate reads, and the wall time.
    """
    t_ns: list[int] = []
    positions: list[Float64[ndarray, " 3"]] = []
    quaternions: list[Float64[ndarray, " 4"]] = []
    replayed: int = 0
    # A count is asked of the feed as a time, or it fetches a window nothing here
    # reads. The in-loop breaks still make the cut, because the feed yields to
    # the end of the window that covers `stop_ns`.
    counted_ns: int | None = feed.stop_ns_after(max_framesets)
    if counted_ns is not None:
        stop_ns = counted_ns if stop_ns is None else min(stop_ns, counted_ns)
    started: float = time.monotonic()
    for frameset in feed.framesets(stop_ns):
        if max_framesets is not None and replayed >= max_framesets:
            break
        if stop_ns is not None and frameset.t_ns > stop_ns:
            break
        replayed += 1
        for _tracked, result in lockstep.push(frameset):
            pose: Float64[ndarray, " 7"] = result.world_from_rig
            t_ns.append(result.t_ns)
            # The slice is a view onto a 7-float buffer the estimator would
            # otherwise keep alive per pose, so it is copied; `np.roll` already
            # returns a new array, so the second copy would be a second one.
            positions.append(pose[0:3].copy())
            quaternions.append(np.roll(pose[3:7], 1))
    wall_s: float = time.monotonic() - started
    estimate: Trajectory = Trajectory(
        t_ns=np.array(t_ns, dtype=np.int64),
        position_m=np.array(positions, dtype=np.float64).reshape(-1, 3),
        quaternion_wxyz=np.array(quaternions, dtype=np.float64).reshape(-1, 4),
    )
    # Whatever is still held never got samples covering it, so it produced no
    # pose: that, and only that, is a lost frameset.
    return SegmentRun(estimate=shift_clock(estimate, feed.export_offset_ns), framesets=replayed, lost=len(lockstep.pending), wall_s=wall_s)


def run_segment(
    manifest: ReferenceManifest,
    segment: ReferenceSegment,
    window_s: float | None = None,
    max_framesets: int | None = None,
) -> SegmentRun:
    """Drive one MSD reference clip through :class:`slam_rs._core.Vio`.

    Args:
        manifest: The reference set, which resolves the dataset's basalt config.
        segment: Manifest entry naming the layers, the IMU model and the device's
            image safe radius.
        window_s: Stop after this many seconds of the clip; None replays it whole.
        max_framesets: Stop after this many framesets; None replays the clip.

    Returns:
        What :func:`_drive` produced over that clip.
    """
    source: LocalSegment = LocalSegment(base_rrd=segment.base_path, gt_rrd=segment.gt_path)
    feed: SegmentFeed
    with open_segment(source, segment.imu) as feed:
        lockstep: Lockstep = Lockstep(vio=_core.Vio(_core.Calibration.from_catalog(feed.cameras, feed.imu), flow_config(manifest, segment)))
        return _drive(feed, lockstep, None if window_s is None else int(window_s * 1e9), max_framesets)
