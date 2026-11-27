"""Track catalog framesets in order, holding frames until their IMU samples arrive.

Replay and fleet measurements share this driver so they use the same frame,
inertial, and timestamp handling. Returned trajectories use the device clock.
"""

import itertools
import time
import warnings
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from jaxtyping import Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from slam_rs import BackendRequest, _core
from slam_rs.catalog_calibration import ImuCalib
from slam_rs.catalog_feed import DEFAULT_WINDOW_S, ROBOCAP_RIG, CameraCalib, CatalogSegment, Frameset, SegmentFeed, dataset_rig, open_segment
from slam_rs.config import SlamConfig, config_text_sha256
from slam_rs.gba_log import gba_keyframes
from slam_rs.reference import ReferenceSegment, RobocapSession, resolved_flow_config
from slam_rs.trajectory import PoseLog, Trajectory, shift_clock

MAX_HELD_FRAMESETS: int = 2
"""Most framesets the hold may ever carry: one refused, plus the one whose samples unblock it."""


class DropReason(StrEnum):
    """Why an input cannot produce a pose; values are persisted unchanged."""

    BEFORE_INITIALIZATION = "insufficient stereo support before initialization"
    IMU_ENDED = "end of stream before IMU coverage"


@dataclass(slots=True, frozen=True)
class DroppedFrameset:
    """An input that cannot produce an estimate, with its device-clock timestamp."""

    t_ns: int
    """Input timestamp."""
    reason: DropReason
    """Why the estimator cannot publish a pose."""


@dataclass(slots=True)
class Lockstep:
    """One estimator, driven frameset by frameset, holding a refused frameset.

    Everything a driver needs to report afterwards is on this value: what was
    pushed, what tracked and how long it took, how many retries it cost, and
    what is still held. The tracked count is ``len(elapsed_ms)``: one entry per
    yielded pose, including the final drain in lag mode.
    """

    vio: _core.Vio
    """The estimator every frameset and every inertial sample goes through."""
    pending: list[Frameset] = field(default_factory=list)
    """Framesets refused for want of IMU, oldest first, waiting for the samples that cover them."""
    imu_samples: int = 0
    """Inertial samples pushed so far."""
    framesets: int = 0
    """Framesets pushed so far."""
    retries: int = 0
    """How many framesets were refused for want of IMU, and so had to be tracked a second time."""
    dropped: list[DroppedFrameset] = field(default_factory=list)
    """Inputs explicitly rejected at initialization or end of stream."""
    elapsed_ms: list[float] = field(default_factory=list)
    """Wall time per yielded pose. Lag mode adds startup frontend time to the
    final drain; calls waiting for stereo do not yield a pose."""
    consumed_ms: list[float] = field(default_factory=list)
    """Wall time of every ``track`` call that consumed a frameset, posed or not (a bootstrapping or
    stereo-waiting frameset included); refusals for want of IMU are not in it."""
    _buffered: Frameset | None = None
    """The accepted frame still waiting for a lagged estimate."""
    _buffered_flow: _core.FlowFrame | None = None
    """Keypoints of that frame, before the next frontend replaces them."""
    _result_flow: _core.FlowFrame | None = None
    """Keypoints paired with the last yielded estimate in lag mode."""
    _startup_ms: float = 0.0
    """First buffered call's cost, included with the final drain's timing."""

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
        self.framesets += 1
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
            result: _core.VioResult = self.vio.track(held.t_ns, held.images, held.masks)
            elapsed_ms: float = 1e3 * (time.monotonic() - started)
            if result.status == _core.VioStatus.NeedMoreImu:
                self.retries += 1
                return
            self.consumed_ms.append(elapsed_ms)
            self.pending.pop(0)
            if result.status == _core.VioStatus.Buffered:
                self._buffered = held
                self._buffered_flow = self.vio.flow_frame()
                self._startup_ms = elapsed_ms
                continue
            completed: Frameset = held
            if self._buffered is not None:
                completed = self._buffered
                self._result_flow = self._buffered_flow
                self._buffered = held
                self._buffered_flow = self.vio.flow_frame()
            assert completed.t_ns == result.t_ns
            if result.status == _core.VioStatus.NoVisualFeatures:
                self.dropped.append(DroppedFrameset(completed.t_ns, DropReason.BEFORE_INITIALIZATION))
                continue
            self.elapsed_ms.append(elapsed_ms)
            yield completed, result

    def flow_frame(self) -> _core.FlowFrame:
        """Keypoints for the most recently yielded estimate, including lag mode."""
        frame: _core.FlowFrame | None = self._result_flow if self._result_flow is not None else self.vio.flow_frame()
        assert frame is not None, "no tracked keypoints before the first accepted frame"
        return frame

    def flush(self) -> Iterator[tuple[Frameset, _core.VioResult]]:
        """Drain the lagged estimate and report inputs left without IMU coverage."""
        for held in self.pending:
            self.dropped.append(DroppedFrameset(held.t_ns, DropReason.IMU_ENDED))
        self.pending.clear()
        started: float = time.monotonic()
        result: _core.VioResult | None = self.vio.flush()
        if result is None:
            return
        elapsed_ms: float = 1e3 * (time.monotonic() - started) + self._startup_ms
        held: Frameset | None = self._buffered
        assert held is not None and held.t_ns == result.t_ns
        self._result_flow = self._buffered_flow
        self._buffered = None
        self._buffered_flow = None
        self._startup_ms = 0.0
        if result.status == _core.VioStatus.NoVisualFeatures:
            self.dropped.append(DroppedFrameset(held.t_ns, DropReason.BEFORE_INITIALIZATION))
            return
        self.elapsed_ms.append(elapsed_ms)
        yield held, result


@runtime_checkable
class RunObserver(Protocol):
    """What :func:`drive` tells a layer writer while the run goes on (:class:`slam_rs.vio_log.VioStage`)."""

    def consumed(self, frameset: Frameset) -> None:
        """Called after each frameset's push, whether or not it yielded a pose (an inertial bootstrap attempt may have run)."""
        ...

    def tracked(self, held: Frameset, result: _core.VioResult) -> None:
        """Called for each pose, in trajectory order, while the estimator still holds that frame's snapshot."""
        ...

    def flushed(self) -> None:
        """Called once after the estimator's flush, before the final trajectory: the map is current."""
        ...


@dataclass(slots=True, frozen=True)
class SegmentRun:
    """What one clip through the whole pipeline produced, on the absolute device clock."""

    estimate: Trajectory
    """Every pose the estimator reported, in frameset order."""
    framesets: int
    """Framesets replayed."""
    lost: int
    """Framesets that produced no pose: waiting for stereo structure or IMU coverage."""
    wall_s: float
    """Wall time the feed loop took: decode plus ``track``, nothing logged."""
    ground_truth: Trajectory
    """Ground truth on the exported device clock."""
    median_tracker_ms: float
    """Median accepted tracker call duration."""
    config_sha256: str
    """SHA-256 of the exact config text the estimator was built from (:func:`~slam_rs.reference.config_text_sha256`).

    Taken where the text is parsed, so a tool that exports it cannot name a
    config the estimator did not read.
    """
    calibration_sha256: str
    """SHA-256 of the calibration JSON the estimator was built from: the recording-derived settings (gravity, readouts,
    lenses) that the config text does not hold."""
    dropped: tuple[DroppedFrameset, ...] = ()
    """Explicitly rejected input timestamps, on the exported device clock."""
    gba_failure: str | None = None
    """Why the global BA failed, when it was asked for and did: ``final`` is then the map after loops and PGO."""
    final: Trajectory | None = None
    """The final trajectory (:func:`final_trajectory`: the global BA, then re-anchoring), at the estimate's timestamps; None with the back end off, where it is the estimate."""
    map_result: _core.MapResult | None = None
    """What :meth:`slam_rs._core.Vio.finish_map` returned (the GBA's keyframes, on the estimator's clock); None with the back end off."""
    gba_keyframes: Trajectory | None = None
    """The map keyframes after the global BA (:func:`slam_rs.gba_log.gba_keyframes`), on the exported device clock; None when the GBA did not run."""


    def uncovered(self) -> int:
        """Framesets that never got the inertial samples that cover them: a run a tool refuses to export."""
        return sum(drop.reason == DropReason.IMU_ENDED for drop in self.dropped)

    def unexplained_loss(self) -> str | None:
        """Describe an incomplete run that cannot replace a published pose layer."""
        if (len(self.estimate) < 2 or self.lost != len(self.dropped)
                or len(self.estimate) + len(self.dropped) != self.framesets
                or len({drop.t_ns for drop in self.dropped}) != len(self.dropped)
                or any(not (
                    drop.reason == DropReason.BEFORE_INITIALIZATION and drop.t_ns < self.estimate.t_ns[0]
                    or drop.reason == DropReason.IMU_ENDED and drop.t_ns > self.estimate.t_ns[-1]
                ) for drop in self.dropped)):
            return f"refusing incomplete pose layer: {len(self.estimate)} poses / {self.framesets} framesets, {self.lost} missing, {len(self.dropped)} explained drops"
        return None


def reanchored(estimate: Trajectory, map_keyframes: _core.MapSnapshot | _core.MapResult | None) -> Trajectory:
    """Every raw pose re-anchored on the map's keyframes on both sides (:func:`slam_rs._core.reanchor`).

    Call it after :meth:`slam_rs._core.Vio.flush`, when every map keyframe is one of the poses. Mid-run it gives the
    corrected history so far only when ``estimate`` reaches the snapshot's newest keyframe; one it does not reach
    raises ``ValueError``.

    Args:
        estimate: The raw causal VIO poses, on the clock the estimator ran on (the map's keyframe ids are on it too).
        map_keyframes: The map after the flush, or what :meth:`slam_rs._core.Vio.finish_map` returned; None (back end
            off) returns ``estimate`` itself.

    Returns:
        One pose per raw pose, at the same timestamps.
    """
    if map_keyframes is None:
        return estimate
    final: Float64[ndarray, "n 7"] = _core.reanchor(estimate.t_ns, estimate.xyzw(), map_keyframes.keyframe_t_ns, map_keyframes.keyframe_poses)
    return Trajectory.from_xyzw(estimate.t_ns, final)


@dataclass(slots=True, frozen=True)
class FinalRun:
    """The run's final trajectory and the map it was re-anchored on."""

    final: Trajectory
    """The raw VIO re-anchored on the map's keyframes, at its timestamps; the raw VIO itself with the back end off."""
    map_result: _core.MapResult | None
    """What :meth:`slam_rs._core.Vio.finish_map` returned; None with the back end off."""
    gba_failure: str | None
    """Why the global BA failed, when it was asked for and did; ``final`` is then re-anchored on the map after loops and PGO."""


def final_trajectory(vio: _core.Vio, estimate: Trajectory, *, run_gba: bool) -> FinalRun:
    """The run's final trajectory (DESIGN §1, §2): the global BA on the map after loops and PGO, then the raw VIO re-anchored on its keyframes.

    Call it once, after the flush: the GBA writes its poses into the map. A GBA that fails costs only the GBA: the
    core returns the map as loops and PGO left it with the reason (``MapResult.gba_failure``), and this warns once.

    Args:
        vio: The flushed estimator.
        estimate: Its raw causal VIO poses, on the estimator's clock.
        run_gba: Run the global BA; without it the final run is re-anchored on the map as loops and PGO left it.

    Returns:
        The final trajectory, the map it stands on and the GBA's failure, if any.
    """
    map_result: _core.MapResult | None = vio.finish_map(run_gba=run_gba)
    gba_failure: str | None = None if map_result is None else map_result.gba_failure
    if gba_failure is not None:
        warnings.warn(f"the global BA failed ({gba_failure}); the final run is re-anchored on the map after loops and PGO", RuntimeWarning, stacklevel=2)
    return FinalRun(final=reanchored(estimate, map_result), map_result=map_result, gba_failure=gba_failure)


def track(framesets: Iterable[Frameset], lockstep: Lockstep, observer: RunObserver | None = None) -> Iterator[_core.VioResult]:
    """Push every frameset through the lockstep, then flush it: the one loop every driver of the estimator runs.

    Args:
        framesets: The framesets, in time order, each with the samples since the previous one.
        lockstep: The estimator and its IMU hold.
        observer: Told of each pose, each frameset's push and the flush (:class:`RunObserver`).

    Yields:
        Each pose's result, in trajectory order.
    """
    for frameset in framesets:
        for held, result in lockstep.push(frameset):
            yield result
            if observer is not None:
                observer.tracked(held, result)
        if observer is not None:
            observer.consumed(frameset)
    for held, result in lockstep.flush():
        yield result
        if observer is not None:
            observer.tracked(held, result)
    if observer is not None:
        observer.flushed()


def drive(
    feed: SegmentFeed,
    lockstep: Lockstep,
    stop_ns: int | None = None,
    max_framesets: int | None = None,
    *,
    config_sha256: str,
    calibration: _core.Calibration,
    run_gba: bool,
    observer: RunObserver | None = None,
) -> SegmentRun:
    """Track framesets with IMU hold/retry; export estimates, the final trajectory (:func:`final_trajectory`) and truth on the device clock.

    ``run_gba`` runs the global BA for the final trajectory; callers that score only the raw VIO pass False and never
    run it. A GBA that fails falls back as :func:`final_trajectory` says and is recorded in ``gba_failure``. An
    ``observer`` sees every frameset, every pose and the flush as they happen (the full result layer); without one
    nothing but the poses is kept.
    """
    # A count is asked of the feed as a time, or it fetches a window nothing here
    # reads. The in-loop cuts still make the count, because the feed yields to
    # the end of the window that covers `stop_ns`.
    counted_ns: int | None = feed.stop_ns_after(max_framesets)
    if counted_ns is not None:
        stop_ns = counted_ns if stop_ns is None else min(stop_ns, counted_ns)
    framesets: Iterator[Frameset] = feed.framesets(stop_ns)
    if stop_ns is not None:
        framesets = itertools.takewhile(lambda frameset: frameset.t_ns <= stop_ns, framesets)
    poses: PoseLog = PoseLog()
    started: float = time.monotonic()
    for result in track(itertools.islice(framesets, max_framesets), lockstep, observer):
        pose: Float64[ndarray, " 7"] | None = result.world_from_rig
        assert pose is not None, f"Tracking result at {result.t_ns} has no pose"
        poses.append(result.t_ns, pose[0:3], pose[[6, 3, 4, 5]])
    wall_s: float = time.monotonic() - started
    estimate: Trajectory = poses.trajectory()
    finished: FinalRun = final_trajectory(lockstep.vio, estimate, run_gba=run_gba)
    keyframes: Trajectory | None = None if finished.map_result is None else gba_keyframes(finished.map_result)
    # Both an uncovered input and a consumed pre-initialization input lack a pose.
    return SegmentRun(
        estimate=shift_clock(estimate, feed.export_offset_ns),
        final=None if finished.map_result is None else shift_clock(finished.final, feed.export_offset_ns),
        map_result=finished.map_result,
        gba_keyframes=None if keyframes is None else shift_clock(keyframes, feed.export_offset_ns),
        gba_failure=finished.gba_failure,
        framesets=lockstep.framesets,
        lost=lockstep.framesets - len(estimate),
        wall_s=wall_s,
        config_sha256=config_sha256,
        calibration_sha256=config_text_sha256(calibration.to_json()),
        ground_truth=shift_clock(feed.ground_truth_between(int(feed.frame_t_ns[0]), int(feed.frame_t_ns[-1])), feed.export_offset_ns),
        median_tracker_ms=float(np.median(lockstep.elapsed_ms)) if lockstep.elapsed_ms else float("nan"),
        dropped=tuple(DroppedFrameset(drop.t_ns + feed.export_offset_ns, drop.reason) for drop in lockstep.dropped),
    )


def raw_vio(calibration: _core.Calibration, flow: _core.VioConfig, backend: BackendRequest) -> _core.Vio:
    """Build the ``Vio`` of a run scored on the raw VIO alone: the gate, the bench and the RoboCap fleet rows.

    The back end is off whatever ``backend.enabled`` says, so its map step and loop queries never run inline on the
    tracking thread, where they would land in the timed ``track`` calls the speed gate compares with its baselines.
    """
    return _core.Vio(calibration, flow.with_backend("off"), backend=backend)


def run_segment(
    settings: SlamConfig,
    segment: ReferenceSegment,
    window_s: float | None = None,
    max_framesets: int | None = None,
    backend: BackendRequest = "auto",
    profile: Literal["reference", "fast"] = "fast",
    catalog: str | None = None,
    source: CatalogSegment | None = None,
) -> SegmentRun:
    """Replay an MSD segment from the catalog with its dataset configuration.

    Decodes on the CPU on purpose: the gate's baselines were measured on the
    single-threaded dav1d path (D28), so NVDEC would move the numbers it scores.
    """
    source = source or CatalogSegment(catalog or settings.catalog_url, segment.dataset_name, segment.segment_id)
    feed: SegmentFeed
    with open_segment(source, profile=dataset_rig(settings.dataset(segment.dataset_name)), decode_device="cpu") as feed:
        flow: _core.VioConfig
        config_text: str
        flow, config_text = resolved_flow_config(settings, segment, profile=profile)
        calibration: _core.Calibration = _core.Calibration.from_catalog(feed.cameras, feed.imu)
        lockstep: Lockstep = Lockstep(vio=raw_vio(calibration, flow, backend))
        # The gate scores the raw VIO only: no GBA, so a GBA failure cannot fail it.
        return drive(
            feed, lockstep, None if window_s is None else int(window_s * 1e9), max_framesets, config_sha256=config_text_sha256(config_text), calibration=calibration, run_gba=False
        )


def robocap_estimator_files(
    settings: SlamConfig, profile: Literal["reference", "fast"] = "reference"
) -> tuple[_core.Calibration, _core.VioConfig, str]:
    """Read the RoboCap calibration and profiled configuration named by the settings.

    Returns:
        Calibration, parsed VIO configuration, and the exact configuration text
        used to identify the run.
    """
    calibration: _core.Calibration = _core.Calibration.from_json((settings.package_root / settings.robocap.calibration).read_text())
    config_text: str = settings.robocap_config_text(profile)
    return calibration, _core.VioConfig.from_json(config_text), config_text


def check_calibration_matches_recording(basalt: _core.Calibration, cameras: tuple[CameraCalib, ...], imu: ImuCalib, downscale: int) -> None:
    """Require the file calibration and catalog statics to describe the same rig.

    Intrinsics, distortion, extrinsics, and IMU parameters are compared at the
    selected image scale. The file must carry no additional camera time offset:
    catalog samples are already aligned, and the feed only shifts their common
    time origin.

    Raises:
        ValueError: If any rig parameter differs beyond its storage precision.
    """
    if basalt.camera_count != len(cameras):
        raise ValueError(f"basalt's calibration has {basalt.camera_count} cameras, the feed selected {len(cameras)}")
    expected: tuple[tuple[int, int], ...] = tuple((camera.width, camera.height) for camera in cameras)
    if tuple(basalt.resolution) != expected:
        raise ValueError(f"basalt's calibration is {list(basalt.resolution)}, the feed decodes {list(expected)} at downscale {downscale}")
    for camera, model, values, extrinsic in zip(cameras, basalt.camera_models, basalt.intrinsics, basalt.imu_T_cam, strict=True):
        if model != camera.model:
            raise ValueError(f"cam {camera.index}: basalt's model is {model!r}, the recording gives {camera.model!r}")
        for name, theirs, mine in zip(("fx", "fy", "cx", "cy"), values[:4], (camera.fx, camera.fy, camera.cx, camera.cy), strict=True):
            # Catalog statics use float32; a hundredth of a pixel means another rig.
            if abs(theirs - mine) > 1e-2:
                raise ValueError(f"cam {camera.index}: basalt's {name} is {theirs}, the recording gives {mine} at downscale {downscale}")
        # Distortion is resolution-invariant; compare at float32 storage precision.
        for number, (theirs, mine) in enumerate(zip(values[4:], camera.distortion.tolist(), strict=True), start=1):
            if abs(theirs - mine) > 1e-6:
                raise ValueError(f"cam {camera.index}: basalt's k{number} is {theirs}, the recording gives {mine}")
        offset_m: float = float(np.abs(extrinsic[:3, 3] - camera.imu_T_cam[:3, 3]).max())
        if offset_m > 1e-4:
            raise ValueError(f"cam {camera.index}: basalt places it {1e3 * offset_m:.3f} mm from where the recording does")
        written_rotation: Rotation = Rotation.from_matrix(extrinsic[:3, :3])
        turn_deg: float = float(np.degrees((written_rotation * Rotation.from_matrix(camera.imu_T_cam[:3, :3]).inv()).magnitude()))
        if turn_deg > 1e-2:
            raise ValueError(f"cam {camera.index}: basalt turns it {turn_deg:.4f} deg from where the recording does")
    for name, theirs, mine in (
        ("imu_update_rate", [basalt.imu_update_rate], imu.frequency_hz),
        ("gyro_noise_std", basalt.gyro_noise_std, imu.gyro_noise_std),
        ("accel_noise_std", basalt.accel_noise_std, imu.accel_noise_std),
        ("gyro_bias_std", basalt.gyro_bias_std, imu.gyro_bias_std),
        ("accel_bias_std", basalt.accel_bias_std, imu.accel_bias_std),
    ):
        if any(abs(value - mine) > 1e-12 for value in theirs):
            raise ValueError(f"basalt's {name} is {theirs}, the catalog gives {mine}")
    if basalt.cam_time_offset_ns != 0:
        raise ValueError(
            f"basalt's calibration carries cam_time_offset_ns {basalt.cam_time_offset_ns}; the feed applies that offset, so the file must not"
        )


def run_robocap(
    settings: SlamConfig,
    session: RobocapSession,
    seconds: float = 0.0,
    window_s: float = DEFAULT_WINDOW_S,
    profile: Literal["reference", "fast"] = "fast",
    catalog: str | None = None,
    backend: BackendRequest = "auto",
) -> SegmentRun:
    """Replay a RoboCap catalog session without logging.

    Decodes on the CPU on purpose: a fleet row compares this machine's
    trajectory with other machines' and with a CPU-decoded regression
    reference, so every machine must see the same pixels; NVDEC rounds them
    differently.
    """
    calibration: _core.Calibration
    flow: _core.VioConfig
    calibration, flow, config_text = robocap_estimator_files(settings, profile=profile)
    feed: SegmentFeed
    with open_segment(
        CatalogSegment(catalog or settings.catalog_url, "robocap", session.segment_id),
        profile=ROBOCAP_RIG,
        window_s=window_s,
        decode_device="cpu",
    ) as feed:
        check_calibration_matches_recording(calibration, feed.cameras, feed.imu, ROBOCAP_RIG.downscale)
        stop_ns: int | None = None if seconds <= 0.0 else int(feed.frame_t_ns[0]) + int(seconds * 1e9)
        # A fleet row scores the raw VIO only, and RoboCap runs on Mali, where no back end runs until a probe-backed run
        # exists (DESIGN Q7): the back end stays off here, as robocap-live builds without it.
        return drive(feed, Lockstep(vio=raw_vio(calibration, flow, backend)), stop_ns, config_sha256=config_text_sha256(config_text), calibration=calibration, run_gba=False)
