"""Track catalog framesets in order, holding frames until their IMU samples arrive.

Replay and fleet measurements share this driver so they use the same frame,
inertial, and timestamp handling. Returned trajectories use the device clock.
"""

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from jaxtyping import Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from slam_rs import _core
from slam_rs.catalog_feed import DEFAULT_WINDOW_S, CameraCalib, CatalogSegment, Frameset, RigProfile, SegmentFeed, open_segment
from slam_rs.reference import (
    ImuParameters,
    ReferenceManifest,
    ReferenceSegment,
    RobocapSession,
    config_text_sha256,
    profiled_config_text,
    resolved_flow_config,
)
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
    ground_truth: Trajectory
    """Ground truth on the exported device clock."""
    median_tracker_ms: float
    """Median accepted tracker call duration."""
    config_sha256: str
    """SHA-256 of the exact config text the estimator was built from (:func:`~slam_rs.reference.config_text_sha256`).

    Taken where the text is parsed, so a tool that exports it cannot name a
    config the estimator did not read.
    """


def _drive(feed: SegmentFeed, lockstep: Lockstep, stop_ns: int | None = None, max_framesets: int | None = None, *, config_sha256: str) -> SegmentRun:
    """Track framesets with IMU hold/retry; export estimates and truth on the device clock."""
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
    return SegmentRun(
        estimate=shift_clock(estimate, feed.export_offset_ns),
        framesets=replayed,
        lost=len(lockstep.pending),
        wall_s=wall_s,
        config_sha256=config_sha256,
        ground_truth=shift_clock(feed.ground_truth_between(int(feed.frame_t_ns[0]), int(feed.frame_t_ns[-1])), feed.export_offset_ns),
        median_tracker_ms=float(np.median(lockstep.elapsed_ms)) if lockstep.elapsed_ms else float("nan"),
    )


def run_segment(
    manifest: ReferenceManifest,
    segment: ReferenceSegment,
    window_s: float | None = None,
    max_framesets: int | None = None,
    gpu: bool = False,
    profile: Literal["reference", "fast"] = "fast",
    catalog: str | None = None,
    source: CatalogSegment | None = None,
) -> SegmentRun:
    """Replay an MSD segment from the catalog with its dataset configuration."""
    source = source or CatalogSegment(catalog or manifest.catalog_url, segment.dataset_name, segment.segment_id)
    feed: SegmentFeed
    with open_segment(source, manifest.dataset(segment.dataset_name).imu) as feed:
        flow: _core.VioConfig
        config_text: str
        flow, config_text = resolved_flow_config(manifest, segment, profile=profile)
        lockstep: Lockstep = Lockstep(vio=_core.Vio(_core.Calibration.from_catalog(feed.cameras, feed.imu), flow, gpu=gpu))
        return _drive(feed, lockstep, None if window_s is None else int(window_s * 1e9), max_framesets, config_sha256=config_text_sha256(config_text))


def robocap_estimator_files(
    manifest: ReferenceManifest, profile: Literal["reference", "fast"] = "reference"
) -> tuple[_core.Calibration, _core.VioConfig, str]:
    """Read the RoboCap calibration and profiled configuration named by the manifest.

    Returns:
        Calibration, parsed VIO configuration, and the exact configuration text
        used to identify the run.
    """
    calibration: _core.Calibration = _core.Calibration.from_json((manifest.package_root / manifest.robocap.calibration).read_text())
    config_text: str = profiled_config_text(manifest.package_root / manifest.robocap.vio_config, profile, manifest.package_root / "configs/profiles")
    return calibration, _core.VioConfig.from_json(config_text), config_text


def check_calibration_matches_recording(basalt: _core.Calibration, cameras: tuple[CameraCalib, ...], imu: ImuParameters, downscale: int) -> None:
    """Require the file calibration and catalog statics to describe the same rig.

    Intrinsics, distortion, extrinsics, and IMU parameters are compared at the
    selected image scale. The estimator calibration must carry no camera time
    offset because the feed has already applied it to the frame timestamps.

    Raises:
        ValueError: If any rig parameter differs beyond its storage precision.
    """
    if basalt.camera_count != len(cameras):
        raise ValueError(f"basalt's calibration has {basalt.camera_count} cameras, the feed selected {len(cameras)}")
    expected: tuple[tuple[int, int], ...] = tuple((camera.width, camera.height) for camera in cameras)
    if tuple(basalt.resolution) != expected:
        raise ValueError(f"basalt's calibration is {list(basalt.resolution)}, the feed decodes {list(expected)} at downscale {downscale}")
    # `Calibration` exposes no intrinsics accessor, so the comparison goes through
    # the JSON round trip written by to_json.
    written: dict[str, Any] = json.loads(basalt.to_json())["value0"]
    for camera, lens, extrinsic in zip(cameras, written["intrinsics"], written["T_imu_cam"], strict=True):
        if lens["camera_type"] != camera.model:
            raise ValueError(f"cam {camera.index}: basalt's model is {lens['camera_type']!r}, the recording gives {camera.model!r}")
        values: dict[str, float] = lens["intrinsics"]
        for name, mine in (("fx", camera.fx), ("fy", camera.fy), ("cx", camera.cx), ("cy", camera.cy)):
            # float32 statics on the recording against float64 in the JSON: a
            # thousandth of a pixel is rounding, a hundredth is a different rig.
            if abs(values[name] - mine) > 1e-2:
                raise ValueError(f"cam {camera.index}: basalt's {name} is {values[name]}, the recording gives {mine} at downscale {downscale}")
        # The distortion is resolution-invariant, so it is compared as stored and
        # to float32's own precision rather than to a pixel's.
        for number, mine in enumerate(camera.distortion.tolist(), start=1):
            if abs(values[f"k{number}"] - mine) > 1e-6:
                raise ValueError(f"cam {camera.index}: basalt's k{number} is {values[f'k{number}']}, the recording gives {mine}")
        translation: Float64[ndarray, " 3"] = np.array([extrinsic["px"], extrinsic["py"], extrinsic["pz"]], dtype=np.float64)
        offset_m: float = float(np.abs(translation - camera.imu_T_cam[:3, 3]).max())
        if offset_m > 1e-4:
            raise ValueError(f"cam {camera.index}: basalt places it {1e3 * offset_m:.3f} mm from where the recording does")
        written_rotation: Rotation = Rotation.from_quat([extrinsic["qx"], extrinsic["qy"], extrinsic["qz"], extrinsic["qw"]])
        turn_deg: float = float(np.degrees((written_rotation * Rotation.from_matrix(camera.imu_T_cam[:3, :3]).inv()).magnitude()))
        if turn_deg > 1e-2:
            raise ValueError(f"cam {camera.index}: basalt turns it {turn_deg:.4f} deg from where the recording does")
    for name, mine in (
        ("imu_update_rate", imu.rate_hz),
        ("gyro_noise_std", imu.gyro_noise_std),
        ("accel_noise_std", imu.accel_noise_std),
        ("gyro_bias_std", imu.gyro_bias_std),
        ("accel_bias_std", imu.accel_bias_std),
    ):
        # A rate is one number, a noise density is one per axis and all three are
        # the same number, which is how Kalibr writes an isotropic model.
        theirs: list[float] = written[name] if isinstance(written[name], list) else [written[name]]
        if any(abs(value - mine) > 1e-12 for value in theirs):
            raise ValueError(f"basalt's {name} is {written[name]}, the manifest gives {mine}")
    if written["cam_time_offset_ns"] != 0:
        raise ValueError(
            f"basalt's calibration carries cam_time_offset_ns {written['cam_time_offset_ns']}; the feed applies that offset, so the file must not"
        )


def run_robocap(
    manifest: ReferenceManifest,
    session: RobocapSession,
    seconds: float = 0.0,
    window_s: float = DEFAULT_WINDOW_S,
    profile: Literal["reference", "fast"] = "fast",
    catalog: str | None = None,
    gpu: bool = False,
) -> SegmentRun:
    """Replay a RoboCap catalog session without logging."""
    calibration: _core.Calibration
    flow: _core.VioConfig
    calibration, flow, config_text = robocap_estimator_files(manifest, profile=profile)
    feed: SegmentFeed
    with open_segment(
        CatalogSegment(catalog or manifest.catalog_url, "robocap", session.segment_id),
        manifest.robocap.imu,
        profile=RigProfile.from_robocap(manifest.robocap),
        window_s=window_s,
    ) as feed:
        check_calibration_matches_recording(calibration, feed.cameras, manifest.robocap.imu, manifest.robocap.downscale)
        stop_ns: int | None = None if seconds <= 0.0 else int(feed.frame_t_ns[0]) + int(seconds * 1e9)
        return _drive(feed, Lockstep(vio=_core.Vio(calibration, flow, gpu=gpu)), stop_ns, config_sha256=config_text_sha256(config_text))
