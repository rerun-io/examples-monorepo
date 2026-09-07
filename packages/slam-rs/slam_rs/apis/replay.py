"""Replay one reference segment through the Rust core and log it to Rerun.

The tool is the end-to-end wiring of everything else in the package: the frozen
manifest picks the segment and the IMU noise model, the feed decodes it on the
pinned CPU ``gray8`` path, the core consumes framesets and IMU batches, and the
result is logged under paths that mirror the dataset so a run sits beside the
ground truth in one viewer.

The core is still a stub, so ``track()`` reports ``NotInitialised`` or
``NeedMoreImu`` and no pose is written. That is the point of running it now: the
plumbing, the timestamps and the ATE report are exercised before the estimator
exists, and the day it starts tracking nothing else has to change.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rerun as rr
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig

from slam_rs import _core
from slam_rs.catalog_feed import CameraCalib, Frameset, LocalSegment, SegmentFeed, open_segment
from slam_rs.reference import ReferenceManifest, ReferenceSegment, load_manifest
from slam_rs.trajectory import AteResult, Trajectory, ate, coverage, write_trajectory

SMOKE_SEGMENT: str = "msd-index__MIO_others__MIO10_short_2_panorama"
"""Default segment: the 7.6 s rotation-dominated panorama from the smoke tier."""
RUN_ENTITY: str = "/world/runs/slam_rs"
"""Where this run's estimate goes, beside the dataset's own ``/world/runs/gt``."""
IMAGE_DOWNSCALE: int = 2
"""Images are logged at half resolution: the viewer does not need full-resolution pixels to show what was fed."""


@dataclass(slots=True)
class Config:
    """Replay a reference segment through the slam-rs core."""

    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Viewer, save and headless behaviour."""
    segment: str = SMOKE_SEGMENT
    """Segment id from ``reference_segments.toml``; also names the IMU parameters used for ``--rrd``."""
    rrd: Path | None = None
    """Base-layer ``.rrd`` to replay instead of the manifest's, keeping ``--segment``'s IMU parameters."""
    gt_rrd: Path | None = None
    """Ground-truth ``.rrd`` for ``--rrd``; the manifest's own path is used when neither is given."""
    max_framesets: int | None = None
    """Stop after this many framesets; None replays the whole segment."""
    frame_stride: int = 1
    """Replay every n-th frameset. Every frame is still decoded: decimated AV1 decode is unreliable."""
    window_s: float = 60.0
    """Longest time window fetched from the catalog in one round trip."""
    output_csv: Path | None = None
    """Where the estimated trajectory is written; defaults to ``data/<segment>/slam_rs.csv``."""


def _log_calibration(cameras: tuple[CameraCalib, ...]) -> None:
    """Log the rig's static geometry so the images sit in the right place in 3D."""
    rr.log("/", rr.ViewCoordinates.RUB, static=True)
    for camera in cameras:
        entity: str = f"/world/rig_00/cam_{camera.index:02d}"
        rr.log(
            entity,
            rr.Transform3D(translation=camera.imu_T_cam[:3, 3], mat3x3=camera.imu_T_cam[:3, :3]),
            static=True,
        )
        image_from_camera: Float64[ndarray, "3 3"] = np.array(
            [[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        rr.log(
            f"{entity}/pinhole",
            rr.Pinhole(image_from_camera=image_from_camera, resolution=[camera.width, camera.height], camera_xyz=rr.ViewCoordinates.RDF),
            static=True,
        )


def _nearest_ground_truth(ground_truth: Trajectory, t_ns: int) -> int:
    """Index of the ground-truth pose closest in time to ``t_ns``."""
    return int(np.abs(ground_truth.t_ns - t_ns).argmin())


def _replay(feed: SegmentFeed, config: Config) -> Trajectory:
    """Drive the core over the feed, logging inputs and any tracked pose.

    Args:
        feed: Open segment feed.
        config: Parsed CLI options.

    Returns:
        The poses the core reported while tracking; possibly empty.
    """
    vio: _core.Vio = _core.Vio(camera_count=len(feed.cameras), min_imu_samples=1)
    imu_t_ns: Int64[ndarray, " n_samples"] = feed.imu_stream.t_ns
    cursor: int = 0
    statuses: dict[str, int] = {}
    pose_t_ns: list[int] = []
    positions: list[Float64[ndarray, " 3"]] = []
    quaternions: list[Float64[ndarray, " 4"]] = []
    started: float = time.monotonic()
    replayed: int = 0
    frameset: Frameset
    for frameset in feed.framesets():
        if config.max_framesets is not None and replayed >= config.max_framesets:
            break
        replayed += 1
        rr.set_time("video_time", duration=np.timedelta64(frameset.t_ns, "ns"))

        # Push every IMU sample up to and including the first one at or after the
        # frame time. Feeding only samples <= t_frame deadlocks a blocking backend
        # on the very first frameset, which is the bug the basalt fork hit.
        lead: int = int(np.searchsorted(imu_t_ns, frameset.t_ns, side="right"))
        lead = min(lead + 1, len(imu_t_ns))
        if lead > cursor:
            vio.push_imu_batch(
                imu_t_ns[cursor:lead],
                np.ascontiguousarray(feed.imu_stream.gyro_rad_s[cursor:lead]),
                np.ascontiguousarray(feed.imu_stream.accel_m_s2[cursor:lead]),
            )
            for sample in range(cursor, lead):
                rr.set_time("video_time", duration=np.timedelta64(int(imu_t_ns[sample]), "ns"))
                rr.log("/world/rig_00/imu_00/gyro", rr.Scalars(feed.imu_stream.gyro_rad_s[sample]))
                rr.log("/world/rig_00/imu_00/accel", rr.Scalars(feed.imu_stream.accel_m_s2[sample]))
            cursor = lead
            rr.set_time("video_time", duration=np.timedelta64(frameset.t_ns, "ns"))

        for camera, image in zip(feed.cameras, frameset.images, strict=True):
            small: UInt8[ndarray, "h w"] = np.ascontiguousarray(image[::IMAGE_DOWNSCALE, ::IMAGE_DOWNSCALE])
            rr.log(f"/world/rig_00/cam_{camera.index:02d}/pinhole/image", rr.Image(small, color_model="L"))

        if feed.ground_truth is not None:
            nearest: int = _nearest_ground_truth(feed.ground_truth, frameset.t_ns)
            quaternion_wxyz: Float64[ndarray, " 4"] = feed.ground_truth.quaternion_wxyz[nearest]
            rr.log(
                "/world/rig_00",
                rr.Transform3D(
                    translation=feed.ground_truth.position_m[nearest],
                    quaternion=rr.Quaternion(xyzw=np.roll(quaternion_wxyz, -1)),
                ),
            )

        result: _core.VioResult = vio.track(frameset.t_ns, frameset.images)
        # A PyO3 enum has no ``name`` and is unhashable (see ``_core.pyi``), so the
        # repr is both the only name it has and the only thing that keys a dict.
        status_name: str = str(result.status)
        statuses[status_name] = statuses.get(status_name, 0) + 1
        if result.status == _core.VioStatus.Tracking:
            pose: Float64[ndarray, " 7"] = result.world_from_rig
            rr.log(f"{RUN_ENTITY}/rig", rr.Transform3D(translation=pose[0:3], quaternion=rr.Quaternion(xyzw=pose[3:7])))
            rr.log(f"{RUN_ENTITY}/velocity", rr.Scalars(result.velocity))
            pose_t_ns.append(result.t_ns)
            positions.append(pose[0:3].copy())
            quaternions.append(np.roll(pose[3:7], 1).copy())

    elapsed: float = time.monotonic() - started
    print(f"{replayed} framesets in {elapsed:.1f} s ({replayed / max(elapsed, 1e-9):.1f} fps), statuses: {statuses}")
    return Trajectory(
        t_ns=np.array(pose_t_ns, dtype=np.int64),
        position_m=np.array(positions, dtype=np.float64).reshape(-1, 3),
        quaternion_wxyz=np.array(quaternions, dtype=np.float64).reshape(-1, 4),
    )


def main(config: Config) -> None:
    """Replay one segment, log it, write the trajectory and report the error.

    Args:
        config: Parsed CLI options.
    """
    manifest: ReferenceManifest = load_manifest()
    segment: ReferenceSegment = manifest.by_id(config.segment)
    source: LocalSegment = LocalSegment(
        base_rrd=config.rrd if config.rrd is not None else segment.base_path,
        gt_rrd=config.gt_rrd if config.gt_rrd is not None else (None if config.rrd is not None else segment.gt_path),
    )
    output_csv: Path = config.output_csv if config.output_csv is not None else Path("data") / segment.segment_id / "slam_rs.csv"
    print(f"replaying {segment.segment_id} ({segment.tier} tier) from {source.base_rrd}")

    with open_segment(source, segment.imu, frame_stride=config.frame_stride, window_s=config.window_s) as feed:
        print(
            f"{len(feed.cameras)} cameras, {len(feed.frame_t_ns)} framesets, {len(feed.imu_stream)} IMU samples, "
            f"{0 if feed.ground_truth is None else len(feed.ground_truth)} ground-truth poses"
        )
        _log_calibration(feed.cameras)
        estimate: Trajectory = _replay(feed, config)
        write_trajectory(output_csv, estimate)
        print(f"{len(estimate)} tracked poses -> {output_csv}")
        if len(estimate) >= 3 and feed.ground_truth is not None:
            result: AteResult = ate(feed.ground_truth, estimate)
            print(result.summary())
            print(f"coverage: {coverage(feed.ground_truth, estimate):.1%}")
        else:
            print("no ATE: the core reported no tracked poses (expected until the estimator lands)")
