"""End-to-end single-sequence ARKitScenes ingestion CLI.

Image, depth, confidence, calibration, and camera poses are orientation-baked.
IMU acceleration and gyro Scalars stay in the original device frame because
they are frame-agnostic scalar channels, not spatial vector archetypes.
"""

import csv
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
from PIL import Image
from rich.console import Console
from rich.progress import Progress, TaskID
from scipy.spatial.transform import Rotation
from simplecv.camera_parameters import Extrinsics, PinholeParameters
from simplecv.camera_parameters import Intrinsics as SimpleCVIntrinsics
from simplecv.rerun_custom_types import CameraDistortion as RerunCameraDistortion
from simplecv.rerun_custom_types import PinholeWithDistortion

from arkitscenes_download.ingest.blueprint import DEPTH_RANGE_MM, make_blueprint
from arkitscenes_download.ingest.clock import (
    CLOCK_ACCEPTANCE_FRACTION,
    ClockAlignment,
    clamped_to_epoch,
    path_timestamps,
    recover_clock_from_packets,
    shared_epoch,
)
from arkitscenes_download.ingest.depth import ArkitDepthConfidence, encode_depth_png, sorted_timestamped_paths
from arkitscenes_download.ingest.distortion import AppleDistortionProvenance, OpenCVRationalFit, opencv_rational_from_apple
from arkitscenes_download.ingest.gt import ArkitMeshSummary, log_arkit_mesh, log_gt_boxes
from arkitscenes_download.ingest.imu import ImuSamples, decode_imu
from arkitscenes_download.ingest.metadata import (
    CameraDistortion,
    CameraExtrinsics,
    PoseSelection,
    decode_camera_distortions_from_packets,
    decode_ultrawide_extrinsics_from_packets,
    select_pose_source,
)
from arkitscenes_download.ingest.mov import (
    MetadataPacket,
    VideoPacketSamples,
    VideoSamples,
    demux_metadata_streams,
    iter_video_samples,
    prepare_video_track,
    track_packet_times,
)
from arkitscenes_download.ingest.paths import (
    CAM_ULTRAWIDE,
    CAM_WIDE,
    CONFIDENCE,
    DEPTH,
    IMU,
    IMU_ACCEL,
    IMU_ATTITUDE,
    IMU_GYRO,
    PINHOLE_ULTRAWIDE,
    PINHOLE_WIDE,
    PINHOLE_WIDE_LOWRES,
    RIG,
    SKY_ANGLE_ULTRAWIDE,
    SKY_ANGLE_WIDE,
    TIMELINE,
    VIDEO_ULTRAWIDE,
    VIDEO_WIDE,
    WORLD,
)
from arkitscenes_download.ingest.recording import atomic_recording
from arkitscenes_download.ingest.rig import (
    DeviceFrameFit,
    Intrinsics,
    Trajectory,
    fit_original_camera_from_device,
    interpolate_sky_angles,
    load_intrinsics,
    load_trajectory,
    measured_orientation_quarter_turns,
    orientation_ambiguity,
    resample_trajectory,
    rotate_intrinsics_sequence,
    rotate_pixels,
    rotate_resolution,
    rotate_trajectory,
    scale_intrinsics,
    sky_angles,
)
from arkitscenes_download.schema import REQUIRED_LAYER_NAMES

BATCH_SIZE: int = 1000
CONSOLE: Console = Console(markup=False)


def verify_rrd(path: Path | list[Path]) -> str:
    """Run the Rerun CLI's cheap structural validity check."""
    paths: list[Path] = [path] if isinstance(path, Path) else path
    result: subprocess.CompletedProcess[str] = subprocess.run(
        ["rerun", "rrd", "verify", *(str(item) for item in paths)], check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        if len(paths) > 1:
            for item in paths:
                single: subprocess.CompletedProcess[str] = subprocess.run(
                    ["rerun", "rrd", "verify", str(item)], check=False, capture_output=True, text=True
                )
                if single.returncode != 0:
                    raise ValueError(f"RRD verification failed for {item}: {single.stderr.strip()}")
            return "ok"
        raise ValueError(f"RRD verification failed for {paths}: {result.stderr.strip()}")
    return result.stdout.strip() or "ok"


@dataclass(frozen=True, slots=True)
class Config:
    """Configuration for ingesting one ARKitScenes sequence."""

    video_id: str = "47332195"
    """ARKitScenes sequence identifier."""
    data_dir: Path = Path("data")
    """Dataset root containing raw/Training."""
    output: Path = Path("data/rrd")
    """Root for eight independently published ``<video_id>/<layer>.rrd`` files."""
    spawn: bool = False
    """Spawn a viewer after saving (disabled for headless use)."""
    keep_transcode_cache: bool = False
    """Keep prepared MP4 tracks for faster repeated development ingests."""


def _send_columns(recording: rr.RecordingStream, path: str, timestamps: np.ndarray, columns: rr.ComponentColumnList, epoch: float) -> None:
    """Send component columns on the shared duration timeline, rebased to the epoch."""
    rr.send_columns(path, indexes=[rr.TimeColumn(TIMELINE, duration=timestamps - epoch)], columns=columns, recording=recording)


def _log_intrinsics(recording: rr.RecordingStream, path: str, intrinsics: Intrinsics, resolution: tuple[int, int], epoch: float) -> None:
    """Log timestamped camera calibration from the epoch onward."""
    keep: np.ndarray = intrinsics.timestamps >= epoch
    timestamps: np.ndarray = clamped_to_epoch(intrinsics.timestamps[keep], epoch)
    resolutions: list[tuple[int, int]] = [resolution] * len(timestamps)
    _send_columns(recording, path, timestamps, rr.Pinhole.columns(image_from_camera=intrinsics.matrices[keep], resolution=resolutions), epoch)


def _log_video(recording: rr.RecordingStream, path: str, video: VideoSamples, clock_offset: float, epoch: float, description: str) -> np.ndarray:
    """Log one prepared track in bounded raw AV1 packet batches."""
    timestamp_batches: list[np.ndarray] = []
    recording.log(path, rr.VideoStream(codec=rr.VideoCodec.AV1), static=True)
    recording.log(path, rr.AnyValues(encoder=video.encoder, encoder_settings=video.settings, ffmpeg_version=video.ffmpeg_version), static=True)
    first_batch: bool = True
    try:
        with Progress(console=CONSOLE, transient=True) as progress:
            task_id: TaskID = progress.add_task(description, total=None)
            for samples in iter_video_samples(video):
                aligned: np.ndarray = samples.timestamps + clock_offset
                if first_batch:
                    aligned = clamped_to_epoch(aligned, epoch)
                    first_batch = False
                samples = VideoPacketSamples(aligned, samples.payloads, samples.is_keyframes)
                if len(samples.timestamps) > 1 and not np.all(np.diff(samples.timestamps) > 0.0):
                    raise ValueError("clock-aligned VideoStream timestamps are not strictly increasing")
                _send_columns(recording, path, samples.timestamps, rr.VideoStream.columns(sample=samples.payloads, is_keyframe=samples.is_keyframes), epoch)  # pyrefly: ignore  # bad-argument-type
                timestamp_batches.append(samples.timestamps)
                progress.advance(task_id, len(samples.timestamps))
    finally:
        if video.delete_after_use:
            video.path.unlink(missing_ok=True)
    return np.concatenate(timestamp_batches)


def _log_sky_angles(recording: rr.RecordingStream, path: str, timestamps: np.ndarray, angles: np.ndarray, epoch: float) -> None:
    """Log per-frame clockwise sky orientation for one camera."""
    _send_columns(recording, path, timestamps, rr.Scalars.columns(scalars=angles), epoch)


def _encode_depth_asset(item: tuple[Path, int]) -> bytes:
    """Decode, orientation-bake, and encode one depth image."""
    asset, quarter_turns = item
    with Image.open(asset) as image:
        depth_hw: np.ndarray = np.asarray(image)
    rotated_hw: np.ndarray = np.rot90(depth_hw, quarter_turns)
    return encode_depth_png(rotated_hw)


def _decode_confidence_asset(item: tuple[Path, int]) -> np.ndarray:
    """Decode and orientation-bake one confidence image."""
    asset, quarter_turns = item
    with Image.open(asset) as image:
        labels_hw: np.ndarray = np.asarray(image, dtype=np.uint8)
    return np.rot90(labels_hw, quarter_turns)


def _log_baked_columns(
    recording: rr.RecordingStream,
    path: str,
    paths: list[Path],
    quarter_turns: int,
    pool: ThreadPoolExecutor,
    description: str,
    worker: Callable[[tuple[Path, int]], object],
    make_columns: Callable[[list[object]], rr.ComponentColumnList],
    epoch: float,
) -> None:
    """Decode orientation-baked assets and send ordered column batches from the epoch onward."""
    all_timestamps: np.ndarray = path_timestamps(paths)
    kept_paths: list[Path] = [asset for asset, timestamp in zip(paths, all_timestamps, strict=True) if timestamp >= epoch]
    if not kept_paths:
        return
    timestamps: np.ndarray = clamped_to_epoch(all_timestamps[all_timestamps >= epoch], epoch)
    with Progress(console=CONSOLE, transient=True) as progress:
        task_id: TaskID = progress.add_task(description, total=len(kept_paths))
        for start in range(0, len(kept_paths), BATCH_SIZE):
            batch_paths: list[Path] = kept_paths[start : start + BATCH_SIZE]
            items: list[tuple[Path, int]] = [(asset, quarter_turns) for asset in batch_paths]
            values: list[object] = list(pool.map(worker, items))
            _send_columns(recording, path, timestamps[start : start + len(batch_paths)], make_columns(values), epoch)
            progress.advance(task_id, len(batch_paths))


def _log_depth(recording: rr.RecordingStream, path: str, paths: list[Path], quarter_turns: int, pool: ThreadPoolExecutor, description: str, epoch: float) -> None:
    """Decode, orientation-bake, and encode depth PNGs in ordered batches."""

    def make_columns(values: list[object]) -> rr.ComponentColumnList:
        blobs: list[bytes] = [value for value in values if isinstance(value, bytes)]
        # TODO(rerun#upstream): drop depth_range once the viewer auto-ranges encoded depth (see DEPTH_RANGE_MM in ingest/blueprint.py).
        return rr.EncodedDepthImage.columns(
            blob=blobs, media_type=["image/png"] * len(blobs), meter=[1000.0] * len(blobs), depth_range=[DEPTH_RANGE_MM] * len(blobs)
        )

    _log_baked_columns(recording, path, paths, quarter_turns, pool, description, _encode_depth_asset, make_columns, epoch)


def _log_confidence(recording: rr.RecordingStream, paths: list[Path], quarter_turns: int, pool: ThreadPoolExecutor, description: str, epoch: float) -> None:
    """Decode confidence labels and log them as segmentation images."""
    path: str = CONFIDENCE
    recording.log(
        path,
        rr.AnnotationContext(
            [
                rr.AnnotationInfo(id=ArkitDepthConfidence.LOW, label="low", color=(255, 0, 0)),
                rr.AnnotationInfo(id=ArkitDepthConfidence.MEDIUM, label="medium", color=(255, 255, 0)),
                rr.AnnotationInfo(id=ArkitDepthConfidence.HIGH, label="high", color=(0, 255, 0)),
            ]
        ),
        static=True,
    )

    def make_columns(values: list[object]) -> rr.ComponentColumnList:
        labels: list[np.ndarray] = [value for value in values if isinstance(value, np.ndarray)]
        # Raw buffers carry no shape: without an explicit ImageFormat column the
        # viewer cannot interpret the pixels and the view renders empty.
        formats: list[rr.datatypes.ImageFormat] = [
            rr.datatypes.ImageFormat(width=label.shape[1], height=label.shape[0], channel_datatype="U8") for label in labels
        ]
        return rr.SegmentationImage.columns(buffer=labels, format=formats)

    _log_baked_columns(recording, path, paths, quarter_turns, pool, description, _decode_confidence_asset, make_columns, epoch)


def _log_rig_grammar(recording: rr.RecordingStream) -> None:
    """Log the static entity grammar exclusively in the base layer."""
    recording.log(WORLD, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    recording.log(RIG, rr.AnyValues(schema_version="arkitscenes:v2", reference="cam_00", num_cameras=2), static=True)
    recording.log(CAM_WIDE, rr.AnyValues(name="wide", kind="rgb"), static=True)
    recording.log(CAM_ULTRAWIDE, rr.AnyValues(name="ultrawide", kind="rgb", extrinsics_source="mebx"), static=True)
    recording.log(IMU, rr.AnyValues(name="imu", kind="imu"), static=True)


def _log_calibration_rig(
    recording: rr.RecordingStream, trajectory: Trajectory, extrinsics: CameraExtrinsics, rig_from_device: Rotation, epoch: float
) -> None:
    """Log transforms exclusively in the calibration layer."""
    _send_columns(
        recording,
        RIG,
        trajectory.timestamps,
        rr.Transform3D.columns(translation=trajectory.translation, quaternion=trajectory.quaternion_xyzw),
        epoch,
    )
    recording.log(CAM_WIDE, rr.Transform3D(translation=(0.0, 0.0, 0.0), quaternion=rr.Quaternion(xyzw=[0.0, 0.0, 0.0, 1.0])), static=True)
    recording.log(CAM_ULTRAWIDE, rr.Transform3D(translation=extrinsics.translation_m, quaternion=rr.Quaternion(xyzw=extrinsics.quaternion_xyzw)), static=True)
    recording.log(IMU, rr.Transform3D(quaternion=rr.Quaternion(xyzw=rig_from_device.as_quat())), static=True)
    recording.log(IMU, rr.AnyValues(values_frame="CoreMotion device; entity transform maps values into baked rig frame"), static=True)


def _log_distortions(
    recording: rr.RecordingStream,
    distortions: list[CameraDistortion],
    pinhole_intrinsics_by_camera: dict[str, SimpleCVIntrinsics],
    quarter_turns: int,
) -> None:
    """Log a standard fit plus exact first-sample Apple provenance statically."""
    camera_paths: dict[str, str] = {
        "wide": PINHOLE_WIDE,
        "ultrawide": PINHOLE_ULTRAWIDE,
    }
    for distortion in distortions:
        dimensions_wh: tuple[int, int] = tuple(int(value) for value in distortion.reference_dimensions_wh)  # type: ignore[assignment]
        center_xy: np.ndarray = rotate_pixels(distortion.center_xy, dimensions_wh, quarter_turns)
        baked_dimensions_wh: tuple[int, int] = rotate_resolution(dimensions_wh, quarter_turns)
        path: str = camera_paths[distortion.camera_name]
        pinhole_intrinsics: SimpleCVIntrinsics = pinhole_intrinsics_by_camera[distortion.camera_name]
        fit: OpenCVRationalFit = opencv_rational_from_apple(
            np.asarray(pinhole_intrinsics.k_matrix, dtype=np.float64),
            distortion.coefficients,
            center_xy,
            baked_dimensions_wh,
            (pinhole_intrinsics.width, pinhole_intrinsics.height),
        )
        camera: PinholeParameters = PinholeParameters(
            name=distortion.camera_name,
            extrinsics=Extrinsics(cam_R_world=np.eye(3, dtype=np.float64), cam_t_world=np.zeros(3, dtype=np.float64)),
            intrinsics=pinhole_intrinsics,
            distortion=fit.distortion,
        )
        standard_components: RerunCameraDistortion | None = PinholeWithDistortion.from_camera(camera).distortion
        if standard_components is None:
            raise AssertionError("Brown-Conrady fit did not produce Rerun distortion components")
        recording.log(
            path,
            standard_components,
            static=True,
        )
        recording.log(
            path,
            AppleDistortionProvenance(distortion.coefficients, distortion.inverse_coefficients),
            static=True,
        )
        recording.log(
            path,
            rr.AnyValues(
                distortion_center_xy=center_xy.tolist(),
                reference_dimensions_wh=list(baked_dimensions_wh),
                distortion_source="mebx_stream_10",
                distortion_temporal_sample_count=distortion.temporal_sample_count,
            ),
            static=True,
        )
        CONSOLE.print(
            f"Distortion fit {distortion.camera_name}: model=brown_conrady center=pinhole_principal_point "
            f"max_residual={fit.max_residual_px:.6f}px"
        )


def _log_imu(recording: rr.RecordingStream, imu: ImuSamples, epoch: float) -> None:
    """Log raw inertial streams and decoded motion attitude from the epoch onward."""
    accel_keep: np.ndarray = imu.accel_timestamps >= epoch
    gyro_keep: np.ndarray = imu.gyro_timestamps >= epoch
    motion_keep: np.ndarray = imu.motion_timestamps >= epoch
    _send_columns(recording, IMU_ACCEL, imu.accel_timestamps[accel_keep], rr.Scalars.columns(scalars=imu.accel_ms2[accel_keep]), epoch)
    _send_columns(recording, IMU_GYRO, imu.gyro_timestamps[gyro_keep], rr.Scalars.columns(scalars=imu.gyro_rads[gyro_keep]), epoch)
    _send_columns(recording, IMU_ATTITUDE, imu.motion_timestamps[motion_keep], rr.Scalars.columns(scalars=imu.attitude_xyzw[motion_keep]), epoch)


def ingest_sequence(config: Config) -> Path:
    """Write eight layer RRDs for one raw sequence and return its output directory."""
    ingest_started: float = time.perf_counter()
    candidates: list[Path] = [config.data_dir / "raw" / split / config.video_id for split in ("Training", "Validation")]
    sequence_dir: Path = next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])
    if not sequence_dir.is_dir():
        raise FileNotFoundError(f"sequence {config.video_id} not found in {candidates[0].parent.parent}")
    mov_path: Path = sequence_dir / f"{config.video_id}.mov"
    metadata_packets: dict[int, list[MetadataPacket]] = demux_metadata_streams(mov_path, (1, 3, 4, 5, 10))
    lowres_depth: list[Path] = sorted_timestamped_paths(sequence_dir / "lowres_depth")
    alignment: ClockAlignment = recover_clock_from_packets(metadata_packets[1], metadata_packets[3], lowres_depth)
    if alignment.within_two_ms_fraction < CLOCK_ACCEPTANCE_FRACTION:
        raise ValueError(f"clock validation failed: only {alignment.within_two_ms_fraction:.3%} within 2 ms")

    CONSOLE.print(
        f"Clock: wide={alignment.offset_seconds:.9f}s ultra={alignment.ultrawide_offset_seconds:.9f}s "
        f"dispersion={alignment.dispersion_seconds * 1e3:.3f}ms drift={alignment.drift_seconds_per_second * 1e6:.3f}us/s "
        f"agreement={alignment.ultrawide_agreement_seconds * 1e3:.3f}ms depth_max={alignment.max_depth_residual_seconds * 1e3:.3f}ms"
    )
    with (config.data_dir / "raw" / "metadata.csv").open(newline="") as metadata_file:
        metadata: dict[str, str] = next(row for row in csv.DictReader(metadata_file) if row["video_id"] == config.video_id)
    sky_direction: str = metadata["sky_direction"]
    trajectory_sparse_unbaked: Trajectory = load_trajectory(sequence_dir / "lowres_wide.traj")
    imu: ImuSamples = decode_imu(mov_path)
    lowres_intrinsics_paths: list[Path] = sorted((sequence_dir / "lowres_wide_intrinsics").glob("*.pincam"))
    lowres_intrinsics: Intrinsics = load_intrinsics(lowres_intrinsics_paths)
    wide_intrinsics: Intrinsics = scale_intrinsics(lowres_intrinsics, 7.5)
    trajectory_unbaked: Trajectory = resample_trajectory(trajectory_sparse_unbaked, lowres_intrinsics.timestamps)
    unbaked_sky_angles: np.ndarray = sky_angles(trajectory_unbaked.quaternion_xyzw)
    quarter_turns: int = measured_orientation_quarter_turns(unbaked_sky_angles)
    orientation_is_ambiguous, circular_spread, boundary_distance = orientation_ambiguity(unbaked_sky_angles)
    warning: str = " WARNING: AMBIGUOUS ORIENTATION" if orientation_is_ambiguous else ""
    CONSOLE.print(
        f"Orientation: label={sky_direction}, source=measured_gravity, quarter_turns={quarter_turns}, "
        f"spread={np.rad2deg(circular_spread):.2f}deg boundary_distance={np.rad2deg(boundary_distance):.2f}deg{warning}"
    )
    ultrawide_intrinsics: Intrinsics = load_intrinsics(sorted((sequence_dir / "ultrawide_intrinsics").glob("*.pincam")))
    wide_intrinsics, wide_resolution = rotate_intrinsics_sequence(wide_intrinsics, (1920, 1440), quarter_turns)
    lowres_intrinsics, lowres_resolution = rotate_intrinsics_sequence(lowres_intrinsics, (256, 192), quarter_turns)
    ultrawide_intrinsics, ultrawide_resolution = rotate_intrinsics_sequence(ultrawide_intrinsics, (640, 480), quarter_turns)
    trajectory_sparse: Trajectory = rotate_trajectory(trajectory_sparse_unbaked, quarter_turns)
    extrinsics: CameraExtrinsics = decode_ultrawide_extrinsics_from_packets(metadata_packets[5])
    baked_from_original: Rotation = Rotation.from_euler("z", -quarter_turns * np.pi / 2.0)
    baked_extrinsics_rotation: Rotation = baked_from_original * Rotation.from_quat(extrinsics.quaternion_xyzw) * baked_from_original.inv()
    extrinsics = CameraExtrinsics(baked_from_original.apply(extrinsics.translation_m), baked_extrinsics_rotation.as_quat())
    pose_selection: PoseSelection = select_pose_source(mov_path, trajectory_sparse_unbaked, metadata_packets[4])
    pose_trajectory_unbaked: Trajectory = pose_selection.trajectory
    pose_source: str = pose_selection.source
    device_fit: DeviceFrameFit = fit_original_camera_from_device(trajectory_sparse_unbaked, imu.accel_timestamps, imu.accel_ms2)
    rig_from_device: Rotation = Rotation.from_euler("z", quarter_turns * np.pi / 2.0) * Rotation.from_quat(device_fit.quaternion_xyzw)
    CONSOLE.print(f"IMU frame: original_cam_from_device={device_fit.quaternion_xyzw.tolist()} residual={np.rad2deg(device_fit.rms_residual_rad):.3f}deg")
    sequence_output: Path = config.output / config.video_id
    sequence_output.mkdir(parents=True, exist_ok=True)

    # Rebase the shared timeline so t=0 is the first instant BOTH cameras have
    # a frame: leading frames of the earlier camera are trimmed at transcode
    # time (the encoder restarts on a keyframe), and every other stream drops
    # its pre-epoch samples so nothing precedes the first visible frame.
    wide_source_times: np.ndarray = track_packet_times(mov_path, 0) + alignment.offset_seconds
    ultrawide_source_times: np.ndarray = track_packet_times(mov_path, 2) + alignment.ultrawide_offset_seconds
    epoch, wide_drop, ultrawide_drop = shared_epoch(wide_source_times, ultrawide_source_times)
    portrait: bool = wide_resolution[0] < wide_resolution[1]
    CONSOLE.print(f"Timeline: epoch={epoch:.6f}s dropped_leading=(wide={wide_drop}, ultrawide={ultrawide_drop}) portrait={portrait}")

    wide_video: VideoSamples = prepare_video_track(mov_path, 0, quarter_turns, config.keep_transcode_cache, drop_leading=wide_drop)
    CONSOLE.print(f"Transcode {config.video_id} wide: {wide_video.transcode_seconds:.2f}s ({wide_video.encoder})")
    ultrawide_video: VideoSamples = prepare_video_track(mov_path, 2, quarter_turns, config.keep_transcode_cache, drop_leading=ultrawide_drop)
    CONSOLE.print(f"Transcode {config.video_id} ultrawide: {ultrawide_video.transcode_seconds:.2f}s ({ultrawide_video.encoder})")

    wide_path: Path = sequence_output / "video_wide.rrd"
    with atomic_recording(wide_path, config.video_id, send_properties=False) as recording:
        wide_video_timestamps: np.ndarray = _log_video(
            recording,
            VIDEO_WIDE,
            wide_video,
            alignment.offset_seconds,
            epoch,
            f"{config.video_id} video-wide",
        )

    ultrawide_path: Path = sequence_output / "video_ultrawide.rrd"
    with atomic_recording(ultrawide_path, config.video_id, send_properties=False) as recording:
        ultrawide_video_timestamps: np.ndarray = _log_video(
            recording,
            VIDEO_ULTRAWIDE,
            ultrawide_video,
            alignment.ultrawide_offset_seconds,
            epoch,
            f"{config.video_id} video-ultrawide",
        )

    trajectory_unbaked = resample_trajectory(pose_trajectory_unbaked, wide_video_timestamps)
    trajectory: Trajectory = rotate_trajectory(trajectory_unbaked, quarter_turns)
    before: int = int(np.count_nonzero(wide_video_timestamps < pose_trajectory_unbaked.timestamps[0]))
    after: int = int(np.count_nonzero(wide_video_timestamps > pose_trajectory_unbaked.timestamps[-1]))
    CONSOLE.print(
        f"Pose coverage: source={pose_source} span={pose_trajectory_unbaked.timestamps[0]:.6f}..{pose_trajectory_unbaked.timestamps[-1]:.6f} before={before} after={after}"
    )

    distortions: list[CameraDistortion] = decode_camera_distortions_from_packets(metadata_packets[10])
    for distortion in distortions:
        CONSOLE.print(
            f"Distortion {distortion.camera_name}: model=apple_radial_poly coefficients={distortion.coefficients.tolist()} "
            f"inverse={distortion.inverse_coefficients.tolist()} center={distortion.center_xy.tolist()} "
            f"reference={distortion.reference_dimensions_wh.tolist()} temporal_samples={distortion.temporal_sample_count}"
        )
    pinhole_intrinsics_by_camera: dict[str, SimpleCVIntrinsics] = {}
    for camera_name, intrinsics, resolution in (
        ("wide", wide_intrinsics, wide_resolution),
        ("ultrawide", ultrawide_intrinsics, ultrawide_resolution),
    ):
        logged_indices: np.ndarray = np.flatnonzero(intrinsics.timestamps >= epoch)
        if len(logged_indices) == 0:
            raise ValueError(f"{camera_name} has no pinhole calibration at or after the shared epoch")
        pinhole_intrinsics_by_camera[camera_name] = SimpleCVIntrinsics.from_k_matrix(
            camera_conventions="RDF",
            k_matrix=intrinsics.matrices[int(logged_indices[0])],
            width=resolution[0],
            height=resolution[1],
        )
    calibration_path: Path = sequence_output / "calibration.rrd"
    with atomic_recording(calibration_path, config.video_id, send_properties=False) as recording:
        _log_intrinsics(recording, PINHOLE_WIDE, wide_intrinsics, wide_resolution, epoch)
        _log_intrinsics(recording, PINHOLE_WIDE_LOWRES, lowres_intrinsics, lowres_resolution, epoch)
        _log_intrinsics(recording, PINHOLE_ULTRAWIDE, ultrawide_intrinsics, ultrawide_resolution, epoch)
        _log_calibration_rig(recording, trajectory, extrinsics, rig_from_device, epoch)
        _log_sky_angles(recording, SKY_ANGLE_WIDE, trajectory.timestamps, sky_angles(trajectory.quaternion_xyzw), epoch)
        ultrawide_sky_angles: np.ndarray = interpolate_sky_angles(trajectory_sparse, ultrawide_video_timestamps)
        _log_sky_angles(recording, SKY_ANGLE_ULTRAWIDE, ultrawide_video_timestamps, ultrawide_sky_angles, epoch)
        _log_distortions(recording, distortions, pinhole_intrinsics_by_camera, quarter_turns)

    confidence: list[Path] = sorted_timestamped_paths(sequence_dir / "confidence")
    depth_started: float = time.perf_counter()
    depth_path: Path = sequence_output / "arkit_depth.rrd"
    with (
        atomic_recording(depth_path, config.video_id, send_properties=False) as recording,
        ThreadPoolExecutor(max_workers=8) as pool,
    ):
        _log_depth(
            recording,
            DEPTH,
            lowres_depth,
            quarter_turns,
            pool,
            f"{config.video_id} depth-lowres",
            epoch,
        )
        _log_confidence(recording, confidence, quarter_turns, pool, f"{config.video_id} confidence", epoch)
    depth_seconds: float = time.perf_counter() - depth_started
    CONSOLE.print(f"Depth {config.video_id}: {depth_seconds:.1f}s ({len(lowres_depth)} arkit + {len(confidence)} conf)")

    imu_path: Path = sequence_output / "imu.rrd"
    with atomic_recording(imu_path, config.video_id, send_properties=False) as recording:
        _log_imu(recording, imu, epoch)

    arkit_mesh_path: Path = sequence_output / "arkit_mesh.rrd"
    with atomic_recording(arkit_mesh_path, config.video_id, send_properties=False) as recording:
        mesh_summary: ArkitMeshSummary = log_arkit_mesh(sequence_dir, config.video_id, recording)

    gt_boxes_path: Path = sequence_output / "gt_boxes.rrd"
    with atomic_recording(gt_boxes_path, config.video_id, send_properties=False) as recording:
        box_count: int = log_gt_boxes(sequence_dir, config.video_id, recording)

    base_path: Path = sequence_output / "base.rrd"
    # base is the one layer that carries recording properties (and the embedded blueprint).
    with atomic_recording(
        base_path,
        config.video_id,
        send_properties=True,
        default_blueprint=make_blueprint(portrait, framing=(mesh_summary.mesh_center_xyz, mesh_summary.bounding_radius_m)),
    ) as recording:
        # Typed, grouped recording properties (segment-table columns property:<group>:<field>).
        # The bar for a field: someone filters or sorts segments by it. Pipeline
        # diagnostics stay in the console log, not the table.
        # Some metadata rows carry visit_id='NA': omit the field (null segment-table
        # cell) rather than invent a sentinel venue id.
        visit_id_field: dict[str, Any] = {} if metadata["visit_id"] == "NA" else {"visit_id": int(metadata["visit_id"])}
        recording.send_property(
            "capture",
            rr.AnyValues(
                **visit_id_field,
                split=metadata["fold"],
                orientation="portrait" if portrait else "landscape",
                orientation_quarter_turns_ccw=int(quarter_turns),
                pose_source=pose_source,
                uptime_epoch_seconds=float(epoch),
                schema_version="arkitscenes:v2",
            ),
        )
        recording.send_property(
            "quality",
            rr.AnyValues(
                pose_fit_translation_rms_m=float(pose_selection.translation_rms_m),
                orientation_ambiguous=bool(orientation_is_ambiguous),
            ),
        )
        recording.send_property("faro", rr.AnyValues(has_scans=metadata["has_laser_scanner_point_clouds"] == "True"))
        _log_rig_grammar(recording)
    layer_paths: list[Path] = [sequence_output / f"{name}.rrd" for name in REQUIRED_LAYER_NAMES]
    verify_rrd(layer_paths)
    verification: str = f"{len(layer_paths)}/{len(REQUIRED_LAYER_NAMES)} layers verified"
    ingest_seconds: float = time.perf_counter() - ingest_started
    CONSOLE.print(
        f"Saved {sequence_output}: videos={len(wide_video_timestamps)}/{len(ultrawide_video_timestamps)}, depth={len(lowres_depth)}, "
        f"IMU={len(imu.accel_timestamps)}/{len(imu.gyro_timestamps)}, boxes={box_count}, verify={verification}"
    )
    CONSOLE.print(f"Ingest {config.video_id} total: {ingest_seconds:.2f}s")
    if config.spawn:
        rr.spawn()
    return sequence_output
