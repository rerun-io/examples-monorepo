"""Native Gen2 sensors, MPS hand measurements and full-lens derived projections."""

import struct
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from projectaria_tools.core.calibration import ImuCalibration

from dataforge import aria, hands, paths, schema, writing
from dataforge.datasets.aria_gen2_pilot_source import Camera, Scene
from dataforge.datasets.hot3d_layers import project_keypoints
from dataforge.datasets.hot3d_vrs import nearest_framesets
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import (
    ImuChannel,
    annotation_context,
    frame_index_column,
    log_camera_node,
    log_dense_pose_track,
    log_imu,
    log_rig_node,
    log_video_stream,
    time_column,
)
from dataforge.timing import SequenceTimer
from dataforge.video_encoding import AV1_CQ, AV1_GOP, FrameSource, encode_frames_to_mp4
from dataforge.vrs import ImageRecord
from dataforge.vrs_hevc import VrsHevcReader, VrsImuReader


def camera_access_units(source: Path, camera: Camera, *, preview: bool) -> Iterator[bytes]:
    """Check each raw record against its own camera census before encoding."""
    records: Iterator[ImageRecord] = VrsHevcReader(source / "video.vrs", camera.stream_id).images()
    if preview:
        records = islice(records, len(camera.times_ns))
    count: int = 0
    for record in records:
        if count >= len(camera.times_ns) or record.capture_timestamp_ns != camera.times_ns[count]:
            raise ValueError(f"{source}/{camera.stream_id}: capture timestamp mismatch at frame {count}")
        yield record.image
        count += 1
    if count != len(camera.times_ns):
        raise ValueError(f"{source}/{camera.stream_id}: {count} records, expected {len(camera.times_ns)}")


def write_motion(recording: rr.RecordingStream, scene: Scene) -> None:
    """All native trajectory/IMU rows, plus explicit missing-pose boundaries."""
    track = scene.trajectory
    keep: Bool[ndarray, "n"] = np.ones(len(track.times_ns), dtype=np.bool_) if scene.stop_ns is None else track.times_ns <= scene.stop_ns
    # A rig with no transform would appear at world origin. Explicit invalid rows
    # hide it before tracking, in large gaps and after the last measured pose.
    missing: list[int] = [min(int(camera.times_ns[0]) for camera in scene.cameras)]
    missing += (track.times_ns[:-1][np.diff(track.times_ns) > 2_000_000] + 1).tolist()
    missing.append(int(track.times_ns[-1]) + 1)
    camera_times: Int64[ndarray, "c"] = np.unique(np.concatenate([camera.times_ns for camera in scene.cameras]))
    missing += camera_times[~np.isfinite(track.at(camera_times)).all(axis=(1, 2))].tolist()
    extra: Int64[ndarray, "e"] = np.asarray(missing, dtype=np.int64)
    extra = extra[~np.isfinite(track.at(extra)).all(axis=(1, 2))]
    if scene.stop_ns is not None:
        extra = extra[extra <= scene.stop_ns]
    times: Int64[ndarray, "t"] = np.union1d(track.times_ns[keep], extra)
    log_dense_pose_track(
        recording,
        schema.rig_path(0),
        times_ns=times,
        frame_indices=nearest_framesets(scene.cameras[0].times_ns, times),
        transforms=track.at(times),
    )
    rr.send_columns(
        schema.rig_path(0) + "/quality",
        indexes=[time_column(track.times_ns[keep]), frame_index_column(nearest_framesets(scene.cameras[0].times_ns, track.times_ns[keep]))],
        columns=rr.Scalars.columns(scalars=track.quality[keep]),
        recording=recording,
    )
    for index, label in enumerate(("imu-left", "imu-right")):
        reader: VrsImuReader = VrsImuReader(scene.source / "video.vrs", f"1202-{index + 1}")
        stamps: list[int] = []
        gyros: list[list[float]] = []
        accels: list[list[float]] = []
        for payload in reader.records(79):
            stamp: int = struct.unpack_from("<q", payload, 11)[0]
            if scene.stop_ns is not None and stamp > scene.stop_ns:
                break
            stamps.append(stamp)
            gyros.append(list(struct.unpack_from("<fff", payload, 55)) if payload[1] else [float("nan")] * 3)
            accels.append(list(struct.unpack_from("<fff", payload, 43)) if payload[0] else [float("nan")] * 3)
        imu: ImuCalibration | None = scene.calibration.get_imu_calib(label)
        if imu is None:
            raise ValueError(f"{scene.source}: missing {label} factory calibration")
        transform: Float64[ndarray, "4 4"] = np.asarray(imu.get_transform_device_imu().to_matrix())
        imu_times: Int64[ndarray, "n"] = np.asarray(stamps, dtype=np.int64)
        if np.any(np.diff(imu_times) <= 0):
            raise ValueError(f"{scene.source}/{label}: unordered IMU timestamps")
        log_imu(
            recording,
            0,
            index,
            name=label,
            gyro=ImuChannel(imu_times, np.asarray(gyros, dtype=np.float64).reshape(-1, 3)),
            accel=ImuChannel(imu_times, np.asarray(accels, dtype=np.float64).reshape(-1, 3)),
            rig_T_imu=rr.Transform3D(translation=transform[:3, 3], mat3x3=transform[:3, :3]),
        )


def write_base(recording: rr.RecordingStream, scene: Scene, identity: SequenceIdentity, timer: SequenceTimer) -> None:
    """Pipe native HEVC to shared AV1 NVENC, then remap MP4 samples to VRS times."""
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True, recording=recording)
    rr.log("/", annotation_context(), static=True, recording=recording)
    log_rig_node(recording, 0, reference=None, num_cameras=5, name="Aria Gen2 device", kind="ego")
    work_root: Path = paths.output_root() / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="aria-gen2-", dir=work_root) as work:
        # Sequential cameras limit seeks against cook_0's read-only NAS source.
        for index, camera in enumerate(scene.cameras):
            clip: Path = Path(work) / f"{camera.stream_id}.mp4"
            with timer.stage("transcode"):
                encode_frames_to_mp4(
                    camera_access_units(scene.source, camera, preview=scene.stop_ns is not None),
                    clip,
                    source=FrameSource("hevc"),
                    fps=camera.fps,
                    gop=AV1_GOP,
                    cq=AV1_CQ,
                    filter_threads=1,
                )
            log_camera_node(
                recording,
                0,
                index,
                aria.fisheye62_from_aria(
                    camera.calibration, rig_T_cam=np.asarray(camera.calibration.get_transform_device_camera().to_matrix()), name=camera.label
                ),
                name=camera.label,
                kind="rgb" if index == 0 else "grayscale",
                image_plane_distance=0.05,
                camera_model="FISHEYE624 (pinhole approximation; full lens in projections)",
            )
            rr.log(
                schema.cam_path(0, index),
                rr.AnyValues(stream_id=camera.stream_id, video_codec="av1", cq=AV1_CQ, gop=AV1_GOP, projection_params=camera.calibration.get_projection_params()),
                static=True,
                recording=recording,
            )
            log_video_stream(
                recording, clip, schema.video_path(0, index), times_ns=camera.times_ns, frame_indices=np.arange(len(camera.times_ns), dtype=np.int64)
            )
            clip.unlink()
    write_motion(recording, scene)
    writing.send_capture_properties(
        recording,
        identity,
        num_frames=len(scene.cameras[0].times_ns),
        num_cameras=5,
        source_revision="AriaGen2PilotDataset v1.0",
        dataset_version="v1.0",
        source_num_frames=scene.cameras[0].source_count,
        source_resolution=pa.array(
            [
                f"{camera.stream_id}:{int(camera.calibration.get_image_size()[0])}x{int(camera.calibration.get_image_size()[1])}"
                for camera in scene.cameras
            ]
        ),
        clock_source="Unshifted VRS capture_timestamp_ns per camera/IMU; MPS tracking_timestamp_us * 1000, device boot origin",
        calibration_source="VRS factory FISHEYE624 including thin prism",
    )
    recording.send_property("episode", rr.AnyValues(sequence=scene.source.name))


def write_hands(recording: rr.RecordingStream, scene: Scene) -> None:
    """Every MPS hand row, including present zero-confidence measurements."""
    batch = scene.hands
    if not len(batch.times_ns):
        return
    frames: Int64[ndarray, "n"] = nearest_framesets(scene.cameras[0].times_ns, batch.times_ns)
    hands.log_keypoints3d(recording, times_ns=batch.times_ns, frame_indices=frames, positions=batch.positions, confidence=batch.confidence)
    for index, side in enumerate(("left", "right")):
        hands.log_hand_confidence(recording, side, times_ns=batch.times_ns, frame_indices=frames, confidence=batch.scores[:, index])
        log_dense_pose_track(
            recording, schema.hand_wrist_path(side), times_ns=batch.times_ns, frame_indices=frames, transforms=batch.wrists[:, index]
        )
        for normal_index, name in enumerate(("palm_normal", "wrist_normal")):
            rr.send_columns(
                f"/world/gt/hands/{side}/{name}",
                indexes=[time_column(batch.times_ns), frame_index_column(frames)],
                columns=rr.AnyValues.columns(
                    normal_world=pa.FixedSizeListArray.from_arrays(pa.array(batch.normals[:, index, normal_index].reshape(-1)), 3)
                ),
                recording=recording,
            )


def write_projections(recording: rr.RecordingStream, scene: Scene) -> None:
    """Reuse HOT3D's full FISHEYE624 projection, on the native hand clock."""
    recording.send_property("projections", rr.AnyValues(derived_from="coco133_xyz", camera_model="FISHEYE624", calibration_source="VRS factory"))
    batch = scene.hands
    if not len(batch.times_ns):
        return
    frames: Int64[ndarray, "n"] = nearest_framesets(scene.cameras[0].times_ns, batch.times_ns)
    for index, camera in enumerate(scene.cameras):
        pixels: Float64[ndarray, "n 133 2"] = project_keypoints(camera.calibration, batch.device_poses, batch.positions)
        hands.log_keypoints2d(
            recording,
            0,
            index,
            path=schema.coco133_uv_projected_path(0, index),
            times_ns=batch.times_ns,
            frame_indices=frames,
            positions=pixels.astype(np.float32),
            confidence=batch.confidence,
        )
