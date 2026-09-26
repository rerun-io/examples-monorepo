"""Native Gen2 sensors, MPS hand measurements and full-lens derived projections."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from projectaria_tools.core.calibration import ImuCalibration

from dataforge import aria, hands, paths, schema, writing
from dataforge.datasets.aria_gen2_pilot_source import IMUS, Scene
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
from dataforge.vrs import census_images


def write_motion(recording: rr.RecordingStream, scene: Scene) -> None:
    """All native trajectory/IMU rows, plus explicit missing-pose boundaries."""
    provider = aria.open_vrs(scene.source / "video.vrs")
    imu_channels: list[tuple[ImuChannel, ImuChannel]] = []
    first_stamp: int = min(int(camera.times_ns[0]) for camera in scene.cameras)
    for stream_id, _ in IMUS:
        gyro, accel = aria.read_imu(provider, stream_id)
        if scene.stop_ns is not None:
            kept: Bool[ndarray, "s"] = gyro.times_ns <= scene.stop_ns
            gyro, accel = ImuChannel(gyro.times_ns[kept], gyro.values_xyz[kept]), ImuChannel(accel.times_ns[kept], accel.values_xyz[kept])
        imu_channels.append((gyro, accel))
        for channel in (gyro, accel):
            if len(channel.times_ns):
                first_stamp = min(first_stamp, int(channel.times_ns[0]))
    track = scene.trajectory
    keep: Bool[ndarray, "n"] = np.ones(len(track.times_ns), dtype=np.bool_) if scene.stop_ns is None else track.times_ns <= scene.stop_ns
    # A rig with no transform would sit at the world origin. It is visible on each run of
    # rows <= 2 ms apart and hidden elsewhere: a NaN row at the first logged sensor stamp (when
    # tracking starts later) and 1 ns after every run's last row.
    run_ends: Int64[ndarray, "r"] = np.flatnonzero(np.append(np.diff(track.times_ns) > 2_000_000, True))
    hide: Int64[ndarray, "h"] = np.append(first_stamp, track.times_ns[run_ends] + 1)
    hide = hide[~np.isfinite(track.at(hide)).all(axis=(1, 2))]
    if scene.stop_ns is not None:
        hide = hide[hide <= scene.stop_ns]
    times: Int64[ndarray, "t"] = np.union1d(track.times_ns[keep], hide)
    log_dense_pose_track(
        recording,
        schema.rig_path(0),
        times_ns=times,
        frame_indices=nearest_framesets(scene.frame_clock, times),
        transforms=track.at(times),
    )
    rr.send_columns(
        schema.rig_path(0) + "/quality",
        indexes=[time_column(track.times_ns[keep]), frame_index_column(nearest_framesets(scene.frame_clock, track.times_ns[keep]))],
        columns=rr.Scalars.columns(scalars=track.quality[keep]),
        recording=recording,
    )
    for index, ((_, label), (gyro, accel)) in enumerate(zip(IMUS, imu_channels, strict=True)):
        imu: ImuCalibration | None = scene.calibration.get_imu_calib(label)
        if imu is None:
            raise ValueError(f"{scene.source}: missing {label} factory calibration")
        transform: Float64[ndarray, "4 4"] = np.asarray(imu.get_transform_device_imu().to_matrix())
        log_imu(
            recording,
            0,
            index,
            name=label,
            gyro=gyro,
            accel=accel,
            rig_T_imu=rr.Transform3D(translation=transform[:3, 3], mat3x3=transform[:3, :3]),
        )


def write_base(recording: rr.RecordingStream, scene: Scene, identity: SequenceIdentity, timer: SequenceTimer) -> None:
    """Pipe native HEVC to shared AV1 NVENC, then remap MP4 samples to VRS times."""
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True, recording=recording)
    rr.log("/", annotation_context(), static=True, recording=recording)
    log_rig_node(recording, 0, reference=None, num_cameras=len(scene.cameras), name="Aria Gen2 device", kind="ego")
    work_root: Path = paths.output_root() / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="aria-gen2-", dir=work_root) as work:
        # Sequential cameras limit seeks against cook_0's read-only NAS source.
        for index, camera in enumerate(scene.cameras):
            clip: Path = Path(work) / f"{camera.stream_id}.mp4"
            with timer.stage("transcode"):
                encode_frames_to_mp4(
                    census_images(
                        scene.vrs.hevc(camera.stream_id).images(),
                        camera.times_ns,
                        camera.source_count,
                        preview=scene.stop_ns is not None,
                        where=f"{scene.source}/{camera.stream_id}",
                    ),
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
                recording, clip, schema.video_path(0, index), times_ns=camera.times_ns, frame_indices=nearest_framesets(scene.frame_clock, camera.times_ns)
            )
            clip.unlink()
    write_motion(recording, scene)
    writing.send_capture_properties(
        recording,
        identity,
        num_frames=len(scene.cameras[0].times_ns),
        num_cameras=len(scene.cameras),
        source_revision="AriaGen2PilotDataset v1.0",
        dataset_version="v1.0",
        source_num_frames=scene.cameras[0].source_count,
        source_resolution=pa.array(
            [
                f"{camera.stream_id}:{int(camera.calibration.get_image_size()[0])}x{int(camera.calibration.get_image_size()[1])}"
                for camera in scene.cameras
            ]
        ),
        clock_source="Unshifted VRS capture_timestamp_ns per camera/IMU; MPS tracking_timestamp_us * 1000, device boot origin; frame_index: nearest slam-front-left stamp, ties earlier",
        calibration_source="VRS factory FISHEYE624 including thin prism",
    )
    recording.send_property("episode", rr.AnyValues(sequence=scene.source.name))


def write_hands(recording: rr.RecordingStream, scene: Scene) -> None:
    """Every MPS hand row, including present zero-confidence measurements."""
    batch = scene.hands
    if not len(batch.times_ns):
        return
    frames: Int64[ndarray, "n"] = batch.frame_indices
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
    for index, camera in enumerate(scene.cameras):
        pixels: Float64[ndarray, "n 133 2"] = project_keypoints(camera.calibration, batch.device_poses, batch.positions)
        hands.log_keypoints2d(
            recording,
            0,
            index,
            path=schema.coco133_uv_projected_path(0, index),
            times_ns=batch.times_ns,
            frame_indices=batch.frame_indices,
            positions=pixels.astype(np.float32),
            confidence=batch.confidence,
        )
