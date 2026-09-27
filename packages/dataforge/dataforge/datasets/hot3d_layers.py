"""HOT3D layer writers; raw VRS and sidecars are read-only."""

from collections.abc import Callable, Iterator
from functools import partial
from itertools import chain
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray

from dataforge import aria, hands, objects, schema, writing
from dataforge.datasets.hot3d_hands import HandBatch, millimetre_wrists
from dataforge.datasets.hot3d_source import DEVICES, RELEASE, AssetInfo, pose_array
from dataforge.datasets.hot3d_vrs import CameraModel, CameraStream, Scene
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import (
    annotation_context,
    frame_index_column,
    log_camera_node,
    log_camera_source,
    log_dense_pose_track,
    log_imu,
    log_rig_node,
    log_video_stream,
    time_column,
)
from dataforge.timing import SequenceTimer
from dataforge.umetrack_hands import HandProfileDoc, log_hand_meshes
from dataforge.video_encoding import AV1_CQ, AV1_GOP, FrameSource, encode_frames_to_mp4, parallel_clips, work_dir
from dataforge.vrs import VrsImageReader, census_images


def write_base(recording: rr.RecordingStream, scene: Scene, identity: SequenceIdentity, timer: SequenceTimer) -> None:
    """Encode every native image once and publish shipped device poses and IMU samples."""
    # PyTurboJPEG is in the dataforge envs only; slam-rs imports this module through the dataset registry.
    from dataforge.jpeg import decode_jpeg_frames, jpeg_frame_source

    rr.log("/", DEVICES[scene.device].view_coordinates, static=True, recording=recording)
    rr.log("/", annotation_context(), static=True, recording=recording)
    log_rig_node(recording, 0, reference=None, num_cameras=len(scene.cameras), name=scene.device, kind="ego")

    def encode(camera: CameraStream, clip: Path) -> None:
        model: CameraModel = camera.model
        reader: VrsImageReader = VrsImageReader(scene.vrs, model.stream_id)

        encoded_images: Iterator[bytes] = census_images(
            reader.images(), camera.times_ns, camera.source_count, preview=scene.stop_ns is not None, where=f"{scene.source}/{model.stream_id}"
        )
        first: bytes = next(encoded_images)
        source: FrameSource = jpeg_frame_source(first)
        if (source.width, source.height) != (model.width, model.height):
            raise ValueError(f"{scene.source}/{model.stream_id}: JPEG dimensions disagree with calibration")
        encode_frames_to_mp4(
            decode_jpeg_frames(chain([first], encoded_images), source=source),
            clip,
            source=source,
            fps=30,
            gop=AV1_GOP,
            cq=AV1_CQ,
            rotate_cw_quarter_turns=1,
            filter_threads=1,
        )

    resolutions: list[str] = []
    with work_dir("hot3d-") as work:
        clips: list[Path] = [work / f"{camera.model.stream_id}.mp4" for camera in scene.cameras]
        jobs: list[tuple[Path, Callable[[], None]]] = [
            (clip, partial(encode, camera, clip)) for camera, clip in zip(scene.cameras, clips, strict=True)
        ]
        with parallel_clips(jobs, timer) as encoded:
            for index, (camera, clip) in enumerate(zip(scene.cameras, encoded, strict=True)):
                model: CameraModel = camera.model
                log_camera_node(
                    recording,
                    0,
                    index,
                    camera.calibration.to_fisheye62(),
                    name=model.label,
                    kind="rgb" if model.stream_id == "214-1" else "grayscale",
                    image_plane_distance=0.05,
                    camera_model="FISHEYE624 (thin-prism omitted)",
                    image_rotation_cw_deg=90,
                )
                resolutions.append(
                    log_camera_source(
                        recording,
                        0,
                        index,
                        name=model.label,
                        source_width=model.width,
                        source_height=model.height,
                        stream_id=model.stream_id,
                        video_codec="av1",
                        cq=AV1_CQ,
                        gop=AV1_GOP,
                    )
                )
                log_video_stream(
                    recording,
                    clip,
                    schema.video_path(0, index),
                    times_ns=camera.times_ns,
                    frame_indices=np.arange(len(camera.times_ns), dtype=np.int64),
                )
    if scene.metadata.have_hand_object_pose_gt:
        log_dense_pose_track(
            recording,
            schema.rig_path(0),
            times_ns=scene.times_ns,
            frame_indices=scene.frame_indices,
            transforms=pose_array(scene.headset, scene.times_ns),
        )
    if DEVICES[scene.device].has_imu:
        calibration: aria.DeviceCalibration = aria.read_device_calibration(scene.vrs)
        # Match the common physical slam-left camera to bridge the factory device
        # origin to the slightly different device origin in the GT calibration.
        factory_camera = calibration.camera("camera-slam-left")
        gt_camera = next(camera.model for camera in scene.cameras if camera.model.stream_id == "1201-1")
        gt_T_factory: Float64[ndarray, "4 4"] = gt_camera.device_T_camera.matrix() @ np.linalg.inv(factory_camera.rig_T_cam.matrix())
        for index, stream in enumerate(aria.IMU_STREAM_IDS):
            gyro, accel = aria.read_imu(scene.vrs, stream, stop_ns=scene.stop_ns)
            transform: Float64[ndarray, "4 4"] = gt_T_factory @ calibration.imu(aria.STREAM_LABELS[stream]).matrix()
            log_imu(
                recording,
                0,
                index,
                gyro=gyro,
                accel=accel,
                name=aria.STREAM_LABELS[stream],
                rig_T_imu=rr.Transform3D(translation=transform[:3, 3], mat3x3=transform[:3, :3]),
            )
    writing.send_capture_properties(
        recording,
        identity,
        num_frames=len(scene.cameras[0].times_ns),
        num_cameras=len(scene.cameras),
        source_revision=f"HOT3D {RELEASE}",
        source_num_frames=scene.cameras[0].source_count,
        source_resolution=pa.array(resolutions),
        clock_source=DEVICES[scene.device].clock_source,
    )
    recording.send_property(
        "episode",
        rr.AnyValues(
            participant_id=scene.metadata.participant_id,
            object_ids=pa.array(scene.metadata.object_uids, type=pa.string()),
            has_gt=scene.metadata.have_hand_object_pose_gt,
        ),
    )


def write_hands(recording: rr.RecordingStream, scene: Scene, profile: HandProfileDoc, batch: HandBatch) -> None:
    """Dense keypoints and UmeTrack parameters, MANO on the label census, and quality flags."""
    times: Int64[ndarray, "n"] = scene.times_ns
    frames: Int64[ndarray, "n"] = scene.frame_indices
    rr.log(schema.hand_profile_path(), rr.TextDocument(profile.text, media_type="application/json"), static=True, recording=recording)
    hands.log_keypoints3d(recording, times_ns=times, frame_indices=frames, positions=batch.positions, confidence=batch.confidence)
    for index, side in enumerate(hands.HAND_SIDES):
        hands.log_hand_confidence(recording, side.name, times_ns=times, frame_indices=frames, confidence=batch.scores[index].astype(np.float64))
        hands.log_joint_angles(recording, side.name, times_ns=times, frame_indices=frames, angles=batch.angles[index])
        log_dense_pose_track(recording, schema.hand_wrist_path(side.name), times_ns=times, frame_indices=frames, transforms=batch.wrists[index])
    for side in hands.HAND_SIDES:
        coefficients: Float32[ndarray, "n 15"] = np.full((len(times), 15), np.nan, dtype=np.float32)
        betas: Float32[ndarray, "n 10"] = np.full((len(times), 10), np.nan, dtype=np.float32)
        translations: Float32[ndarray, "n 3"] = np.full((len(times), 3), np.nan, dtype=np.float32)
        rotations: Float64[ndarray, "n 4"] = np.full((len(times), 4), np.nan)
        for index, stamp in enumerate(times):
            row = scene.mano.get(int(stamp))
            if row is not None and side.key in row.hand_poses:
                pose = row.hand_poses[side.key]
                coefficients[index] = pose.pose
                betas[index] = pose.betas
                translations[index] = pose.wrist_xform.t_xyz
                rotations[index] = pose.wrist_xform.q_wxyz
        rr.send_columns(
            schema.hand_mano_path(side.name),
            indexes=[time_column(times), frame_index_column(frames)],
            columns=rr.AnyValues.columns(
                pca_coefficients=pa.FixedSizeListArray.from_arrays(pa.array(coefficients.reshape(-1)), 15),
                betas=pa.FixedSizeListArray.from_arrays(pa.array(betas.reshape(-1)), 10),
                wrist_translation=pa.FixedSizeListArray.from_arrays(pa.array(translations.reshape(-1)), 3),
                wrist_quaternion_wxyz=pa.FixedSizeListArray.from_arrays(pa.array(rotations.reshape(-1)), 4),
            ),
            recording=recording,
        )
    for name, streams in scene.masks.items():
        for stream, (stamps, flags) in streams.items():
            camera_index: int = next(index for index, camera in enumerate(scene.cameras) if camera.model.stream_id == stream)
            path: str = schema.quality_flag_path(0, camera_index, name)
            rr.log(path, rr.AnyValues(stream_id=stream), static=True, recording=recording)
            selected: Int64[ndarray, "n"] = np.searchsorted(times, stamps)
            rr.send_columns(
                path,
                indexes=[time_column(stamps), frame_index_column(frames[selected])],
                columns=rr.Scalars.columns(scalars=flags.astype(np.float64)),
                recording=recording,
            )


def write_projections(recording: rr.RecordingStream, scene: Scene, batch: HandBatch) -> None:
    """Write derived lens-model pixels on the same census as the 3D joints."""
    world_T_device: Float64[ndarray, "n 4 4"] = pose_array(scene.headset, scene.times_ns)
    hands.log_projections(
        recording,
        (
            ((0, cam), aria.project_to_calibration(camera.calibration, world_T_device, batch.positions).astype(np.float32))
            for cam, camera in enumerate(scene.cameras)
        ),
        times_ns=scene.times_ns,
        frame_indices=scene.frame_indices,
        confidence=batch.confidence,
        camera_model="FISHEYE624",
    )


def write_hand_meshes(recording: rr.RecordingStream, scene: Scene, profile: HandProfileDoc, batch: HandBatch) -> None:
    """Skin hands with finite angles and a positive score; every other row is an empty mesh."""
    for index, side in enumerate(hands.HAND_SIDES):
        log_hand_meshes(
            recording,
            side,
            profile.model,
            batch.angles[index],
            millimetre_wrists(batch.wrists[index]),
            np.isfinite(batch.angles[index]).all(axis=1) & (batch.scores[index] > 0.0),
            times_ns=scene.times_ns,
            frame_indices=scene.frame_indices,
        )


def write_object_poses(recording: rr.RecordingStream, scene: Scene, census: dict[str, AssetInfo]) -> None:
    """Write native IDs and dense poses, invalidating every census gap."""
    for alias in scene.metadata.object_uids:
        transforms: Float64[ndarray, "n 4 4"] = pose_array(scene.objects.get(alias, {}), scene.times_ns)
        confidence: Float32[ndarray, "n"] = np.isfinite(transforms).all(axis=(1, 2)).astype(np.float32)
        rr.log(schema.objects_path(alias), rr.AnyValues(instance_id=alias, name=census[alias].instance_name), static=True, recording=recording)
        objects.log_object_pose(
            recording,
            alias,
            times_ns=scene.times_ns,
            frame_indices=scene.frame_indices,
            transforms=transforms,
            confidence=confidence,
            missing="invalidate",
        )


def write_object_meshes(recording: rr.RecordingStream, scene: Scene, assets: Path) -> None:
    """Write native geometry and confidence-driven visibility without owning poses."""
    for alias in scene.metadata.object_uids:
        transforms: Float64[ndarray, "n 4 4"] = pose_array(scene.objects.get(alias, {}), scene.times_ns)
        posed: Bool[ndarray, "n"] = np.isfinite(transforms).all(axis=(1, 2))
        asset: rr.Asset3D = rr.Asset3D(
            contents=objects.strip_texture_transform((assets / f"{alias}.glb").read_bytes()), media_type="model/gltf-binary"
        )
        objects.log_object_mesh(
            recording,
            alias,
            times_ns=scene.times_ns,
            frame_indices=scene.frame_indices,
            asset=asset,
            confidence=posed.astype(np.float32),
            posed=posed,
            trust_threshold=0.0,
        )
