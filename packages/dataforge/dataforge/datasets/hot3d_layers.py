"""HOT3D layer writers; raw VRS and sidecars are read-only."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray
from projectaria_tools.core.calibration import CameraCalibration

from dataforge import aria, hands, paths, schema, writing
from dataforge.datasets.hot3d_hands import HandBatch, evaluate_hands
from dataforge.datasets.hot3d_source import DEVICES, URL_LIST_DATE, pose_array
from dataforge.datasets.hot3d_vrs import CameraModel, CameraStream, Scene
from dataforge.datasets.show3d_hands import HAND_SIDES, HandProfileDoc, read_hand_profile
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
from dataforge.video_encoding import AV1_CQ, AV1_GOP, FrameSource, decode_jpeg_frames, encode_frames_to_mp4, jpeg_frame_source, parallel_clips
from dataforge.vrs import VrsImageReader


def write_base(recording: rr.RecordingStream, scene: Scene, identity: SequenceIdentity, timer: SequenceTimer) -> None:
    """Encode every native image once and publish shipped device poses and IMU samples."""
    rr.log("/", DEVICES[scene.device].view_coordinates, static=True, recording=recording)
    rr.log("/", annotation_context(), static=True, recording=recording)
    log_rig_node(recording, 0, reference=None, num_cameras=len(scene.cameras), name=scene.device, kind="ego")
    work_root: Path = paths.output_root() / "work"
    work_root.mkdir(parents=True, exist_ok=True)

    def encode(camera: CameraStream, clip: Path) -> None:
        model: CameraModel = camera.model
        reader: VrsImageReader = VrsImageReader(scene.source / "recording.vrs", model.stream_id)

        def images() -> Iterator[bytes]:
            # Every record must carry the projectaria-tools stamp of the same index (the
            # record <-> video-sample proof); a preview decodes only its prefix.
            seen: int = 0
            for record in reader.images():
                if seen < len(camera.times_ns):
                    if record.capture_timestamp_ns != camera.times_ns[seen]:
                        raise ValueError(f"{scene.source}/{model.stream_id}: capture timestamp mismatch at frame {seen}")
                    yield record.image
                seen += 1
            if seen != camera.source_count:
                raise ValueError(f"{scene.source}/{model.stream_id}: {seen} image records, expected {camera.source_count}")

        encoded_images: Iterator[bytes] = images()
        first: bytes = next(encoded_images)
        source: FrameSource = jpeg_frame_source(first)
        if (source.width, source.height) != (model.width, model.height):
            raise ValueError(f"{scene.source}/{model.stream_id}: JPEG dimensions disagree with calibration")
        count: int = encode_frames_to_mp4(
            decode_jpeg_frames(chain([first], encoded_images), source=source),
            clip,
            source=source,
            fps=30,
            gop=AV1_GOP,
            cq=AV1_CQ,
            rotate_cw_quarter_turns=1,
            filter_threads=1,
        )
        if count != len(camera.times_ns):
            raise ValueError(f"{scene.source}/{model.stream_id}: encoded {count}, expected {len(camera.times_ns)}")

    with TemporaryDirectory(prefix="hot3d-", dir=work_root) as work:
        clips: list[Path] = [Path(work) / f"{camera.model.stream_id}.mp4" for camera in scene.cameras]
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
                    aria.fisheye62_from_aria(
                        camera.calibration, rig_T_cam=np.asarray(camera.calibration.get_transform_device_camera().to_matrix()), name=model.label
                    ),
                    name=model.label,
                    kind="rgb" if model.stream_id == "214-1" else "grayscale",
                    image_plane_distance=0.05,
                    camera_model="FISHEYE624 (thin-prism omitted)",
                    image_rotation_cw_deg=90,
                )
                rr.log(
                    schema.cam_path(0, index),
                    rr.AnyValues(
                        source_width=model.width, source_height=model.height, stream_id=model.stream_id, video_codec="av1", cq=AV1_CQ, gop=AV1_GOP
                    ),
                    static=True,
                    recording=recording,
                )
                log_video_stream(
                    recording,
                    clip,
                    schema.video_path(0, index),
                    times_ns=camera.times_ns,
                    frame_indices=np.arange(len(camera.times_ns), dtype=np.int64),
                )
                clip.unlink()
    if scene.metadata.have_hand_object_pose_gt:
        log_dense_pose_track(
            recording,
            schema.rig_path(0),
            times_ns=scene.times_ns,
            frame_indices=scene.frame_indices,
            transforms=pose_array(scene.headset, scene.times_ns),
        )
    if DEVICES[scene.device].has_imu:
        calibration = scene.provider.get_device_calibration()
        if calibration is None:
            raise ValueError(f"{scene.source}: Aria IMU needs VRS calibration")
        # Match the common physical slam-left camera to bridge the factory device
        # origin to the slightly different device origin in the GT calibration.
        factory_camera = calibration.get_camera_calib("camera-slam-left")
        if factory_camera is None:
            raise ValueError(f"{scene.source}: missing factory slam-left calibration")
        gt_camera = next(camera.model for camera in scene.cameras if camera.model.stream_id == "1201-1")
        gt_T_factory: Float64[ndarray, "4 4"] = gt_camera.device_T_camera.matrix() @ np.linalg.inv(
            factory_camera.get_transform_device_camera().to_matrix()
        )
        for index, stream in enumerate(aria.IMU_STREAM_IDS):
            channels: aria.ImuSamples = aria.read_imu(scene.provider, stream)
            imu = calibration.get_imu_calib(aria.STREAM_LABELS[stream])
            if imu is None:
                raise ValueError(f"{scene.source}: missing {stream} calibration")
            transform: Float64[ndarray, "4 4"] = gt_T_factory @ imu.get_transform_device_imu().to_matrix()
            selected: list[ImuChannel] = []
            for channel in channels:
                keep: Bool[ndarray, "n"] = (
                    np.ones(len(channel.times_ns), dtype=np.bool_) if scene.stop_ns is None else channel.times_ns <= scene.stop_ns
                )
                selected.append(ImuChannel(channel.times_ns[keep], channel.values_xyz[keep]))
            log_imu(
                recording,
                0,
                index,
                gyro=selected[0],
                accel=selected[1],
                name=aria.STREAM_LABELS[stream],
                rig_T_imu=rr.Transform3D(translation=transform[:3, 3], mat3x3=transform[:3, :3]),
            )
    writing.send_capture_properties(
        recording,
        identity,
        num_frames=len(scene.cameras[0].times_ns),
        num_cameras=len(scene.cameras),
        source_revision=f"HOT3D v4.0.0; URL list {URL_LIST_DATE}",
        url_list_date=URL_LIST_DATE,
        source_num_frames=scene.cameras[0].source_count,
        source_resolution=pa.array([f"{camera.model.stream_id}:{camera.model.width}x{camera.model.height}" for camera in scene.cameras]),
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
    for index, side in enumerate(HAND_SIDES):
        hands.log_hand_confidence(recording, side.name, times_ns=times, frame_indices=frames, confidence=batch.scores[index].astype(np.float64))
        hands.log_joint_angles(recording, side.name, times_ns=times, frame_indices=frames, angles=batch.angles[index])
        log_dense_pose_track(recording, schema.hand_wrist_path(side.name), times_ns=times, frame_indices=frames, transforms=batch.wrists[index])
    for side in HAND_SIDES:
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


@dataclass(frozen=True, slots=True)
class ProjectedKeypoints:
    """Full-precision lens projections before the Float32 logging boundary."""

    positions: Float64[ndarray, "n 133 2"]
    """Rotated image pixels; rejected joints are NaN."""
    confidence: Float32[ndarray, "n 133"]
    """Source joint confidence, zero for rejected projections."""


def project_keypoints(
    calibration: CameraCalibration,
    world_T_device: Float64[ndarray, "n 4 4"],
    positions: Float32[ndarray, "n 133 3"],
    confidence: Float32[ndarray, "n 133"],
) -> ProjectedKeypoints:
    """Project dense world joints through the shipped camera model.

    Args:
        calibration: Full lens model in the logged image orientation.
        world_T_device: Float64[ndarray, "n 4 4"] GT poses, NaN when missing.
        positions: Float32[ndarray, "n 133 3"] world metres.
        confidence: Float32[ndarray, "n 133"] shipped joint confidence.

    Returns:
        Float64 pixels and Float32 confidence, with NaN/0 for invalid joints.
    """
    positions, confidence = hands.confidence_rule(positions, confidence)
    pixels: Float64[ndarray, "n 133 2"] = np.full((*positions.shape[:2], 2), np.nan)
    scores: Float32[ndarray, "n 133"] = np.zeros_like(confidence)
    device_T_camera: Float64[ndarray, "4 4"] = np.asarray(calibration.get_transform_device_camera().to_matrix())
    # Loop locals stay unannotated: beartype would rebuild a jaxtyping checker per joint in the dev env.
    for index, pose in enumerate(world_T_device):
        if not np.isfinite(pose).all():
            continue
        cam_T_world = np.linalg.inv(pose @ device_T_camera)
        for joint in np.flatnonzero(np.isfinite(positions[index]).all(axis=1)):
            point = cam_T_world[:3, :3] @ positions[index, joint] + cam_T_world[:3, 3]
            if point[2] <= 0.0:
                continue
            pixel = calibration.project(point)
            if pixel is not None and np.isfinite(pixel).all():
                pixels[index, joint] = pixel
                scores[index, joint] = confidence[index, joint]
    return ProjectedKeypoints(pixels, scores)


def write_projections(recording: rr.RecordingStream, scene: Scene, batch: HandBatch) -> None:
    """Write derived lens-model pixels on the same census as the 3D joints."""
    recording.send_property("projections", rr.AnyValues(derived_from="coco133_xyz", camera_model="FISHEYE624"))
    world_T_device: Float64[ndarray, "n 4 4"] = pose_array(scene.headset, scene.times_ns)
    for cam, camera in enumerate(scene.cameras):
        projected: ProjectedKeypoints = project_keypoints(camera.calibration, world_T_device, batch.positions, batch.confidence)
        hands.log_keypoints2d(
            recording,
            0,
            cam,
            path=schema.coco133_uv_projected_path(0, cam),
            times_ns=scene.times_ns,
            frame_indices=scene.frame_indices,
            positions=projected.positions.astype(np.float32),
            confidence=projected.confidence,
        )


class LayerWriter:
    """One conversion's dispatcher; hand profile and FK are evaluated once for hand layers and projections."""

    def __init__(self, scene: Scene, identity: SequenceIdentity, timer: SequenceTimer) -> None:
        self.scene: Scene = scene
        self.identity: SequenceIdentity = identity
        self.timer: SequenceTimer = timer
        self.profile: HandProfileDoc | None = None
        self.batch: HandBatch | None = None

    def write_layer(self, layer: str, recording: rr.RecordingStream) -> None:
        """Dispatch disjoint layer owners, preparing shared data only when needed."""
        if layer in (paths.HAND_POSE_LAYER, paths.PROJECTIONS_LAYER) and self.profile is None:
            self.profile = read_hand_profile(self.scene.source / "umetrack_hand_user_profile.json")
            self.batch = evaluate_hands(self.profile.model, self.scene.hands, self.scene.times_ns)
        writers: dict[str, Callable[[], None]] = {
            paths.BASE_LAYER: partial(write_base, recording, self.scene, self.identity, self.timer),
        }
        if self.profile is not None and self.batch is not None:
            writers[paths.PROJECTIONS_LAYER] = partial(write_projections, recording, self.scene, self.batch)
            writers[paths.HAND_POSE_LAYER] = partial(write_hands, recording, self.scene, self.profile, self.batch)
        if layer not in writers:
            raise ValueError(f"unknown HOT3D layer {layer}")
        writers[layer]()
