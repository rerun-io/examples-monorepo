"""UmeTrack sensors, hand parameters, skinned meshes and lens-model projections."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Bool, Float32, Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.camera_parameters import Extrinsics, Fisheye62Parameters, Intrinsics
from simplecv.sensors.camera.fisheye62 import project_fisheye62
from simplecv.umetrack_temp.generic_hand_model_numpy import skin_landmarks, wrist_for_hand

from dataforge import hands, logging_toolkit, schema, writing
from dataforge.datasets.umetrack_remote import REPOSITORY, REVISION
from dataforge.datasets.umetrack_source import FISHEYE62, Camera, SequenceData
from dataforge.identity import SequenceIdentity
from dataforge.timing import SequenceTimer
from dataforge.umetrack_hands import log_hand_meshes
from dataforge.video_encoding import AV1_CQ, AV1_GOP, parallel_clips, transcode_mp4, work_dir
from dataforge.world_up import WORLD_UP_VIEW_COORDINATES


def rig_cameras(scene: SequenceData) -> list[Fisheye62Parameters]:
    """Return the four typed cameras with rig-from-camera extrinsics."""
    cameras: list[Fisheye62Parameters] = []
    for index, source in enumerate(scene.labels.cameras):
        transform: Float64[ndarray, "4 4"] = scene.rig_T_cam[index]
        camera: Fisheye62Parameters = Fisheye62Parameters(
            name=f"cam_{index:02}",
            distortion=source.lens(),
            extrinsics=Extrinsics(world_R_cam=transform[:3, :3], world_t_cam=transform[:3, 3]),
            intrinsics=Intrinsics.from_focal_principal_point(
                camera_conventions="RDF",
                fl_x=source.fx,
                fl_y=source.fy,
                cx=source.cx,
                cy=source.cy,
                width=source.ImageSizeX,
                height=source.ImageSizeY,
            ),
        )
        cameras.append(camera)
    return cameras


def write_geometry(recording: rr.RecordingStream, scene: SequenceData, identity: SequenceIdentity) -> None:
    """Publish camera calibration, rig dropouts, provenance and measured world up."""
    rr.log("/world", WORLD_UP_VIEW_COORDINATES["+y"], static=True, recording=recording)
    rr.log("/", logging_toolkit.annotation_context(), static=True, recording=recording)
    logging_toolkit.log_rig_node(recording, 0, reference=schema.CAM0_REFERENCE, num_cameras=4, name="UmeTrack headset", kind="ego")
    resolutions: list[str] = []
    for index, camera in enumerate(rig_cameras(scene)):
        source: Camera = scene.labels.cameras[index]
        logging_toolkit.log_camera_node(
            recording, 0, index, camera, name=camera.name, kind="grayscale", image_plane_distance=0.05, camera_model=FISHEYE62
        )
        # Each camera is one tile of the source's four-camera mosaic, cropped and re-encoded.
        resolutions.append(
            logging_toolkit.log_camera_source(
                recording,
                0,
                index,
                name=camera.name,
                source_width=source.ImageSizeX,
                source_height=source.ImageSizeY,
                video_codec="av1",
                cq=AV1_CQ,
                gop=AV1_GOP,
            )
        )
        coefficients: dict[str, float | None] = dict(
            k1=source.k1,
            k2=source.k2,
            k3=source.k3,
            k4=source.k4,
            k5=source.k5,
            k6=source.k6,
            p1=source.p1,
            p2=source.p2,
            p3=source.p3,
            p4=source.p4,
        )
        rr.log(
            schema.pinhole_path(0, index),
            rr.AnyValues(
                drop_untyped_nones=True,
                **coefficients,
                source_camera_angle_deg=float(scene.labels.camera_angles[index]),
            ),
            static=True,
            recording=recording,
        )
    quaternions: Float64[ndarray, "n 4"] = np.full((scene.count, 4), np.nan, dtype=np.float64)
    if scene.tracked.any():
        quaternions[scene.tracked] = Rotation.from_matrix(scene.world_T_rig[scene.tracked, :3, :3]).as_quat()
    logging_toolkit.log_pose_track(
        recording,
        schema.rig_path(0),
        times_ns=scene.times_ns,
        frame_indices=scene.frame_indices,
        translations_xyz=scene.world_T_rig[:, :3, 3],
        quaternions_xyzw=quaternions,
    )
    rr.send_columns(
        schema.rig_path(0),
        indexes=[logging_toolkit.time_column(scene.times_ns), logging_toolkit.frame_index_column(scene.frame_indices)],
        columns=rr.AnyValues.columns(untracked=~scene.tracked),
        recording=recording,
    )
    writing.send_capture_properties(
        recording,
        identity,
        num_frames=scene.count,
        num_cameras=4,
        source_revision=f"github.com/{REPOSITORY}@{REVISION}",
        source_resolution=pa.array(resolutions),
        source_num_frames=scene.source_num_frames,
        clock_source="mp4_container_pts",
        fps=scene.fps,
        world_up_axis="+y",
        headset_up=pa.array(scene.headset_up),
        headset_up_spread_deg=scene.headset_up_spread_deg,
    )
    domain, interaction, split, user, _ = identity.parts
    recording.send_property("episode", rr.AnyValues(domain=domain, interaction=interaction, split=split, user=user))


def write_base(recording: rr.RecordingStream, scene: SequenceData, identity: SequenceIdentity, timer: SequenceTimer) -> None:
    """Crop four grayscale streams and remux them with the source presentation times."""
    write_geometry(recording, scene, identity)
    with work_dir("umetrack-") as work:
        clips: list[Path] = [work / f"cam_{index:02}.mp4" for index in range(4)]

        def encode(index: int, clip: Path) -> None:
            transcode_mp4(
                scene.source.with_suffix(".mp4"), clip, gop=AV1_GOP, cq=AV1_CQ, fps=scene.fps, frames=scene.count, gray=True, crop=scene.crop(index)
            )

        jobs: list[tuple[Path, Callable[[], None]]] = [(clip, partial(encode, index, clip)) for index, clip in enumerate(clips)]
        with parallel_clips(jobs, timer) as ready:
            for index, clip in enumerate(ready):
                logging_toolkit.log_video_stream(
                    recording, clip, schema.video_path(0, index), times_ns=scene.times_ns, frame_indices=scene.frame_indices
                )


@dataclass(frozen=True, slots=True)
class HandKeypoints:
    """Shared world-space COCO rows for hand_pose and projections."""

    positions: Float32[ndarray, "n 133 3"]
    """World metres, NaN for missing joints."""
    confidence: Float32[ndarray, "n 133"]
    """Source confidence, zero for missing joints."""


def hand_keypoints(scene: SequenceData) -> HandKeypoints:
    """Skin confidence-positive landmarks once for both layer writers."""
    landmarks: Float32[ndarray, "n 2 21 3"] = np.full((scene.count, 2, 21, 3), np.nan, dtype=np.float32)
    confidence: Float32[ndarray, "n 2"] = scene.labels.hand_confidences
    for hand_index in range(2):
        valid: Bool[ndarray, "n"] = confidence[:, hand_index] > 0
        angles: Float32[ndarray, "k 22"] = scene.labels.joint_angles[:, hand_index][valid]
        wrists: Float32[ndarray, "k 4 4"] = scene.labels.wrist_transforms[:, hand_index][valid]
        if valid.any():
            landmarks[valid, hand_index] = skin_landmarks(
                scene.labels.hand_model, angles, wrist_for_hand(wrists, hand_index)
            ) * np.float32(0.001)
    return HandKeypoints(*hands.confidence_rule(*hands.coco133_from_hands(landmarks, confidence)))


def write_hands(recording: rr.RecordingStream, scene: SequenceData, keypoints: HandKeypoints) -> None:
    """Publish shared hand keypoints and sparse source parameters."""
    rr.log(schema.hand_profile_path(), rr.TextDocument(scene.profile_text, media_type="application/json"), static=True, recording=recording)
    confidence: Float32[ndarray, "n 2"] = scene.labels.hand_confidences
    for hand_index, side in enumerate(hands.HAND_SIDES):
        valid: Bool[ndarray, "n"] = confidence[:, hand_index] > 0
        angles: Float32[ndarray, "k 22"] = scene.labels.joint_angles[:, hand_index][valid]
        wrists: Float32[ndarray, "k 4 4"] = scene.labels.wrist_transforms[:, hand_index][valid]
        hands.log_hand_confidence(
            recording, side.name, times_ns=scene.times_ns, frame_indices=scene.frame_indices, confidence=confidence[:, hand_index].astype(np.float64)
        )
        if valid.any():
            hands.log_joint_angles(recording, side.name, times_ns=scene.times_ns[valid], frame_indices=scene.frame_indices[valid], angles=angles)
            logging_toolkit.log_pose_track(
                recording,
                schema.hand_wrist_path(side.name),
                times_ns=scene.times_ns[valid],
                frame_indices=scene.frame_indices[valid],
                translations_xyz=wrists[:, :3, 3] * np.float32(0.001),
                quaternions_xyzw=Rotation.from_matrix(wrists[:, :3, :3]).as_quat(),
            )
    hands.log_keypoints3d(
        recording, times_ns=scene.times_ns, frame_indices=scene.frame_indices, positions=keypoints.positions, confidence=keypoints.confidence
    )


def write_meshes(recording: rr.RecordingStream, scene: SequenceData) -> None:
    """Skin confidence-positive hands with umetrack_hands.log_hand_meshes; every absent row is cleared."""
    for hand_index, side in enumerate(hands.HAND_SIDES):
        log_hand_meshes(
            recording,
            side,
            scene.labels.hand_model,
            scene.labels.joint_angles[:, hand_index],
            scene.labels.wrist_transforms[:, hand_index],
            scene.labels.hand_confidences[:, hand_index] > 0,
            times_ns=scene.times_ns,
            frame_indices=scene.frame_indices,
        )


def write_projections(recording: rr.RecordingStream, scene: SequenceData, keypoints: HandKeypoints) -> None:
    """Write lens-model projections on the hand-pose clock, clearing untracked frames."""

    def pixels_by_camera() -> Iterator[tuple[tuple[int, int], Float32[ndarray, "n 133 2"]]]:
        """Project one camera at a time, as the writer logs it."""
        for index, camera in enumerate(rig_cameras(scene)):
            world_T_cam: Float64[ndarray, "n 4 4"] = scene.world_T_rig @ scene.rig_T_cam[index]
            # Invert the rigid pose: R transpose times (world point minus camera origin).
            xyz_cam: Float64[ndarray, "n 133 3"] = np.einsum(
                "nji,nkj->nki", world_T_cam[:, :3, :3], keypoints.positions - world_T_cam[:, None, :3, 3]
            )
            projected: Float64[ndarray, "p 2"] = project_fisheye62(xyz_cam.reshape(-1, 3), camera)
            yield (0, index), projected.reshape(scene.count, 133, 2).astype(np.float32)

    hands.log_projections(
        recording,
        pixels_by_camera(),
        times_ns=scene.times_ns,
        frame_indices=scene.frame_indices,
        confidence=keypoints.confidence,
        camera_model=FISHEYE62,
    )