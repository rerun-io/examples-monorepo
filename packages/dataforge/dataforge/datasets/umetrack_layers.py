"""UmeTrack sensors, hand parameters and lens-model projections."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Bool, Complex128, Float32, Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.camera_parameters import Extrinsics, Fisheye62Parameters, Intrinsics, KannalaBrandtDistortion, apply_radial_tangential_distortion
from simplecv.umetrack_temp.generic_hand_model_numpy import skin_landmarks, wrist_for_hand

from dataforge import hands, logging_toolkit, schema, writing
from dataforge.datasets.umetrack_source import Camera, SequenceData
from dataforge.identity import SequenceIdentity
from dataforge.timing import SequenceTimer
from dataforge.umetrack_hands import HAND_SIDES
from dataforge.video_encoding import AV1_CQ, AV1_GOP, parallel_clips, transcode_mp4_gray
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
    logging_toolkit.log_rig_node(recording, 0, reference="cam_00", num_cameras=4, name="UmeTrack headset", kind="ego")
    for index, camera in enumerate(rig_cameras(scene)):
        source: Camera = scene.labels.cameras[index]
        logging_toolkit.log_camera_node(
            recording, 0, index, camera, name=camera.name, kind="grayscale", image_plane_distance=0.05, camera_model="FishEye62"
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
    source: Camera = scene.labels.cameras[0]
    writing.send_capture_properties(
        recording,
        identity,
        num_frames=scene.count,
        num_cameras=4,
        source_revision="github.com/facebookresearch/UmeTrack_data@main (mirrored 2025-10-31)",
        source_resolution=f"{source.ImageSizeX * 4}x{source.ImageSizeY}",
        source_num_frames=scene.source_num_frames,
        clock_source="mp4_container_pts",
        fps=scene.fps,
        world_up_axis="+y",
        headset_up=pa.array(scene.headset_up),
        headset_up_spread_deg=scene.headset_up_spread_deg,
    )
    domain, interaction, split, user, _ = identity.parts
    recording.send_property("episode", rr.AnyValues(domain=domain, interaction=interaction, split=split, user=user))


def write_base(recording: rr.RecordingStream, scene: SequenceData, identity: SequenceIdentity, timer: SequenceTimer, work_root: Path) -> None:
    """Crop four grayscale streams and remux them with the source presentation times."""
    write_geometry(recording, scene, identity)
    work_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="umetrack-", dir=work_root) as work:
        clips: list[Path] = [Path(work) / f"cam_{index:02}.mp4" for index in range(4)]

        def encode(index: int, clip: Path) -> None:
            transcode_mp4_gray(
                scene.source.with_suffix(".mp4"), clip, gop=AV1_GOP, cq=AV1_CQ, fps=scene.fps, frames=scene.count, crop=scene.crop(index)
            )

        jobs: list[tuple[Path, Callable[[], None]]] = [(clip, partial(encode, index, clip)) for index, clip in enumerate(clips)]
        with parallel_clips(jobs, timer) as ready:
            for index, clip in enumerate(ready):
                logging_toolkit.log_video_stream(
                    recording, clip, schema.video_path(0, index), times_ns=scene.times_ns, frame_indices=scene.frame_indices
                )
                clip.unlink()


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
    xyz: Float32[ndarray, "n 133 3"] = np.full((scene.count, 133, 3), np.nan, dtype=np.float32)
    conf: Float32[ndarray, "n 133"] = np.zeros((scene.count, 133), dtype=np.float32)
    for index in range(scene.count):
        xyz[index], conf[index] = hands.coco133_from_hands(landmarks[index], confidence[index])
    return HandKeypoints(*hands.confidence_rule(xyz, conf))


def write_hands(recording: rr.RecordingStream, scene: SequenceData, keypoints: HandKeypoints | None = None) -> None:
    """Skin only confidence-positive landmarks and publish sparse source parameters."""
    rr.log(schema.hand_profile_path(), rr.TextDocument(scene.profile_text, media_type="application/json"), static=True, recording=recording)
    if keypoints is None:
        keypoints = hand_keypoints(scene)
    confidence: Float32[ndarray, "n 2"] = scene.labels.hand_confidences
    for hand_index, side in enumerate(HAND_SIDES):
        valid: Bool[ndarray, "n"] = confidence[:, hand_index] > 0
        angles: Float32[ndarray, "k 22"] = scene.labels.joint_angles[:, hand_index][valid]
        wrists: Float32[ndarray, "k 4 4"] = scene.labels.wrist_transforms[:, hand_index][valid]
        hands.log_hand_confidence(
            recording, side, times_ns=scene.times_ns, frame_indices=scene.frame_indices, confidence=confidence[:, hand_index].astype(np.float64)
        )
        if valid.any():
            hands.log_joint_angles(recording, side, times_ns=scene.times_ns[valid], frame_indices=scene.frame_indices[valid], angles=angles)
            logging_toolkit.log_pose_track(
                recording,
                schema.hand_wrist_path(side),
                times_ns=scene.times_ns[valid],
                frame_indices=scene.frame_indices[valid],
                translations_xyz=wrists[:, :3, 3] * np.float32(0.001),
                quaternions_xyzw=Rotation.from_matrix(wrists[:, :3, :3]).as_quat(),
            )
    hands.log_keypoints3d(
        recording, times_ns=scene.times_ns, frame_indices=scene.frame_indices, positions=keypoints.positions, confidence=keypoints.confidence
    )


def project_fisheye62(
    xyz_cam: Float64[ndarray, "n 3"], camera: Fisheye62Parameters
) -> tuple[Float64[ndarray, "n 2"], Bool[ndarray, "n"]]:
    """Project Float64[n,3] camera metres to Float64[n,2] pixels and a Bool[n] validity mask.

    The first radial derivative root bounds the monotonic field of view. Call once
    per camera with all frames flattened, so its bound is computed only once.
    Invalid inputs, rear rays, folded rays and out-of-image pixels become NaN.
    """
    lens: KannalaBrandtDistortion | None = camera.distortion
    assert lens is not None
    radial: tuple[float, ...] = (lens.k1, lens.k2, lens.k3, lens.k4, lens.k5, lens.k6)
    # The derivative is a degree-six polynomial in theta squared.
    roots: Complex128[ndarray, "r"] = np.polynomial.polynomial.polyroots(
        [1.0, *((2 * i + 1) * k for i, k in enumerate(radial, 1))]
    ).astype(np.complex128)
    bounds: list[float] = [float(np.sqrt(root.real)) for root in roots if abs(root.imag) < 1e-10 and 0 < root.real <= (np.pi / 2) ** 2]
    theta_max: float = min(bounds, default=float(np.pi / 2))
    radius: Float64[ndarray, "n"] = np.hypot(xyz_cam[:, 0], xyz_cam[:, 1])
    theta: Float64[ndarray, "n"] = np.arctan2(radius, xyz_cam[:, 2])
    valid: Bool[ndarray, "n"] = np.isfinite(xyz_cam).all(axis=1) & (xyz_cam[:, 2] > 0) & (theta < theta_max)
    scale: Float64[ndarray, "n"] = np.divide(theta, radius, out=np.zeros_like(theta), where=radius > 0)
    normalized: Float64[ndarray, "n 2"] = xyz_cam[:, :2] * scale[:, None]
    normalized[~valid] = 0.0
    distorted: Float64[ndarray, "n 2"] = apply_radial_tangential_distortion(lens, normalized)
    intrinsics: Intrinsics = camera.intrinsics
    pixels: Float64[ndarray, "n 2"] = distorted * [intrinsics.fl_x, intrinsics.fl_y] + [intrinsics.cx, intrinsics.cy]
    valid &= np.isfinite(pixels).all(axis=1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < intrinsics.width)
    valid &= (pixels[:, 1] >= 0) & (pixels[:, 1] < intrinsics.height)
    pixels[~valid] = np.nan
    return pixels, valid


def write_projections(recording: rr.RecordingStream, scene: SequenceData, keypoints: HandKeypoints | None = None) -> None:
    """Write lens-model projections on the hand-pose clock, clearing untracked frames."""
    if keypoints is None:
        keypoints = hand_keypoints(scene)
    recording.send_property("projections", rr.AnyValues(derived_from="coco133_xyz", camera_model="FishEye62"))
    for index, camera in enumerate(rig_cameras(scene)):
        world_T_cam: Float64[ndarray, "n 4 4"] = scene.world_T_rig @ scene.rig_T_cam[index]
        # Invert the rigid pose: R transpose times (world point minus camera origin).
        xyz_cam: Float64[ndarray, "n 133 3"] = np.einsum(
            "nji,nkj->nki", world_T_cam[:, :3, :3], keypoints.positions - world_T_cam[:, None, :3, 3]
        )
        xyz_cam[~scene.tracked] = np.nan
        projected: tuple[Float64[ndarray, "p 2"], Bool[ndarray, "p"]] = project_fisheye62(xyz_cam.reshape(-1, 3), camera)
        pixels: Float32[ndarray, "n 133 2"] = projected[0].reshape(scene.count, 133, 2).astype(np.float32)
        hands.log_projected_keypoints2d(
            recording, 0, index, times_ns=scene.times_ns, frame_indices=scene.frame_indices, positions=pixels, confidence=keypoints.confidence
        )
