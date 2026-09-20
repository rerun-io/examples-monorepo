"""SHOW3D measured hands and derived meshes on the shared frame clock."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import pyarrow as pa
import rerun as rr
from einops import rearrange
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray
from scipy.spatial.transform import Rotation
from serde import serde
from simplecv.data.skeleton.assembly_hands import assembly21_to_coco133
from simplecv.data.skeleton.coco_133 import COCO_133_ID2NAME, COCO_133_IDS, COCO_133_LINKS
from simplecv.rerun_custom_types import Points2DWithConfidence, Points3DWithConfidence, confidence_scores_to_rgb
from simplecv.umetrack_temp.generic_hand_model_numpy import (
    LEFT_HAND_INDEX,
    NUM_JOINTS_PER_HAND,
    NUM_LANDMARKS_PER_HAND,
    RIGHT_HAND_INDEX,
    HandModelNumpy,
    skin_mesh,
    wrist_for_hand,
)

from dataforge import schema, writing
from dataforge.datasets.show3d_source import HAND_POSE_VERSION, HEADSET_CAMERAS, FrameClock, FrameInfo, agrees_with_frame, read_json, sparse_rows
from dataforge.identity import SequenceIdentity


@serde
@dataclass(frozen=True, slots=True)
class HandPose:
    """One hand; each optional measurement has its own availability."""

    confidence: float
    """Tracking confidence, including zero on lost frames."""
    joint_angles: Float32[ndarray, "22"] | None
    """UmeTrack joint angles in radians."""
    wrist_rotation: Float32[ndarray, "3 3"] | None
    """World-from-wrist rotation."""
    wrist_translation: Float32[ndarray, "3"] | None
    """World wrist position in millimetres."""
    landmarks_3d_mm: Float32[ndarray, "21 3"] | None
    """World landmarks in millimetres."""
    landmarks_2d: dict[str, list[list[float] | None]] | None
    """Headset pixels, with null entries outside the image."""

    def __post_init__(self) -> None:
        if not isfinite(self.confidence):
            raise ValueError("hand confidence must be finite")
        if (self.wrist_rotation is None) != (self.wrist_translation is None):
            raise ValueError("wrist rotation and translation must be present together")
        for camera, landmarks in (self.landmarks_2d or {}).items():
            if camera not in {c.source_name for c in HEADSET_CAMERAS} or len(landmarks) != NUM_LANDMARKS_PER_HAND:
                raise ValueError("UV landmarks require a headset camera and 21 entries")
            if any(point is not None and (len(point) != 2 or not all(isfinite(value) for value in point)) for point in landmarks):
                raise ValueError("UV landmarks must be finite pixel pairs or null")


@serde
@dataclass(frozen=True, slots=True)
class HandFrame(FrameInfo):
    """A source frame with left (0) and right (1) hand measurements."""

    hand_poses: dict[Literal["0", "1"], HandPose]
    """Both hands, including confidence-zero records."""

    def __post_init__(self) -> None:
        if set(self.hand_poses) != {"0", "1"}:
            raise ValueError("hand frame must contain hands 0 and 1")


@dataclass(frozen=True, slots=True)
class HandSide:
    """Source identity and rendering settings for one hand."""

    key: Literal["0", "1"]
    """Source hand key."""
    name: str
    """Schema side name."""
    model_index: int
    """UmeTrack handedness index."""
    albedo: tuple[int, int, int, int]
    """Mesh RGBA color."""


HAND_SIDES: tuple[HandSide, HandSide] = (
    HandSide("0", "left", LEFT_HAND_INDEX, (90, 160, 240, 110)),
    HandSide("1", "right", RIGHT_HAND_INDEX, (240, 170, 130, 110)),
)

def read_hand_frames(hand_path: Path, clock: FrameClock) -> list[HandFrame]:
    """Decode once and align hand records with the metadata frame clock."""
    frames: dict[str, HandFrame] = read_json(hand_path, dict[str, HandFrame])
    if len(frames) != clock.info.num_frames:
        raise ValueError(f"{hand_path}: {len(frames)} hand frames != scene census {clock.info.num_frames}")
    for key, frame in frames.items():
        if key != str(frame.index):
            raise ValueError(f"{hand_path}: frame key {key} disagrees with index {frame.index}")
    selected: list[HandFrame] = []
    for base in clock.frames:
        hand: HandFrame | None = frames.get(str(base.index))
        if hand is None or not agrees_with_frame(hand, base):
            raise ValueError(f"{hand_path}: hand frame {base.index} disagrees with base sidecars")
        selected.append(hand)
    return selected


@serde
@dataclass(frozen=True, slots=True)
class HandProfile:
    """Full typed subject model; preserve its source text separately."""

    hand_model: HandModelNumpy
    """Float32 UmeTrack rest geometry and int64 topology."""

    def __post_init__(self) -> None:
        if len(self.hand_model.landmark_rest_positions) != NUM_LANDMARKS_PER_HAND:
            raise ValueError("hand profile requires 21 rest landmarks")
        if len(self.hand_model.joint_rotation_axes) != NUM_JOINTS_PER_HAND:
            raise ValueError("hand profile requires 22 joint rotation axes")


class HandProfileDoc(NamedTuple):
    """Verbatim profile text and its validated model."""

    text: str
    model: HandModelNumpy


def read_hand_profile(path: Path) -> HandProfileDoc:
    """Read and validate a profile once, retaining its verbatim document."""
    text: str = path.read_text()
    profile: HandProfile = read_json(path, HandProfile, text=text)
    return HandProfileDoc(text, profile.hand_model)



SKINNING_BATCH_SIZE: int = 256
"""Bound skinning workspace to less than 20 MB."""
MESH_CONFIDENCE: float = 0.5
"""Skin a hand only above this confidence: the Hub README's default threshold ("filters most solver
failures without throwing away usable data"); ``> 0`` includes "low-quality frames you usually want to drop"."""


def high_confidence_coverage(confidence: list[float]) -> float:
    """Fraction of scene frames whose confidence is strictly greater than 0.5."""
    return sum(value > 0.5 for value in confidence) / len(confidence)


def write_hand_pose_layer(identity: SequenceIdentity, clock: FrameClock, selected: list[HandFrame], profile_text: str, target: Path) -> None:
    """Publish aligned measured hands and validated profile text on the base clocks."""
    with writing.atomic_recording(target, recording_id=identity.recording_id, send_properties=False) as recording:
        rr.log(
            "/",
            rr.AnnotationContext(
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label="Coco Wholebody", color=(0, 0, 255)),
                    keypoint_annotations=[rr.AnnotationInfo(id=point, label=name) for point, name in COCO_133_ID2NAME.items()],
                    keypoint_connections=COCO_133_LINKS,
                )
            ),
            static=True,
            recording=recording,
        )
        rr.log(
            schema.hand_profile_path(),
            rr.TextDocument(profile_text, media_type="application/json"),
            static=True,
            recording=recording,
        )
        n_frames: int = len(selected)
        xyz: Float32[ndarray, "n 133 3"] = np.full((n_frames, 133, 3), np.nan, dtype=np.float32)
        conf: Float32[ndarray, "n 133"] = np.zeros((n_frames, 133), dtype=np.float32)
        for frame_index, frame in enumerate(selected):
            landmarks_lr: Float32[ndarray, "2 21 3"] = np.full((2, 21, 3), np.nan, dtype=np.float32)
            for hand_index, side in enumerate(HAND_SIDES):
                pose: HandPose = frame.hand_poses[side.key]
                if pose.landmarks_3d_mm is not None:
                    landmarks_lr[hand_index] = pose.landmarks_3d_mm * np.float32(0.001)
            # Checked UmeTrack LANDMARK against Assembly-Hands HAND_ID2NAME: tips 0–4,
            # wrist 5, thumb 6–7, finger joints 8–19, palm 20 have the same order.
            xyz[frame_index] = assembly21_to_coco133(landmarks_lr)[:, :3]
            for hand_index, side in enumerate(HAND_SIDES):
                pose = frame.hand_poses[side.key]
                if pose.landmarks_3d_mm is not None:
                    offset: int = 91 + hand_index * 21
                    conf[frame_index, offset : offset + 21] = np.float32(pose.confidence)
                    conf[frame_index, 9 + hand_index] = np.float32(pose.confidence)
                    if not np.isfinite(xyz[frame_index, offset + 1]).all():
                        conf[frame_index, offset + 1] = np.float32(0.0)
        flat_xyz: Float32[ndarray, "n 3"] = rearrange(xyz, "f k d -> (f k) d")
        flat_conf: Float32[ndarray, "n"] = rearrange(conf, "f k -> (f k)")
        colors: UInt8[ndarray, "n 3"] = confidence_scores_to_rgb(flat_conf[None, :, None])[0]
        rr.log(
            schema.coco133_xyz_path(),
            Points3DWithConfidence.from_fields(class_ids=0, keypoint_ids=COCO_133_IDS, show_labels=False, radii=0.004),
            static=True,
            recording=recording,
        )
        rr.send_columns(
            schema.coco133_xyz_path(),
            indexes=clock.indexes(slice(None)),
            columns=Points3DWithConfidence.columns(positions=flat_xyz, confidences=flat_conf, colors=colors).partition([133] * n_frames),
            recording=recording,
        )
        for camera in HEADSET_CAMERAS:
            uv: Float32[ndarray, "n 133 2"] = np.full((n_frames, 133, 2), np.nan, dtype=np.float32)
            uv_conf: Float32[ndarray, "n 133"] = np.zeros((n_frames, 133), dtype=np.float32)
            for frame_index, frame in enumerate(selected):
                pixels_lr: Float32[ndarray, "2 21 3"] = np.full((2, 21, 3), np.nan, dtype=np.float32)
                pixels_lr[:, :, 2] = np.float32(0.0)
                for hand_index, side in enumerate(HAND_SIDES):
                    pose = frame.hand_poses[side.key]
                    pixels: list[list[float] | None] | None = (pose.landmarks_2d or {}).get(camera.source_name)
                    if pixels is not None:
                        pixels_lr[hand_index, :, :2] = np.asarray(
                            [point if point is not None else [np.nan, np.nan] for point in pixels], dtype=np.float32
                        )
                uv[frame_index] = assembly21_to_coco133(pixels_lr)[:, :2]
                for hand_index, side in enumerate(HAND_SIDES):
                    offset = 91 + hand_index * 21
                    uv_conf[frame_index, offset : offset + 21] = np.float32(frame.hand_poses[side.key].confidence)
                    uv_conf[frame_index, 9 + hand_index] = np.float32(frame.hand_poses[side.key].confidence)
                uv_conf[frame_index, ~np.isfinite(uv[frame_index]).all(axis=1)] = np.float32(0.0)
            path: str = schema.coco133_uv_path(camera.rig, camera.cam)
            rr.log(
                path,
                Points2DWithConfidence.from_fields(class_ids=0, keypoint_ids=COCO_133_IDS, show_labels=False, radii=3.0),
                static=True,
                recording=recording,
            )
            flat_uv: Float32[ndarray, "n 2"] = rearrange(uv, "f k d -> (f k) d")
            flat_uv_conf: Float32[ndarray, "n"] = rearrange(uv_conf, "f k -> (f k)")
            rr.send_columns(
                path,
                indexes=clock.indexes(slice(None)),
                columns=Points2DWithConfidence.columns(positions=flat_uv, confidences=flat_uv_conf).partition([133] * n_frames),
                recording=recording,
            )
        coverage: dict[str, pa.Array] = {}
        for side in HAND_SIDES:
            poses: list[HandPose] = [frame.hand_poses[side.key] for frame in selected]
            confidence: list[float] = [pose.confidence for pose in poses]
            rr.send_columns(
                schema.hand_confidence_path(side.name),
                indexes=clock.indexes(slice(None)),
                columns=rr.Scalars.columns(scalars=confidence),
                recording=recording,
            )
            coverage[f"coverage_{side.name}_high_conf"] = pa.array([high_confidence_coverage(confidence)], type=pa.float64())
            positions, values = sparse_rows(poses, lambda pose: pose.joint_angles)
            angles: pa.Array = pa.array([value.tolist() for value in values], type=pa.list_(pa.float32(), NUM_JOINTS_PER_HAND))
            clock.send_sparse(recording, schema.hand_joint_angles_path(side.name), positions, rr.AnyValues.columns(joint_angles=angles))
            positions, values = sparse_rows(poses, lambda pose: pose.wrist_rotation)
            rotations: Float64[ndarray, "n 4"] = Rotation.from_matrix(np.asarray(values, dtype=np.float64).reshape(-1, 3, 3)).as_quat()
            translations: Float32[ndarray, "n 3"] = np.asarray(
                [poses[i].wrist_translation for i in positions], dtype=np.float32
            ).reshape(-1, 3) * np.float32(0.001)
            clock.send_sparse(
                recording, schema.hand_wrist_path(side.name), positions,
                rr.Transform3D.columns(translation=translations, quaternion=rotations),
            )
        recording.send_property("hand_pose", rr.AnyValues(version=pa.array([HAND_POSE_VERSION], type=pa.string()), **coverage))


def write_hand_mesh_layer(identity: SequenceIdentity, clock: FrameClock, frames: list[HandFrame], model: HandModelNumpy, target: Path) -> None:
    """Skin trusted hands in bounded batches; hand_pose owns the annotation context.

    The source ships a wrist and joint angles for many frames it marks with confidence 0
    (the tracker lost the hand) and for low-confidence frames whose landmarks float far
    from any hand. Those rows are kept verbatim in ``hand_pose``; this derived layer skins
    only frames with a wrist and confidence > ``MESH_CONFIDENCE``, and writes an empty
    vertex row on every other frame so the viewer's latest-at never holds a stale mesh.
    """
    with writing.atomic_recording(target, recording_id=identity.recording_id, send_properties=False) as recording:
        for side in HAND_SIDES:
            path: str = schema.hand_mesh_path(side.name)
            rr.log(
                path,
                rr.Mesh3D.from_fields(triangle_indices=model.mesh_triangles, albedo_factor=side.albedo),
                static=True,
                recording=recording,
            )
            poses: list[HandPose] = [frame.hand_poses[side.key] for frame in frames]
            trusted: list[bool] = [pose.wrist_rotation is not None and pose.confidence > MESH_CONFIDENCE for pose in poses]
            if any(pose.joint_angles is None for pose, ok in zip(poses, trusted, strict=True) if ok):
                raise ValueError(f"{identity.sequence_key}: posed {side.name} hand lacks joint angles")
            for start in range(0, len(frames), SKINNING_BATCH_SIZE):
                stop: int = min(start + SKINNING_BATCH_SIZE, len(frames))
                batch_poses: list[HandPose] = [poses[i] for i in range(start, stop) if trusted[i]]
                vertices: Float32[ndarray, "k v 3"] = np.zeros((0, len(model.mesh_vertices), 3), dtype=np.float32)
                if batch_poses:
                    angles: Float32[ndarray, "k 22"] = np.asarray([pose.joint_angles for pose in batch_poses], dtype=np.float32)
                    wrists: Float32[ndarray, "k 4 4"] = np.zeros((len(batch_poses), 4, 4), dtype=np.float32)
                    wrists[:, :3, :3] = np.asarray([pose.wrist_rotation for pose in batch_poses], dtype=np.float32)
                    wrists[:, :3, 3] = np.asarray([pose.wrist_translation for pose in batch_poses], dtype=np.float32)
                    wrists[:, 3, 3] = 1.0
                    vertices = skin_mesh(model, angles, wrist_for_hand(wrists, side.model_index)) * np.float32(0.001)
                lengths: list[int] = [len(model.mesh_vertices) if trusted[i] else 0 for i in range(start, stop)]
                rr.send_columns(
                    path,
                    indexes=clock.indexes(slice(start, stop)),
                    columns=rr.Mesh3D.columns(vertex_positions=vertices.reshape(-1, 3)).partition(lengths),
                    recording=recording,
                )
