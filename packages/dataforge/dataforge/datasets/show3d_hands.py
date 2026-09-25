"""SHOW3D measured hands and derived meshes on the shared frame clock."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Bool, Float32, Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from serde import serde
from simplecv.umetrack_temp.generic_hand_model_numpy import (
    NUM_JOINTS_PER_HAND,
    NUM_LANDMARKS_PER_HAND,
    HandModelNumpy,
)

from dataforge import hands, logging_toolkit, schema, writing
from dataforge.datasets.show3d_source import (
    DEFAULT_CONFIDENCE,
    HAND_POSE_VERSION,
    HEADSET_CAMERAS,
    FrameClock,
    FrameInfo,
    agrees_with_frame,
    read_json,
    sparse_rows,
)
from dataforge.identity import SequenceIdentity
from dataforge.umetrack_hands import HAND_SIDES, log_hand_meshes


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

    @property
    def trusted(self) -> bool:
        """Above the Hub's default threshold; the one place that rule lives."""
        return self.confidence > DEFAULT_CONFIDENCE

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

    hand_poses: dict[str, HandPose]
    """Both hands, including confidence-zero records."""

    def __post_init__(self) -> None:
        if set(self.hand_poses) != {"0", "1"}:
            raise ValueError("hand frame must contain hands 0 and 1")


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


def high_confidence_coverage(confidence: list[float]) -> float:
    """Fraction of scene frames whose confidence is strictly greater than 0.5."""
    return sum(value > 0.5 for value in confidence) / len(confidence)


def write_hand_pose_layer(identity: SequenceIdentity, clock: FrameClock, selected: list[HandFrame], profile_text: str, target: Path) -> None:
    """Publish aligned measured hands and validated profile text on the base clocks."""
    with writing.atomic_recording(target, recording_id=identity.recording_id, send_properties=False) as recording:
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
            for hand_index, _side in enumerate(HAND_SIDES):
                pose: HandPose = frame.hand_poses[str(hand_index)]
                if pose.landmarks_3d_mm is not None and pose.trusted:
                    landmarks_lr[hand_index] = pose.landmarks_3d_mm * np.float32(0.001)
            # Checked UmeTrack LANDMARK against Assembly-Hands HAND_ID2NAME: tips 0–4,
            # wrist 5, thumb 6–7, finger joints 8–19, palm 20 have the same order.
            confidence_lr: Float32[ndarray, "2"] = np.array(
                [frame.hand_poses[str(hand_index)].confidence for hand_index, _side in enumerate(HAND_SIDES)], dtype=np.float32
            )
            xyz[frame_index], conf[frame_index] = hands.coco133_from_hands(landmarks_lr, confidence_lr)
        # The shared writer zeroes the confidence of every non-finite slot (e.g. an undefined thumb-base midpoint).
        hands.log_keypoints3d(recording, times_ns=clock.times_ns, frame_indices=clock.frame_indices, positions=xyz, confidence=conf)
        for camera in HEADSET_CAMERAS:
            uv: Float32[ndarray, "n 133 2"] = np.full((n_frames, 133, 2), np.nan, dtype=np.float32)
            uv_conf: Float32[ndarray, "n 133"] = np.zeros((n_frames, 133), dtype=np.float32)
            for frame_index, frame in enumerate(selected):
                pixels_lr: Float32[ndarray, "2 21 2"] = np.full((2, 21, 2), np.nan, dtype=np.float32)
                for hand_index, _side in enumerate(HAND_SIDES):
                    pose = frame.hand_poses[str(hand_index)]
                    pixels: list[list[float] | None] | None = (pose.landmarks_2d or {}).get(camera.source_name)
                    if pixels is not None and pose.trusted:
                        pixels_lr[hand_index, :, :2] = np.asarray(
                            [point if point is not None else [np.nan, np.nan] for point in pixels], dtype=np.float32
                        )
                confidence_lr = np.array([frame.hand_poses[str(hand_index)].confidence for hand_index, side in enumerate(HAND_SIDES)], dtype=np.float32)
                uv[frame_index], uv_conf[frame_index] = hands.coco133_from_hands(pixels_lr, confidence_lr)
            hands.log_keypoints2d(
                recording, camera.rig, camera.cam, times_ns=clock.times_ns, frame_indices=clock.frame_indices, positions=uv, confidence=uv_conf
            )
        coverage: dict[str, pa.Array] = {}
        for hand_index, side in enumerate(HAND_SIDES):
            poses: list[HandPose] = [frame.hand_poses[str(hand_index)] for frame in selected]
            confidence: list[float] = [pose.confidence for pose in poses]
            hands.log_hand_confidence(
                recording, side, times_ns=clock.times_ns, frame_indices=clock.frame_indices,
                confidence=np.asarray(confidence, dtype=np.float64),
            )
            coverage[f"coverage_{side}_high_conf"] = pa.array([high_confidence_coverage(confidence)], type=pa.float64())
            positions, values = sparse_rows(poses, lambda pose: pose.joint_angles)
            angles: Float32[ndarray, "n 22"] = np.asarray(values, dtype=np.float32).reshape(-1, NUM_JOINTS_PER_HAND)
            hands.log_joint_angles(recording, side, times_ns=clock.times_ns[positions], frame_indices=clock.frame_indices[positions], angles=angles)
            positions, values = sparse_rows(poses, lambda pose: pose.wrist_rotation)
            rotations: Float64[ndarray, "n 4"] = Rotation.from_matrix(np.asarray(values, dtype=np.float64).reshape(-1, 3, 3)).as_quat()
            translations: Float32[ndarray, "n 3"] = np.asarray(
                [poses[i].wrist_translation for i in positions], dtype=np.float32
            ).reshape(-1, 3) * np.float32(0.001)
            if positions:
                logging_toolkit.log_pose_track(
                    recording, schema.hand_wrist_path(side), times_ns=clock.times_ns[positions],
                    frame_indices=clock.frame_indices[positions], translations_xyz=translations, quaternions_xyzw=rotations,
                )
        recording.send_property("hand_pose", rr.AnyValues(version=pa.array([HAND_POSE_VERSION], type=pa.string()), **coverage))


def write_hand_mesh_layer(identity: SequenceIdentity, clock: FrameClock, frames: list[HandFrame], model: HandModelNumpy, target: Path) -> None:
    """Skin trusted hands in bounded batches; hand_pose owns the annotation context.

    The source ships a wrist and joint angles for many frames it marks with confidence 0
    (the tracker lost the hand) and for low-confidence frames whose landmarks float far
    from any hand. Those rows are kept verbatim in ``hand_pose``; this derived layer skins
    only frames with a wrist and confidence > ``DEFAULT_CONFIDENCE``, and writes an empty
    vertex row on every other frame so the viewer's latest-at never holds a stale mesh.
    """
    with writing.atomic_recording(target, recording_id=identity.recording_id, send_properties=False) as recording:
        for hand_index, side in enumerate(HAND_SIDES):
            poses: list[HandPose] = [frame.hand_poses[str(hand_index)] for frame in frames]
            trusted: Bool[ndarray, "n"] = np.array([pose.wrist_rotation is not None and pose.trusted for pose in poses], dtype=bool)
            angles: Float32[ndarray, "n 22"] = np.zeros((len(poses), NUM_JOINTS_PER_HAND), dtype=np.float32)
            wrists: Float32[ndarray, "n 4 4"] = np.zeros((len(poses), 4, 4), dtype=np.float32)
            for row in np.flatnonzero(trusted):
                pose: HandPose = poses[row]
                if pose.joint_angles is None:
                    raise ValueError(f"{identity.sequence_key}: posed {side} hand lacks joint angles")
                angles[row] = pose.joint_angles
                wrists[row, :3, :3] = pose.wrist_rotation
                wrists[row, :3, 3] = pose.wrist_translation
                wrists[row, 3, 3] = 1.0
            log_hand_meshes(recording, side, hand_index, model, angles, wrists, trusted, times_ns=clock.times_ns, frame_indices=clock.frame_indices)
