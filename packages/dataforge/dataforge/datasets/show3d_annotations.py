"""Independent SHOW3D hand_pose, captions, and searchable properties layers."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TypeVar

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Float32, Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.umetrack_temp.generic_hand_model_numpy import LANDMARK, NUM_JOINTS_PER_HAND, NUM_LANDMARKS_PER_HAND, UME_HAND_CONNECTIONS

from dataforge import schema, writing
from dataforge.datasets.show3d_annotation_source import HAND_SIDES, Caption, HandFrame, HandPose
from dataforge.datasets.show3d_source import CAMERAS, CAPTIONS_VERSION, HAND_POSE_VERSION, FrameClock, IndexRow
from dataforge.identity import SequenceIdentity


def high_confidence_coverage(confidence: list[float]) -> float:
    """Fraction of scene frames whose confidence is strictly greater than 0.5."""
    return sum(value > 0.5 for value in confidence) / len(confidence)


@dataclass(frozen=True, slots=True)
class EpisodeProperties:
    """Stable string schema for searchable scene metadata."""

    subject_id: str
    """Upstream subject."""
    split: str
    """Dataset split."""
    object_alias: str
    """Object token from the scene ID."""
    action: str
    """Action from the scene ID."""
    hand: str
    """Caption hand selection, or empty."""
    overall_caption: str
    """Caption summary, or empty."""
    hand_pose_version: str
    """Available hand annotation version, or empty."""
    object_pose_version: str
    """Available object annotation version, or empty."""
    captions_version: str
    """Available caption version, or empty."""


def write_properties_layer(identity: SequenceIdentity, source: IndexRow, caption: Caption | None, target: Path) -> None:
    """Publish one episode property chunk with a stable string schema."""
    values: EpisodeProperties = EpisodeProperties(
        subject_id=source.subject_id,
        split=source.split,
        object_alias=source.object_alias,
        action=source.action,
        hand=caption.hand if caption else "",
        overall_caption=caption.overall_caption if caption else "",
        hand_pose_version=HAND_POSE_VERSION if source.has_hand_pose else "",
        object_pose_version="v1" if source.has_object_pose else "",
        captions_version=CAPTIONS_VERSION if source.has_caption else "",
    )
    with writing.atomic_recording(target, recording_id=identity.recording_id, send_properties=False) as recording:
        recording.send_property("episode", rr.AnyValues(**{f.name: pa.array([getattr(values, f.name)], pa.string()) for f in fields(values)}))


T = TypeVar("T")


def sparse_rows(poses: list[HandPose], getter: Callable[[HandPose], T | None]) -> tuple[list[int], list[T]]:  # noqa: UP047
    """Select sparse measurements and their positions on the shared clock."""
    positions: list[int] = []
    values: list[T] = []
    for position, pose in enumerate(poses):
        value: T | None = getter(pose)
        if value is not None:
            positions.append(position)
            values.append(value)
    return positions, values


def write_captions_layer(identity: SequenceIdentity, caption: Caption, target: Path) -> None:
    """Publish the static instruction and caption provenance."""
    with writing.atomic_recording(target, recording_id=identity.recording_id, send_properties=False) as recording:
        rr.log(schema.instruction_path(), rr.TextDocument(caption.markdown(), media_type="text/markdown"), static=True, recording=recording)
        recording.send_property(
            "captions", rr.AnyValues(version=pa.array([CAPTIONS_VERSION], type=pa.string()), hand=pa.array([caption.hand], type=pa.string()))
        )


def send_sparse(recording: rr.RecordingStream, path: str, clock: FrameClock, positions: list[int], columns: Iterable[rr.ComponentColumn]) -> None:
    """Send available rows on both recording clocks; omit absent measurements."""
    if positions:
        rr.send_columns(path, indexes=clock.indexes(positions), columns=columns, recording=recording)


def write_hand_pose_layer(identity: SequenceIdentity, clock: FrameClock, selected: list[HandFrame], profile_text: str, target: Path) -> None:
    """Publish aligned measured hands and validated profile text on the base clocks."""
    with writing.atomic_recording(target, recording_id=identity.recording_id, send_properties=False) as recording:
        rr.log(
            "/",
            rr.AnnotationContext(
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=1, label="UmeTrack hand"),
                    keypoint_annotations=[rr.AnnotationInfo(id=int(point), label=point.name) for point in LANDMARK],
                    keypoint_connections=sorted(UME_HAND_CONNECTIONS),
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
        coverage: dict[str, pa.Array] = {}
        for key, side in HAND_SIDES:
            poses: list[HandPose] = [frame.hand_poses[key] for frame in selected]
            root: str = schema.hands_path(side)
            confidence: list[float] = [pose.confidence for pose in poses]
            rr.send_columns(
                root + "/confidence", indexes=clock.indexes(slice(None)), columns=rr.Scalars.columns(scalars=confidence), recording=recording
            )
            coverage[f"coverage_{side}_high_conf"] = pa.array([high_confidence_coverage(confidence)], type=pa.float64())
            positions, values = sparse_rows(poses, lambda pose: pose.landmarks_3d_mm)
            rr.log(
                root + "/landmarks",
                rr.Points3D.from_fields(class_ids=1, keypoint_ids=list(range(NUM_LANDMARKS_PER_HAND)), show_labels=False, radii=0.004),
                static=True,
                recording=recording,
            )
            points: Float32[ndarray, "n 3"] = np.asarray(values, dtype=np.float32).reshape(-1, 3) * np.float32(0.001)
            send_sparse(
                recording,
                root + "/landmarks",
                clock,
                positions,
                rr.Points3D.columns(positions=points).partition([NUM_LANDMARKS_PER_HAND] * len(positions)),
            )
            positions, values = sparse_rows(poses, lambda pose: pose.joint_angles)
            angles: pa.Array = pa.array([value.tolist() for value in values], type=pa.list_(pa.float32(), NUM_JOINTS_PER_HAND))
            send_sparse(recording, root + "/joint_angles", clock, positions, rr.AnyValues.columns(joint_angles=angles))
            positions, values = sparse_rows(poses, lambda pose: pose.wrist_rotation)
            rotations: Float64[ndarray, "n 4"] = np.asarray([Rotation.from_matrix(value).as_quat() for value in values], dtype=np.float64).reshape(
                -1, 4
            )
            translations: Float32[ndarray, "n 3"] = np.asarray([poses[i].wrist_translation for i in positions], dtype=np.float32).reshape(
                -1, 3
            ) * np.float32(0.001)
            send_sparse(recording, root + "/wrist", clock, positions, rr.Transform3D.columns(translation=translations, quaternion=rotations))
            for camera in [c for c in CAMERAS if c.rig == 1]:
                positions, values = sparse_rows(
                    poses, lambda pose, camera=camera: pose.landmarks_2d.get(camera.source_name) if pose.landmarks_2d is not None else None
                )
                path: str = schema.hand_uv_path(camera.rig, camera.cam, side)
                rr.log(
                    path,
                    rr.Points2D.from_fields(class_ids=1, keypoint_ids=list(range(NUM_LANDMARKS_PER_HAND)), show_labels=False, radii=3.0),
                    static=True,
                    recording=recording,
                )
                uv: Float32[ndarray, "n 2"] = np.array(
                    [[point if point is not None else [np.nan, np.nan] for point in row] for row in values], dtype=np.float32
                ).reshape(-1, 2)
                send_sparse(recording, path, clock, positions, rr.Points2D.columns(positions=uv).partition([NUM_LANDMARKS_PER_HAND] * len(positions)))
        recording.send_property("hand_pose", rr.AnyValues(version=pa.array([HAND_POSE_VERSION], type=pa.string()), **coverage))
