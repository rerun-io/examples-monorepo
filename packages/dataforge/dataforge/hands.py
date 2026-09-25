"""Shared hand measurements; positions are metres or shipped image pixels.

Readers retain simplecv's ``assembly21_to_coco133`` mapping: COCO thumb-base
slots 92/113 are midpoints, and wrist slots 9/10 copy hand wrists 91/112.
These derived slots carry the source hand confidence, as SHOW3D does. Class 0
uses the root COCO-133 AnnotationContext. No writer projects 3D into 2D.
"""

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Bool, Float32, Float64, UInt8
from numpy import ndarray
from simplecv.data.skeleton.coco133_layers import COCO133_ROI_COLORS, COCO133_ROI_LABELS, Coco133RoiLayer
from simplecv.data.skeleton.coco_133 import COCO_133_ID2NAME, COCO_133_IDS, COCO_133_LINKS
from simplecv.rerun_custom_types import Points2DWithConfidence, Points3DWithConfidence, confidence_scores_to_rgb

from dataforge import schema


def confidence_rule(
    positions: Float32[ndarray, "t 133 d"], confidence: Float32[ndarray, "t 133"] | None
) -> tuple[Float32[ndarray, "t 133 d"], Float32[ndarray, "t 133"]]:
    """Apply the port-wide keypoint rule to dense COCO rows.

    A present joint keeps its shipped confidence, or 1.0 when the source ships none
    (explicit None). A joint with any non-finite coordinate is missing: every
    coordinate becomes NaN and its confidence 0.0.

    Args:
        positions: Float32[ndarray, "t 133 d"], metres (d=3) or shipped pixels (d=2).
        confidence: Float32[ndarray, "t 133"] or None.

    Returns:
        The masked positions and the Float32[ndarray, "t 133"] confidence.
    """
    if confidence is not None and confidence.shape != positions.shape[:2]:
        raise ValueError("confidence must match the keypoint rows")
    valid: Bool[ndarray, "t 133"] = np.isfinite(positions).all(axis=-1)
    masked: Float32[ndarray, "t 133 d"] = np.where(valid[..., None], positions, np.float32(np.nan))
    scores: Float32[ndarray, "t 133"] = np.where(valid, np.float32(1.0) if confidence is None else confidence, np.float32(0.0))
    return masked, scores


def log_keypoints3d(
    recording: rr.RecordingStream,
    indexes: list[rr.TimeColumn],
    positions: Float32[ndarray, "t 133 3"],
    confidence: Float32[ndarray, "t 133"] | None,
) -> None:
    """Write dense COCO rows; explicit None means the source ships no confidence.

    Args:
        recording: Destination stream.
        indexes: Source timelines.
        positions: Float32[ndarray, "t 133 3"], metres; missing slots contain NaN.
        confidence: Float32[ndarray, "t 133"] or None; preserve shipped values.
    """
    ruled: tuple[Float32[ndarray, "t 133 3"], Float32[ndarray, "t 133"]] = confidence_rule(positions, confidence)
    xyz: Float32[ndarray, "t 133 3"] = ruled[0]
    flat_conf: Float32[ndarray, "n"] = ruled[1].reshape(-1)
    colors: UInt8[ndarray, "n 3"] = confidence_scores_to_rgb(flat_conf[None, :, None])[0]
    rr.log(
        schema.coco133_xyz_path(),
        Points3DWithConfidence.from_fields(class_ids=0, keypoint_ids=COCO_133_IDS, show_labels=False, radii=0.004),
        static=True,
        recording=recording,
    )
    rr.send_columns(
        schema.coco133_xyz_path(),
        indexes=indexes,
        columns=Points3DWithConfidence.columns(positions=xyz.reshape(-1, 3), confidences=flat_conf, colors=colors).partition([133] * len(positions)),
        recording=recording,
    )


def log_keypoints2d(
    recording: rr.RecordingStream,
    path: str,
    indexes: list[rr.TimeColumn],
    positions: Float32[ndarray, "t 133 2"],
    confidence: Float32[ndarray, "t 133"] | None,
) -> None:
    """Write shipped Float32[t,133,2] pixels with Float32[t,133] confidence or explicit None."""
    ruled: tuple[Float32[ndarray, "t 133 2"], Float32[ndarray, "t 133"]] = confidence_rule(positions, confidence)
    uv: Float32[ndarray, "t 133 2"] = ruled[0]
    scores: Float32[ndarray, "t 133"] = ruled[1]
    rr.log(
        path,
        Points2DWithConfidence.from_fields(class_ids=0, keypoint_ids=COCO_133_IDS, show_labels=False, radii=3.0),
        static=True,
        recording=recording,
    )
    rr.send_columns(
        path,
        indexes=indexes,
        columns=Points2DWithConfidence.columns(positions=uv.reshape(-1, 2), confidences=scores.reshape(-1)).partition([133] * len(positions)),
        recording=recording,
    )


def log_hand_confidence(recording: rr.RecordingStream, side: str, indexes: list[rr.TimeColumn], confidence: list[float]) -> None:
    """Preserve the source's per-hand scalar on every frame."""
    rr.send_columns(schema.hand_confidence_path(side), indexes=indexes, columns=rr.Scalars.columns(scalars=confidence), recording=recording)


def log_joint_angles(recording: rr.RecordingStream, side: str, indexes: list[rr.TimeColumn], angles: Float32[ndarray, "t j"]) -> None:
    """Write shipped Float32[ndarray, 't j'] joint parameters on their sparse clock."""
    if len(angles):
        rr.send_columns(
            schema.hand_joint_angles_path(side),
            indexes=indexes,
            columns=rr.AnyValues.columns(joint_angles=pa.array(angles.tolist(), type=pa.list_(pa.float32(), angles.shape[1]))),
            recording=recording,
        )


def log_wrist(
    recording: rr.RecordingStream,
    side: str,
    indexes: list[rr.TimeColumn],
    translations: Float32[ndarray, "t 3"],
    quaternions: Float64[ndarray, "t 4"],
) -> None:
    """Write Float32[t,3] metre translations and Float64[t,4] xyzw quaternions on a sparse clock."""
    if len(translations):
        rr.send_columns(
            schema.hand_wrist_path(side),
            indexes=indexes,
            columns=rr.Transform3D.columns(translation=translations, quaternion=quaternions),
            recording=recording,
        )


def annotation_context() -> rr.AnnotationContext:
    """Root classes every layer relies on: the COCO-133 skeleton (class 0) and the §13 box labels (100-103)."""
    return rr.AnnotationContext(
        [
            rr.ClassDescription(
                info=rr.AnnotationInfo(id=0, label="Coco Wholebody", color=(0, 0, 255)),
                keypoint_annotations=[rr.AnnotationInfo(id=point, label=name) for point, name in COCO_133_ID2NAME.items()],
                keypoint_connections=COCO_133_LINKS,
            ),
            *(
                rr.ClassDescription(info=rr.AnnotationInfo(id=int(layer), label=COCO133_ROI_LABELS[layer], color=COCO133_ROI_COLORS[layer]))
                for layer in Coco133RoiLayer
            ),
        ]
    )
