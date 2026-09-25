"""Shared hand measurements; positions are metres or shipped image pixels.

Two adapters, one policy. ``coco133_from_hands`` uses simplecv's ``assembly21_to_coco133``
mapping: COCO thumb-base slots 92/113 are wrist–thumb CMC midpoints (palm is unused), and
wrist slots 9/10 copy hand wrists 91/112. These derived slots carry the source hand
confidence, as SHOW3D does. ``coco133_from_coco_hands`` serves sources that ship
COCO-WholeBody hand order (HO-Cap): they get their shipped slots and nothing derived.
Class 0 uses the root COCO-133 AnnotationContext. Derived projections use a separate entity from shipped 2D.
"""

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from simplecv.data.skeleton.assembly_hands import assembly21_to_coco133
from simplecv.data.skeleton.coco_133 import COCO_133_IDS, LEFT_HAND_IDX, RIGHT_HAND_IDX
from simplecv.rerun_custom_types import Points2DWithConfidence, Points3DWithConfidence, confidence_scores_to_rgb

from dataforge import schema
from dataforge.logging_toolkit import frame_index_column, time_column

HAND_OF_SLOT: Int64[ndarray, "133"] = np.full(133, -1, dtype=np.int64)
"""Owning hand index, or -1 for uncovered COCO slots."""
HAND_OF_SLOT[9] = HAND_OF_SLOT[LEFT_HAND_IDX] = 0
HAND_OF_SLOT[10] = HAND_OF_SLOT[RIGHT_HAND_IDX] = 1


HAND_ALBEDO: dict[str, tuple[int, int, int, int]] = {"left": (90, 160, 240, 110), "right": (240, 170, 130, 110)}
"""Shared per-side mesh RGBA."""


def coco133_from_coco_hands(joints_lr: Float32[ndarray, "2 21 d"]) -> Float32[ndarray, "133 d"]:
    """Copy Float32[2,21,d] left/right COCO hands; missing joints enter as NaN."""
    result: Float32[ndarray, "133 d"] = np.full((133, joints_lr.shape[-1]), np.nan, dtype=np.float32)
    result[LEFT_HAND_IDX] = joints_lr[0]
    result[RIGHT_HAND_IDX] = joints_lr[1]
    return result


def coco133_from_hands(
    landmarks_lr: Float32[ndarray, "2 21 d"], confidence_lr: Float32[ndarray, "2"]
) -> tuple[Float32[ndarray, "133 d"], Float32[ndarray, "133"]]:
    """Map Float32[2,21,d] Assembly hands and Float32[2] confidences to COCO.

    d is 2 (shipped pixels) or 3 (metres); absent hands contain NaN.
    Writers apply confidence_rule to clear confidence on non-finite slots.
    """
    dimensions: int = landmarks_lr.shape[-1]
    if dimensions not in (2, 3):
        raise ValueError("hand landmarks must have 2 or 3 coordinates")
    padded: Float32[ndarray, "2 21 3"] = np.zeros((2, 21, 3), dtype=np.float32)
    padded[:, :, :dimensions] = landmarks_lr
    positions: Float32[ndarray, "133 d"] = assembly21_to_coco133(padded)[:, :dimensions]
    confidence: Float32[ndarray, "133"] = np.zeros(133, dtype=np.float32)
    covered: Bool[ndarray, "133"] = HAND_OF_SLOT >= 0
    confidence[covered] = confidence_lr[HAND_OF_SLOT[covered]]
    return positions, confidence


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
    *,
    times_ns: Int64[ndarray, "t"],
    frame_indices: Int64[ndarray, "t"],
    positions: Float32[ndarray, "t 133 3"],
    confidence: Float32[ndarray, "t 133"] | None,
) -> None:
    """Write dense COCO rows with confidence colours; None means no shipped confidence.

    Args:
        recording: Destination stream.
        times_ns: Int64[ndarray, "t"] video times in nanoseconds.
        frame_indices: Int64[ndarray, "t"] source indices.
        positions: Float32[ndarray, "t 133 3"], metres; missing slots contain NaN.
        confidence: Float32[ndarray, "t 133"] or None; preserve shipped values.
    """
    positions, scores = confidence_rule(positions, confidence)
    flat_conf: Float32[ndarray, "n"] = scores.reshape(-1)
    colors: UInt8[ndarray, "n 3"] = confidence_scores_to_rgb(flat_conf[None, :, None])[0]
    rr.log(
        schema.coco133_xyz_path(),
        Points3DWithConfidence.from_fields(class_ids=0, keypoint_ids=COCO_133_IDS, show_labels=False, radii=0.004),
        static=True,
        recording=recording,
    )
    rr.send_columns(
        schema.coco133_xyz_path(),
        indexes=[time_column(times_ns), frame_index_column(frame_indices)],
        columns=Points3DWithConfidence.columns(positions=positions.reshape(-1, 3), confidences=flat_conf, colors=colors).partition([133] * len(positions)),
        recording=recording,
    )


def log_keypoints2d(
    recording: rr.RecordingStream,
    rig: int,
    cam: int,
    *,
    path: str | None = None,
    times_ns: Int64[ndarray, "t"],
    frame_indices: Int64[ndarray, "t"],
    positions: Float32[ndarray, "t 133 2"],
    confidence: Float32[ndarray, "t 133"] | None,
) -> None:
    """Write Float32[t,133,2] pixels and confidence; path defaults to shipped 2D."""
    path = schema.coco133_uv_path(rig, cam) if path is None else path
    positions, scores = confidence_rule(positions, confidence)
    rr.log(
        path,
        Points2DWithConfidence.from_fields(class_ids=0, keypoint_ids=COCO_133_IDS, show_labels=False, radii=3.0),
        static=True,
        recording=recording,
    )
    rr.send_columns(
        path,
        indexes=[time_column(times_ns), frame_index_column(frame_indices)],
        columns=Points2DWithConfidence.columns(positions=positions.reshape(-1, 2), confidences=scores.reshape(-1)).partition([133] * len(positions)),
        recording=recording,
    )


def log_hand_confidence(
    recording: rr.RecordingStream, side: str, *, times_ns: Int64[ndarray, "t"], frame_indices: Int64[ndarray, "t"], confidence: Float64[ndarray, "t"]
) -> None:
    """Preserve Float64[ndarray, "t"] per-hand confidence on both Int64[t] clocks."""
    rr.send_columns(
        schema.hand_confidence_path(side),
        indexes=[time_column(times_ns), frame_index_column(frame_indices)],
        columns=rr.Scalars.columns(scalars=confidence),
        recording=recording,
    )


def log_joint_angles(
    recording: rr.RecordingStream, side: str, *, times_ns: Int64[ndarray, "t"], frame_indices: Int64[ndarray, "t"], angles: Float32[ndarray, "t j"]
) -> None:
    """Write shipped Float32[ndarray, 't j'] joint parameters on their sparse clock."""
    if len(angles):
        rr.send_columns(
            schema.hand_joint_angles_path(side),
            indexes=[time_column(times_ns), frame_index_column(frame_indices)],
            columns=rr.AnyValues.columns(joint_angles=pa.FixedSizeListArray.from_arrays(pa.array(angles.reshape(-1)), angles.shape[1])),
            recording=recording,
        )
