"""One upper-body display skeleton coloured by support-adjusted confidence."""

from dataclasses import dataclass

import numpy as np
import rerun as rr
from dataforge.logging_toolkit import time_column
from jaxtyping import Float32, Int64, UInt8
from numpy import ndarray
from simplecv.data.skeleton.coco_133 import COCO_133_LINKS

DISPLAY_JOINTS: Int64[ndarray, "k"] = np.r_[3:9, 11:13, 91:133]
"""Ears, shoulders, elbows, hips and hands; lower-body data stays in the source layers."""
DISPLAY_LINKS: list[tuple[int, int]] = [
    ({9: 91, 10: 112}.get(a, a), {9: 91, 10: 112}.get(b, b))
    for a, b in COCO_133_LINKS
    if {9: 91, 10: 112}.get(a, a) in DISPLAY_JOINTS and {9: 91, 10: 112}.get(b, b) in DISPLAY_JOINTS
]


@dataclass(frozen=True, slots=True)
class SkeletonStyle:
    colors: UInt8[ndarray, "n 133 4"]
    edge_colors: UInt8[ndarray, "n e 4"]


def skeleton_style(confidence: Float32[ndarray, "n 133"]) -> SkeletonStyle:
    """Red at 0.7, yellow at 0.85, green at 1.0; a link uses its weaker endpoint."""
    def colors(scores: Float32[ndarray, "n j"]) -> UInt8[ndarray, "n j 4"]:
        scaled = np.clip((scores - 0.7) / 0.3, 0.0, 1.0)
        return np.stack([np.minimum(2.0 - 2.0 * scaled, 1.0) * 255,
                         np.minimum(2.0 * scaled, 1.0) * 255,
                         np.zeros_like(scaled), np.full_like(scaled, 255)], axis=-1).astype(np.uint8)
    edges = np.asarray(DISPLAY_LINKS, dtype=np.int64)
    return SkeletonStyle(colors(confidence), colors(confidence[:, edges].min(axis=-1)))


def log_display_skeleton(
    recording: rr.RecordingStream,
    path: str,
    positions: Float32[ndarray, "n 133 2"] | Float32[ndarray, "n 133 3"],
    confidence: Float32[ndarray, "n 133"],
    stamps: Int64[ndarray, "n"],
) -> None:
    """Use the same confidence colours and fixed widths in every view."""
    style: SkeletonStyle = skeleton_style(confidence)
    edges: Int64[ndarray, "e 2"] = np.asarray(DISPLAY_LINKS, dtype=np.int64)
    dimension: int = positions.shape[-1]
    point_type = rr.Points2D if dimension == 2 else rr.Points3D
    line_type = rr.LineStrips2D if dimension == 2 else rr.LineStrips3D
    # Retain the inspectable topology, but omit keypoint_ids on Points so the
    # viewer uses only our explicit, per-edge styled lines.
    recording.log(path, rr.AnnotationContext(rr.ClassDescription(info=0, keypoint_connections=DISPLAY_LINKS)), static=True)
    recording.log(path, point_type.from_fields(show_labels=False), static=True)
    recording.log(
        path,
        rr.AnyValues(
            coco133_joint_ids=list(range(133)),
            skeleton_links=DISPLAY_LINKS,
            confidence_semantics="Support-adjusted exo score; Quest hands assigned 1.0 because tracking supplies no per-joint confidence",
        ),
        static=True,
    )
    rr.send_columns(
        path,
        indexes=[time_column(stamps)],
        columns=[
            *point_type.columns(
                positions=positions.reshape(-1, dimension),
                colors=style.colors.reshape(-1, 4),
                radii=np.full(positions.shape[0] * 133, 3.0 if dimension == 2 else 0.006, dtype=np.float32),
            ).partition([133] * len(stamps)),
            *rr.AnyValues.columns(confidence=confidence.reshape(-1)).partition([133] * len(stamps)),
        ],
        recording=recording,
    )
    rr.send_columns(
        path,
        indexes=[time_column(stamps)],
        columns=line_type.columns(
            strips=positions[:, edges].reshape(-1, 2, dimension),
            colors=style.edge_colors.reshape(-1, 4),
            radii=np.full(len(stamps) * len(edges), -1.2, dtype=np.float32),
        ).partition([len(edges)] * len(stamps)),
        recording=recording,
    )
