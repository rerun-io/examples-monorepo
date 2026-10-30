"""One display skeleton, with camera-supported and inferred joints distinguished."""

from dataclasses import dataclass

import numpy as np
import rerun as rr
from dataforge.logging_toolkit import time_column
from jaxtyping import Bool, Float32, Int64, UInt8
from numpy import ndarray
from simplecv.data.skeleton.coco_133 import COCO_133_LINKS

DISPLAY_JOINTS: Int64[ndarray, "k"] = np.r_[3:9, 11:23, 91:133]
DISPLAY_LEFT_JOINTS: frozenset[int] = frozenset({*range(91, 112), 3, 5, 7, 11, 13, 15, 17, 18, 19})
DISPLAY_LINKS: list[tuple[int, int]] = [
    ({9: 91, 10: 112}.get(a, a), {9: 91, 10: 112}.get(b, b))
    for a, b in COCO_133_LINKS
    if {9: 91, 10: 112}.get(a, a) in DISPLAY_JOINTS and {9: 91, 10: 112}.get(b, b) in DISPLAY_JOINTS
]


@dataclass(frozen=True, slots=True)
class SkeletonStyle:
    colors: UInt8[ndarray, "n 133 4"]
    """Magenta/white for observed joints; translucent grey for inferred joints."""
    edge_colors: UInt8[ndarray, "n e 4"]
    """An edge is grey whenever either endpoint is inferred."""
    observed_edges: Bool[ndarray, "n e"]
    """Both endpoints have camera support."""


def skeleton_style(observed: Bool[ndarray, "n 133"]) -> SkeletonStyle:
    """Colour every joint and edge according to its view-specific support."""
    side_colors: UInt8[ndarray, "133 4"] = np.array(
        [(255, 64, 255, 255) if joint in DISPLAY_LEFT_JOINTS else (255, 255, 255, 255) for joint in range(133)],
        dtype=np.uint8,
    )
    edges: Int64[ndarray, "e 2"] = np.asarray(DISPLAY_LINKS, dtype=np.int64)
    edge_observed: Bool[ndarray, "n e"] = observed[:, edges].all(axis=-1)
    return SkeletonStyle(
        np.where(observed[..., None], side_colors, np.array([125, 135, 145, 190], np.uint8)),
        np.where(edge_observed[..., None], side_colors[edges[:, 0]], np.array([105, 115, 125, 150], np.uint8)),
        edge_observed,
    )


def log_display_skeleton(
    recording: rr.RecordingStream,
    path: str,
    positions: Float32[ndarray, "n 133 2"] | Float32[ndarray, "n 133 3"],
    observed: Bool[ndarray, "n 133"],
    stamps: Int64[ndarray, "n"],
) -> None:
    """Write explicit edges so mixed observed/inferred links can have distinct colours and widths."""
    style: SkeletonStyle = skeleton_style(observed)
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
            observed_semantics="Camera is an inlier for this point; OAK hands use Quest tracking, OAK body is inferred",
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
                radii=np.where(observed, 3.0 if dimension == 2 else 0.006, 1.5 if dimension == 2 else 0.003).reshape(-1),
            ).partition([133] * len(stamps)),
            *rr.AnyValues.columns(observed=observed.reshape(-1)).partition([133] * len(stamps)),
        ],
        recording=recording,
    )
    rr.send_columns(
        path,
        indexes=[time_column(stamps)],
        columns=line_type.columns(
            strips=positions[:, edges].reshape(-1, 2, dimension),
            colors=style.edge_colors.reshape(-1, 4),
            radii=np.where(style.observed_edges, -1.2, -0.7).reshape(-1),
        ).partition([len(edges)] * len(stamps)),
        recording=recording,
    )
