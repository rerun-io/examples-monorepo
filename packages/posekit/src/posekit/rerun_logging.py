"""Rerun logging helpers for posekit image and video predictions."""

from __future__ import annotations

import numpy as np
import rerun as rr
from jaxtyping import Float32, UInt8, UInt16
from numpy import ndarray
from simplecv.rerun_custom_types import Points2DWithConfidence

from posekit.skeletons import KeypointSkeleton

PERSON_COLORS: UInt8[ndarray, "palette 3"] = np.asarray(
    [
        [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
        [148, 103, 189], [140, 86, 75], [227, 119, 194], [23, 190, 207],
    ],
    dtype=np.uint8,
)


def person_color(person_idx: int) -> tuple[int, int, int]:
    """Return the stable display color for one person index."""
    color_rgb: UInt8[ndarray, "3"] = PERSON_COLORS[person_idx % PERSON_COLORS.shape[0]]
    return int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2])


def log_skeleton_annotation_context(skeleton: KeypointSkeleton, *, entity_path: str = "video") -> None:
    """Log a Rerun annotation context describing the skeleton's keypoints and links."""
    rr.log(
        entity_path,
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label=skeleton.name),
                    keypoint_annotations=[rr.AnnotationInfo(id=idx, label=name) for idx, name in enumerate(skeleton.keypoint_names)],
                    keypoint_connections=list(skeleton.links),
                )
            ]
        ),
        static=True,
    )


def log_person_bbox(box_xyxy: Float32[ndarray, "4"], person_idx: int, *, entity_path: str = "image") -> None:
    """Log one person's box with its stable display color.

    Args:
        box_xyxy: Float32 box with shape ``(4,)`` in ``xyxy`` pixel order.
        person_idx: Stable person or track index selecting the display color.
        entity_path: Parent entity holding the ``person_<idx>/bbox`` children.
    """
    rr.log(
        f"{entity_path}/person_{person_idx}/bbox",
        rr.Boxes2D(
            array=box_xyxy.reshape(1, 4),
            array_format=rr.Box2DFormat.XYXY,
            labels=f"person_{person_idx}",
            colors=person_color(person_idx),
        ),
    )


def log_person_points2d(
    entity_path: str,
    xy: Float32[ndarray, "p 2"],
    confidence: Float32[ndarray, "p"],
    threshold: float,
    *,
    keypoint_ids: UInt16[ndarray, "p"] | None = None,
    class_ids: int | None = None,
) -> None:
    """Log one person's points colored by confidence, masking low-confidence positions.

    Point colors follow the shared red -> yellow -> green confidence gradient and
    the raw (unmasked) scores ride along as queryable confidence components.
    """
    visible_xy: Float32[ndarray, "p 2"] = xy.copy()
    visible_xy[confidence < threshold] = np.nan
    rr.log(
        entity_path,
        Points2DWithConfidence(
            positions=visible_xy,
            confidences=confidence,
            keypoint_ids=keypoint_ids,
            class_ids=class_ids,
            # Negative radii are screen-space UI points, so they stay visible at every zoom level.
            radii=-2.5,
        ),
    )


def log_topdown_predictions(
    boxes_xyxy: Float32[ndarray, "n 4"],
    keypoints_xy: Float32[ndarray, "n k 2"],
    scores: Float32[ndarray, "n k"],
    skeleton: KeypointSkeleton,
    keypoint_threshold: float,
) -> None:
    """Log sparse keypoints with their skeleton annotation context."""
    log_skeleton_annotation_context(skeleton, entity_path="image")
    keypoint_ids: UInt16[ndarray, "k"] = np.arange(keypoints_xy.shape[1], dtype=np.uint16)
    for person_idx in range(boxes_xyxy.shape[0]):
        log_person_bbox(boxes_xyxy[person_idx], person_idx)
        log_person_points2d(
            f"image/person_{person_idx}/keypoints",
            keypoints_xy[person_idx],
            scores[person_idx],
            keypoint_threshold,
            keypoint_ids=keypoint_ids,
            class_ids=0,
        )


def log_dense_predictions(
    boxes_xyxy: Float32[ndarray, "n 4"],
    landmarks_xy: Float32[ndarray, "n p 2"],
    visibility: Float32[ndarray, "n p"],
    visibility_threshold: float,
) -> None:
    """Log MAMMA dense landmarks for each person."""
    for person_idx in range(boxes_xyxy.shape[0]):
        log_person_bbox(boxes_xyxy[person_idx], person_idx)
        log_person_points2d(
            f"image/person_{person_idx}/landmarks",
            landmarks_xy[person_idx],
            visibility[person_idx],
            visibility_threshold,
        )


def log_approximate_skeletons(
    joints_xy: Float32[ndarray, "n j 2"],
    strips_by_person: list[Float32[ndarray, "b 2 2"]],
) -> None:
    """Log approximate skeleton strips and joints for each person."""
    for person_idx, strips in enumerate(strips_by_person):
        rr.log(
            f"image/person_{person_idx}/skeleton",
            rr.LineStrips2D(strips=strips, colors=(255, 255, 255), radii=-1.5),
        )
        rr.log(
            f"image/person_{person_idx}/joints",
            rr.Points2D(positions=joints_xy[person_idx], colors=person_color(person_idx), radii=-4.0, show_labels=False),
        )
