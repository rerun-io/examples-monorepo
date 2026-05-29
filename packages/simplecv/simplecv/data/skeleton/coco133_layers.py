from __future__ import annotations

from enum import IntEnum
from typing import Final


class Coco133AnnotationLayer(IntEnum):
    """Semantic layers used when logging COCO-133 keypoints."""

    GT = 0
    RAW_2D = 1
    TRACKED_2D = 2
    PROJECTED_2D = 3
    OPTIMIZED_2D = 4
    TRIANGULATED_3D = 5


COCO133_LAYER_LABELS: Final[dict[Coco133AnnotationLayer, str]] = {
    Coco133AnnotationLayer.GT: "coco133_gt",
    Coco133AnnotationLayer.RAW_2D: "coco133_raw2d",
    Coco133AnnotationLayer.TRACKED_2D: "coco133_tracked2d",
    Coco133AnnotationLayer.PROJECTED_2D: "coco133_projected2d",
    Coco133AnnotationLayer.OPTIMIZED_2D: "coco133_optimized2d",
    Coco133AnnotationLayer.TRIANGULATED_3D: "coco133_triangulated3d",
}

# Skeleton link colours should avoid the red/yellow/green spectrum that encodes per-keypoint confidence.
COCO133_LAYER_COLORS: Final[dict[Coco133AnnotationLayer, tuple[int, int, int]]] = {
    Coco133AnnotationLayer.GT: (30, 64, 255),  # deep blue for annotated ground truth
    Coco133AnnotationLayer.RAW_2D: (59, 130, 246),  # azure
    Coco133AnnotationLayer.TRACKED_2D: (165, 105, 255),  # violet
    Coco133AnnotationLayer.PROJECTED_2D: (217, 70, 239),  # magenta
    Coco133AnnotationLayer.OPTIMIZED_2D: (148, 163, 255),  # periwinkle
    Coco133AnnotationLayer.TRIANGULATED_3D: (14, 165, 233),  # cyan to distinguish derived 3d results
}

COCO133_PREDICTION_LAYER_TO_PATH: Final[dict[Coco133AnnotationLayer, str]] = {
    Coco133AnnotationLayer.RAW_2D: "raw",
    Coco133AnnotationLayer.TRACKED_2D: "tracked",
    Coco133AnnotationLayer.PROJECTED_2D: "projected",
    Coco133AnnotationLayer.OPTIMIZED_2D: "optimized",
}


class Coco133RoiLayer(IntEnum):
    """Semantic regions-of-interest used for auxiliary COCO-133 overlays."""

    LEFT_HAND = 100
    RIGHT_HAND = 101
    FULL_BODY = 102
    FACE = 103


COCO133_ROI_LABELS: Final[dict[Coco133RoiLayer, str]] = {
    Coco133RoiLayer.LEFT_HAND: "left_hand",
    Coco133RoiLayer.RIGHT_HAND: "right_hand",
    Coco133RoiLayer.FULL_BODY: "full_body",
    Coco133RoiLayer.FACE: "face",
}


COCO133_ROI_COLORS: Final[dict[Coco133RoiLayer, tuple[int, int, int]]] = {
    Coco133RoiLayer.LEFT_HAND: (16, 185, 129),  # teal
    Coco133RoiLayer.RIGHT_HAND: (249, 115, 22),  # orange
    Coco133RoiLayer.FULL_BODY: (59, 130, 246),  # azure
    Coco133RoiLayer.FACE: (236, 72, 153),  # pink
}
