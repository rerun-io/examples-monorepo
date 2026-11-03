"""One upper-body display skeleton coloured by support-adjusted confidence."""

import numpy as np
import rerun as rr
from dataforge.hands import log_keypoints2d, log_keypoints3d
from jaxtyping import Float32, Int64
from numpy import ndarray
from simplecv.data.skeleton.coco_133 import COCO_133_IDS, COCO_133_LINKS
from simplecv.rerun_custom_types import Points3DWithConfidence

DISPLAY_JOINTS: Int64[ndarray, "k"] = np.r_[3:9, 11:13, 91:133]
"""Ears, shoulders, elbows, hips and hands; lower-body data stays in the source layers."""
WRIST_TO_HAND_ROOT: dict[int, int] = {9: 91, 10: 112}
"""Body wrists drawn at the hand roots, so arm links meet the hand skeletons."""
DISPLAY_LINKS: list[tuple[int, int]] = [
    (WRIST_TO_HAND_ROOT.get(a, a), WRIST_TO_HAND_ROOT.get(b, b))
    for a, b in COCO_133_LINKS
    if WRIST_TO_HAND_ROOT.get(a, a) in DISPLAY_JOINTS and WRIST_TO_HAND_ROOT.get(b, b) in DISPLAY_JOINTS
]
SKELETON_LINK_RGB: tuple[int, int, int] = (0, 0, 255)
"""Links of the display skeleton and of the model keypoints; an annotation class without a colour draws them in a colour from its path."""
CONFIDENCE_3D_COLOR_FLOOR: float = 0.2
"""Red end of the colour scale for 3D (Kineo) confidence; green is 1.0. The 10th percentile of COCO body joints 3-16 over the 26
easy26-r3 keypoints layers is 0.199 (p5 0.158, p50 0.485, p95 0.698), so about one body joint in ten shows pure red. The earlier
floor of 0.7 was on the detector scale: 95% of the body joints fell below it."""


def log_display_skeleton(
    recording: rr.RecordingStream,
    path: str,
    positions: Float32[ndarray, "n 133 2"] | Float32[ndarray, "n 133 3"],
    confidence: Float32[ndarray, "n 133"],
    stamps: Int64[ndarray, "n"],
) -> None:
    """Use shared typed keypoints: confidence-coloured points and blue annotation links."""
    recording.log(
        path,
        rr.AnnotationContext(rr.ClassDescription(
            info=rr.AnnotationInfo(id=0, color=SKELETON_LINK_RGB),
            keypoint_connections=DISPLAY_LINKS,
        )),
        static=True,
    )
    if positions.shape[-1] == 3:
        recording.log(path, Points3DWithConfidence.from_fields(class_ids=0, keypoint_ids=COCO_133_IDS,
                      show_labels=False, radii=rr.Radius.ui_points(3.0)), static=True)
        log_keypoints3d(recording, path=path, times_ns=stamps, frame_indices=None,
                        positions=positions, confidence=confidence, color_range=(CONFIDENCE_3D_COLOR_FLOOR, 1.0), describe=False)
    else:
        log_keypoints2d(recording, path=path, times_ns=stamps, frame_indices=None,
                        positions=positions, confidence=confidence, color_range=(CONFIDENCE_3D_COLOR_FLOOR, 1.0))
