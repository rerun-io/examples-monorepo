"""One upper-body display skeleton coloured by support-adjusted confidence."""

import numpy as np
import rerun as rr
from dataforge.hands import log_keypoints2d, log_keypoints3d
from jaxtyping import Float32, Int64
from numpy import ndarray
from simplecv.data.skeleton.coco_133 import COCO_133_LINKS

DISPLAY_JOINTS: Int64[ndarray, "k"] = np.r_[3:9, 11:13, 91:133]
"""Ears, shoulders, elbows, hips and hands; lower-body data stays in the source layers."""
DISPLAY_LINKS: list[tuple[int, int]] = [
    ({9: 91, 10: 112}.get(a, a), {9: 91, 10: 112}.get(b, b))
    for a, b in COCO_133_LINKS
    if {9: 91, 10: 112}.get(a, a) in DISPLAY_JOINTS and {9: 91, 10: 112}.get(b, b) in DISPLAY_JOINTS
]


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
            info=rr.AnnotationInfo(id=0, color=(0, 0, 255)),
            keypoint_connections=DISPLAY_LINKS,
        )),
        static=True,
    )
    if positions.shape[-1] == 3:
        log_keypoints3d(recording, path=path, times_ns=stamps, frame_indices=None,
                        positions=positions, confidence=confidence, color_range=(0.7, 1.0))
    else:
        log_keypoints2d(recording, path=path, times_ns=stamps, frame_indices=None,
                        positions=positions, confidence=confidence, color_range=(0.7, 1.0))
