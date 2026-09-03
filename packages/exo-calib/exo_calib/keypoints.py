"""Per-camera Stage B keypoint records and the AssemblyHands-X confidence post-processing."""

from dataclasses import dataclass
from typing import Literal, NamedTuple, TypeAlias, get_args

import numpy as np
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray

from exo_calib.confidence import median_filter_keypoints, modulate_crop_edge_confidence, threshold_rescale_confidence

# AssemblyHands-X gate: temporal median window (frames), crop-edge margin (crop
# pixels), and the confidence threshold before rescaling to [0, 1]. Stage B's
# overlay and the refinement use the same values, so the two cannot drift.
AHX_MEDIAN_WINDOW: int = 5
AHX_MARGIN_PX: float = 32.0
AHX_CONF_TAU: float = 0.15

BoxSource: TypeAlias = Literal["yolox", "tracked", "none"]
"""Per-frame person-box provenance: detected, projected from the tracked skeleton, or absent."""
BOX_SOURCE_BY_NAME: dict[str, BoxSource] = {source: source for source in get_args(BoxSource)}
"""Narrows a stored provenance string back to :data:`BoxSource` (``None`` for anything else)."""

STAGE_B_FRAME_BUDGET: int = 1000
"""Frames Stage B processes per view, spread evenly over the capture; shorter captures use every frame."""


class GatedKeypoints(NamedTuple):
    """Keypoints after the AssemblyHands-X post-processing: median-filtered pixels and the gated confidences."""

    xy: Float64[ndarray, "*batch 133 2"]
    conf: Float64[ndarray, "*batch 133"]


@dataclass(slots=True)
class CameraKeypoints:
    """Raw Stage B output for one camera."""

    sample_indices: Int64[ndarray, "t"]
    """Decoder sample indices of the processed frames."""
    times_ns: Int64[ndarray, "t"]
    """Timeline timestamps (nanoseconds) of the processed frames."""
    kp_xy: Float32[ndarray, "t 133 2"]
    """Image-space keypoints; NaN where no detection."""
    conf: Float32[ndarray, "t 133"]
    """Raw pose-network keypoint confidences; 0 where no detection."""
    bbox_xyxy: Float32[ndarray, "t 4"]
    """Person box per frame; NaN where no detection."""
    box_source: tuple[BoxSource, ...]
    """Per-frame box provenance, index-aligned with ``bbox_xyxy``."""
    crop_origin_xy: Float32[ndarray, "t 2"]
    """Image-space origin of the model crop rectangle."""
    crop_size_wh: Float32[ndarray, "t 2"]
    """Image-space size of the model crop rectangle."""
    crop_input_wh: Int64[ndarray, "2"]
    """Pose model crop input size (width, height) the crop rectangles map into;
    the AssemblyHands-X margin rule operates in this pixel frame."""
    video_wh: Int64[ndarray, "2"]
    """Native video frame size (width, height)."""


def sampled_frame_indices(num_samples: int, budget: int) -> Int64[ndarray, "t"]:
    """Spread at most ``budget`` frames evenly over a capture of ``num_samples`` frames.

    Returns:
        Int64 sample indices with shape ``(t,)``, every frame when the capture fits the budget.
    """
    if budget < 1:
        raise ValueError("budget must be positive")
    if num_samples < 1:
        raise ValueError("the capture has no frames")
    return np.linspace(0, num_samples - 1, min(budget, num_samples)).round().astype(np.int64)


def postprocess_confidences(cam: CameraKeypoints, median_window: int, margin_px: float, conf_tau: float) -> GatedKeypoints:
    """Apply the AssemblyHands-X 2D post-processing to one camera's keypoints.

    Args:
        cam: Raw Stage B output for one camera.
        median_window: Temporal median-filter window in frames.
        margin_px: Crop-edge confidence falloff width in crop pixels.
        conf_tau: Inclusive confidence cutoff before linear rescaling.

    Returns:
        Median-filtered keypoints and the crop-margin-modulated, threshold-rescaled
        confidences, both on the full frame grid (shape ``(t, 133, …)``).
    """
    crop_wh: tuple[int, int] = (int(cam.crop_input_wh[0]), int(cam.crop_input_wh[1]))
    filtered_xy: Float64[ndarray, "t 133 2"] = median_filter_keypoints(cam.kp_xy.astype(np.float64), window=median_window)
    num_frames: int = filtered_xy.shape[0]
    post_conf: Float64[ndarray, "t 133"] = np.zeros((num_frames, 133), dtype=np.float64)
    for t in range(num_frames):
        if not np.isfinite(cam.crop_origin_xy[t]).all():
            continue
        size_wh: Float64[ndarray, "2"] = cam.crop_size_wh[t].astype(np.float64)
        kp_crop_xy: Float64[ndarray, "133 2"] = (filtered_xy[t] - cam.crop_origin_xy[t].astype(np.float64)) / size_wh * np.asarray(crop_wh, dtype=np.float64)
        modulated: Float64[ndarray, "133"] = modulate_crop_edge_confidence(kp_crop_xy, cam.conf[t].astype(np.float64), crop_wh, margin_px)
        post_conf[t] = threshold_rescale_confidence(modulated, tau=conf_tau)
    return GatedKeypoints(xy=filtered_xy, conf=post_conf)


def stack_postprocessed(
    per_camera: dict[str, CameraKeypoints], names: tuple[str, ...], median_window: int, margin_px: float, conf_tau: float
) -> GatedKeypoints:
    """Post-process every camera and stack onto the common frame grid (shape ``(v, t, 133, …)``).

    Raises:
        ValueError: If the cameras were not processed on the same frame grid.
    """
    sample_indices: Int64[ndarray, "t"] = per_camera[names[0]].sample_indices
    for name in names[1:]:
        if not np.array_equal(per_camera[name].sample_indices, sample_indices):
            raise ValueError(f"camera {name} was processed on a different frame grid than {names[0]}")
    gated: list[GatedKeypoints] = [postprocess_confidences(per_camera[name], median_window, margin_px, conf_tau) for name in names]
    return GatedKeypoints(xy=np.stack([g.xy for g in gated]), conf=np.stack([g.conf for g in gated]))
