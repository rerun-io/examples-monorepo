"""AssemblyHands-X post-processing of Stage B keypoints (temporal median, crop-margin downweight, threshold rescale), and
``undistort_views``: every view's keypoints in its pinhole pixels."""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray

from exo_calib.ops.confidence import median_filter_keypoints, modulate_crop_edge_confidence, threshold_rescale_confidence
from exo_calib.ops.lens import undistort
from exo_calib.types import Lens, MultiViewKeypoints

# AssemblyHands-X gate: temporal median window (VIDEO frames, not records), crop-edge margin (crop pixels), and the
# confidence threshold before rescaling to [0, 1]. The refinement and its overlay use the same values, so the two cannot drift.
AHX_MEDIAN_WINDOW: int = 5
AHX_MARGIN_PX: float = 32.0
AHX_CONF_TAU: float = 0.15


class GatedKeypoints(NamedTuple):
    """Keypoints after the AssemblyHands-X post-processing: median-filtered pixels and the gated confidences."""

    xy: Float64[ndarray, "*batch 133 2"]
    conf: Float64[ndarray, "*batch 133"]


def postprocess_confidences(
    kp_xy: Float32[ndarray, "t 133 2"],
    conf: Float32[ndarray, "t 133"],
    crop_origin_xy: Float32[ndarray, "t 2"],
    crop_size_wh: Float32[ndarray, "t 2"],
    crop_input_wh: tuple[int, int],
    frame_index: Int64[ndarray, " t"],
    median_window: int = AHX_MEDIAN_WINDOW,
    margin_px: float = AHX_MARGIN_PX,
    conf_tau: float = AHX_CONF_TAU,
) -> GatedKeypoints:
    """Gate one camera's keypoints over time (frames without a crop rectangle get zero confidence).

    ``frame_index`` is each record's video frame: the median mixes only records within ``median_window // 2`` video
    frames of each other, so under a sparse frame budget the pixels stay the model's own output for that frame.

    Returns:
        Median-filtered keypoints and the crop-margin-modulated, threshold-rescaled confidences, shape ``(t, 133, …)``.
    """
    filtered_xy: Float64[ndarray, "t 133 2"] = median_filter_keypoints(kp_xy.astype(np.float64), frame_index, window=median_window)
    post_conf: Float64[ndarray, "t 133"] = np.zeros(filtered_xy.shape[:2], dtype=np.float64)
    for t in range(filtered_xy.shape[0]):
        if not np.isfinite(crop_origin_xy[t]).all():
            continue
        kp_crop_xy: Float64[ndarray, "133 2"] = (filtered_xy[t] - crop_origin_xy[t]) / crop_size_wh[t].astype(np.float64) * np.asarray(crop_input_wh, dtype=np.float64)
        modulated: Float64[ndarray, "133"] = modulate_crop_edge_confidence(kp_crop_xy, conf[t].astype(np.float64), crop_input_wh, margin_px)
        post_conf[t] = threshold_rescale_confidence(modulated, tau=conf_tau)
    return GatedKeypoints(xy=filtered_xy, conf=post_conf)


def stack_postprocessed(records: Sequence[MultiViewKeypoints]) -> GatedKeypoints:
    """Gate every camera of a time-ordered record sequence onto one grid (shape ``(v, t, 133, …)``)."""
    frame_index: Int64[ndarray, " t"] = np.array([r.frame_index for r in records], dtype=np.int64)
    gated: list[GatedKeypoints] = [
        postprocess_confidences(
            np.stack([r.kp_xy[v] for r in records]),
            np.stack([r.conf[v] for r in records]),
            np.stack([r.crop_origin_xy[v] for r in records]),
            np.stack([r.crop_size_wh[v] for r in records]),
            records[0].crop_input_wh,
            frame_index,
        )
        for v in range(len(records[0].views))
    ]
    return GatedKeypoints(xy=np.stack([g.xy for g in gated]), conf=np.stack([g.conf for g in gated]))


def undistort_views(lenses: Sequence[Lens], xy: Float64[ndarray, "v t 133 2"], K: Float64[ndarray, "v 3 3"]) -> Float64[ndarray, "v t 133 2"]:
    """Every view's keypoints in its pinhole pixels (through its lens, same ``K``): the pixels the rig's geometry works in."""
    return np.stack([undistort(lens, g.reshape(-1, 2), k).reshape(g.shape) for lens, g, k in zip(lenses, xy, K, strict=True)])
