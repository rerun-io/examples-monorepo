"""GPU postprocessing for YOLOX-style detector outputs."""

import torch
from jaxtyping import Float
from torch import Tensor

from posekit.predictions import BoxDetections


def decode_yolox_head_outputs(
    boxes_xyxy: Float[Tensor, "b anchors 4"],
    person_scores: Float[Tensor, "b anchors"],
    *,
    resize_ratios: Float[Tensor, "b"],
    score_thr: float = 0.3,
    nms_thr: float = 0.45,
) -> BoxDetections:
    """Threshold and NMS-filter decoded YOLOX head outputs on GPU.

    One flattened threshold + one ``batched_nms`` call covers the whole frame
    batch (a single host sync instead of one per frame). Output rows keep the
    per-frame score-descending order the per-frame NMS produced.

    Args:
        boxes_xyxy: Decoded boxes in detector-input coordinates.
        person_scores: Person-class confidence per anchor.
        resize_ratios: Per-frame letterbox resize ratios from
            :func:`posekit.ops.letterbox.letterbox_frames`.
        score_thr: Minimum person score.
        nms_thr: Non-maximum suppression IoU threshold.

    Returns:
        Flattened image-space detections across the frame batch.
    """
    from torchvision.ops import batched_nms

    scores: Float[Tensor, "b anchors"] = person_scores.float()
    boxes: Float[Tensor, "b anchors 4"] = boxes_xyxy.float() / resize_ratios[:, None, None]
    keep_mask: Tensor = scores > float(score_thr)
    frame_ids: Tensor = torch.arange(int(boxes.shape[0]), device=boxes.device)[:, None].expand_as(keep_mask)
    flat_boxes: Float[Tensor, "m 4"] = boxes[keep_mask]
    if int(flat_boxes.shape[0]) == 0:
        return BoxDetections.empty(boxes.device)
    flat_scores: Float[Tensor, "m"] = scores[keep_mask]
    flat_frames: Tensor = frame_ids[keep_mask]
    keep: Tensor = batched_nms(flat_boxes, flat_scores, flat_frames, float(nms_thr))
    order: Tensor = keep[torch.argsort(flat_frames[keep], stable=True)]
    return BoxDetections(flat_boxes[order].contiguous(), flat_scores[order].contiguous(), flat_frames[order].contiguous())
