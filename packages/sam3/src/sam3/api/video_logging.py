"""Rerun logging of SAM3 video outputs, shared by the batch, chunk and stream demos."""

import numpy as np
import rerun as rr
import torch
from jaxtyping import Float32, Int, Shaped, UInt8, UInt16
from numpy import ndarray
from torch import Tensor

from sam3.viz_constants import BOX_PALETTE

BOX_DRAW_ORDER: int = 6


def segmentation_id_map(masks: Shaped[Tensor, "n h w"]) -> UInt8[ndarray, "h w"] | UInt16[ndarray, "h w"]:
    """Instance id per pixel (0 = background, instance i -> i + 1; the highest id wins where masks overlap).

    Built on the masks' device, so only the id map, not every float mask, crosses to the host.

    Args:
        masks: Per-instance mask scores or booleans; a pixel belongs to an instance at >= 0.5.

    Returns:
        uint8 map for up to 255 instances, uint16 beyond.
    """
    ids: Int[Tensor, "n 1 1"] = torch.arange(1, masks.shape[0] + 1, device=masks.device, dtype=torch.int32)[:, None, None]
    id_map: Int[Tensor, "h w"] = torch.where(masks >= 0.5, ids, 0).amax(dim=0)
    if masks.shape[0] <= 255:
        return id_map.to(torch.uint8).cpu().numpy()
    return id_map.cpu().numpy().astype(np.uint16)


def log_frame_outputs(
    frame_idx: int,
    frame_timestamps_ns: Int[ndarray, "num_frames"],
    processed_outputs: dict,
    frame_shape: tuple[int, int],
) -> None:
    """
    Log segmentation masks and bounding boxes for a video frame to Rerun.

    Args:
        frame_idx: Current frame index (used as timeline sequence).
        frame_timestamps_ns: Array of frame timestamps in nanoseconds from the video asset.
        processed_outputs: Dict containing 'masks', 'boxes', and optionally 'scores'.
        frame_shape: Tuple of (height, width) for the frame.
    """
    # Set time on video_time timeline to match the video asset
    if frame_idx < len(frame_timestamps_ns):
        rr.set_time("video_time", duration=1e-9 * frame_timestamps_ns[frame_idx])
    rr.set_time("frame_idx", sequence=frame_idx)

    raw_masks = processed_outputs.get("masks")
    raw_boxes = processed_outputs.get("boxes")
    raw_scores = processed_outputs.get("scores")

    if raw_masks is None or len(raw_masks) == 0:
        return

    masks: Shaped[Tensor, "n h w"] | Shaped[Tensor, "h w"] = torch.as_tensor(raw_masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if tuple(masks.shape[1:]) != frame_shape:
        raise ValueError(f"masks of shape {tuple(masks.shape)} do not match the frame {frame_shape}")
    num_instances: int = masks.shape[0]

    # Build per-instance colors (used for boxes)
    colors: UInt8[ndarray, "k 4"] = np.asarray(
        [BOX_PALETTE[idx % BOX_PALETTE.shape[0]] for idx in range(num_instances)],
        dtype=np.uint8,
    )

    # Log segmentation (video is already logged as static asset)
    rr.log("video/segmentation", rr.SegmentationImage(segmentation_id_map(masks)))

    # Log bounding boxes if present
    if raw_boxes is not None:
        boxes_np: Float32[ndarray, "n 4"] = np.asarray(torch.as_tensor(raw_boxes).detach().cpu().numpy(), dtype=np.float32)
        if boxes_np.ndim == 1:
            boxes_np = boxes_np[None, :]

        class_ids: Int[ndarray, "n"] = np.arange(1, boxes_np.shape[0] + 1, dtype=np.int32)

        labels: list[str] | None = None
        if raw_scores is not None:
            scores_np: Float32[ndarray, "n"] = np.asarray(torch.as_tensor(raw_scores).detach().cpu().numpy(), dtype=np.float32).reshape(-1)
            labels = [f"{score:.2f}" for score in scores_np.tolist()]

        rr.log(
            "video/boxes",
            rr.Boxes2D(
                array=boxes_np,
                array_format=rr.Box2DFormat.XYXY,
                class_ids=class_ids,
                labels=labels,
                colors=colors[:, :3],
                show_labels=True,
                draw_order=BOX_DRAW_ORDER,
            ),
        )
