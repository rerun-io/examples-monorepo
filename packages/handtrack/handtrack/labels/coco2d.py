"""DetNet labels from 2D COCO-133 hand keypoints (datasets that ship projected keypoints, not a hand model)."""

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float32, Int64
from torch import Tensor

from handtrack.geometry.letterbox import Letterbox
from handtrack.labels.circles import enclosing_circles
from handtrack.labels.validity import HandLabel

COCO_SLOTS: dict[int, int] = {91: 5, 93: 6, 94: 7, 95: 0, 96: 8, 97: 9, 98: 10, 99: 1, 100: 11, 101: 12, 102: 13, 103: 2, 104: 14, 105: 15, 106: 16, 107: 3, 108: 17, 109: 18, 110: 19, 111: 4}
"""COCO-133 left-hand slot -> our landmark (right hand: slot + 21). Slot 92 is the wrist/thumb-CMC midpoint; the palm centre (20) has none."""
RIGHT_HAND_SLOT_OFFSET: int = 21
MIN_VISIBLE_COCO_KEYPOINTS: int = 16
"""Present when at least 16 of the 20 COCO hand keypoints are in the image: the base rule's 17 of 21 (81 %) for 20 points."""


@dataclass(frozen=True, slots=True)
class Coco2dHands:
    """Per-image DetNet labels for both hands (slot 0 = left), in the net frame."""

    image_valid: Bool[Tensor, 'n']
    """False when either hand's tracking is not trusted, so the image gives no supervision."""
    hand_label: Int64[Tensor, 'n 2']
    """HandLabel per hand from the count of keypoints inside the source image."""
    net_xy: Float32[Tensor, 'n 2 21 2']
    """Keypoints in net pixels; NaN where the dataset has none (always landmark 20, the palm centre)."""
    in_front: Bool[Tensor, 'n 2 21']
    """Keypoints that count: finite, confidence above zero and inside the source image."""
    circles: Float32[Tensor, 'n 2 3']
    """Enclosing circle (cx, cy, r) of the counted keypoints in net pixels; NaN for a hand without one."""


def coco133_hands(
    uv: Float32[Tensor, 'n 133 2'],
    confidence: Float32[Tensor, 'n 133'],
    trusted: Bool[Tensor, 'n 2'],
    letterbox: Letterbox,
) -> Coco2dHands:
    """Label both hands of n images from source-pixel COCO-133 keypoints with the base visibility rules."""
    slots: Int64[Tensor, '20'] = torch.tensor(list(COCO_SLOTS))
    landmarks: Int64[Tensor, '20'] = torch.tensor(list(COCO_SLOTS.values()))
    count: int = uv.shape[0]
    source: Float32[Tensor, 'n 2 21 2'] = torch.full((count, 2, 21, 2), torch.nan, dtype=torch.float32)
    weight: Float32[Tensor, 'n 2 21'] = torch.zeros((count, 2, 21), dtype=torch.float32)
    for side in (0, 1):
        source[:, side, landmarks] = uv[:, slots + side * RIGHT_HAND_SLOT_OFFSET].float()
        weight[:, side, landmarks] = confidence[:, slots + side * RIGHT_HAND_SLOT_OFFSET].float()
    inside: Bool[Tensor, 'n 2 21'] = ((source[..., 0] >= -0.5) & (source[..., 0] < letterbox.source_width - 0.5)
                                      & (source[..., 1] >= -0.5) & (source[..., 1] < letterbox.source_height - 0.5))
    counted: Bool[Tensor, 'n 2 21'] = source.isfinite().all(dim=-1) & (weight > 0) & inside
    net_xy: Float32[Tensor, 'n 2 21 2'] = letterbox.to_net(source)
    visible: Int64[Tensor, 'n 2'] = counted.sum(dim=-1)
    hand_label: Int64[Tensor, 'n 2'] = torch.where(visible >= MIN_VISIBLE_COCO_KEYPOINTS, int(HandLabel.PRESENT),
                                                   torch.where(visible == 0, int(HandLabel.ABSENT), int(HandLabel.PARTIAL)))
    circles: Float32[Tensor, 'n 2 3'] = torch.from_numpy(enclosing_circles(torch.nan_to_num(net_xy).numpy(), counted.numpy()))
    return Coco2dHands(image_valid=trusted.all(dim=-1), hand_label=hand_label, net_xy=net_xy, in_front=counted, circles=circles)
