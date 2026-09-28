"""Dataset validity is distinct from a projected hand's visibility label."""

from enum import IntEnum

import torch
from jaxtyping import Bool, Float32, Int64
from torch import Tensor

MIN_VISIBLE_KEYPOINTS: int = 17
SHOW3D_CONFIDENCE_THRESHOLD: float = 0.1


class HandLabel(IntEnum):
    """Presence supervision: partial hands contribute no presence loss."""

    ABSENT = 0
    PRESENT = 1
    PARTIAL = 2


def classify_visibility(visible: Int64[Tensor, '*b']) -> Int64[Tensor, '*b']:
    """Classify counts of keypoints in front and inside the image or crop."""
    return torch.where(visible >= MIN_VISIBLE_KEYPOINTS, HandLabel.PRESENT, torch.where(visible == 0, HandLabel.ABSENT, HandLabel.PARTIAL))


def umetrack_hands(confidence: Float32[Tensor, 'f 2'], headset_tracked: Bool[Tensor, 'f']) -> tuple[Bool[Tensor, 'f'], Bool[Tensor, 'f 2']]:
    """Return frame validity and binary hand labels; conf-0 hands are absent."""
    if not bool(((confidence == 0) | (confidence == 1)).all()):
        raise ValueError('UmeTrack confidence must be binary (0 or 1)')
    return headset_tracked.clone(), confidence == 1


def show3d_hands(
    confidence: Float32[Tensor, 'f 2'],
    has_pose: Bool[Tensor, 'f 2'],
    inside: Int64[Tensor, 'f c 2'],
    headset_valid: Bool[Tensor, 'f'],
) -> tuple[Bool[Tensor, 'f c'], Bool[Tensor, 'f c 2']]:
    """Keep an image only when both hands are labelled or known absent.

    Label flags describe available hand labels even on invalid images; callers
    must apply image_valid before using them for supervision.
    """
    labelled: Bool[Tensor, 'f c 2'] = ((confidence > SHOW3D_CONFIDENCE_THRESHOLD) & has_pose)[:, None, :].expand_as(inside)
    absent: Bool[Tensor, 'f c 2'] = has_pose[:, None, :] & (inside == 0)
    return headset_valid[:, None] & (labelled | absent).all(dim=-1), labelled
