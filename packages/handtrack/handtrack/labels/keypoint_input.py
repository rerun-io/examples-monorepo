"""The 63 interleaved KeyNet features and the untracked-hand convention.

An untracked hand receives ``torch.zeros(63, device=device)``. ZERO_INPUT is
that vector on CPU; construct it on the caller's device for accelerator use.
"""

import torch
from einops import rearrange
from jaxtyping import Float32
from simplecv.umetrack_temp.generic_hand_model_torch import LANDMARK, HandModelTorch
from torch import Tensor

from handtrack.labels.crops import CROP_SIZE
from handtrack.labels.heatmaps import DISTANCE_RANGE_MM

ZERO_INPUT: Float32[Tensor, '63'] = torch.zeros(63)
"""All-zero keypoint features mean the hand is not tracked yet."""


def hand_scale(model: HandModelTorch, generic: HandModelTorch) -> float:
    """Least-squares scale from generic to subject wrist-centred rest landmarks."""
    subject: Float32[Tensor, '21 3'] = model.landmark_rest_positions - model.landmark_rest_positions[LANDMARK.WRIST_JOINT]
    reference: Float32[Tensor, '21 3'] = generic.landmark_rest_positions.to(subject.device)
    reference = reference - reference[LANDMARK.WRIST_JOINT]
    return float((subject * reference).sum() / reference.square().sum())


def relative_distances(points_cam: Float32[Tensor, '*b 21 3'], phi: Float32[Tensor, '*b']) -> Float32[Tensor, '*b 21']:
    """Convert radial distances in metres to centred, scale-normalised millimetres."""
    distance: Float32[Tensor, '*b 21'] = torch.linalg.vector_norm(points_cam, dim=-1)
    return (distance - distance.mean(dim=-1, keepdim=True)) * 1000.0 / phi[..., None]


def keypoint_input(points_crop: Float32[Tensor, 'b 21 2'], d_rel_mm: Float32[Tensor, 'b 21']) -> Float32[Tensor, 'b 63']:
    """Interleave (u,v,d) using crop pixel centres, including any prior mirroring.

    Our distance scaling d=d_rel_mm/DISTANCE_RANGE_MM keeps all three features
    O(1). Coordinates outside the crop stay unclipped for tracking history.
    """
    uv: Float32[Tensor, 'b 21 2'] = (points_crop + 0.5) / CROP_SIZE
    return rearrange(torch.cat((uv, (d_rel_mm / DISTANCE_RANGE_MM)[..., None]), dim=-1), 'b k feature -> b (k feature)')


def add_input_noise(vectors: Float32[Tensor, 'b 63'], generator: torch.Generator, uv_std: float, d_std: float) -> Float32[Tensor, 'b 63']:
    """Add independent Gaussian noise in normalised feature units; do not clamp."""
    noise: Float32[Tensor, 'b 63'] = torch.randn((vectors.shape[0], 63), generator=generator, device=vectors.device)
    noise[..., 0::3] *= uv_std
    noise[..., 1::3] *= uv_std
    noise[..., 2::3] *= d_std
    return vectors + noise
