"""Differentiable low-LOD SOMA surface landmarks and shared head attachment."""

import torch
from jaxtyping import Float32
from torch import Tensor

# SOMA-X v0.3.0 low topology, neutral asset SHA256 5829af4a5333abdba5ad449360b868faacee99d395971e62f35c0e7169432b63.
# Visually checked at the anterior ear-canal/tragus region, not the outer helix.
EAR_VERTICES = (676, 649, 675, 2880, 2853, 2879)


def ear_vertices(
    rest: Float32[Tensor, "b v 3"],
    weights: Float32[Tensor, "v j"],
    inverse_bind: Float32[Tensor, "b j 4 4"],
    transforms: Float32[Tensor, "n j 4 4"],
) -> Float32[Tensor, "n 2 3"]:
    """Skin six vertices and average three per ear without evaluating the full mesh."""
    selected = rest[:, EAR_VERTICES, :]
    homogeneous = torch.cat((selected, torch.ones_like(selected[..., :1])), dim=-1)
    blended = torch.einsum("vj,njkl->nvkl", weights[list(EAR_VERTICES)], transforms @ inverse_bind)
    points = torch.einsum("nvij,nvj->nvi", blended, homogeneous.expand(len(transforms), -1, -1))[..., :3]
    return points.reshape(-1, 2, 3, 3).mean(dim=2)


def rotation_delta(vector: Float32[Tensor, "3"]) -> Float32[Tensor, "3 3"]:
    """SO(3) exponential with finite derivatives at the zero initialization."""
    x, y, z = vector.unbind()
    zero = x * 0.0
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    return torch.matrix_exp(skew)


# COCO 17..22: left big toe / small toe / heel, then right.
# SOMA v0.3.0 low mesh: distal medial and lateral toes, posterior heel surface.
# Right-hand IDs mirror the left topology at +2204. See the foot mapping render.
FOOT_VERTICES = (
    (2034, 2144, 2026),
    (1968, 1970, 1965),
    (2155, 2152, 2039),
    (4238, 4348, 4230),
    (4172, 4174, 4169),
    (4359, 4356, 4243),
)


def foot_vertices(
    rest: Float32[Tensor, "b v 3"],
    weights: Float32[Tensor, "v j"],
    inverse_bind: Float32[Tensor, "b j 4 4"],
    transforms: Float32[Tensor, "n j 4 4"],
) -> Float32[Tensor, "n 6 3"]:
    """Skin only the eighteen validated foot vertices and average three per landmark."""
    indices = [vertex for group in FOOT_VERTICES for vertex in group]
    selected = rest[:, indices, :]
    homogeneous = torch.cat((selected, torch.ones_like(selected[..., :1])), dim=-1)
    blended = torch.einsum("vj,njkl->nvkl", weights[indices], transforms @ inverse_bind)
    points = torch.einsum("nvij,nvj->nvi", blended, homogeneous.expand(len(transforms), -1, -1))[..., :3]
    return points.reshape(-1, 6, 3, 3).mean(dim=2)
