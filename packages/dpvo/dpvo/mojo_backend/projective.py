"""Fused inference camera geometry; differentiable training keeps projective_ops."""

import torch
from jaxtyping import Float32, Int64
from torch import Tensor

from dpvo.mojo_backend._launch import launch


def flow_magnitude(
    poses: Float32[Tensor, "1 frames 7"],
    patches: Float32[Tensor, "1 patches 3 ps ps"],
    intrinsics: Float32[Tensor, "1 frames 4"],
    source: Int64[Tensor, "edges"],
    target: Int64[Tensor, "edges"],
    patch: Int64[Tensor, "edges"],
    *,
    beta: float,
) -> Float32[Tensor, "1 edges ps ps"]:
    """Blend full-motion and translation-only pixel flow, as projective_ops.flow_mag.

    Uses each edge's source and target intrinsics and clamps homogeneous Z to 0.1
    before projection. This inference operation does not define an autograd rule.
    """
    size: int = patches.shape[-1]
    flow: Float32[Tensor, "edges ps ps"] = torch.empty(source.numel(), size, size, dtype=torch.float32, device=poses.device)
    if source.numel():
        launch(
            "flow_magnitude", (beta,),
            (poses.view(-1, 7), patches.view(-1, 3, size, size), intrinsics.view(-1, 4), source, target, patch, flow),
        )
    return flow[None]
