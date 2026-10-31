"""Soft anatomical foot constraints in a Y-up world."""


import torch
from jaxtyping import Float
from torch import Tensor


def foot_pose_penalty(poses: Float[Tensor, "n 2 3"], axes: Float[Tensor, "2 3 3"], toe_joints: bool = False) -> Float[Tensor, ""]:
    """Soft local rotation-vector bounds; toes allow flex -20/+60° and yaw/roll ±10°."""
    angles = torch.einsum("nfi,fij->nfj", poses, axes)
    lower = torch.deg2rad(poses.new_tensor([-20.0, -10.0, -10.0] if toe_joints else [-35.0, -35.0, -25.0]))
    upper = torch.deg2rad(poses.new_tensor([60.0, 10.0, 10.0] if toe_joints else [50.0, 35.0, 25.0]))
    excess = torch.relu(lower - angles) + torch.relu(angles - upper)
    return 0.04 * (angles / 0.5).square().mean() + (excess / 0.1).square().mean()


def symmetry_penalty(lengths: Float[Tensor, "n 8"]) -> Float[Tensor, ""]:
    """Match paired upper arm, forearm, thigh and shin lengths in 1 cm units."""
    return ((lengths[:, 0::2] - lengths[:, 1::2]) / 0.01).square().mean()
