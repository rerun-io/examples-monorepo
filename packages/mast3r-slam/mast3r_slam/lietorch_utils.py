import einops
import lietorch
import torch
from jaxtyping import Float
from torch import Tensor


def as_SE3(X: lietorch.Sim3 | lietorch.SE3) -> lietorch.SE3:
    """Convert a Sim3 or SE3 pose to SE3 by discarding the scale component."""
    if isinstance(X, lietorch.SE3):
        return X
    t: Float[Tensor, "... 3"]
    q: Float[Tensor, "... 4"]
    s: Float[Tensor, "... 1"]
    t, q, s = einops.rearrange(X.data.detach().cpu(), "... c -> (...) c").split([3, 4, 1], -1)
    world_T_cam: lietorch.SE3 = lietorch.SE3(torch.cat([t, q], dim=-1))
    return world_T_cam
