import einops
import lietorch
import torch


def as_SE3(X: lietorch.Sim3 | lietorch.SE3) -> lietorch.SE3:
    """Convert a Sim3 or SE3 pose to SE3 by discarding the scale component."""
    if isinstance(X, lietorch.SE3):
        return X
    t: torch.Tensor
    q: torch.Tensor
    s: torch.Tensor
    t, q, s = einops.rearrange(X.data.detach().cpu(), "... c -> (...) c").split([3, 4, 1], -1)
    T_WC: lietorch.SE3 = lietorch.SE3(torch.cat([t, q], dim=-1))
    return T_WC
