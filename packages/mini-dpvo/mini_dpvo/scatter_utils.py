import torch
from jaxtyping import Float, Int
from torch import Tensor


def _expand_index(index: Int[Tensor, "n"], src: Float[Tensor, "*shape"], dim: int) -> Float[Tensor, "*shape"]:
    view: list[int] = [1] * src.dim()
    view[dim] = index.shape[0]
    expanded: Float[Tensor, "*shape"] = index.view(*view).expand_as(src)
    return expanded


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 1,
    dim_size: int | None = None,
) -> torch.Tensor:
    index: torch.Tensor = index.long()
    if dim_size is None:
        dim_size: int = int(index.max().item()) + 1 if index.numel() > 0 else 0

    out_shape: list[int] = list(src.shape)
    out_shape[dim] = dim_size
    out: torch.Tensor = src.new_zeros(out_shape)
    if dim_size == 0 or index.numel() == 0:
        return out

    expanded_index: torch.Tensor = _expand_index(index, src, dim)
    out.scatter_add_(dim, expanded_index, src)
    return out


def scatter_softmax(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 1,
    dim_size: int | None = None,
) -> torch.Tensor:
    index: torch.Tensor = index.long()
    if dim_size is None:
        dim_size: int = int(index.max().item()) + 1 if index.numel() > 0 else 0
    if dim_size == 0 or index.numel() == 0:
        return torch.zeros_like(src)

    expanded_index: torch.Tensor = _expand_index(index, src, dim)
    max_shape: list[int] = list(src.shape)
    max_shape[dim] = dim_size
    max_per_group: torch.Tensor = torch.full(max_shape, -torch.inf, dtype=src.dtype, device=src.device)
    max_per_group.scatter_reduce_(dim, expanded_index, src, reduce="amax", include_self=True)
    gathered_max: torch.Tensor = max_per_group.gather(dim, expanded_index)

    exp: torch.Tensor = torch.exp(src - gathered_max)
    denom: torch.Tensor = scatter_sum(exp, index, dim=dim, dim_size=dim_size)
    result: torch.Tensor = exp / denom.gather(dim, expanded_index).clamp_min(torch.finfo(src.dtype).tiny)
    return result
