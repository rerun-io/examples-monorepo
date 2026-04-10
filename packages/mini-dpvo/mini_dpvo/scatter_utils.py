import torch


def _expand_index(index: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
    view = [1] * src.dim()
    view[dim] = index.shape[0]
    return index.view(*view).expand_as(src)


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 1,
    dim_size: int | None = None,
) -> torch.Tensor:
    index = index.long()
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0

    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = src.new_zeros(out_shape)
    if dim_size == 0 or index.numel() == 0:
        return out

    out.scatter_add_(dim, _expand_index(index, src, dim), src)
    return out


def scatter_softmax(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 1,
    dim_size: int | None = None,
) -> torch.Tensor:
    index = index.long()
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
    if dim_size == 0 or index.numel() == 0:
        return torch.zeros_like(src)

    expanded_index = _expand_index(index, src, dim)
    max_shape = list(src.shape)
    max_shape[dim] = dim_size
    max_per_group = torch.full(max_shape, -torch.inf, dtype=src.dtype, device=src.device)
    max_per_group.scatter_reduce_(dim, expanded_index, src, reduce="amax", include_self=True)
    gathered_max = max_per_group.gather(dim, expanded_index)

    exp = torch.exp(src - gathered_max)
    denom = scatter_sum(exp, index, dim=dim, dim_size=dim_size)
    return exp / denom.gather(dim, expanded_index).clamp_min(torch.finfo(src.dtype).tiny)
