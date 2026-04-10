"""Pure-PyTorch scatter operations replacing the ``pytorch-scatter`` C++ extension.

This module provides index-based aggregation primitives (sum and softmax) that
operate along a specified dimension.  They are used throughout DPVO for
accumulating per-edge quantities into per-node (pose or patch) buffers --
e.g. assembling the normal equations in bundle adjustment, or computing
attention-weighted messages in the :class:`~mini_dpvo.blocks.SoftAgg` module.

The implementations rely only on :func:`torch.Tensor.scatter_add_` and
:func:`torch.Tensor.scatter_reduce_`, so no custom CUDA kernels or external
packages are needed.
"""

import torch
from jaxtyping import Float, Int
from torch import Tensor


def _expand_index(index: Int[Tensor, "n"], src: Float[Tensor, "*shape"], dim: int) -> Float[Tensor, "*shape"]:
    """Broadcast a 1-D index tensor so it can be used with ``scatter_add_``.

    ``scatter_add_`` requires the index tensor to have the same number of
    dimensions as the source.  This helper reshapes ``index`` to size-1 on
    every dimension except ``dim``, then expands it to match ``src``.

    Args:
        index: 1-D tensor of destination indices, length ``n``.
        src: Source tensor whose shape the expanded index must match.
        dim: The dimension along which scattering will occur.

    Returns:
        Index tensor with the same shape as ``src``, suitable for
        ``scatter_add_`` or ``gather``.
    """
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
    """Sum elements of ``src`` into bins defined by ``index`` along ``dim``.

    This is a drop-in replacement for ``torch_scatter.scatter_add`` used in
    the original DPVO codebase.  For each position ``i`` along ``dim``,
    ``src[..., i, ...]`` is added to ``out[..., index[i], ...]``.

    Args:
        src: Source tensor containing values to scatter.
        index: 1-D index tensor mapping each element along ``dim`` to a
            destination bin.  Must have the same length as ``src.shape[dim]``.
        dim: Dimension along which to scatter (default 1, the edge dim in
            DPVO tensors).
        dim_size: Size of the output along ``dim``.  If ``None``, inferred
            as ``max(index) + 1``.

    Returns:
        Tensor with the same shape as ``src`` except along ``dim``, which
        has size ``dim_size``.  Each bin contains the sum of the
        corresponding source elements.
    """
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
    """Compute group-wise softmax over ``src`` using ``index`` to define groups.

    For each unique value ``g`` in ``index``, this computes a numerically
    stable softmax over all elements in ``src`` that belong to group ``g``
    along dimension ``dim``.

    The algorithm follows three steps for numerical stability:

    1. Subtract the per-group maximum (computed via ``scatter_reduce_``
       with ``reduce="amax"``) to prevent overflow in the exponent.
    2. Exponentiate the shifted values.
    3. Divide by the per-group sum of exponentials (computed via
       :func:`scatter_sum`).

    This is used by :class:`~mini_dpvo.blocks.SoftAgg` to compute
    attention weights within each spatial or temporal neighborhood.

    Args:
        src: Source tensor containing logits / raw scores.
        index: 1-D index tensor defining group membership along ``dim``.
        dim: Dimension along which groups are defined (default 1).
        dim_size: Number of groups.  If ``None``, inferred from ``index``.

    Returns:
        Tensor of the same shape as ``src`` where elements within each
        group sum to 1.
    """
    index: torch.Tensor = index.long()
    if dim_size is None:
        dim_size: int = int(index.max().item()) + 1 if index.numel() > 0 else 0
    if dim_size == 0 or index.numel() == 0:
        return torch.zeros_like(src)

    expanded_index: torch.Tensor = _expand_index(index, src, dim)
    max_shape: list[int] = list(src.shape)
    max_shape[dim] = dim_size
    # Step 1: per-group max for numerical stability
    max_per_group: torch.Tensor = torch.full(max_shape, -torch.inf, dtype=src.dtype, device=src.device)
    max_per_group.scatter_reduce_(dim, expanded_index, src, reduce="amax", include_self=True)
    gathered_max: torch.Tensor = max_per_group.gather(dim, expanded_index)

    # Step 2: exponentiate shifted values (max subtracted)
    exp: torch.Tensor = torch.exp(src - gathered_max)
    # Step 3: normalize by per-group sum
    denom: torch.Tensor = scatter_sum(exp, index, dim=dim, dim_size=dim_size)
    result: torch.Tensor = exp / denom.gather(dim, expanded_index).clamp_min(torch.finfo(src.dtype).tiny)
    return result
