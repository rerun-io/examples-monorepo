"""The two PyTorch3D 0.7 ops TRiGS calls, in plain torch, so PyTorch3D is never built.

recipes/open4dgs-trigs-src patch 0001 points every ``from pytorch3d.ops import ...`` in ``trigs`` here. The
signatures and return values follow ``pytorch3d.ops.knn_points`` and ``pytorch3d.ops.sample_farthest_points``
for the arguments TRiGS passes (padded batches with ``lengths``, per-batch ``K`` for farthest-point sampling).
"""

from typing import NamedTuple

import torch
from jaxtyping import Float, Int64
from torch import Tensor


class KNN(NamedTuple):
    """``pytorch3d.ops.knn.KNN``: squared distances, indices into ``p2``, and the neighbours when asked for."""

    dists: Float[Tensor, "n p1 k"]
    idx: Int64[Tensor, "n p1 k"]
    knn: Float[Tensor, "n p1 k d"] | None


KNN_MEMORY_BUDGET: int = 1 << 30
"""Bytes of one chunk's distance matrix in ``knn_points`` (1 GiB): rows per chunk = budget / (batch * p2 * 4 B)."""


def knn_points(
    p1: Float[Tensor, "n p1 d"],
    p2: Float[Tensor, "n p2 d"],
    lengths1: Int64[Tensor, "n"] | None = None,
    lengths2: Int64[Tensor, "n"] | None = None,
    norm: int = 2,
    K: int = 1,
    return_nn: bool = False,
    chunk_size: int | None = None,
    memory_budget: int = KNN_MEMORY_BUDGET,
) -> KNN:
    """K nearest neighbours of each ``p1`` point among the valid ``p2`` points, sorted nearest first.

    Distances are squared L2 (``norm=2``) or L1 (``norm=1``), as in PyTorch3D. Rows of ``p1`` past ``lengths1``
    and neighbour slots past ``lengths2`` get distance 0 and index -1. ``p1`` is processed in chunks so one
    chunk's ``(n, chunk, p2)`` distance matrix stays within ``memory_budget`` bytes (``chunk_size`` overrides);
    the square and the padding mask are applied in place, so no second copy of that matrix exists.
    """
    if norm not in (1, 2):
        raise ValueError(f"norm must be 1 or 2, got {norm}")
    n, n_p1, _ = p1.shape
    n_p2 = p2.shape[1]
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if lengths1 is None:
        lengths1 = torch.full((n,), n_p1, dtype=torch.int64, device=p1.device)
    if lengths2 is None:
        lengths2 = torch.full((n,), n_p2, dtype=torch.int64, device=p1.device)
    k_eff = min(K, n_p2)
    dists = torch.zeros((n, n_p1, K), dtype=p1.dtype, device=p1.device)
    idx = torch.full((n, n_p1, K), -1, dtype=torch.int64, device=p1.device)
    invalid_p2 = torch.arange(n_p2, device=p1.device)[None, :] >= lengths2[:, None]
    any_invalid: bool = bool(invalid_p2.any())
    rows: int = chunk_size if chunk_size is not None else max(1, memory_budget // max(1, n * n_p2 * p1.element_size()))
    for start in range(0, n_p1, rows):
        q = p1[:, start : start + rows]
        d = torch.cdist(q, p2).square_() if norm == 2 else torch.cdist(q, p2, p=1.0)
        if any_invalid:
            d.masked_fill_(invalid_p2[:, None, :], float("inf"))
        chunk_d, chunk_i = d.topk(k_eff, dim=-1, largest=False, sorted=True)
        del d
        dists[:, start : start + rows, :k_eff] = chunk_d
        idx[:, start : start + rows, :k_eff] = chunk_i
    # Slots past a batch item's valid p2 count (inf distance) and rows past lengths1 carry no neighbour.
    empty_slot = torch.isinf(dists) | (torch.arange(n_p1, device=p1.device)[None, :, None] >= lengths1[:, None, None])
    dists = dists.masked_fill(empty_slot, 0.0)
    idx = idx.masked_fill(empty_slot, -1)
    neighbours = None
    if return_nn:
        gathered = torch.gather(p2, 1, idx.clamp(min=0).reshape(n, -1)[..., None].expand(-1, -1, p2.shape[-1]))
        neighbours = gathered.reshape(n, n_p1, K, p2.shape[-1]).masked_fill(empty_slot[..., None], 0.0)
    return KNN(dists=dists, idx=idx, knn=neighbours)


def sample_farthest_points(
    points: Float[Tensor, "n p d"],
    lengths: Int64[Tensor, "n"] | None = None,
    K: int | list[int] | Int64[Tensor, "n"] = 50,
    random_start_point: bool = False,
) -> tuple[Float[Tensor, "n k d"], Int64[Tensor, "n k"]]:
    """Iterative farthest-point sampling per batch item, PyTorch3D semantics.

    Each item keeps ``K[i]`` of its first ``lengths[i]`` points (fewer when it has fewer); the output is padded to
    ``max(K)`` with index -1 and zero points. The first pick is point 0, or a random valid point with
    ``random_start_point``. All batch items advance together, one pick per iteration, on the points' device.
    """
    n, n_p, _ = points.shape
    device = points.device
    if lengths is None:
        lengths = torch.full((n,), n_p, dtype=torch.int64, device=device)
    if isinstance(K, int):
        k_per = torch.full((n,), K, dtype=torch.int64, device=device)
    elif isinstance(K, list):
        k_per = torch.tensor(K, dtype=torch.int64, device=device)
    else:
        k_per = K.to(device=device, dtype=torch.int64)
    k_per = torch.minimum(k_per, lengths)
    k_max = int(k_per.max().item()) if n > 0 else 0
    idx = torch.full((n, k_max), -1, dtype=torch.int64, device=device)
    if k_max == 0:
        return points.new_zeros((n, 0, points.shape[-1])), idx
    valid = torch.arange(n_p, device=device)[None, :] < lengths[:, None]
    batch = torch.arange(n, device=device)
    start = torch.rand(n, device=device) * lengths.clamp(min=1) if random_start_point else torch.zeros(n, device=device)
    current = start.long()
    min_dist = torch.full((n, n_p), float("inf"), dtype=points.dtype, device=device).masked_fill(~valid, -1.0)
    for step in range(k_max):
        active = step < k_per
        idx[:, step] = torch.where(active, current, torch.full_like(current, -1))
        d = (points - points[batch, current][:, None, :]).square().sum(-1)
        min_dist = torch.minimum(min_dist, d.masked_fill(~valid, -1.0))
        current = min_dist.argmax(dim=1)
    sampled = torch.gather(points, 1, idx.clamp(min=0)[..., None].expand(-1, -1, points.shape[-1]))
    sampled = sampled.masked_fill((idx < 0)[..., None], 0.0)
    return sampled, idx
