"""Inference-only fallback for the one optional GVHMR KNN helper."""

from __future__ import annotations


def knn_points(p1, p2, *, K: int = 1, return_nn: bool = False):
    """Return the PyTorch3D-compatible tuple using native PyTorch operations."""

    import torch

    if p1.ndim != 3 or p2.ndim != 3 or p1.shape[0] != p2.shape[0]:
        raise ValueError("p1 and p2 must have shapes (N, P, D) and (N, Q, D).")
    if K < 1 or p2.shape[1] < K:
        raise ValueError(f"K must be in [1, {p2.shape[1]}], got {K}.")
    distances = torch.cdist(p1, p2).square()
    squared_distances, indices = distances.topk(K, dim=-1, largest=False, sorted=True)
    neighbors = None
    if return_nn:
        expanded = p2[:, None].expand(-1, p1.shape[1], -1, -1)
        gather_index = indices[..., None].expand(-1, -1, -1, p2.shape[-1])
        neighbors = torch.gather(expanded, 2, gather_index)
    return squared_distances, indices, neighbors
