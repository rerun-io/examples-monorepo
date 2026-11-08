"""simple-knn's `distCUDA2` (bkerbl/simple-knn@f155ec0) on the Mojo kernels: 2DGS's initial scale per SfM point."""

import torch
from jaxtyping import Float32
from torch import Tensor

from hexels_mojo._launch import launch

BOX_SIZE: int = 1024


def distCUDA2(points: Float32[Tensor, "p 3"]) -> Float32[Tensor, "p"]:  # noqa: N802 (upstream's name)
    """Mean squared distance from every point to its 3 nearest others, approximated as upstream does (see mojo/knn.mojo)."""
    points = points.contiguous()
    count: int = points.shape[0]
    # cub's DeviceReduce in upstream starts from {0, 0, 0}, so the box always contains the origin.
    bounds: Tensor = torch.cat((points.amin(0).clamp_max(0.0), points.amax(0).clamp_min(0.0))).contiguous()
    codes: Tensor = torch.empty(count, dtype=torch.int64, device=points.device)
    launch("knn", (0,), (points, bounds, codes))
    # cub::DeviceRadixSort::SortPairs is stable: ties keep index order.
    order: Tensor = torch.sort(codes, stable=True).indices
    boxes: Tensor = torch.empty((count + BOX_SIZE - 1) // BOX_SIZE, 6, dtype=torch.float32, device=points.device)
    dists: Tensor = torch.zeros(count, dtype=torch.float32, device=points.device)
    launch("knn", (1,), (points, order, boxes, dists))
    return dists
