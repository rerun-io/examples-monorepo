"""GPU weighted-DLT triangulation — port of ``optimization/utils/triangulation_functions.py``.

The original flattened to CPU for a batched SVD; here the right null vector
comes from a batched 4x4 ``eigh`` of AᵀA on the GPU (smallest eigenvector ==
last right-singular vector), which is far cheaper than SVD at this size and
never leaves the device.
"""

from __future__ import annotations

import torch
from jaxtyping import Bool, Float32


def triangulate_points(
    pts2d_per_cam: list[Float32[torch.Tensor, "n 3"]],
    k_per_cam: list[Float32[torch.Tensor, "3 3"]],
    world_to_cam_per_cam: list[Float32[torch.Tensor, "4 4"]],
    vis_per_cam: list[Float32[torch.Tensor, "n"]] | None = None,
    img_size_px: float = 512.0,
    vis_thresh: float = 0.0,
    clamp_sigma: tuple[float, float] = (1.0, 24.0),
) -> tuple[Float32[torch.Tensor, "n 3"], Bool[torch.Tensor, "n"]]:
    """Triangulate one tick's landmarks from multiple cameras.

    Args:
        pts2d_per_cam: Per-camera ``[x_px, y_px, log_variance]`` landmarks.
        k_per_cam: Per-camera intrinsics (engine resolution).
        world_to_cam_per_cam: Per-camera world->cam transforms.
        vis_per_cam: Per-camera visibility in [0, 1]; points at/below
            ``vis_thresh`` are excluded for that camera.
        img_size_px: Crop size that scales the log-variance to pixel sigma
            (matches the original: ``sigma = exp(0.5*logvar) * img_size/2``).
        clamp_sigma: Pixel-sigma clamp range.

    Returns:
        World points ``[n, 3]`` and a validity mask (>= 2 contributing cameras).
    """
    device: torch.device = k_per_cam[0].device
    n_cams: int = len(pts2d_per_cam)
    n: int = pts2d_per_cam[0].shape[0]

    proj: Float32[torch.Tensor, "c 3 4"] = torch.stack(
        [k_per_cam[c] @ world_to_cam_per_cam[c][:3, :] for c in range(n_cams)], dim=0
    )
    pts2d: Float32[torch.Tensor, "c n 3"] = torch.stack(pts2d_per_cam, dim=0)
    vis: Float32[torch.Tensor, "c n"] = (
        torch.stack(vis_per_cam, dim=0) if vis_per_cam is not None else torch.ones(n_cams, n, device=device)
    )

    valid: Bool[torch.Tensor, "c n"] = (vis > vis_thresh) & (pts2d[..., :2].abs().sum(-1) > 0)
    valid_f: Float32[torch.Tensor, "c n"] = valid.float()

    sigma: Float32[torch.Tensor, "c n"] = (torch.exp(0.5 * pts2d[..., 2]) * (img_size_px / 2.0)).clamp(*clamp_sigma)
    w: Float32[torch.Tensor, "c n"] = (1.0 / (sigma * sigma)) * valid_f
    w_sqrt: Float32[torch.Tensor, "c n"] = torch.sqrt(w + 1e-12)

    u: Float32[torch.Tensor, "c n"] = pts2d[..., 0]
    v: Float32[torch.Tensor, "c n"] = pts2d[..., 1]
    p0: Float32[torch.Tensor, "c 4"] = proj[:, 0, :]
    p1: Float32[torch.Tensor, "c 4"] = proj[:, 1, :]
    p2: Float32[torch.Tensor, "c 4"] = proj[:, 2, :]

    # Rows of the DLT system per point: [n, 2c, 4].
    r1: Float32[torch.Tensor, "c n 4"] = (u.unsqueeze(-1) * p2.unsqueeze(1) - p0.unsqueeze(1)) * w_sqrt.unsqueeze(-1)
    r2: Float32[torch.Tensor, "c n 4"] = (v.unsqueeze(-1) * p2.unsqueeze(1) - p1.unsqueeze(1)) * w_sqrt.unsqueeze(-1)
    a: Float32[torch.Tensor, "n c2 4"] = torch.cat([r1, r2], dim=0).permute(1, 0, 2).contiguous()

    # Smallest eigenvector of AtA == right-singular vector of A.
    ata: Float32[torch.Tensor, "n 4 4"] = a.transpose(1, 2) @ a
    eigvecs: Float32[torch.Tensor, "n 4 4"] = torch.linalg.eigh(ata.double()).eigenvectors.float()
    xh: Float32[torch.Tensor, "n 4"] = eigvecs[:, :, 0]
    xw: Float32[torch.Tensor, "n 3"] = xh[:, :3] / (xh[:, 3:4] + 1e-8)

    ok: Bool[torch.Tensor, "n"] = valid.sum(dim=0) >= 2
    xw = xw.masked_fill(~ok.unsqueeze(-1), 0.0)
    return xw, ok
