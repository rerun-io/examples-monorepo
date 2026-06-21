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
    vis_thresh: float = 0.5,
    clamp_sigma: tuple[float, float] = (1.0, 24.0),
) -> tuple[Float32[torch.Tensor, "n 3"], Bool[torch.Tensor, "n"]]:
    """Triangulate one tick's landmarks from multiple cameras.

    Args:
        pts2d_per_cam: Per-camera ``[x_px, y_px, log_variance]`` landmarks.
        k_per_cam: Per-camera intrinsics (engine resolution).
        world_to_cam_per_cam: Per-camera world->cam transforms.
        vis_per_cam: Per-camera visibility in [0, 1]; points at/below
            ``vis_thresh`` are excluded for that camera (0.5: only confident
            landmarks triangulate — occluded ones poisoned the anchors).
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


def triangulate_gated(
    pts2d_per_cam: list[Float32[torch.Tensor, "n 3"]],
    k_per_cam: list[Float32[torch.Tensor, "3 3"]],
    world_to_cam_per_cam: list[Float32[torch.Tensor, "4 4"]],
    vis_per_cam: list[Float32[torch.Tensor, "n"]],
    reproj_thresh_px: float = 40.0,
    min_cams: int = 2,
    vis_thresh: float = 0.5,
) -> tuple[list[Float32[torch.Tensor, "n"]], Float32[torch.Tensor, "n 3"], Bool[torch.Tensor, "n"]]:
    """Triangulate, then iteratively DROP the worst camera whose confident
    landmarks reproject with mean error > ``reproj_thresh_px``, re-triangulating,
    while > ``min_cams`` cameras survive.

    Multi-view consistency gate (port of the original's
    ``propagate_ids_via_reprojection``): when a per-camera SAM2 mask swaps to the
    OTHER person during contact, that camera's landmarks reproject hundreds of px
    from the (mostly-correct) 3D estimate, so a generous threshold isolates it
    cleanly while leaving correctly-tracked cameras (a few px) untouched. The
    returned gated visibility zeros dropped cameras so they neither triangulate
    NOR pull the downstream fit's reprojection term. No-op when all cameras agree
    (the common case, incl. single-person), so single-person fits are unchanged.
    """
    n_cams: int = len(pts2d_per_cam)
    gated: list[Float32[torch.Tensor, "n"]] = [v.clone() for v in vis_per_cam]
    proj: list[Float32[torch.Tensor, "3 4"]] = [k_per_cam[c] @ world_to_cam_per_cam[c][:3, :] for c in range(n_cams)]
    active: list[int] = list(range(n_cams))
    points3d: Float32[torch.Tensor, "n 3"]
    valid: Bool[torch.Tensor, "n"]
    points3d, valid = triangulate_points(pts2d_per_cam, k_per_cam, world_to_cam_per_cam, gated, vis_thresh=vis_thresh)
    while len(active) > min_cams:
        xh: Float32[torch.Tensor, "n 4"] = torch.cat([points3d, torch.ones_like(points3d[:, :1])], dim=-1)
        worst_c: int = -1
        worst_err: float = reproj_thresh_px
        for c in active:
            conf: Bool[torch.Tensor, "n"] = (gated[c] > vis_thresh) & valid
            if int(conf.sum().item()) < 4:
                continue
            uvw: Float32[torch.Tensor, "n 3"] = xh @ proj[c].T
            uv: Float32[torch.Tensor, "n 2"] = uvw[:, :2] / (uvw[:, 2:3].abs() + 1e-6)
            merr: float = float((uv - pts2d_per_cam[c][:, :2]).norm(dim=-1)[conf].mean().item())
            if merr > worst_err:
                worst_err, worst_c = merr, c
        if worst_c < 0:
            break
        active.remove(worst_c)
        gated[worst_c] = torch.zeros_like(gated[worst_c])
        points3d, valid = triangulate_points(pts2d_per_cam, k_per_cam, world_to_cam_per_cam, gated, vis_thresh=vis_thresh)
    return gated, points3d, valid
