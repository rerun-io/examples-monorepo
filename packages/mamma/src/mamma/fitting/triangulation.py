"""GPU weighted-DLT triangulation — port of ``optimization/utils/triangulation_functions.py``.

The original flattened to CPU for a batched SVD; here the right null vector
comes from a batched 4x4 ``eigh`` of AᵀA on the GPU (smallest eigenvector ==
last right-singular vector), which is far cheaper than SVD at this size and
never leaves the device.
"""

from __future__ import annotations

import torch
from jaxtyping import Bool, Float32

PRIOR_REF_WH: tuple[int, int] = (1024, 752)
"""Reference image size (engine width x height) of the temporal prior's tolerance."""
PRIOR_THRESH_REF_PX: float = 40.0
"""How far (pixels at :data:`PRIOR_REF_WH`) a camera's landmarks may lie from the previous tick's points and still seed
the prior candidate. Wrong-person cameras sat 56-468 px from their own person on MultiMama_3, correct ones 2-8 px."""


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

    proj: Float32[torch.Tensor, "c 3 4"] = torch.stack([k_per_cam[c] @ world_to_cam_per_cam[c][:3, :] for c in range(n_cams)], dim=0)
    pts2d: Float32[torch.Tensor, "c n 3"] = torch.stack(pts2d_per_cam, dim=0)
    vis: Float32[torch.Tensor, "c n"] = torch.stack(vis_per_cam, dim=0) if vis_per_cam is not None else torch.ones(n_cams, n, device=device)

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
    prior: tuple[Float32[torch.Tensor, "n 3"], Bool[torch.Tensor, "n"]] | None = None,
    prior_thresh: float = PRIOR_THRESH_REF_PX,
    image_wh_per_cam: list[tuple[int, int]] | None = None,
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

    ``prior`` (the person's points from the previous tick) proposes a second candidate: the cameras that agree with
    it (within ``prior_thresh`` pixels at the 1024x752 reference resolution, see :data:`PRIOR_THRESH_REF_PX`) seed a
    triangulation, every camera agreeing with that triangulation is re-admitted, and the greedy drop runs from there.
    The candidate kept is the one the current cameras agree with more (a soft vote, 1 per exact camera falling to 0 at
    ``reproj_thresh_px``); the prior only breaks a tie, by proximity. When the masks of half the cameras slide onto another person during contact, the all-camera
    start is already wrong and its greedy drop removes good cameras (MultiMama_3: 5-8 of 14 cameras); a prior that
    lags a moving majority loses the vote instead of vetoing it. A seed with fewer than ``min_cams`` cameras or no
    valid point is not a candidate.
    """
    n_cams: int = len(pts2d_per_cam)
    proj: Float32[torch.Tensor, "c 3 4"] = torch.stack([k_per_cam[c] @ world_to_cam_per_cam[c][:3, :] for c in range(n_cams)])
    pts2d: Float32[torch.Tensor, "c n 2"] = torch.stack(pts2d_per_cam)[..., :2]

    observed: list[bool] = (torch.stack([v.abs().sum() for v in vis_per_cam]) > 0).tolist()

    def greedy(
        start: list[Float32[torch.Tensor, "n"]], excluded: frozenset[int] = frozenset()
    ) -> tuple[list[Float32[torch.Tensor, "n"]], Float32[torch.Tensor, "n 3"], Bool[torch.Tensor, "n"]]:
        gated: list[Float32[torch.Tensor, "n"]] = [v.clone() for v in start]
        active: list[int] = [c for c in range(n_cams) if c not in excluded]
        points3d, valid = triangulate_points(pts2d_per_cam, k_per_cam, world_to_cam_per_cam, gated, vis_thresh=vis_thresh)
        while len(active) > min_cams:
            # All cameras' confident-landmark counts + mean reprojection errors in
            # one batched pass and ONE host read per round (was 2 syncs/camera).
            stats: list[list[float]] = _camera_errors(points3d, valid, proj, pts2d, gated, vis_thresh)
            worst_c: int = -1
            worst_err: float = reproj_thresh_px
            for c in active:
                if stats[0][c] < 4:
                    continue
                if stats[1][c] > worst_err:
                    worst_err, worst_c = stats[1][c], c
            if worst_c < 0:
                break
            active.remove(worst_c)
            gated[worst_c] = torch.zeros_like(gated[worst_c])
            points3d, valid = triangulate_points(pts2d_per_cam, k_per_cam, world_to_cam_per_cam, gated, vis_thresh=vis_thresh)
        return gated, points3d, valid

    unseeded = greedy(vis_per_cam)
    if prior is None:
        return unseeded
    # Prior residuals at the reference resolution (per axis, so an anisotropic resize changes nothing).
    scale: Float32[torch.Tensor, "c 1 2"] | None = None
    if image_wh_per_cam is not None:
        scale = torch.tensor([[PRIOR_REF_WH[0] / w, PRIOR_REF_WH[1] / h] for w, h in image_wh_per_cam], dtype=pts2d.dtype, device=pts2d.device)[
            :, None, :
        ]
    prior_err: list[list[float]] = _camera_errors(prior[0], prior[1], proj, pts2d, vis_per_cam, vis_thresh, scale)
    seed_cams: list[int] = [c for c in range(n_cams) if prior_err[0][c] >= 4 and prior_err[1][c] <= prior_thresh]
    judged: list[int] = [c for c in range(n_cams) if observed[c] and prior_err[0][c] >= 4]
    if len(seed_cams) < min_cams or seed_cams == judged:
        return unseeded  # too little support, or every judged camera agrees with the prior: the same start as unseeded
    seed_vis: list[Float32[torch.Tensor, "n"]] = [vis_per_cam[c] if c in seed_cams else torch.zeros_like(vis_per_cam[c]) for c in range(n_cams)]
    seed_points, seed_valid = triangulate_points(pts2d_per_cam, k_per_cam, world_to_cam_per_cam, seed_vis, vis_thresh=vis_thresh)
    if not bool(seed_valid.any()):
        return unseeded
    seed_err: list[list[float]] = _camera_errors(seed_points, seed_valid, proj, pts2d, vis_per_cam, vis_thresh)
    left_out: frozenset[int] = frozenset(c for c in range(n_cams) if observed[c] and not (seed_err[0][c] >= 4 and seed_err[1][c] <= reproj_thresh_px))
    seeded = greedy([torch.zeros_like(v) if c in left_out else v for c, v in enumerate(vis_per_cam)], left_out)
    # Soft vote of every current camera: 1 for exact agreement, falling to 0 at the gate threshold (a blend of two people
    # sits inside the threshold for many cameras but agrees exactly with none).
    support: list[float] = []
    for _, points3d, valid in (unseeded, seeded):
        err: list[list[float]] = _camera_errors(points3d, valid, proj, pts2d, vis_per_cam, vis_thresh)
        support.append(sum(max(0.0, 1.0 - err[1][c] / reproj_thresh_px) for c in range(n_cams) if err[0][c] >= 4))
    if abs(support[1] - support[0]) > 1e-3:
        return seeded if support[1] > support[0] else unseeded
    both: Bool[torch.Tensor, "n"] = unseeded[2] & seeded[2] & prior[1]
    distances: list[float] = torch.stack([((points3d - prior[0]).norm(dim=-1) * both).sum() for _, points3d, _ in (unseeded, seeded)]).tolist()
    return seeded if distances[1] < distances[0] else unseeded


def _camera_errors(
    points3d: Float32[torch.Tensor, "n 3"],
    valid: Bool[torch.Tensor, "n"],
    proj: Float32[torch.Tensor, "c 3 4"],
    pts2d: Float32[torch.Tensor, "c n 2"],
    vis: list[Float32[torch.Tensor, "n"]],
    vis_thresh: float,
    scale: Float32[torch.Tensor, "c 1 2"] | None = None,
) -> list[list[float]]:
    """``[counts, mean errors]`` per camera: confident landmarks and their mean reprojection error (px) against
    ``points3d``, in one batched pass and ONE host read."""
    xh: Float32[torch.Tensor, "n 4"] = torch.cat([points3d, torch.ones_like(points3d[:, :1])], dim=-1)
    uvw: Float32[torch.Tensor, "c n 3"] = xh @ proj.transpose(1, 2)
    uv: Float32[torch.Tensor, "c n 2"] = uvw[..., :2] / (uvw[..., 2:3].abs() + 1e-6)
    conf: Bool[torch.Tensor, "c n"] = (torch.stack(vis) > vis_thresh) & valid
    residual: Float32[torch.Tensor, "c n 2"] = uv - pts2d if scale is None else (uv - pts2d) * scale
    err: Float32[torch.Tensor, "c n"] = torch.where(conf, residual.norm(dim=-1), torch.zeros_like(uv[..., 0]))
    count: Float32[torch.Tensor, "c"] = conf.sum(dim=-1).float()
    return torch.stack([count, err.sum(dim=-1) / count.clamp(min=1.0)]).tolist()
