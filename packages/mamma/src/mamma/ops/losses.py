"""Fitting losses — ports of the subset of ``optimization/losses/losses.py``
used by the streaming sliding-window fitter (geman-mcclure reprojection,
3D anchor MSE, shape prior, temporal smoothness).
"""

from __future__ import annotations

import torch
from jaxtyping import Float32


def geman_mcclure(
    target: Float32[torch.Tensor, "... d"],
    pred: Float32[torch.Tensor, "... d"],
    delta: float,
    weight: Float32[torch.Tensor, "... 1"] | float = 1.0,
    uncertainties: Float32[torch.Tensor, "... 1"] | float = 1.0,
) -> Float32[torch.Tensor, ""]:
    """Robust Geman-McClure penalty (original ``geman_mcclure_loss``)."""
    x_squared = ((target - pred) / uncertainties) ** 2
    loss = weight * x_squared / (delta**2 + x_squared)
    return loss.mean()


def reprojection_loss(
    pts2d_per_cam: list[Float32[torch.Tensor, "t n 3"]],
    pts3d_world: Float32[torch.Tensor, "t n 3"],
    k_per_cam: list[Float32[torch.Tensor, "3 3"]],
    world_to_cam_per_cam: list[Float32[torch.Tensor, "4 4"]],
    vis_per_cam: list[Float32[torch.Tensor, "t n"]] | None,
    vis_clip_value: float = 0.8,
    img_size_px: float = 512.0,
    weight: float = 1.0,
    delta_px: float = 8.0,
) -> Float32[torch.Tensor, ""]:
    """Geman-McClure 2D reprojection over all cameras (original ``proj_pts_loss``).

    The landmark third channel is a log-variance; sigma in crop pixels is
    ``exp(0.5*logvar)/2*img_size`` clamped to [1, 50] (the original stored
    sigma directly; we convert from logvar here — same operating range).
    """
    total: Float32[torch.Tensor, ""] = pts3d_world.new_zeros(())
    n_cams: int = len(pts2d_per_cam)
    for cam_id in range(n_cams):
        w2c: Float32[torch.Tensor, "4 4"] = world_to_cam_per_cam[cam_id]
        pts_cam: Float32[torch.Tensor, "t n 3"] = pts3d_world @ w2c[:3, :3].T + w2c[:3, 3]
        proj: Float32[torch.Tensor, "t n 3"] = pts_cam @ k_per_cam[cam_id].T
        proj = proj / proj[..., 2:3].clamp(min=1e-6)
        pts2d: Float32[torch.Tensor, "t n 3"] = pts2d_per_cam[cam_id]
        in_img: Float32[torch.Tensor, "t n 1"] = (pts2d[..., :2].sum(-1, keepdim=True) > 0).float()
        sigma: Float32[torch.Tensor, "t n 1"] = (torch.exp(0.5 * pts2d[..., 2:3]) / 2.0 * img_size_px).clamp(1.0, 50.0)
        point_weight: Float32[torch.Tensor, "t n 1"] = weight * in_img
        if vis_per_cam is not None:
            point_weight = torch.clip(vis_per_cam[cam_id][..., None], min=vis_clip_value) * point_weight
        error: Float32[torch.Tensor, ""] = geman_mcclure(
            proj[..., :2], pts2d[..., :2], delta=delta_px, weight=point_weight, uncertainties=sigma
        )
        # Sync-free NaN/Inf guard: `if torch.isnan(error)` forces a host sync
        # per camera per iteration (and invalidates CUDA-graph capture).
        error = torch.nan_to_num(error, nan=0.0, posinf=0.0, neginf=0.0)
        total = total + error / n_cams
    return total


def anchor3d_loss(
    pts3d_pred: Float32[torch.Tensor, "t n 3"],
    pts3d_anchor: Float32[torch.Tensor, "t n 3"],
    valid: Float32[torch.Tensor, "t n"],
    weight: float = 1.0,
) -> Float32[torch.Tensor, ""]:
    """Saturating MSE to triangulated anchors over valid points.

    For small residuals this is exactly the original ``l2_loss_3d_points`` MSE
    (delta**2 * x/(delta**2 + x) ~= x for x << delta**2); a mis-triangulated
    anchor's influence saturates at delta**2 instead of growing unboundedly —
    self-occluded windows (bend-overs) produce valid-but-displaced anchors that
    plain MSE at this weight would force the fit to chase.
    """
    delta_m: float = 0.1
    diff: Float32[torch.Tensor, "t n"] = ((pts3d_pred - pts3d_anchor) ** 2).sum(-1)
    # eigh on a degenerate-but-valid system (>=2 near-collinear cameras) can
    # emit NaN anchors; NaN * 0-mask is still NaN, so scrub before masking
    # (same guard as reprojection_loss).
    diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
    robust: Float32[torch.Tensor, "t n"] = delta_m**2 * diff / (delta_m**2 + diff)
    denom: Float32[torch.Tensor, ""] = valid.sum().clamp(min=1.0)
    return weight * (robust * valid).sum() / denom


def shape_prior_loss(betas: Float32[torch.Tensor, "1 nb"], weight: float = 1.0) -> Float32[torch.Tensor, ""]:
    """L2 shrinkage on shape coefficients (original ``body_shape_prior_loss``)."""
    return weight * torch.mean(betas**2)


def temporal_smoothness_loss(
    trans: Float32[torch.Tensor, "t 3"],
    pose_aa: Float32[torch.Tensor, "t p"],
    weight_trans: float = 1.0,
    weight_pose: float = 1.0,
) -> Float32[torch.Tensor, ""]:
    """First-difference smoothness over the window (aa variant of the originals)."""
    if trans.shape[0] < 2:
        return trans.new_zeros(())
    trans_term: Float32[torch.Tensor, ""] = ((trans[1:] - trans[:-1]) ** 2).sum(-1).mean()
    pose_term: Float32[torch.Tensor, ""] = ((pose_aa[1:] - pose_aa[:-1]) ** 2).mean()
    return weight_trans * trans_term + weight_pose * pose_term


def floor_penetration_loss(
    verts: Float32[torch.Tensor, "t n 3"],
    weight: float = 1.0,
) -> Float32[torch.Tensor, ""]:
    """One-sided penalty on vertices below the calibrated ground plane (z=0).

    The original ma_3d carries full floor-contact machinery; this is the safe
    subset for a causal fitter — penetration is always wrong (jump landings
    measured 60-70mm underground without it), while airborne frames are
    untouched because the penalty is one-sided.
    """
    below: Float32[torch.Tensor, "t n"] = torch.relu(-verts[..., 2])
    return weight * (below**2).mean()


def acceleration_loss(
    pts3d: Float32[torch.Tensor, "t n 3"],
    weight: float = 1.0,
    dt: float = 1.0 / 30.0,
) -> Float32[torch.Tensor, ""]:
    """Second-difference (acceleration) penalty on 3D point trajectories.

    Port of the original ``acceleration_loss`` (applied to the sampled SMPL-X
    points): constant-velocity motion costs nothing, so unlike first-difference
    smoothness this bridges bad-observation windows without lagging runs or
    jumps.
    """
    if pts3d.shape[0] < 3:
        return pts3d.new_zeros(())
    vel: Float32[torch.Tensor, "tv n 3"] = (pts3d[1:] - pts3d[:-1]) / dt
    acc: Float32[torch.Tensor, "ta n 3"] = (vel[1:] - vel[:-1]) / dt
    return weight * acc.pow(2).mean()


def floor_contact_loss(
    pts3d: Float32[torch.Tensor, "t n 3"],
    floor_contact: Float32[torch.Tensor, "t n"],
    weight: float = 1.0,
) -> Float32[torch.Tensor, ""]:
    """Pull landmarks MammaNet predicts in floor contact toward world z=0.

    Probability-weighted z**2 (probabilities are camera-fused and zeroed below
    0.25 upstream). The original DAG carries these same per-landmark
    predictions; using them is what keeps a bent-over fit planted instead of
    floating on biased landmarks.
    """
    denom: Float32[torch.Tensor, ""] = floor_contact.sum().clamp(min=1.0)
    return weight * (floor_contact * pts3d[..., 2].pow(2)).sum() / denom
