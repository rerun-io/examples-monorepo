"""Confidence-weighted human-keypoint bundle adjustment (Kineo-style, VGGT-init).

This refines a *coarse* multi-camera rig — the kind our VGGT + MoGe-v2 pre-pass
recovers from uncalibrated synced video — using the dense 2D body landmarks
MammaNet already produces, weighted by their detector confidence. It follows the
Kineo recipe (Javerliat et al., 2025, arXiv:2510.24464; ``liris-xr/kineo``):

  minimize over {camera extrinsics, focal lengths, (optional distortion),
                 per-correspondence 3D points}
      sum_{cam c, sample s}  vis_{c,s} * Huber( ||proj_c(X_s) - x_{c,s}|| / sigma_{c,s} )

The two crucial departures from the released Kineo code, both motivated by the
project author's own analysis in ``liris-xr/kineo`` issue #7 ("the BA portion can
often fail catastrophically ... use a coarse init from VGGT then refine"):

  1. **VGGT initialization, not SfM/MST.** The cameras start from our metric,
     gravity-aligned VGGT+MoGe pre-pass instead of essential-matrix SfM, so the
     BA only has to *refine* a good rig rather than discover one from scratch.
  2. **Anchor-to-init regularization.** Kineo's released BA has no term keeping
     the cameras near their initialization (only an L2 on distortion coeffs), so
     a poorly-constrained sequence can drift arbitrarily. We add a soft penalty
     on rotation/translation/focal drift from the VGGT init — exactly the
     "keep some regularization to prevent a big drift from the original estimate"
     strategy the Kineo author recommends in that thread (and the same idea as
     "Look Ma, No Hands" penalizing deviation from a coarse prediction).

Gauge: camera 0 is locked (fixes the 6-DoF pose freedom); the global scale gauge
is pinned by camera 0's fixed metric translation plus the translation anchor, so
the refined rig stays in the metric, Z-up frame the pre-pass established.

The confidence model mirrors mamma's own :func:`mamma.fitting.losses.reprojection_loss`
exactly (log-variance -> pixel sigma, visibility clipped at ``vis_clip``) so the
calibration the fitter consumes is refined in the fitter's own error metric.

Pure-torch and dependency-light (no roma/kornia/pytorch3d/VGGT) so it unit-tests
in isolation; SO(3) exp and the Brown-Conrady forward model are implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from jaxtyping import Bool, Float32


def so3_exp(omega: Float32[torch.Tensor, "n 3"]) -> Float32[torch.Tensor, "n 3 3"]:
    """Batched SO(3) exponential map (Rodrigues' formula).

    Args:
        omega: Axis-angle rotation vectors ``[n, 3]`` (radians; magnitude = angle).

    Returns:
        Rotation matrices ``[n, 3, 3]``. ``omega = 0`` maps to the identity.
        Uses the unnormalized skew matrix with sinc-style coefficients so the
        gradient at ``omega = 0`` is the (nonzero) skew operator — normalizing
        the axis (``omega / ||omega||``) instead would give a zero gradient at 0
        (``||·||`` has zero subgradient there) and freeze the rotation at init.
    """
    theta2: Float32[torch.Tensor, "n 1"] = (omega * omega).sum(dim=-1, keepdim=True)
    theta: Float32[torch.Tensor, "n 1"] = torch.sqrt(theta2 + 1e-12)
    wx, wy, wz = omega[:, 0], omega[:, 1], omega[:, 2]
    zero: Float32[torch.Tensor, "n"] = torch.zeros_like(wx)
    k: Float32[torch.Tensor, "n 3 3"] = torch.stack(
        [zero, -wz, wy, wz, zero, -wx, -wy, wx, zero], dim=-1
    ).reshape(-1, 3, 3)  # skew(omega), unnormalized
    eye: Float32[torch.Tensor, "3 3"] = torch.eye(3, device=omega.device, dtype=omega.dtype)
    a: Float32[torch.Tensor, "n 1 1"] = (torch.sin(theta) / theta).unsqueeze(-1)
    b: Float32[torch.Tensor, "n 1 1"] = ((1.0 - torch.cos(theta)) / (theta2 + 1e-12)).unsqueeze(-1)
    return eye + a * k + b * (k @ k)


def project_points(
    pts3d_world: Float32[torch.Tensor, "s 3"],
    rot_world_to_cam: Float32[torch.Tensor, "3 3"],
    trans_world_to_cam: Float32[torch.Tensor, "3"],
    k_matrix: Float32[torch.Tensor, "3 3"],
    dist_coeffs: Float32[torch.Tensor, "5"] | None = None,
) -> tuple[Float32[torch.Tensor, "s 2"], Float32[torch.Tensor, "s"]]:
    """Project world points into one camera (pinhole + optional Brown-Conrady).

    Brown-Conrady coefficient order is ``[k1, k2, p1, p2, k3]`` (the simplecv /
    OpenCV convention shared by ``simplecv.camera_parameters.BrownConradyDistortion``).

    Returns:
        Pixel coordinates ``[s, 2]`` and camera-frame depth ``[s]`` (z, for a
        behind-camera mask).
    """
    pts_cam: Float32[torch.Tensor, "s 3"] = pts3d_world @ rot_world_to_cam.T + trans_world_to_cam
    depth: Float32[torch.Tensor, "s"] = pts_cam[:, 2]
    z: Float32[torch.Tensor, "s 1"] = depth.unsqueeze(-1).clamp(min=1e-6)
    xn: Float32[torch.Tensor, "s"] = pts_cam[:, 0] / z[:, 0]
    yn: Float32[torch.Tensor, "s"] = pts_cam[:, 1] / z[:, 0]

    if dist_coeffs is not None:
        k1, k2, p1, p2, k3 = (dist_coeffs[0], dist_coeffs[1], dist_coeffs[2], dist_coeffs[3], dist_coeffs[4])
        r2: Float32[torch.Tensor, "s"] = xn * xn + yn * yn
        radial: Float32[torch.Tensor, "s"] = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xn = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
        yn = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn

    fx: Float32[torch.Tensor, ""] = k_matrix[0, 0]
    fy: Float32[torch.Tensor, ""] = k_matrix[1, 1]
    cx: Float32[torch.Tensor, ""] = k_matrix[0, 2]
    cy: Float32[torch.Tensor, ""] = k_matrix[1, 2]
    u: Float32[torch.Tensor, "s"] = fx * xn + cx
    v: Float32[torch.Tensor, "s"] = fy * yn + cy
    return torch.stack([u, v], dim=-1), depth


@dataclass(frozen=True, slots=True)
class BundleAdjustConfig:
    """Operating point for :func:`bundle_adjust_cameras`."""

    optimize_focal: bool = True
    """Refine each camera's focal length (fx == fy)."""
    optimize_principal_point: bool = False
    """Refine the principal point (off: trust the centered VGGT/MoGe init)."""
    optimize_distortion: bool = False
    """Refine Brown-Conrady coefficients. Off by default — iPhone footage is
    near-pinhole and the GT calibration we compare against carries no distortion,
    so enabling it would bias the fairness of that comparison."""
    optimize_rotation: bool = True
    """Refine camera rotations (all but the locked camera 0)."""
    optimize_translation: bool = True
    """Refine camera translations (all but the locked camera 0)."""
    n_iters: int = 40
    """Iteration budget: the outer loop runs ``max(2, n_iters // 20 + 1)`` LBFGS
    ``.step()`` calls (each up to 20 inner iterations), early-stopping on the
    loss plateau."""
    lr: float = 1.0
    """LBFGS learning rate (strong-Wolfe line search scales the step)."""
    huber_delta: float = 3.0
    """Huber transition in *sigma units* on the sigma-normalized reprojection
    residual (≈ a 3-sigma inlier band)."""
    vis_clip: float = 0.2
    """Floor on per-landmark visibility weight (matches the fitter's
    ``vis_clip_value``: even self-occluded landmarks contribute at 0.2x)."""
    img_size_px: float = 512.0
    """Crop size scaling log-variance to pixel sigma (matches the fitter)."""
    clamp_sigma: tuple[float, float] = (1.0, 50.0)
    """Pixel-sigma clamp range (matches the fitter's reprojection loss)."""
    anchor_rot_weight: float = 50.0
    """Soft penalty on per-camera rotation drift (rad^2) from the VGGT init."""
    anchor_trans_weight: float = 50.0
    """Soft penalty on per-camera translation drift (m^2) from the VGGT init."""
    anchor_focal_weight: float = 10.0
    """Soft penalty on fractional focal-length drift from the VGGT/MoGe init."""
    distortion_reg_weight: float = 1.0
    """L2 shrinkage on distortion coefficients (only when ``optimize_distortion``)."""
    min_obs_views: int = 2
    """A correspondence must be visible in at least this many cameras to be kept."""


@dataclass(frozen=True, slots=True)
class BundleAdjustResult:
    """Refined rig + diagnostics from :func:`bundle_adjust_cameras`."""

    k_per_cam: Float32[torch.Tensor, "c 3 3"]
    """Refined intrinsics (same resolution as the inputs)."""
    world_to_cam_per_cam: Float32[torch.Tensor, "c 4 4"]
    """Refined world->camera extrinsics."""
    dist_per_cam: Float32[torch.Tensor, "c 5"]
    """Brown-Conrady coefficients ``[k1, k2, p1, p2, k3]`` (zeros if not optimized)."""
    points3d: Float32[torch.Tensor, "s 3"]
    """Final optimized 3D correspondences (world frame)."""
    rmse_px_before: float
    """Confidence-weighted reprojection RMSE (px) at the initialization."""
    rmse_px_after: float
    """Confidence-weighted reprojection RMSE (px) after refinement."""
    loss_history: list[float]
    """Per-iteration total loss."""
    n_samples: int = field(default=0)
    """Number of multi-view correspondences used."""


def _weighted_rmse_px(
    pts3d: Float32[torch.Tensor, "s 3"],
    kps2d: Float32[torch.Tensor, "c s 3"],
    weight: Float32[torch.Tensor, "c s"],
    rot: Float32[torch.Tensor, "c 3 3"],
    trans: Float32[torch.Tensor, "c 3"],
    k_per_cam: Float32[torch.Tensor, "c 3 3"],
    dist_per_cam: Float32[torch.Tensor, "c 5"] | None,
) -> float:
    """Confidence-weighted RMS pixel reprojection error (diagnostic, no grad)."""
    n_cams: int = kps2d.shape[0]
    sq_sum: Float32[torch.Tensor, ""] = pts3d.new_zeros(())
    w_sum: Float32[torch.Tensor, ""] = pts3d.new_zeros(())
    for c in range(n_cams):
        proj, depth = project_points(pts3d, rot[c], trans[c], k_per_cam[c], None if dist_per_cam is None else dist_per_cam[c])
        err2: Float32[torch.Tensor, "s"] = ((proj - kps2d[c, :, :2]) ** 2).sum(-1)
        w: Float32[torch.Tensor, "s"] = weight[c] * (depth > 1e-6).float()
        w = torch.nan_to_num(w * torch.isfinite(err2).float(), nan=0.0)
        err2 = torch.nan_to_num(err2, nan=0.0, posinf=0.0, neginf=0.0)
        sq_sum = sq_sum + (w * err2).sum()
        w_sum = w_sum + w.sum()
    return float(torch.sqrt(sq_sum / w_sum.clamp(min=1.0)).item())


def bundle_adjust_cameras(
    k_per_cam: Float32[torch.Tensor, "c 3 3"],
    world_to_cam_per_cam: Float32[torch.Tensor, "c 4 4"],
    kps2d_per_cam: Float32[torch.Tensor, "c s 3"],
    vis_per_cam: Float32[torch.Tensor, "c s"],
    points3d_init: Float32[torch.Tensor, "s 3"],
    config: BundleAdjustConfig | None = None,
) -> BundleAdjustResult:
    """Refine a coarse rig by confidence-weighted human-keypoint bundle adjustment.

    All pixel quantities (``k_per_cam``, ``kps2d_per_cam``) must be in the *same*
    resolution; the result's ``k_per_cam`` is in that resolution too. The 2D
    landmark third channel is a per-point log-variance (as MammaNet emits).

    Args:
        k_per_cam: Coarse intrinsics ``[c, 3, 3]`` (from the VGGT/MoGe pre-pass).
        world_to_cam_per_cam: Coarse extrinsics ``[c, 4, 4]`` (metric, Z-up).
        kps2d_per_cam: ``[c, s, 3]`` = ``[x_px, y_px, log_variance]`` per camera
            per correspondence ``s`` (a flattened (frame, landmark) selection).
        vis_per_cam: ``[c, s]`` per-landmark visibility in ``[0, 1]``.
        points3d_init: ``[s, 3]`` initial 3D points (e.g. weighted-DLT triangulation
            with the coarse rig).
        config: Optimization settings.

    Returns:
        A :class:`BundleAdjustResult` with refined cameras, 3D points, and the
        before/after weighted reprojection RMSE.
    """
    if config is None:
        config = BundleAdjustConfig()
    device: torch.device = k_per_cam.device
    dtype: torch.dtype = torch.float32
    n_cams: int = int(k_per_cam.shape[0])

    k_init: Float32[torch.Tensor, "c 3 3"] = k_per_cam.to(device, dtype).clone()
    w2c_init: Float32[torch.Tensor, "c 4 4"] = world_to_cam_per_cam.to(device, dtype).clone()
    rot_init: Float32[torch.Tensor, "c 3 3"] = w2c_init[:, :3, :3].contiguous()
    trans_init: Float32[torch.Tensor, "c 3"] = w2c_init[:, :3, 3].contiguous()
    f_init: Float32[torch.Tensor, "c"] = (k_init[:, 0, 0] + k_init[:, 1, 1]) * 0.5

    kps2d: Float32[torch.Tensor, "c s 3"] = kps2d_per_cam.to(device, dtype)
    vis: Float32[torch.Tensor, "c s"] = vis_per_cam.to(device, dtype)

    # Confidence weight per (cam, sample): visibility (clipped) * in-image. The
    # sigma (log-variance -> pixel) term is applied separately as a residual
    # normalizer (resid / sigma before the Huber, below), matching
    # mamma.fitting.losses.reprojection_loss.
    in_img: Float32[torch.Tensor, "c s"] = (kps2d[..., :2].abs().sum(-1) > 0).float()
    sigma: Float32[torch.Tensor, "c s"] = (torch.exp(0.5 * kps2d[..., 2]) / 2.0 * config.img_size_px).clamp(*config.clamp_sigma)
    weight: Float32[torch.Tensor, "c s"] = torch.clip(vis, min=config.vis_clip) * in_img

    # Drop correspondences seen by < min_obs_views cameras (no constraint otherwise).
    obs_mask: Bool[torch.Tensor, "c s"] = (vis > config.vis_clip) & (in_img > 0)
    keep: Bool[torch.Tensor, "s"] = obs_mask.sum(0) >= config.min_obs_views
    kps2d = kps2d[:, keep]
    weight = weight[:, keep]
    sigma = sigma[:, keep]
    pts3d_init: Float32[torch.Tensor, "s 3"] = points3d_init.to(device, dtype)[keep].clone()
    n_samples: int = int(pts3d_init.shape[0])

    # --- Free parameters --------------------------------------------------------
    # Rotation as a local SO(3) delta (init 0) about the VGGT rotation: keeps the
    # anchor penalty a clean ||delta||^2 and the parameterization minimal.
    requires_rot: bool = config.optimize_rotation and n_cams > 1
    requires_trans: bool = config.optimize_translation and n_cams > 1
    domega: torch.Tensor = torch.zeros(n_cams, 3, device=device, dtype=dtype, requires_grad=requires_rot)
    dtrans: torch.Tensor = torch.zeros(n_cams, 3, device=device, dtype=dtype, requires_grad=requires_trans)
    log_f_scale: torch.Tensor = torch.zeros(n_cams, device=device, dtype=dtype, requires_grad=config.optimize_focal)
    dpp: torch.Tensor = torch.zeros(n_cams, 2, device=device, dtype=dtype, requires_grad=config.optimize_principal_point)
    dist: torch.Tensor = torch.zeros(n_cams, 5, device=device, dtype=dtype, requires_grad=config.optimize_distortion)
    pts3d_opt: torch.Tensor = pts3d_init.clone().requires_grad_(True)

    # Camera 0 is the gauge anchor: zero out its extrinsic gradients each step.
    lock0_rot: bool = requires_rot
    lock0_trans: bool = requires_trans

    params: list[torch.Tensor] = [pts3d_opt]
    for p, on in ((domega, requires_rot), (dtrans, requires_trans), (log_f_scale, config.optimize_focal), (dpp, config.optimize_principal_point), (dist, config.optimize_distortion)):
        if on:
            params.append(p)

    def build_cameras() -> tuple[Float32[torch.Tensor, "c 3 3"], Float32[torch.Tensor, "c 3"], Float32[torch.Tensor, "c 3 3"], torch.Tensor | None]:
        rot: Float32[torch.Tensor, "c 3 3"] = so3_exp(domega) @ rot_init
        trans: Float32[torch.Tensor, "c 3"] = trans_init + dtrans
        f: Float32[torch.Tensor, "c"] = f_init * torch.exp(log_f_scale)
        k: Float32[torch.Tensor, "c 3 3"] = k_init.clone()
        k = k.clone()
        k[:, 0, 0] = f
        k[:, 1, 1] = f
        k[:, 0, 2] = k_init[:, 0, 2] + dpp[:, 0]
        k[:, 1, 2] = k_init[:, 1, 2] + dpp[:, 1]
        d: torch.Tensor | None = dist if config.optimize_distortion else None
        return rot, trans, k, d

    def reprojection_and_anchor() -> torch.Tensor:
        rot, trans, k, d = build_cameras()
        total: torch.Tensor = pts3d_opt.new_zeros(())
        w_total: torch.Tensor = pts3d_opt.new_zeros(())
        for c in range(n_cams):
            proj, depth = project_points(pts3d_opt, rot[c], trans[c], k[c], None if d is None else d[c])
            resid: Float32[torch.Tensor, "s"] = torch.linalg.norm(proj - kps2d[c, :, :2], dim=-1) / sigma[c]
            robust: Float32[torch.Tensor, "s"] = torch.nn.functional.huber_loss(
                resid, torch.zeros_like(resid), reduction="none", delta=config.huber_delta
            )
            w: Float32[torch.Tensor, "s"] = weight[c] * (depth > 1e-6).float()
            valid: Float32[torch.Tensor, "s"] = torch.isfinite(robust).float()
            robust = torch.nan_to_num(robust, nan=0.0, posinf=0.0, neginf=0.0)
            total = total + (w * valid * robust).sum()
            w_total = w_total + (w * valid).sum()
        loss: torch.Tensor = total / w_total.clamp(min=1.0)

        # Anchor-to-init regularization (the issue-#7 / Look-Ma-No-Hands term).
        if requires_rot:
            loss = loss + config.anchor_rot_weight * (domega**2).sum()
        if requires_trans:
            loss = loss + config.anchor_trans_weight * (dtrans**2).sum()
        if config.optimize_focal:
            loss = loss + config.anchor_focal_weight * (log_f_scale**2).sum()
        if config.optimize_distortion:
            loss = loss + config.distortion_reg_weight * (dist**2).mean()
        return loss

    rmse_before: float = _weighted_rmse_px(
        pts3d_init, kps2d, weight, rot_init, trans_init, k_init, None
    )

    loss_history: list[float] = []
    if params and n_samples > 0:
        # max_iter > 1 lets LBFGS accumulate curvature within a .step() call and
        # rescale the (very differently-scaled) point / rotation / translation /
        # focal gradients — a single steepest-descent step at lr=1 just overshoots
        # the large-magnitude translation gradient and the line search rejects it.
        optimizer = torch.optim.LBFGS(
            params, lr=config.lr, max_iter=20, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss: torch.Tensor = reprojection_and_anchor()
            loss.backward()
            if lock0_rot and domega.grad is not None:
                domega.grad[0].zero_()
            if lock0_trans and dtrans.grad is not None:
                dtrans.grad[0].zero_()
            return loss

        prev: float = float("inf")
        for _ in range(max(2, config.n_iters // 20 + 1)):
            loss_val: torch.Tensor = optimizer.step(closure)
            cur: float = float(loss_val.item())
            loss_history.append(cur)
            if not torch.isfinite(loss_val):
                break
            if abs(prev - cur) < 1e-10:
                break
            prev = cur

    with torch.no_grad():
        rot, trans, k, d = build_cameras()
        w2c_out: Float32[torch.Tensor, "c 4 4"] = w2c_init.clone()
        w2c_out[:, :3, :3] = rot
        w2c_out[:, :3, 3] = trans
        dist_out: Float32[torch.Tensor, "c 5"] = dist.detach().clone()
        rmse_after: float = _weighted_rmse_px(pts3d_opt.detach(), kps2d, weight, rot, trans, k, d)

    return BundleAdjustResult(
        k_per_cam=k.detach().clone(),
        world_to_cam_per_cam=w2c_out.detach().clone(),
        dist_per_cam=dist_out,
        points3d=pts3d_opt.detach().clone(),
        rmse_px_before=rmse_before,
        rmse_px_after=rmse_after,
        loss_history=loss_history,
        n_samples=n_samples,
    )
