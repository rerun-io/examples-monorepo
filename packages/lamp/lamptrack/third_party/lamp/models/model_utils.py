# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pure math helpers for the LAMP lifter model.

The functions here are stateless and keep tensors on the caller's device, which
makes them usable inside regular eager inference and CUDA Graph capture.
"""
# pyright: reportPrivateImportUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportConstantRedefinition=false, reportPossiblyUnboundVariable=false, reportMissingParameterType=false, reportUnknownParameterType=false

from __future__ import annotations

import torch
import torch.nn.functional as F

# Constants

# Gravity direction in the world frame. Stored as a Python tuple so the model
# can register a device-local buffer once.
GRAVITY_DIRECTION_VIO: tuple[float, float, float] = (0.0, 0.0, -1.0)

# Constant rotation that makes -z the gravity axis when aligning camera poses
# to the gravity-world frame. The model registers this as a buffer.
R_CG_CGZ: tuple[tuple[tuple[float, float, float], ...], ...] = (
    ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)),
)


# SE(3) helpers


def inverse_se3(T: torch.Tensor) -> torch.Tensor:
    """SE(3) inverse for a `(..., 4, 4)` rigid transform."""
    R = T[..., :3, :3]
    t = T[..., :3, 3:]
    R_inv = R.transpose(-1, -2)
    t_inv = -R_inv @ t
    T_inv = T.clone()
    T_inv[..., :3, :3] = R_inv
    T_inv[..., :3, 3:] = t_inv
    return T_inv


def se3_transform_points(T: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Apply rigid transform `T` to `points`."""
    return points @ T[..., :3, :3].transpose(-1, -2) + T[..., :3, 3].unsqueeze(-2)


def reject_vector_a_from_b(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return `a - proj_b(a)`, the component of `a` orthogonal to `b`."""
    b_norm = torch.sqrt((b**2).sum(-1, keepdim=True))
    b_unit = b / b_norm
    a_proj = b_unit * (a * b_unit).sum(-1, keepdim=True)
    return a - a_proj


# Rotation representations


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Zhou et al. (2019) 6D rotation to 3x3 matrix via Gram-Schmidt."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


# Gravity-aligned canonical frame


def gravity_align_T_world_cam(
    T_world_cam: torch.Tensor,
    gravity_w: torch.Tensor,
    R_cg_cgz: torch.Tensor,
) -> torch.Tensor:
    """Build a gravity-aligned canonical local frame at `T_world_cam`."""
    assert T_world_cam.dim() > 1, f"expected dim > 1, got shape {T_world_cam.shape}"
    dim = T_world_cam.dim()
    R_wc = T_world_cam[..., :3, :3]

    # Broadcast gravity_w over the batch dims of R_wc.
    dir_shape = [1] * (dim - 2) + [3]
    g_w = gravity_w.view(dir_shape).to(dtype=R_wc.dtype)  # dtype cast only
    g_w = g_w.expand_as(R_wc[..., 1])

    # Forward vector (z) orthogonal to gravity.
    d3 = reject_vector_a_from_b(a=R_wc[..., 2], b=g_w)
    # Tiny offset so cross of two parallel vectors below doesn't degenerate
    # to zero. Build directly on the right device+dtype.
    d3_offset = torch.zeros_like(d3)
    d3_offset[..., 1] = 1e-3
    d3_is_zeros = (d3 == 0.0).all(dim=-1, keepdim=True).expand_as(d3)
    d3 = torch.where(d3_is_zeros, d3 + d3_offset, d3)

    d2 = torch.linalg.cross(d3, g_w, dim=-1)
    R_wcg = torch.cat([g_w.unsqueeze(-1), d2.unsqueeze(-1), d3.unsqueeze(-1)], -1)
    R_world_cg = F.normalize(R_wcg, p=2.0, dim=-2)

    T_out = T_world_cam.clone()
    T_out[..., :3, :3] = R_world_cg @ R_cg_cgz.transpose(-1, -2)
    return T_out


def get_T_x_c(
    Ts_wc: torch.Tensor,
    gravity_w: torch.Tensor,
    R_cg_cgz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a snippet-local canonical frame using the LAST camera pose."""
    # Use the last camera pose as the local-frame origin.
    T_w_x_anchor = Ts_wc[:, -1, ...]  # (B, 4, 4)
    T_w_x = gravity_align_T_world_cam(
        T_w_x_anchor, gravity_w=gravity_w, R_cg_cgz=R_cg_cgz
    )
    T_w_x = T_w_x.unsqueeze(1)  # (B, 1, 4, 4)
    T_x_w = inverse_se3(T_w_x)
    T_x_c = T_x_w @ Ts_wc
    return T_x_c, T_w_x


# Camera unprojection (2D pixel → 3D ray)


def pinhole_unproject(uv: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Pinhole 2D→3D ray with z=1."""
    assert uv.ndim in (3, 4), f"uv must be (B,N,2) or (B,T,N,2), got {uv.shape}"
    assert params.ndim in (2, 3), f"params must be (B,4) or (B,T,4), got {params.shape}"
    assert params.shape[-1] == 4

    has_t = uv.ndim == 4
    if has_t:
        T_, N = uv.shape[1], uv.shape[2]
        uv = uv.reshape(-1, N, 2)
    params = params.reshape(-1, params.shape[-1])

    B, N = uv.shape[0], uv.shape[1]
    fx_fy = params[:, 0:2].reshape(B, 1, 2)
    cx_cy = params[:, 2:4].reshape(B, 1, 2)
    uv_dist = (uv - cx_cy) / fx_fy
    ray = torch.cat([uv_dist, uv.new_ones(B, N, 1)], dim=2)

    if has_t:
        ray = ray.reshape(B // T_, T_, N, 3)
    return ray


def _sign_plus(x: torch.Tensor) -> torch.Tensor:
    """Returns +1 for x >= 0, -1 for x < 0. Capture-safe (no .item)."""
    return 2.0 * (x >= 0.0).to(x.dtype) - 1.0


def fisheye624_unproject(
    uv: torch.Tensor,
    params: torch.Tensor,
    max_iters: int = 5,
) -> torch.Tensor:
    """FisheyeRadTanThinPrism 2D -> 3D ray via Newton inverse."""
    assert uv.ndim == 3 or uv.ndim == 4, "Expected batched input shaped Bx(T)xNx2"
    assert uv.shape[-1] == 2
    assert params.ndim == 2 or params.ndim == 3
    assert params.shape[-1] == 16 or params.shape[-1] == 15
    eps = 1e-6

    T_ = -1
    if uv.ndim == 4:
        T_, N = uv.shape[1], uv.shape[2]
        uv = uv.reshape(-1, N, 2)
        params = params.reshape(-1, params.shape[-1])

    B, N = uv.shape[0], uv.shape[1]

    if params.shape[-1] == 15:
        fx_fy = params[:, 0].reshape(B, 1, 1)
        cx_cy = params[:, 1:3].reshape(B, 1, 2)
    else:
        fx_fy = params[:, 0:2].reshape(B, 1, 2)
        cx_cy = params[:, 2:4].reshape(B, 1, 2)

    uv_dist = (uv - cx_cy) / fx_fy

    xr_yr = uv_dist.clone()
    for _ in range(max_iters):
        uv_dist_est = xr_yr.clone()
        p0 = params[:, -6].reshape(B, 1)
        p1 = params[:, -5].reshape(B, 1)
        xr = xr_yr[:, :, 0].reshape(B, N)
        yr = xr_yr[:, :, 1].reshape(B, N)
        xr_yr_sq = torch.square(xr_yr)
        xr_sq = xr_yr_sq[:, :, 0].reshape(B, N)
        yr_sq = xr_yr_sq[:, :, 1].reshape(B, N)
        rd_sq = xr_sq + yr_sq
        uv_dist_est[:, :, 0] = uv_dist_est[:, :, 0] + (
            (2.0 * xr_sq + rd_sq) * p0 + 2.0 * xr * yr * p1
        )
        uv_dist_est[:, :, 1] = uv_dist_est[:, :, 1] + (
            (2.0 * yr_sq + rd_sq) * p1 + 2.0 * xr * yr * p0
        )
        # Thin prism.
        s0 = params[:, -4].reshape(B, 1)
        s1 = params[:, -3].reshape(B, 1)
        s2 = params[:, -2].reshape(B, 1)
        s3 = params[:, -1].reshape(B, 1)
        rd_4 = torch.square(rd_sq)
        uv_dist_est[:, :, 0] = uv_dist_est[:, :, 0] + (s0 * rd_sq + s1 * rd_4)
        uv_dist_est[:, :, 1] = uv_dist_est[:, :, 1] + (s2 * rd_sq + s3 * rd_4)
        # 2x2 Jacobian (manual entries — torch.inverse is ~10x slower).
        duv = uv.new_ones(B, N, 2, 2)
        duv[:, :, 0, 0] = 1.0 + 6.0 * xr_yr[:, :, 0] * p0 + 2.0 * xr_yr[:, :, 1] * p1
        offdiag = 2.0 * (xr_yr[:, :, 0] * p1 + xr_yr[:, :, 1] * p0)
        duv[:, :, 0, 1] = offdiag
        duv[:, :, 1, 0] = offdiag
        duv[:, :, 1, 1] = 1.0 + 6.0 * xr_yr[:, :, 1] * p1 + 2.0 * xr_yr[:, :, 0] * p0
        xr_yr_sq_norm = xr_yr_sq[:, :, 0] + xr_yr_sq[:, :, 1]
        temp1 = 2.0 * (s0 + 2.0 * s1 * xr_yr_sq_norm)
        duv[:, :, 0, 0] = duv[:, :, 0, 0] + xr_yr[:, :, 0] * temp1
        duv[:, :, 0, 1] = duv[:, :, 0, 1] + xr_yr[:, :, 1] * temp1
        temp2 = 2.0 * (s2 + 2.0 * s3 * xr_yr_sq_norm)
        duv[:, :, 1, 0] = duv[:, :, 1, 0] + xr_yr[:, :, 0] * temp2
        duv[:, :, 1, 1] = duv[:, :, 1, 1] + xr_yr[:, :, 1] * temp2
        # Manual 2x2 inverse + matmul.
        mat = duv.reshape(-1, 2, 2)
        a = mat[:, 0, 0].reshape(-1, 1, 1)
        b = mat[:, 0, 1].reshape(-1, 1, 1)
        c = mat[:, 1, 0].reshape(-1, 1, 1)
        d = mat[:, 1, 1].reshape(-1, 1, 1)
        det = 1.0 / ((a * d) - (b * c))
        top = torch.cat([d, -b], dim=2)
        bot = torch.cat([-c, a], dim=2)
        inv = (det * torch.cat([top, bot], dim=1)).reshape(B, N, 2, 2)
        diff = uv_dist - uv_dist_est
        a = inv[:, :, 0, 0]
        b = inv[:, :, 0, 1]
        c = inv[:, :, 1, 0]
        d = inv[:, :, 1, 1]
        e = diff[:, :, 0]
        f = diff[:, :, 1]
        step = torch.stack([a * e + b * f, c * e + d * f], dim=-1)
        xr_yr = xr_yr + step

    xr_yr_norm = xr_yr.norm(p=2, dim=2).reshape(B, N, 1)
    th = xr_yr_norm.clone()
    for _ in range(max_iters):
        th_radial = uv.new_ones(B, N, 1)
        dthd_th = uv.new_ones(B, N, 1)
        for k in range(6):
            r_k = params[:, -12 + k].reshape(B, 1, 1)
            th_radial = th_radial + (r_k * torch.pow(th, 2 + k * 2))
            dthd_th = dthd_th + ((3.0 + 2.0 * k) * r_k * torch.pow(th, 2 + k * 2))
        th_radial = th_radial * th
        step = (xr_yr_norm - th_radial) / dthd_th
        step = torch.where(dthd_th.abs() > eps, step, _sign_plus(step) * eps * 10.0)
        th = th + step

    close_to_zero = torch.logical_and(th.abs() < eps, xr_yr_norm.abs() < eps)
    ray_dir = torch.where(close_to_zero, xr_yr, torch.tan(th) / xr_yr_norm * xr_yr)
    ray = torch.cat([ray_dir, uv.new_ones(B, N, 1)], dim=2)

    if T_ > 0:
        ray = ray.reshape(B // T_, T_, N, 3)
    return ray


# Camera ray → Plücker (used by the lifter's input embedding)


def get_cam_ray(
    cam_params: torch.Tensor,
    kp_2ds: torch.Tensor,
    T_x_c: torch.Tensor | None = None,
) -> torch.Tensor:
    """2D keypoint → Plücker ray (direction, normal, score) in local frame."""
    is_pinhole = cam_params.shape[-1] == 4
    if is_pinhole:
        ray_dirs = pinhole_unproject(kp_2ds[..., :2], cam_params)
    else:
        ray_dirs = fisheye624_unproject(kp_2ds[..., :2], cam_params)

    ray_dirs = F.normalize(ray_dirs, p=2.0, dim=-1, eps=1e-6)
    # `zeros_like` already lives on `ray_dirs`'s device; no `.to()` needed.
    origins = torch.zeros_like(ray_dirs)
    if T_x_c is not None:
        origins = se3_transform_points(T_x_c, origins)
        ray_dirs = ray_dirs @ T_x_c[..., :3, :3].transpose(-1, -2)
        ray_dirs = F.normalize(ray_dirs, p=2.0, dim=-1, eps=1e-6)

    plucker_normal = torch.cross(origins, ray_dirs, dim=-1)
    rays = torch.cat([ray_dirs, plucker_normal], dim=-1)
    scores = kp_2ds[..., -1]
    # Zero out rays with no detection.
    rays = torch.where(scores.unsqueeze(-1) < 1e-5, torch.zeros_like(rays), rays)
    return torch.cat([rays, scores.unsqueeze(-1)], dim=-1)


# SMPL output helpers


def transform_smpl_params(
    root_orient: torch.Tensor,
    transl: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    smpl_t_pose_pelvis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a rigid transform (R, t) to the SMPL root pose, fixing the T-pose pelvis offset."""
    smpl_t_pose_pelvis = smpl_t_pose_pelvis.unsqueeze(-2).unsqueeze(-1)  # (B, 1, 3, 1)
    transl = transl.unsqueeze(-1).float()  # (B, T, 3, 1)
    t = t.unsqueeze(-1)  # (B, T, 3, 1)
    transl = R @ (smpl_t_pose_pelvis + transl) + t - smpl_t_pose_pelvis
    root_orient = R @ root_orient
    return root_orient, transl.squeeze(-1)


def matrix_to_axis_angle(rotmat: torch.Tensor) -> torch.Tensor:
    """Rotation matrix → axis-angle (Rodrigues) representation."""
    if rotmat.shape[-2:] != (3, 3):
        raise ValueError(f"rotmat must end in shape (3, 3); got {rotmat.shape}")

    m00 = rotmat[..., 0, 0]
    m01 = rotmat[..., 0, 1]
    m02 = rotmat[..., 0, 2]
    m10 = rotmat[..., 1, 0]
    m11 = rotmat[..., 1, 1]
    m12 = rotmat[..., 1, 2]
    m20 = rotmat[..., 2, 0]
    m21 = rotmat[..., 2, 1]
    m22 = rotmat[..., 2, 2]

    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                dim=-1,
            ),
            min=0.0,
        )
    )

    # Four quaternion candidates in wxyz order, each using a different largest
    # component for numerical stability. Select the best-conditioned candidate
    # with tensor ops only, so this remains CUDA-graph friendly.
    quat_by_wxyz = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m01 + m10, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m01 + m10, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m02 + m20, m12 + m21, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    quat_candidates = quat_by_wxyz / (2.0 * q_abs.clamp(min=0.1).unsqueeze(-1))
    best = torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4).to(
        dtype=rotmat.dtype
    )
    quat = (quat_candidates * best.unsqueeze(-1)).sum(dim=-2)
    quat = F.normalize(quat, p=2.0, dim=-1, eps=1e-12)
    quat = torch.where(quat[..., :1] < 0.0, -quat, quat)

    quat_xyz = quat[..., 1:]
    sin_half = torch.linalg.norm(quat_xyz, dim=-1, keepdim=True)
    half_angle = torch.atan2(sin_half, quat[..., :1])
    angle = 2.0 * half_angle
    small = angle.abs() < 1e-6
    safe_angle = torch.where(small, torch.ones_like(angle), angle)
    sin_half_over_angle = torch.where(
        small,
        0.5 - (angle * angle) / 48.0,
        torch.sin(half_angle) / safe_angle,
    )
    return quat_xyz / sin_half_over_angle


def smpl_forward_joints_lamp_outputs(
    smpl_model,  # smplx.SMPL — duck-typed so import isn't required at module load
    betas: torch.Tensor,
    body_pose_rotmat: torch.Tensor,
    global_orient_rotmat: torch.Tensor,
    transl: torch.Tensor,
    return_verts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Wrap `smplx.SMPL.forward` for rotation-matrix (B, T) inputs; returns LAMP joints/verts."""
    B, T = body_pose_rotmat.shape[:2]
    body_pose_aa = matrix_to_axis_angle(body_pose_rotmat).reshape(B * T, -1)
    global_orient_aa = matrix_to_axis_angle(global_orient_rotmat).reshape(B * T, -1)
    betas_bt = betas.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
    transl_bt = transl.reshape(B * T, -1)

    smpl_out = smpl_model.forward(
        betas=betas_bt,
        body_pose=body_pose_aa,
        global_orient=global_orient_aa,
        transl=transl_bt,
        return_verts=return_verts,
    )
    smpl_joints = smpl_out.joints[:, :24].reshape(B, T, 24, 3)
    smpl_verts: torch.Tensor | None = None
    if return_verts:
        smpl_verts = smpl_out.vertices.reshape(B, T, 6890, 3)
    return smpl_joints, smpl_verts
