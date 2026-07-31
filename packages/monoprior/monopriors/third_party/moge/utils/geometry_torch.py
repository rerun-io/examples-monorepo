from typing import *

import torch
import torch.nn.functional as F

from .geometry_numpy import solve_optimal_focal_shift, solve_optimal_shift


def normalized_view_plane_uv(width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    "UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height
    
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv


def recover_focal_shift(points: torch.Tensor, mask: torch.Tensor = None, focal: torch.Tensor = None, downsample_size: Tuple[int, int] = (64, 64)):
    """
    Recover the depth map and FoV from a point map with unknown z shift and focal.

    Note that it assumes:
    - the optical center is at the center of the map
    - the map is undistorted
    - the map is isometric in the x and y directions

    ### Parameters:
    - `points: torch.Tensor` of shape (..., H, W, 3)
    - `downsample_size: Tuple[int, int]` in (height, width), the size of the downsampled map. Downsampling produces approximate solution and is efficient for large maps.

    ### Returns:
    - `focal`: torch.Tensor of shape (...) the estimated focal length, relative to the half diagonal of the map
    - `shift`: torch.Tensor of shape (...) Z-axis shift to translate the point map to camera space
    """
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]
    diagonal = (height ** 2 + width ** 2) ** 0.5

    points = points.reshape(-1, *shape[-3:])
    mask = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    focal = focal.reshape(-1) if focal is not None else None
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)  # (H, W, 2)

    points_lr = F.interpolate(points.permute(0, 3, 1, 2), downsample_size, mode='nearest').permute(0, 2, 3, 1)
    uv_lr = F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode='nearest').squeeze(0).permute(1, 2, 0)
    mask_lr = None if mask is None else F.interpolate(mask.to(torch.float32).unsqueeze(1), downsample_size, mode='nearest').squeeze(1) > 0
    
    uv_lr_np = uv_lr.cpu().numpy()
    points_lr_np = points_lr.detach().cpu().numpy()
    focal_np = focal.cpu().numpy() if focal is not None else None
    mask_lr_np = None if mask is None else mask_lr.cpu().numpy()
    optim_shift, optim_focal = [], []
    for i in range(points.shape[0]):
        points_lr_i_np = points_lr_np[i] if mask is None else points_lr_np[i][mask_lr_np[i]]
        uv_lr_i_np = uv_lr_np if mask is None else uv_lr_np[mask_lr_np[i]]
        if uv_lr_i_np.shape[0] < 2:
            optim_focal.append(1)
            optim_shift.append(0)
            continue
        if focal is None:
            optim_shift_i, optim_focal_i = solve_optimal_focal_shift(uv_lr_i_np, points_lr_i_np)
            optim_focal.append(float(optim_focal_i))
        else:
            optim_shift_i = solve_optimal_shift(uv_lr_i_np, points_lr_i_np, focal_np[i])
        optim_shift.append(float(optim_shift_i))
    optim_shift = torch.tensor(optim_shift, device=points.device, dtype=points.dtype).reshape(shape[:-3])

    if focal is None:
        optim_focal = torch.tensor(optim_focal, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    else:
        optim_focal = focal.reshape(shape[:-3])

    return optim_focal, optim_shift


def intrinsics_from_focal_center(
    fx: torch.Tensor,
    fy: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
) -> torch.Tensor:
    """Build OpenCV intrinsics from broadcast-compatible tensor inputs."""
    fx, fy, cx, cy = torch.broadcast_tensors(fx, fy, cx, cy)
    zeros, ones = torch.zeros_like(fx), torch.ones_like(fx)
    return torch.stack(
        [
            fx,
            zeros,
            cx,
            zeros,
            fy,
            cy,
            zeros,
            zeros,
            ones,
        ],
        dim=-1,
    ).unflatten(-1, (3, 3))


def _uv_map(height: int, width: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    u = torch.linspace(0.5 / width, 1.0 - 0.5 / width, width, dtype=dtype, device=device)
    v = torch.linspace(0.5 / height, 1.0 - 0.5 / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing="xy")
    return torch.stack([u, v], dim=2)


def _unproject_cv(uv: torch.Tensor, depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    intrinsics = torch.cat(
        [
            torch.cat(
                [
                    intrinsics,
                    torch.zeros((*intrinsics.shape[:-2], 3, 1), dtype=intrinsics.dtype, device=intrinsics.device),
                ],
                dim=-1,
            ),
            torch.tensor([[0, 0, 0, 1]], dtype=intrinsics.dtype, device=intrinsics.device).expand(*intrinsics.shape[:-2], 1, 4),
        ],
        dim=-2,
    )
    transform = intrinsics
    points = torch.cat([uv, torch.ones((*uv.shape[:-1], 1), dtype=uv.dtype, device=uv.device)], dim=-1) * depth[..., None]
    points = torch.cat([points, torch.ones((*points.shape[:-1], 1), dtype=uv.dtype, device=uv.device)], dim=-1)
    points = points @ torch.linalg.inv(transform).mT
    points = points[..., :3]
    return points


def depth_map_to_point_map(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Unproject a normalized-intrinsics depth map into camera-space points."""
    height, width = depth.shape[-2:]
    uv = _uv_map(height, width, dtype=depth.dtype, device=depth.device)
    points = _unproject_cv(uv, depth, intrinsics=intrinsics[..., None, :, :])
    return points
