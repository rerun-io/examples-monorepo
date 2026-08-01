from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from scipy.optimize import OptimizeResult, least_squares
from torch import Tensor


def _solve_focal_shift(
    uv: Float[np.ndarray, "... 2"],
    xyz: Float[np.ndarray, "... 3"],
    focal: float | None = None,
) -> tuple[float, float]:
    """Solve ``min |focal * xy / (z + shift) - uv|`` for shift, and for focal when unknown.

    Args:
        uv: Float view-plane coordinates shaped ``... 2``.
        xyz: Float affine point coordinates shaped ``... 3``.
        focal: Known focal relative to half the image diagonal, or ``None`` to solve for it.

    Returns:
        Optimal shift and focal as floats.
    """
    uv_n2: Float[np.ndarray, "n 2"] = uv.reshape(-1, 2)
    xy_n2: Float[np.ndarray, "n 2"] = xyz[..., :2].reshape(-1, 2)
    z_n: Float[np.ndarray, "n"] = xyz[..., 2].reshape(-1)

    def closed_form_focal(xy_projected_n2: Float[np.ndarray, "n 2"]) -> np.floating[np.generic]:
        return (xy_projected_n2 * uv_n2).sum() / np.square(xy_projected_n2).sum()

    def residual(shift_1: Float[np.ndarray, "1"]) -> Float[np.ndarray, "errors"]:
        xy_projected_n2: Float[np.ndarray, "n 2"] = xy_n2 / (z_n + shift_1)[:, None]
        current_focal: float | np.floating[np.generic] = closed_form_focal(xy_projected_n2) if focal is None else focal
        error: Float[np.ndarray, "errors"] = (current_focal * xy_projected_n2 - uv_n2).ravel()
        return error

    solution: OptimizeResult = least_squares(residual, x0=0.0, ftol=1e-3, method="lm")
    optimal_shift: Float[np.ndarray, ""] = solution.x.squeeze().astype(np.float32)
    optimal_focal: float | np.floating[np.generic] = (
        closed_form_focal(xy_n2 / (z_n + optimal_shift)[:, None]) if focal is None else focal
    )
    return float(optimal_shift), float(optimal_focal)


def normalized_view_plane_uv(
    width: int | Int[Tensor, ""],
    height: int | Int[Tensor, ""],
    aspect_ratio: float | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Float[Tensor, "h w 2"]:
    """Build centered view-plane coordinates normalized by image diagonal.

    Args:
        width: Image width as an integer or scalar integer tensor.
        height: Image height as an integer or scalar integer tensor.
        aspect_ratio: Image width divided by height, or ``None`` to derive it.
        dtype: Output floating-point dtype.
        device: Output Torch device.

    Returns:
        Float UV coordinate tensor shaped ``h w 2``.
    """
    resolved_aspect_ratio: float | Float[Tensor, ""] = width / height if aspect_ratio is None else aspect_ratio
    span_x: float | Float[Tensor, ""] = resolved_aspect_ratio / (1 + resolved_aspect_ratio**2) ** 0.5
    span_y: float | Float[Tensor, ""] = 1 / (1 + resolved_aspect_ratio**2) ** 0.5
    u_w: Float[Tensor, "w"] = torch.linspace(
        -span_x * (width - 1) / width,
        span_x * (width - 1) / width,
        width,
        dtype=dtype,
        device=device,
    )
    v_h: Float[Tensor, "h"] = torch.linspace(
        -span_y * (height - 1) / height,
        span_y * (height - 1) / height,
        height,
        dtype=dtype,
        device=device,
    )
    uv_grid: tuple[Float[Tensor, "h w"], Float[Tensor, "h w"]] = torch.meshgrid(u_w, v_h, indexing="xy")
    u_hw: Float[Tensor, "h w"] = uv_grid[0]
    v_hw: Float[Tensor, "h w"] = uv_grid[1]
    uv_hw2: Float[Tensor, "h w 2"] = torch.stack([u_hw, v_hw], dim=-1)
    return uv_hw2


def recover_focal_shift(
    points: Float[Tensor, "*batch h w 3"],
    mask: Bool[Tensor, "*batch h w"] | None = None,
    focal: Float[Tensor, "*batch"] | None = None,
    downsample_size: tuple[int, int] = (64, 64),
) -> tuple[Float[Tensor, "*batch"], Float[Tensor, "*batch"]]:
    """Recover focal length and Z shift from an affine point map.

    The optical center is assumed to be centered, the map undistorted, and X/Y projection isometric.

    Args:
        points: Float point tensor shaped ``*batch h w 3``.
        mask: Optional bool validity tensor shaped ``*batch h w``.
        focal: Optional known float focal tensor shaped ``*batch`` relative to half the image diagonal.
        downsample_size: ``(height, width)`` used by the CPU least-squares solve.

    Returns:
        Float focal and shift tensors, each shaped ``*batch``.
    """
    original_shape: torch.Size = points.shape
    height: int = points.shape[-3]
    width: int = points.shape[-2]
    points_bhw3: Float[Tensor, "batch h w 3"] = points.reshape(-1, *original_shape[-3:])
    mask_bhw: Bool[Tensor, "batch h w"] | None = None if mask is None else mask.reshape(-1, *original_shape[-3:-1])
    focal_b: Float[Tensor, "batch"] | None = focal.reshape(-1) if focal is not None else None
    uv_hw2: Float[Tensor, "h w 2"] = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)

    points_lr_bhw3: Float[Tensor, "batch sample_h sample_w 3"] = F.interpolate(
        points_bhw3.permute(0, 3, 1, 2),
        downsample_size,
        mode="nearest",
    ).permute(0, 2, 3, 1)
    uv_lr_hw2: Float[Tensor, "sample_h sample_w 2"] = (
        F.interpolate(uv_hw2.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode="nearest").squeeze(0).permute(1, 2, 0)
    )
    mask_lr_bhw: Bool[Tensor, "batch sample_h sample_w"] | None = (
        None if mask_bhw is None else F.interpolate(mask_bhw.to(torch.float32).unsqueeze(1), downsample_size, mode="nearest").squeeze(1) > 0
    )

    uv_lr_np: Float[np.ndarray, "sample_h sample_w 2"] = uv_lr_hw2.cpu().numpy()
    points_lr_np: Float[np.ndarray, "batch sample_h sample_w 3"] = points_lr_bhw3.detach().cpu().numpy()
    focal_np: Float[np.ndarray, "batch"] | None = focal_b.cpu().numpy() if focal_b is not None else None
    mask_lr_np: Bool[np.ndarray, "batch sample_h sample_w"] | None = mask_lr_bhw.cpu().numpy() if mask_lr_bhw is not None else None
    optimal_shifts: list[float] = []
    optimal_focals: list[float] = []
    for batch_index in range(points_bhw3.shape[0]):
        points_lr_i_np: Float[np.ndarray, "n 3"] = (
            points_lr_np[batch_index].reshape(-1, 3) if mask_lr_np is None else points_lr_np[batch_index][mask_lr_np[batch_index]]
        )
        uv_lr_i_np: Float[np.ndarray, "n 2"] = uv_lr_np.reshape(-1, 2) if mask_lr_np is None else uv_lr_np[mask_lr_np[batch_index]]
        if uv_lr_i_np.shape[0] < 2:
            optimal_focals.append(1.0)
            optimal_shifts.append(0.0)
            continue
        focal_i: float | None = None if focal_np is None else float(focal_np[batch_index])
        optimal_shift_i: float
        optimal_focal_i: float
        optimal_shift_i, optimal_focal_i = _solve_focal_shift(uv_lr_i_np, points_lr_i_np, focal_i)
        optimal_focals.append(optimal_focal_i)
        optimal_shifts.append(optimal_shift_i)
    optimal_shift: Float[Tensor, "*batch"] = torch.tensor(
        optimal_shifts,
        device=points.device,
        dtype=points.dtype,
    ).reshape(original_shape[:-3])

    optimal_focal: Float[Tensor, "*batch"]
    if focal_b is None:
        optimal_focal = torch.tensor(optimal_focals, device=points.device, dtype=points.dtype).reshape(original_shape[:-3])
    else:
        optimal_focal = focal_b.reshape(original_shape[:-3])

    return optimal_focal, optimal_shift


def intrinsics_from_focal_center(
    fx: Float[Tensor, "*batch"],
    fy: Float[Tensor, "*batch"],
    cx: Float[Tensor, "*batch"],
    cy: Float[Tensor, "*batch"],
) -> Float[Tensor, "*batch 3 3"]:
    """Build OpenCV intrinsics from broadcast-compatible tensor inputs.

    Args:
        fx: Float horizontal focal tensor shaped ``*batch``.
        fy: Float vertical focal tensor shaped ``*batch``.
        cx: Float horizontal principal-point tensor shaped ``*batch``.
        cy: Float vertical principal-point tensor shaped ``*batch``.

    Returns:
        Float normalized intrinsics tensor shaped ``*batch 3 3``.
    """
    broadcast_values: tuple[Tensor, ...] = torch.broadcast_tensors(fx, fy, cx, cy)
    fx = cast(Float[Tensor, "*batch"], broadcast_values[0])
    fy = cast(Float[Tensor, "*batch"], broadcast_values[1])
    cx = cast(Float[Tensor, "*batch"], broadcast_values[2])
    cy = cast(Float[Tensor, "*batch"], broadcast_values[3])
    zeros: Float[Tensor, "*batch"] = torch.zeros_like(fx)
    ones: Float[Tensor, "*batch"] = torch.ones_like(fx)
    intrinsics: Float[Tensor, "*batch 3 3"] = torch.stack(
        [fx, zeros, cx, zeros, fy, cy, zeros, zeros, ones],
        dim=-1,
    ).unflatten(-1, (3, 3))
    return intrinsics


def _uv_map(height: int, width: int, *, dtype: torch.dtype, device: torch.device) -> Float[Tensor, "h w 2"]:
    u_w: Float[Tensor, "w"] = torch.linspace(0.5 / width, 1.0 - 0.5 / width, width, dtype=dtype, device=device)
    v_h: Float[Tensor, "h"] = torch.linspace(0.5 / height, 1.0 - 0.5 / height, height, dtype=dtype, device=device)
    uv_grid: tuple[Float[Tensor, "h w"], Float[Tensor, "h w"]] = torch.meshgrid(u_w, v_h, indexing="xy")
    u_hw: Float[Tensor, "h w"] = uv_grid[0]
    v_hw: Float[Tensor, "h w"] = uv_grid[1]
    return torch.stack([u_hw, v_hw], dim=2)


def _unproject_cv(
    uv_hw2: Float[Tensor, "h w 2"],
    depth_bhw: Float[Tensor, "*batch h w"],
    intrinsics_b133: Float[Tensor, "*batch 1 3 3"],
) -> Float[Tensor, "*batch h w 3"]:
    padded_intrinsics: Float[Tensor, "*batch 4 4"] = torch.cat(
        [
            torch.cat(
                [
                    intrinsics_b133,
                    torch.zeros((*intrinsics_b133.shape[:-2], 3, 1), dtype=intrinsics_b133.dtype, device=intrinsics_b133.device),
                ],
                dim=-1,
            ),
            torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=intrinsics_b133.dtype, device=intrinsics_b133.device).expand(
                *intrinsics_b133.shape[:-2], 1, 4
            ),
        ],
        dim=-2,
    )
    points_bhw3: Float[Tensor, "*batch h w 3"] = (
        torch.cat(
            [uv_hw2, torch.ones((*uv_hw2.shape[:-1], 1), dtype=uv_hw2.dtype, device=uv_hw2.device)],
            dim=-1,
        )
        * depth_bhw[..., None]
    )
    points_bhw4: Float[Tensor, "*batch h w 4"] = torch.cat(
        [points_bhw3, torch.ones((*points_bhw3.shape[:-1], 1), dtype=uv_hw2.dtype, device=uv_hw2.device)],
        dim=-1,
    )
    points_bhw4 = points_bhw4 @ torch.linalg.inv(padded_intrinsics).mT
    points_bhw3 = points_bhw4[..., :3]
    return points_bhw3


def depth_map_to_point_map(
    depth: Float[Tensor, "*batch h w"],
    intrinsics: Float[Tensor, "*batch 3 3"],
) -> Float[Tensor, "*batch h w 3"]:
    """Unproject a normalized-intrinsics depth map into camera-space points.

    Args:
        depth: Float depth tensor shaped ``*batch h w``.
        intrinsics: Float normalized intrinsics tensor shaped ``*batch 3 3``.

    Returns:
        Float camera-space point tensor shaped ``*batch h w 3``.
    """
    height: int = depth.shape[-2]
    width: int = depth.shape[-1]
    uv_hw2: Float[Tensor, "h w 2"] = _uv_map(height, width, dtype=depth.dtype, device=depth.device)
    points_bhw3: Float[Tensor, "*batch h w 3"] = _unproject_cv(uv_hw2, depth, intrinsics_b133=intrinsics[..., None, :, :])
    return points_bhw3
