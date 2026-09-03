"""DPT head utilities: position embeddings and interpolation helpers."""

import torch
import torch.nn as nn
from jaxtyping import Float, Float32
from torch import Tensor


def position_grid_to_embed(
    pos_grid: Float[Tensor, "height width 2"], embed_dim: int, omega_0: float = 100.0
) -> Float32[Tensor, "height width features"]:
    """Convert a 2D position grid (H, W, 2) to sinusoidal embeddings (H, W, C).

    Args:
        pos_grid: 2D coordinate tensor (H, W, 2).
        embed_dim: Output embedding dimension.

    Returns:
        Position embeddings (H, W, embed_dim).
    """
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat: Float[Tensor, "positions 2"] = pos_grid.reshape(-1, grid_dim)

    emb_x: Float32[Tensor, "positions half_features"] = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)
    emb_y: Float32[Tensor, "positions half_features"] = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)

    emb: Float32[Tensor, "positions features"] = torch.cat([emb_x, emb_y], dim=-1)
    return emb.view(H, W, embed_dim)


def make_sincos_pos_embed(embed_dim: int, pos: Float[Tensor, "positions"], omega_0: float = 100.0) -> Float32[Tensor, "positions features"]:
    """Generate 1D sin/cos position embeddings."""
    assert embed_dim % 2 == 0
    omega: Float32[Tensor, "half_features"] = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega

    pos = pos.reshape(-1)
    out: Float[Tensor, "positions half_features"] = torch.einsum("m,d->md", pos, omega)

    emb_sin: Float[Tensor, "positions half_features"] = torch.sin(out)
    emb_cos: Float[Tensor, "positions half_features"] = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb.float()


def create_uv_grid(
    width: int,
    height: int,
    aspect_ratio: float | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Float[Tensor, "height width 2"]:
    """Create a normalized UV grid (width, height, 2).

    Normalized by aspect ratio so the diagonal length is 1.
    """
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height

    x_coords: Float[Tensor, "width"] = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords: Float[Tensor, "height"] = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)

    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    uv_grid = torch.stack((uu, vv), dim=-1)
    return uv_grid


def custom_interpolate(
    x: Float[Tensor, "batch channels height width"],
    size: tuple[int, int] | None = None,
    scale_factor: float | None = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> Float[Tensor, "batch channels output_height output_width"]:
    """Interpolation that chunks large tensors to avoid integer overflow."""
    if size is None:
        assert scale_factor is not None
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736
    total = size[0] * size[1] * x.shape[0] * x.shape[1]

    if total > INT_MAX:
        chunks: tuple[Float[Tensor, "chunk channels height width"], ...] = torch.chunk(x, chunks=(total // INT_MAX) + 1, dim=0)
        outs: list[Float[Tensor, "chunk channels output_height output_width"]] = [
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks
        ]
        return torch.cat(outs, dim=0).contiguous()

    return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)
