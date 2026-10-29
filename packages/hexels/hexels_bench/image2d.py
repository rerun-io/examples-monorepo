"""2D Hexel image fits (Image-GS benchmark) as a 2D hexel PLY (2D addendum of docs/hexel-ply.md).

Upstream stores a fit as `render_parameters_<step>.pt` / `hexels_raw-fp32.pt`:
`{"method": "hexel", "layout": "dense15" | "flat8", "scale_mode": "log", "height", "width",
"parameters": {positions (n,2) in [0,1], log_scales (n,2), rotations (n,1) radians, features (n,3),
color_gradients (n,2,3) [dense15], softness (n,) already softplus'ed [dense15]}}`. Decoding
(`scripts/image_compression_stable.py` L.111-150) feeds `rasterize_hexels_sum` pixel-space centers
`positions * (W, H)`, `exp(log_scales)`, the angle, colors, gradients with `color_grad_scale = 0.1`,
and softness (10.0 for flat8).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float32
from numpy import ndarray

from hexels_bench.hexel_ply import write_float_ply

PROPERTIES_2D: tuple[str, ...] = (
    "x", "y", "su", "sv", "theta", "sharpness",
    "c0_r", "c0_g", "c0_b",
    "gx_r", "gx_g", "gx_b",
    "gy_r", "gy_g", "gy_b",
)  # fmt: skip
COLOR_GRAD_SCALE: float = 0.1
FLAT_SOFTNESS: float = 10.0


@dataclass(frozen=True, slots=True)
class HexelImage:
    """A decoded-ready 2D Hexel fit, in image pixel units."""

    width: int
    """Image width in pixels."""
    height: int
    """Image height in pixels."""
    centers_xy: Float32[ndarray, "n 2"]
    """Hexel centers in pixels: upstream positions * (W, H)."""
    scales_uv: Float32[ndarray, "n 2"]
    """exp(log_scales), pixels."""
    angles: Float32[ndarray, "n"]
    """Rotation of the hexel's u axis, radians."""
    sharpness: Float32[ndarray, "n"]
    """Softness (sigmoid slope); 10.0 for flat fits."""
    base_rgb: Float32[ndarray, "n 3"]
    """Base color."""
    grad_x_rgb: Float32[ndarray, "n 3"]
    """Raw color gradient along image x (the decoder scales it by 0.1)."""
    grad_y_rgb: Float32[ndarray, "n 3"]
    """Raw color gradient along image y."""

    def to_rows(self) -> Float32[ndarray, "n 15"]:
        return np.ascontiguousarray(
            np.concatenate(
                [self.centers_xy, self.scales_uv, self.angles[:, None], self.sharpness[:, None], self.base_rgb, self.grad_x_rgb, self.grad_y_rgb],
                axis=1,
            ),
            dtype=np.float32,
        )


def hexel_image_from_fields(width: int, height: int, fields: dict[str, ndarray]) -> HexelImage:
    """Map upstream's stored 2D fields (log-scale `dense15` or `flat8`) to pixel-space hexels.

    The same mapping as upstream's decoder (`image_compression_stable.render_image`): centers
    `positions * (W, H)`, scales `exp(log_scales)`, softness as stored (10.0 for flat fits).
    """
    positions: Float32[ndarray, "n 2"] = np.asarray(fields["positions"], dtype=np.float32)
    count: int = int(positions.shape[0])
    gradients: Float32[ndarray, "n 2 3"] = (
        np.asarray(fields["color_gradients"], dtype=np.float32) if "color_gradients" in fields else np.zeros((count, 2, 3), dtype=np.float32)
    )
    softness: Float32[ndarray, "n"] = (
        np.asarray(fields["softness"], dtype=np.float32).reshape(count) if "softness" in fields else np.full(count, FLAT_SOFTNESS, dtype=np.float32)
    )
    return HexelImage(
        width=width,
        height=height,
        centers_xy=positions * np.array([width, height], dtype=np.float32),
        scales_uv=np.exp(np.asarray(fields["log_scales"], dtype=np.float32)),
        angles=np.asarray(fields["rotations"], dtype=np.float32).reshape(count),
        sharpness=softness,
        base_rgb=np.asarray(fields["features"], dtype=np.float32),
        grad_x_rgb=np.ascontiguousarray(gradients[:, 0, :]),
        grad_y_rgb=np.ascontiguousarray(gradients[:, 1, :]),
    )


def load_image_fit(path: Path) -> HexelImage:
    """Read an upstream 2D Hexel fit: raw FP32 `.pt`, or a stored `.hxm`/`.hxc` file decoded by upstream.

    Stored files go through upstream's own `scripts.decode_image.load_representation` (hexels env),
    so the WGSL renderer sees exactly the fields upstream's CUDA decoder sees.
    """
    if path.suffix == ".pt":
        payload: dict = torch.load(path, map_location="cpu", weights_only=True)
        tensors: dict[str, torch.Tensor] = payload["parameters"]
    else:
        from scripts.decode_image import load_representation

        payload, tensors, legacy = load_representation(path)
        if legacy:
            raise ValueError(f"{path}: legacy reciprocal-scale file; only log-scale Hexel fits have a WGSL decoder")
    if payload.get("method") != "hexel" or payload.get("layout") not in ("dense15", "flat8") or payload.get("scale_mode", "log") != "log":
        raise ValueError(f"{path}: expected a log-scale dense15/flat8 Hexel fit, got {payload.get('method')}/{payload.get('layout')}")
    fields: dict[str, ndarray] = {name: value.float().numpy() for name, value in tensors.items()}
    return hexel_image_from_fields(int(payload["width"]), int(payload["height"]), fields)


def write_hexel_image_ply(path: Path, image: HexelImage, source: str, step: int) -> None:
    comments: list[str] = ["hexels_format 1", "hexels_variant 2d", f"hexels_source {source}", f"hexels_step {step}"]
    comments += [f"hexels_image {image.width} {image.height}", f"hexels_color_grad_scale {COLOR_GRAD_SCALE}"]
    write_float_ply(path, comments, PROPERTIES_2D, image.to_rows())
