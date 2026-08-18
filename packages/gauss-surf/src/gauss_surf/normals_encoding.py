"""Pure lossless PNG encoding for signed camera-space normal maps."""

from io import BytesIO

import numpy as np
from jaxtyping import Float32, UInt8
from numpy import ndarray
from PIL import Image

from gauss_surf.render_io import encode_rgb_png


def to_away_from_camera(normals_toward_hw3: Float32[ndarray, "h w 3"]) -> Float32[ndarray, "h w 3"]:
    """Convert MoGe's toward-camera RDF normals to the gaussurf target convention.

    Args:
        normals_toward_hw3: float32 toward-camera normals shaped ``h w 3``.

    Returns:
        float32 away-from-camera RDF normals shaped ``h w 3``.
    """
    if normals_toward_hw3.ndim != 3 or normals_toward_hw3.shape[-1] != 3:
        raise ValueError("Normals must have shape (H, W, 3)")
    normals_away_hw3: Float32[ndarray, "h w 3"] = np.negative(normals_toward_hw3).astype(np.float32, copy=False)
    return normals_away_hw3


def encode_normals_png(normals_hw3: Float32[ndarray, "h w 3"]) -> bytes:
    """Encode signed float32 normals as a lossless RGB PNG.

    Args:
        normals_hw3: float32 normal components shaped ``h w 3`` in ``[-1, 1]``.

    Returns:
        PNG bytes containing the rounded ``(normal + 1) / 2 * 255`` mapping.
    """
    if normals_hw3.ndim != 3 or normals_hw3.shape[-1] != 3:
        raise ValueError("Normals must have shape (H, W, 3)")
    if not np.all(np.isfinite(normals_hw3)):
        raise ValueError("Normals must contain only finite values")
    if np.any(normals_hw3 < -1.0) or np.any(normals_hw3 > 1.0):
        raise ValueError("Normal components must lie within [-1, 1]")

    rgb_hw3: UInt8[ndarray, "h w 3"] = np.rint((normals_hw3 + 1.0) / 2.0 * 255.0).astype(np.uint8)
    return encode_rgb_png(rgb_hw3)


def decode_normal_codes(rgb_hw3: UInt8[ndarray, "h w 3"]) -> Float32[ndarray, "h w 3"]:
    """Decode quantized normal RGB codes and restore central code 128 to zero."""
    if rgb_hw3.ndim != 3 or rgb_hw3.shape[-1] != 3 or rgb_hw3.dtype != np.uint8:
        raise ValueError("Normal codes must be uint8 with shape (H, W, 3)")
    decoded_hw3: Float32[ndarray, "h w 3"] = rgb_hw3.astype(np.float32) / 255.0 * 2.0 - 1.0
    decoded_hw3[rgb_hw3 == 128] = 0.0
    return decoded_hw3


def decode_normals_png(png_bytes: bytes) -> Float32[ndarray, "h w 3"]:
    """Decode a lossless RGB normal PNG into signed float32 components.

    Args:
        png_bytes: PNG bytes written by :func:`encode_normals_png`.

    Returns:
        float32 normal components shaped ``h w 3`` in ``[-1, 1]``.
    """
    with Image.open(BytesIO(png_bytes)) as image:
        rgb_hw3: UInt8[ndarray, "h w 3"] = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return decode_normal_codes(rgb_hw3)
