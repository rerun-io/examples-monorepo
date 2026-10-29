"""16-bit RGB PNG, the render and ground-truth interchange format of the comparison (run-layout.md)."""

from pathlib import Path

import cv2
import numpy as np
from jaxtyping import Float32, UInt16
from numpy import ndarray


def write_png16(path: Path, rgb: UInt16[ndarray, "h w 3"]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.ascontiguousarray(rgb[:, :, ::-1])):
        raise OSError(f"could not write {path}")


def read_png16(path: Path) -> UInt16[ndarray, "h w 3"]:
    """Read a 16-bit RGB PNG; an 8-bit PNG is widened exactly (k -> 257 k)."""
    image: ndarray | None = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"{path}: not an RGB(A) PNG")
    if image.dtype == np.uint8:
        image = image.astype(np.uint16) * np.uint16(257)
    if image.dtype != np.uint16:
        raise ValueError(f"{path}: unsupported dtype {image.dtype}")
    return np.ascontiguousarray(image[:, :, 2::-1])


def to_unit_float(rgb: UInt16[ndarray, "h w 3"]) -> Float32[ndarray, "h w 3"]:
    return rgb.astype(np.float32) / np.float32(65535.0)


def from_unit_float(rgb: Float32[ndarray, "h w 3"]) -> UInt16[ndarray, "h w 3"]:
    """Clamp to [0, 1] once and quantize to 16 bits (round to nearest)."""
    return np.round(np.clip(rgb, 0.0, 1.0) * np.float32(65535.0)).astype(np.uint16)
