"""Cube-face pinholes of the Insta360 cube-face layer (6-face cubemap), and spirula's depth/normal prior encodings.

Every face is a full N x N 90 deg pinhole (fx = fy = N/2, principal point at the image centre with integer pixel
centres: cx = cy = (N - 1) / 2), named ``cube6_<front0|front1|left|right|up|down>``. MoGe-2 runs on each face as
it is, so a prior matches its training image pixel for pixel.
"""

import re
from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float32, Float64, UInt8, UInt16
from numpy import ndarray

FACE_NAME_PATTERN: re.Pattern[str] = re.compile(r"^cube6_(front0|front1|left|right|up|down)$")
"""Face names of the 6-face cubemap layer (each camera node's static ``name``)."""


@dataclass(frozen=True, slots=True)
class CubeFace:
    """One full, centred virtual pinhole of the cubemap layer."""

    entity: str
    """Camera node, e.g. ``/world/rig_00/cam_03``."""
    name: str
    """Face name ``cube6_<direction>`` (the node's static ``name``)."""
    k_matrix: Float64[ndarray, "3 3"]
    """Pinhole intrinsics, pixels, integer pixel centres."""
    width: int
    """Image width, pixels."""
    height: int
    """Image height, pixels."""

    def __post_init__(self) -> None:
        """Validate the name and that the face is a full centred 90 deg pinhole.

        Raises:
            ValueError: For another name, a non-square face, fx or fy other than N/2, or an off-centre principal point.
        """
        if FACE_NAME_PATTERN.match(self.name) is None:
            raise ValueError(f"{self.entity}: face name {self.name!r} is not cube6_<front0|front1|left|right|up|down>")
        centre_px: float = (self.width - 1) / 2.0
        expected: Float64[ndarray, "4"] = np.array([self.width / 2.0, self.width / 2.0, centre_px, centre_px])
        actual: Float64[ndarray, "4"] = np.array([self.k_matrix[0, 0], self.k_matrix[1, 1], self.k_matrix[0, 2], self.k_matrix[1, 2]])
        if self.height != self.width or not np.allclose(actual, expected, atol=1e-3):
            raise ValueError(f"{self.name}: {self.width}x{self.height} with fx fy cx cy {actual} is not a full centred 90 deg face {expected}")


def depth_png_mm(depth_m: Float32[ndarray, "h w"], valid: Bool[ndarray, "h w"]) -> UInt16[ndarray, "h w"]:
    """16-bit z-depth in millimetres for spirula's ``depths/``: 0 = invalid, beyond 65.535 m = invalid.

    Args:
        depth_m: Float32 z-depth in metres, shape ``h w``.
        valid: Bool ``h w`` validity mask.

    Returns:
        UInt16 ``h w`` depth in millimetres.
    """
    depth_mm: Float32[ndarray, "h w"] = np.round(depth_m * 1000.0).astype(np.float32)
    keep: Bool[ndarray, "h w"] = valid & np.isfinite(depth_mm) & (depth_mm > 0.0) & (depth_mm <= 65535.0)
    return np.where(keep, depth_mm, 0.0).astype(np.uint16)


def normal_png_rgb(normals: Float32[ndarray, "h w 3"], valid: Bool[ndarray, "h w"]) -> UInt8[ndarray, "h w 3"]:
    """RGB normals for spirula's ``normals/``: ``127.5 + 127.5 n`` in the OpenCV camera frame, black = none.

    Args:
        normals: Float32 unit normals ``h w 3`` in the camera frame (RDF), facing the camera.
        valid: Bool ``h w`` validity mask.

    Returns:
        UInt8 ``h w 3`` RGB image.
    """
    encoded: Float32[ndarray, "h w 3"] = np.clip(np.round(127.5 + 127.5 * normals), 0.0, 255.0).astype(np.float32)
    return np.where(valid[..., None], encoded, 0.0).astype(np.uint8)
