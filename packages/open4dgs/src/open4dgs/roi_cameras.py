"""Person-centred virtual cameras: each real camera rotated about its optical centre to look at the people, and zoomed.

A pinhole rotated about its centre and given a new focal length sees the same rays, so the virtual image is an exact
reprojection of the (undistorted) source: virtual pixel -> ray -> source pixel is a homography. ``catalog_clip`` folds
this map and the lens undistortion into one ``grid_sample`` grid, so the frames are resampled once.

The virtual camera:

- looks at the centre of the angular box of the points that must stay in view (all people's mesh vertices over the
  window, or any ``n x 3`` world points such as skeleton joints; found by starting at their centroid and re-aiming),
  by the smallest rotation that takes the optical axis there, so the image "up" stays as close as possible to the
  source's;
- has a square-pixel focal equal to the source's (``sqrt(fx fy)``): at its centre it samples the scene at the source's
  angular pixel density at the source's principal point, and the source is denser everywhere off-axis, so it never
  upsamples;
- is just large enough that the points' box, padded by ``pad`` of its size on every side, fits about a CENTRED
  principal point (sizes rounded up to even numbers);
- when ``max_side`` caps it, is resized (antialiased) from that sampling grid to the capped size.

Pixel convention as ``catalog_clip``: pixel centres at integers, a ``w``-wide image spans ``[-0.5, w - 0.5]``, so the
centred principal point is ``((w - 1) / 2, (h - 1) / 2)``.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float32, Float64
from numpy import ndarray
from torch import Tensor

AIM_ITERATIONS: int = 8
"""Re-aims at the projected box centre after the centroid (each one shrinks the off-centre error by about 100x)."""


@dataclass(frozen=True, slots=True)
class VirtualCamera:
    """A virtual pinhole that shares a real camera's optical centre (see the module docstring)."""

    source_K: Float64[ndarray, "3 3"]
    """The real camera's undistorted pinhole at its stored size."""
    source_size: tuple[int, int]
    """The real camera's stored ``(width, height)``."""
    cam_R_virtual: Float64[ndarray, "3 3"]
    """Rotation from virtual-camera axes to real-camera axes (OpenCV RDF both): ``world_T_virtual = world_T_cam @ R``."""
    sample_K: Float64[ndarray, "3 3"]
    """Virtual intrinsics at ``sample_size`` (focal = the source's, principal point centred)."""
    sample_size: tuple[int, int]
    """``(width, height)`` of the sampling grid (native angular density)."""
    K: Float64[ndarray, "3 3"]
    """Virtual intrinsics at ``size`` (the returned images)."""
    size: tuple[int, int]
    """Returned ``(width, height)``: ``sample_size`` capped by ``max_side``."""

    def world_T_virtual(self, world_T_cam: Float64[ndarray, "4 4"]) -> Float64[ndarray, "4 4"]:
        """The virtual camera's camera-to-world pose."""
        rotation: Float64[ndarray, "4 4"] = np.eye(4)
        rotation[:3, :3] = self.cam_R_virtual
        return world_T_cam.astype(np.float64) @ rotation


def minimal_rotation_to(direction: Float64[ndarray, "3"]) -> Float64[ndarray, "3 3"]:
    """The smallest rotation ``R`` with ``R @ [0, 0, 1] = direction / |direction|`` (Rodrigues about ``z x d``)."""
    d: Float64[ndarray, "3"] = direction / np.linalg.norm(direction)
    axis: Float64[ndarray, "3"] = np.cross([0.0, 0.0, 1.0], d)
    sin: float = float(np.linalg.norm(axis))
    cos: float = float(d[2])
    if sin < 1e-12:
        if cos > 0.0:
            return np.eye(3)
        raise ValueError("the points lie straight behind the camera")
    k: Float64[ndarray, "3"] = axis / sin
    cross: Float64[ndarray, "3 3"] = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + sin * cross + (1.0 - cos) * cross @ cross


def _even_ceil(value: float) -> int:
    """The smallest even integer >= ``value`` (a 2e-3 px slack absorbs float32 point noise), at least 2."""
    return max(2, 2 * math.ceil((value - 2e-3) / 2.0))


def _even_round(value: float) -> int:
    return max(2, 2 * round(value / 2.0))


def centred_K(focal: float, size: tuple[int, int]) -> Float64[ndarray, "3 3"]:
    """Square-pixel intrinsics with the principal point at the centre of a ``size`` image."""
    return np.array([[focal, 0.0, (size[0] - 1) / 2.0], [0.0, focal, (size[1] - 1) / 2.0], [0.0, 0.0, 1.0]])


def _tangent_box(in_cam: Float64[ndarray, "n 3"], cam_R_virtual: Float64[ndarray, "3 3"]) -> tuple[Float64[ndarray, "2"], Float64[ndarray, "2"]]:
    """Min and max of ``x / z, y / z`` of the points in the virtual camera; raises if one is not in front of it."""
    in_virtual: Float64[ndarray, "n 3"] = in_cam @ cam_R_virtual  # R^T p for each row
    if not bool((in_virtual[:, 2] > 1e-6).all()):
        raise ValueError("some points are not in front of the virtual camera (field of view of 180 degrees or more)")
    tangent: Float64[ndarray, "n 2"] = in_virtual[:, :2] / in_virtual[:, 2:3]
    return tangent.min(axis=0), tangent.max(axis=0)


def fit_virtual_camera(
    points_world: Float32[ndarray, "n 3"],
    K: Float64[ndarray, "3 3"],
    world_T_cam: Float64[ndarray, "4 4"],
    size: tuple[int, int],
    pad: float = 0.15,
    max_side: int | None = None,
) -> VirtualCamera:
    """The virtual camera of the real camera ``(K, world_T_cam, size)`` that keeps ``points_world`` (NaN rows ignored) in view.

    Raises:
        ValueError: If no point is finite, or some point is not in front of the virtual camera.
    """
    points: Float64[ndarray, "n 3"] = points_world.astype(np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise ValueError("no finite points to fit a virtual camera to")
    cam_T_world: Float64[ndarray, "4 4"] = np.linalg.inv(world_T_cam.astype(np.float64))
    in_cam: Float64[ndarray, "n 3"] = points @ cam_T_world[:3, :3].T + cam_T_world[:3, 3]
    # Start at the centroid, then re-aim at the centre of the projected box: a centroid-centred symmetric box wastes
    # the side the points do not reach (MAMMA dance: the top 40 % of the image was empty).
    rotation: Float64[ndarray, "3 3"] = minimal_rotation_to(in_cam.mean(axis=0))
    lo, hi = _tangent_box(in_cam, rotation)
    for _ in range(AIM_ITERATIONS):
        middle: Float64[ndarray, "2"] = (lo + hi) / 2.0
        if float(np.abs(middle).max()) < 1e-9:
            break
        rotation = minimal_rotation_to(rotation @ np.array([middle[0], middle[1], 1.0]))
        lo, hi = _tangent_box(in_cam, rotation)
    margin: Float64[ndarray, "2"] = (hi - lo) * pad
    half: Float64[ndarray, "2"] = np.maximum(np.abs(lo - margin), np.abs(hi + margin))
    focal: float = math.sqrt(float(K[0, 0]) * float(K[1, 1]))
    sample_size: tuple[int, int] = (_even_ceil(2.0 * half[0] * focal), _even_ceil(2.0 * half[1] * focal))
    out_size: tuple[int, int] = sample_size
    if max_side is not None and max(sample_size) > max_side:
        scale: float = max_side / max(sample_size)
        out_size = (min(_even_round(sample_size[0] * scale), max_side), min(_even_round(sample_size[1] * scale), max_side))
    sample_K: Float64[ndarray, "3 3"] = centred_K(focal, sample_size)
    # The resize from sample_size to out_size scales x and y by w'/w and h'/h: the centred principal point stays centred.
    out_K: Float64[ndarray, "3 3"] = np.array(
        [
            [focal * out_size[0] / sample_size[0], 0.0, (out_size[0] - 1) / 2.0],
            [0.0, focal * out_size[1] / sample_size[1], (out_size[1] - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return VirtualCamera(
        source_K=K.astype(np.float64),
        source_size=size,
        cam_R_virtual=rotation,
        sample_K=sample_K,
        sample_size=sample_size,
        K=out_K,
        size=out_size,
    )


def virtual_to_source_pixels(camera: VirtualCamera, device: torch.device) -> Float64[Tensor, "h w 2"]:
    """For each pixel of the sampling grid (``sample_size``), the undistorted source pixel that sees the same ray.

    Rays that point behind the source camera get NaN.
    """
    width, height = camera.sample_size
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float64, device=device), torch.arange(width, dtype=torch.float64, device=device), indexing="ij"
    )
    pixels: Float64[Tensor, "h w 3"] = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)
    # source pixel ~ K_s R K_v^-1 [x, y, 1]: one homography.
    homography: Float64[ndarray, "3 3"] = camera.source_K @ camera.cam_R_virtual @ np.linalg.inv(camera.sample_K)
    projected: Float64[Tensor, "h w 3"] = pixels @ torch.as_tensor(homography.T, dtype=torch.float64, device=device)
    depth: Float64[Tensor, "h w 1"] = projected[..., 2:3]
    source: Float64[Tensor, "h w 2"] = projected[..., :2] / depth
    return torch.where(depth > 1e-12, source, torch.full_like(source, float("nan")))
