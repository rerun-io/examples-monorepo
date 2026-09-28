"""Differentiable lens models for a camera rig: unclipped Fisheye62 (UmeTrack) and plain pinhole (SHOW3D).

Points are in metres. Projection never clips at the field of view or the image edge: labels need the
position of keypoints outside the image, and the pose fit needs a smooth residual. Use ``in_front``
for the z > 0 test.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float32
from torch import Tensor

_R2_MAX: float = torch.pi**2
"""simplecv's Fisheye62 formula clips the squared normalised radius at pi^2."""


@dataclass(frozen=True, slots=True)
class CameraRig:
    """The calibrated cameras of one rig, stacked on a leading camera axis ``c``.

    ``fisheye62`` holds ``[k1..k6, p1, p2]`` per camera in simplecv's order; ``None`` means every camera is a pinhole.
    """

    names: tuple[str, ...]
    """Camera entity paths, e.g. ``/world/rig_00/cam_00``."""
    image_size: Float32[Tensor, "c 2"]
    """(width, height) in pixels."""
    cam_from_rig: Float32[Tensor, "c 4 4"]
    focal: Float32[Tensor, "c 2"]
    principal: Float32[Tensor, "c 2"]
    fisheye62: Float32[Tensor, "c 8"] | None

    def __post_init__(self) -> None:
        n: int = len(self.names)
        shapes: list[tuple[int, ...]] = [tuple(self.image_size.shape), tuple(self.cam_from_rig.shape), tuple(self.focal.shape), tuple(self.principal.shape)]
        if shapes != [(n, 2), (n, 4, 4), (n, 2), (n, 2)] or (self.fisheye62 is not None and tuple(self.fisheye62.shape) != (n, 8)):
            raise ValueError(f"CameraRig tensors do not match {n} cameras: {shapes}")

    def to(self, device: torch.device | str) -> "CameraRig":
        return CameraRig(
            names=self.names,
            image_size=self.image_size.to(device),
            cam_from_rig=self.cam_from_rig.to(device),
            focal=self.focal.to(device),
            principal=self.principal.to(device),
            fisheye62=None if self.fisheye62 is None else self.fisheye62.to(device),
        )


def transform_points(a_from_b: Float32[Tensor, "*batch 4 4"], points_b: Float32[Tensor, "*batch n 3"]) -> Float32[Tensor, "*batch n 3"]:
    """Apply a rigid transform to a batch of point sets."""
    return torch.einsum("...ij,...nj->...ni", a_from_b[..., :3, :3], points_b) + a_from_b[..., None, :3, 3]


def world_to_cameras(rig: CameraRig, world_from_rig: Float32[Tensor, "*batch 4 4"], points_world: Float32[Tensor, "*batch n 3"]) -> Float32[Tensor, "*batch c n 3"]:
    """World points into every camera of the rig: p_cam = cam_from_rig · rig_from_world · p_world."""
    # Rᵀ(p − t) on row vectors is (p − t) @ R.
    points_rig: Float32[Tensor, "*batch n 3"] = (points_world - world_from_rig[..., None, :3, 3]) @ world_from_rig[..., :3, :3]
    return transform_points(rig.cam_from_rig, points_rig.unsqueeze(-3))


def project(rig: CameraRig, points_cam: Float32[Tensor, "*batch c n 3"]) -> Float32[Tensor, "*batch c n 2"]:
    """Pixel coordinates in each camera's own image, unclipped; the camera axis is the third from last."""
    x: Float32[Tensor, "*batch c n"] = points_cam[..., 0]
    y: Float32[Tensor, "*batch c n"] = points_cam[..., 1]
    z: Float32[Tensor, "*batch c n"] = points_cam[..., 2]
    if rig.fisheye62 is None:
        # A point at z = 0 has no pinhole image; clamp so it stays finite (in_front rejects it).
        z_safe: Float32[Tensor, "*batch c n"] = torch.where(z.abs() < 1e-9, torch.full_like(z, 1e-9), z)
        normalized: Float32[Tensor, "*batch c n 2"] = torch.stack([x / z_safe, y / z_safe], dim=-1)
    else:
        # The epsilon keeps atan2(r, z) / r and its gradient finite on the optical axis (r = 0).
        radius: Float32[Tensor, "*batch c n"] = torch.sqrt(x * x + y * y + 1e-12)
        scale: Float32[Tensor, "*batch c n"] = torch.atan2(radius, z) / radius
        theta_xy: Float32[Tensor, "*batch c n 2"] = torch.stack([x * scale, y * scale], dim=-1)
        k1, k2, k3, k4, k5, k6, p1, p2 = rig.fisheye62[:, :, None].unbind(dim=1)  # each (c, 1), broadcasting over n
        r2: Float32[Tensor, "*batch c n"] = (theta_xy * theta_xy).sum(dim=-1).clamp(max=_R2_MAX)
        radial: Float32[Tensor, "*batch c n"] = 1 + r2 * (k1 + r2 * (k2 + r2 * (k3 + r2 * (k4 + r2 * (k5 + r2 * k6)))))
        u: Float32[Tensor, "*batch c n"] = theta_xy[..., 0] * radial
        v: Float32[Tensor, "*batch c n"] = theta_xy[..., 1] * radial
        uv2: Float32[Tensor, "*batch c n"] = u * u + v * v
        normalized = torch.stack([u + 2 * p2 * u * v + p1 * (uv2 + 2 * u * u), v + 2 * p1 * u * v + p2 * (uv2 + 2 * v * v)], dim=-1)
    return normalized * rig.focal[:, None, :] + rig.principal[:, None, :]


def in_front(points_cam: Float32[Tensor, "*batch 3"]) -> Bool[Tensor, "*batch"]:
    """Finite points with z > 0."""
    return torch.isfinite(points_cam).all(dim=-1) & (points_cam[..., 2] > 0)


def inside_image(rig: CameraRig, pixels: Float32[Tensor, "*batch c n 2"]) -> Bool[Tensor, "*batch c n"]:
    """Pixels inside [0, W) x [0, H) of their camera; a non-finite pixel fails both bounds."""
    size: Float32[Tensor, "c 1 2"] = rig.image_size[:, None, :]
    return ((pixels >= 0) & (pixels < size)).all(dim=-1)
