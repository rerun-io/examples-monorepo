"""Ray <-> pixel maps on the viewing sphere, in torch, for resampling between 360 cameras and virtual views.

Every camera here maps unit rays in its own frame (x right, y down, z forward) to pixels with pixel centres at
integers, and back. ``project`` returns the pixels and a validity flag (the ray lies in the camera's field);
``unproject`` returns unit rays and a flag. The models:

- :class:`MeiCamera`: the unified (Mei) sphere with the Insta360 factory distortion (``mei_v3`` / ``mei_v6``), the
  same rules as kornia-staging's ``camera::mei`` (its Rust core states the formulas).
- :class:`Kb4Camera`: Kannala-Brandt (COLMAP ``OPENCV_FISHEYE``), with an optional image-circle radius.
- :class:`EquirectCamera`: a full ``2:1`` equirectangular panorama.
- :class:`PinholeCamera`: an undistorted pinhole (cube faces).

:func:`sample_sphere` renders a virtual view from several cameras that share one centre (scene at infinity): each
view ray takes the camera that sees it closest to that camera's own axis, the rule a dual-fisheye stitch uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Float64
from numpy import ndarray
from torch import Tensor

MeiModel: TypeAlias = Literal["mei_v3", "mei_v6"]


def pixel_grid(height: int, width: int, device: torch.device | str, scale: tuple[float, float] = (1.0, 1.0)) -> Float64[Tensor, "n 2"]:
    """Pixel centres of an ``height x width`` grid, row-major, in the coordinates of an image ``scale = (sx, sy)`` times larger.

    Cell ``(j, i)`` of the grid covers image pixels ``[i sx - 0.5, (i + 1) sx - 0.5)`` across, so its centre is
    ``((i + 0.5) sx - 0.5, (j + 0.5) sy - 0.5)`` (scale 1: the integer pixel centres).
    """
    ys: Float64[Tensor, " h"] = (torch.arange(height, dtype=torch.float64, device=device) + 0.5) * scale[1] - 0.5
    xs: Float64[Tensor, " w"] = (torch.arange(width, dtype=torch.float64, device=device) + 0.5) * scale[0] - 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)


def _normalize(rays: Float[Tensor, "n 3"]) -> Float[Tensor, "n 3"]:
    return rays / rays.norm(dim=1, keepdim=True).clamp_min(1e-12)


@dataclass(frozen=True, slots=True)
class MeiCamera:
    """Unified (Mei) sphere camera with the Insta360 factory distortion."""

    model: MeiModel
    """``mei_v3`` (``k1 k2 k3 p1 p2``) or ``mei_v6`` (``d0 .. d12``)."""
    fx: float
    """Focal length along x, pixels."""
    fy: float
    """Focal length along y, pixels."""
    cx: float
    """Principal point x, pixels (centres at integers)."""
    cy: float
    """Principal point y, pixels."""
    xi: float
    """Sphere offset."""
    half_field_rad: float
    """Largest angle off the optical axis that projects, radians."""
    distortion: tuple[float, ...]
    """The distortion coefficients in the model's order."""

    @classmethod
    def from_calibration(cls, model: str, fx: float, fy: float, cx: float, cy: float, coefficients: list[float]) -> MeiCamera:
        """Build from a calibration as dataforge writes it: ``coefficients = xi, half_field_rad, distortion...``.

        Raises:
            ValueError: On an unknown model or a wrong coefficient count.
        """
        expected: dict[str, int] = {"mei_v3": 7, "mei_v6": 15}
        if model not in expected:
            raise ValueError(f"not a Mei camera model: {model!r}")
        if len(coefficients) != expected[model]:
            raise ValueError(f"{model} takes {expected[model]} coefficients, got {len(coefficients)}")
        mei_model: MeiModel = "mei_v3" if model == "mei_v3" else "mei_v6"
        return cls(mei_model, fx, fy, cx, cy, float(coefficients[0]), float(coefficients[1]), tuple(float(c) for c in coefficients[2:]))

    def _distort(self, m: Float[Tensor, "n 2"]) -> Float[Tensor, "n 2"]:
        mx: Float[Tensor, " n"] = m[:, 0]
        my: Float[Tensor, " n"] = m[:, 1]
        r2: Float[Tensor, " n"] = mx * mx + my * my
        d: tuple[float, ...] = self.distortion
        radial: tuple[float, ...] = d[:3] if self.model == "mei_v3" else d[:5]
        gain: Float[Tensor, " n"] = torch.ones_like(r2)
        power: Float[Tensor, " n"] = torch.ones_like(r2)
        for k in radial:
            power = power * r2
            gain = gain + k * power
        if self.model == "mei_v3":
            p1, p2 = d[3], d[4]
            xd: Float[Tensor, " n"] = mx * gain + 2.0 * p1 * mx * my + p2 * (r2 + 2.0 * mx * mx)
            yd: Float[Tensor, " n"] = my * gain + p1 * (r2 + 2.0 * my * my) + 2.0 * p2 * mx * my
        else:
            a: Float[Tensor, " n"] = d[5] + d[7] * r2
            b: Float[Tensor, " n"] = d[6] + d[8] * r2
            xd = mx * gain + (r2 + 2.0 * mx * mx) * a + 2.0 * mx * my * b + d[9] * r2 + d[11] * r2 * r2
            yd = my * gain + (r2 + 2.0 * my * my) * b + 2.0 * mx * my * a + d[10] * r2 + d[12] * r2 * r2
        return torch.stack([xd, yd], dim=1)

    def project(self, rays: Float[Tensor, "n 3"]) -> tuple[Float[Tensor, "n 2"], Bool[Tensor, " n"]]:
        """Pixels of camera-frame rays; invalid more than ``half_field_rad`` off the axis."""
        norm: Float[Tensor, " n"] = rays.norm(dim=1)
        den: Float[Tensor, " n"] = rays[:, 2] + self.xi * norm
        valid: Bool[Tensor, " n"] = (norm > 0) & (rays[:, 2] >= norm * math.cos(self.half_field_rad)) & (den > 0)
        m: Float[Tensor, "n 2"] = rays[:, :2] / torch.where(valid, den, torch.ones_like(den))[:, None]
        distorted: Float[Tensor, "n 2"] = self._distort(m)
        pixels: Float[Tensor, "n 2"] = torch.stack([self.fx * distorted[:, 0] + self.cx, self.fy * distorted[:, 1] + self.cy], dim=1)
        return pixels, valid

    def unproject(self, pixels: Float[Tensor, "n 2"], iterations: int = 20) -> tuple[Float[Tensor, "n 3"], Bool[Tensor, " n"]]:
        """Unit rays of pixels: the distortion inverted by Newton (finite-difference Jacobian), then the closed-form lift.

        Invalid where Newton does not converge to 1e-6 normalized units, outside the sphere's one-to-one region, or
        more than ``half_field_rad`` off the axis. Meant for static tables: compute in float64.
        """
        target: Float[Tensor, "n 2"] = torch.stack([(pixels[:, 0] - self.cx) / self.fx, (pixels[:, 1] - self.cy) / self.fy], dim=1)
        m: Float[Tensor, "n 2"] = target.clone()
        step: float = 1e-7
        step_x: Float[Tensor, "2"] = torch.tensor([step, 0.0], dtype=m.dtype, device=m.device)
        step_y: Float[Tensor, "2"] = torch.tensor([0.0, step], dtype=m.dtype, device=m.device)
        for _ in range(iterations):
            base: Float[Tensor, "n 2"] = self._distort(m)
            residual: Float[Tensor, "n 2"] = base - target
            dx: Float[Tensor, "n 2"] = (self._distort(m + step_x) - base) / step
            dy: Float[Tensor, "n 2"] = (self._distort(m + step_y) - base) / step
            det: Float[Tensor, " n"] = dx[:, 0] * dy[:, 1] - dy[:, 0] * dx[:, 1]
            det = torch.where(det.abs() > 1e-12, det, torch.full_like(det, 1e-12))
            m = (
                m
                - torch.stack([dy[:, 1] * residual[:, 0] - dy[:, 0] * residual[:, 1], -dx[:, 1] * residual[:, 0] + dx[:, 0] * residual[:, 1]], dim=1)
                / det[:, None]
            )
        converged: Bool[Tensor, " n"] = (self._distort(m) - target).abs().amax(dim=1) < 1e-6
        r2: Float[Tensor, " n"] = (m * m).sum(dim=1)
        disc: Float[Tensor, " n"] = 1.0 + (1.0 - self.xi * self.xi) * r2
        lam: Float[Tensor, " n"] = (self.xi + disc.clamp_min(0.0).sqrt()) / (1.0 + r2)
        rays: Float[Tensor, "n 3"] = torch.stack([lam * m[:, 0], lam * m[:, 1], lam - self.xi], dim=1)
        valid: Bool[Tensor, " n"] = converged & (disc >= 0) & (rays[:, 2] >= math.cos(self.half_field_rad)) & torch.isfinite(rays).all(dim=1)
        return _normalize(rays), valid


@dataclass(frozen=True, slots=True)
class Kb4Camera:
    """Kannala-Brandt fisheye (COLMAP ``OPENCV_FISHEYE``): ``r = f theta (1 + k1 theta^2 + .. + k4 theta^8)``."""

    fx: float
    """Focal length along x, pixels."""
    fy: float
    """Focal length along y, pixels."""
    cx: float
    """Principal point x, pixels (centres at integers)."""
    cy: float
    """Principal point y, pixels."""
    k: tuple[float, float, float, float]
    """``k1 .. k4``."""
    radius_px: float = math.inf
    """Image-circle radius around the principal point; pixels beyond it are outside the field."""

    def scaled(self, factor: float) -> Kb4Camera:
        """The same lens on an image ``factor`` times the size (pixel centres at integers on both)."""
        return Kb4Camera(
            self.fx * factor, self.fy * factor, (self.cx + 0.5) * factor - 0.5, (self.cy + 0.5) * factor - 0.5, self.k, self.radius_px * factor
        )

    def _theta_d(self, theta: Float[Tensor, " n"]) -> Float[Tensor, " n"]:
        t2: Float[Tensor, " n"] = theta * theta
        return theta * (1.0 + t2 * (self.k[0] + t2 * (self.k[1] + t2 * (self.k[2] + t2 * self.k[3]))))

    def project(self, rays: Float[Tensor, "n 3"]) -> tuple[Float[Tensor, "n 2"], Bool[Tensor, " n"]]:
        """Pixels of camera-frame rays; invalid outside the image circle or past the polynomial's monotone range."""
        rho: Float[Tensor, " n"] = rays[:, :2].norm(dim=1)
        theta: Float[Tensor, " n"] = torch.atan2(rho, rays[:, 2])
        theta_d: Float[Tensor, " n"] = self._theta_d(theta)
        scale: Float[Tensor, " n"] = torch.where(
            rho > 1e-12, theta_d / rho.clamp_min(1e-12), torch.ones_like(rho) / rays[:, 2].abs().clamp_min(1e-12)
        )
        pixels: Float[Tensor, "n 2"] = torch.stack([self.fx * rays[:, 0] * scale + self.cx, self.fy * rays[:, 1] * scale + self.cy], dim=1)
        radius: Float[Tensor, " n"] = torch.hypot(pixels[:, 0] - self.cx, pixels[:, 1] - self.cy)
        # The polynomial folds back past its maximum; only its rising branch is the lens.
        rising: Bool[Tensor, " n"] = (self._theta_d(theta + 1e-4) - theta_d) > 0
        valid: Bool[Tensor, " n"] = (radius < self.radius_px) & rising & (rays.norm(dim=1) > 0)
        return pixels, valid

    def unproject(self, pixels: Float[Tensor, "n 2"], iterations: int = 30) -> tuple[Float[Tensor, "n 3"], Bool[Tensor, " n"]]:
        """Unit rays of pixels: ``theta`` from ``theta_d`` by Newton; invalid outside the image circle or unconverged."""
        x: Float[Tensor, " n"] = (pixels[:, 0] - self.cx) / self.fx
        y: Float[Tensor, " n"] = (pixels[:, 1] - self.cy) / self.fy
        theta_d: Float[Tensor, " n"] = torch.hypot(x, y)
        theta: Float[Tensor, " n"] = theta_d.clone()
        for _ in range(iterations):
            t2: Float[Tensor, " n"] = theta * theta
            slope: Float[Tensor, " n"] = 1.0 + t2 * (3 * self.k[0] + t2 * (5 * self.k[1] + t2 * (7 * self.k[2] + t2 * 9 * self.k[3])))
            theta = (theta - (self._theta_d(theta) - theta_d) / slope.clamp_min(1e-9)).clamp(0.0, math.pi)
        converged: Bool[Tensor, " n"] = (self._theta_d(theta) - theta_d).abs() < 1e-6
        ratio: Float[Tensor, " n"] = torch.where(theta_d > 1e-12, torch.sin(theta) / theta_d.clamp_min(1e-12), torch.ones_like(theta_d))
        rays: Float[Tensor, "n 3"] = torch.stack([x * ratio, y * ratio, torch.cos(theta)], dim=1)
        radius: Float[Tensor, " n"] = torch.hypot(pixels[:, 0] - self.cx, pixels[:, 1] - self.cy)
        return _normalize(rays), converged & (radius < self.radius_px)


@dataclass(frozen=True, slots=True)
class EquirectCamera:
    """Full equirectangular panorama: column ``u`` = longitude ``atan2(x, z)``, row ``v`` = latitude ``asin(y)`` (+y down)."""

    width: int
    """Panorama width, pixels (twice the height)."""
    height: int
    """Panorama height, pixels."""

    def project(self, rays: Float[Tensor, "n 3"]) -> tuple[Float[Tensor, "n 2"], Bool[Tensor, " n"]]:
        """Pixels of rays; every non-zero ray is in the field."""
        unit: Float[Tensor, "n 3"] = _normalize(rays)
        lon: Float[Tensor, " n"] = torch.atan2(unit[:, 0], unit[:, 2])
        lat: Float[Tensor, " n"] = torch.asin(unit[:, 1].clamp(-1.0, 1.0))
        u: Float[Tensor, " n"] = (lon / (2 * math.pi) + 0.5) * self.width - 0.5
        v: Float[Tensor, " n"] = (lat / math.pi + 0.5) * self.height - 0.5
        return torch.stack([u, v], dim=1), rays.norm(dim=1) > 0

    def unproject(self, pixels: Float[Tensor, "n 2"]) -> tuple[Float[Tensor, "n 3"], Bool[Tensor, " n"]]:
        """Unit rays of pixels; every pixel maps."""
        lon: Float[Tensor, " n"] = ((pixels[:, 0] + 0.5) / self.width - 0.5) * 2 * math.pi
        lat: Float[Tensor, " n"] = ((pixels[:, 1] + 0.5) / self.height - 0.5) * math.pi
        rays: Float[Tensor, "n 3"] = torch.stack([torch.cos(lat) * torch.sin(lon), torch.sin(lat), torch.cos(lat) * torch.cos(lon)], dim=1)
        return rays, torch.ones_like(lon, dtype=torch.bool)


@dataclass(frozen=True, slots=True)
class PinholeCamera:
    """Undistorted square pinhole of ``size x size`` pixels and horizontal field ``fov_deg``."""

    size: int
    """Image width and height, pixels."""
    fov_deg: float = 90.0
    """Full field of view across the image, degrees."""

    @property
    def focal(self) -> float:
        """Focal length, pixels."""
        return self.size / 2 / math.tan(math.radians(self.fov_deg) / 2)

    def project(self, rays: Float[Tensor, "n 3"]) -> tuple[Float[Tensor, "n 2"], Bool[Tensor, " n"]]:
        """Pixels of rays in front of the camera; invalid behind it or off the image."""
        centre: float = (self.size - 1) / 2
        z: Float[Tensor, " n"] = rays[:, 2].clamp_min(1e-12)
        pixels: Float[Tensor, "n 2"] = rays[:, :2] / z[:, None] * self.focal + centre
        inside: Bool[Tensor, " n"] = ((pixels > -0.5) & (pixels < self.size - 0.5)).all(dim=1)
        return pixels, (rays[:, 2] > 0) & inside

    def unproject(self, pixels: Float[Tensor, "n 2"]) -> tuple[Float[Tensor, "n 3"], Bool[Tensor, " n"]]:
        """Unit rays of pixels."""
        centre: float = (self.size - 1) / 2
        rays: Float[Tensor, "n 3"] = torch.cat([(pixels - centre) / self.focal, torch.ones_like(pixels[:, :1])], dim=1)
        return _normalize(rays), torch.ones_like(pixels[:, 0], dtype=torch.bool)


SphereCamera: TypeAlias = MeiCamera | Kb4Camera | EquirectCamera | PinholeCamera
"""Any central camera here: unit rays in its frame <-> pixels (centres at integers)."""


def cube_face_rotations() -> Float64[ndarray, "6 3 3"]:
    """``rig_R_face`` of the six 90-degree cube faces: +Z, -Z, -X, +X, -Y (up), +Y (down).

    Image down is +Y on the four side faces; image right is +X on the up and down faces (the Insta360 cube-face layer layout).
    """
    faces: list[list[list[int]]] = [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
        [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
    ]
    # Rows above are face axes (x, y, z) in the rig; rig_R_face has them as columns.
    return np.asarray(faces, dtype=np.float64).transpose(0, 2, 1)


def look_at(direction: Float64[ndarray, "3"], up: Float64[ndarray, "3"]) -> Float64[ndarray, "3 3"]:
    """``rig_R_view`` of a view whose +Z looks along ``direction``, its +Y as close to ``up`` as possible.

    FullCircle's ``look_at_camZ``: ``x = up x z``, ``y = z x x``, falling back to ``up = +X`` when ``direction`` is within
    about 8 degrees of ``up``.
    """
    z: Float64[ndarray, "3"] = direction / np.linalg.norm(direction)
    if abs(float(np.dot(z, up / np.linalg.norm(up)))) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    x: Float64[ndarray, "3"] = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y: Float64[ndarray, "3"] = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


@dataclass(frozen=True, slots=True)
class SphereView:
    """Where each pixel of a virtual view samples a set of co-centred source cameras."""

    grids: Float[Tensor, "n_src h w 2"]
    """``grid_sample`` coordinates (``align_corners=False``) of each view pixel in each source image."""
    source: Float[Tensor, "n_src h w"]
    """1 where that source supplies the pixel, else 0 (at most one source per pixel; none = outside every field)."""


def view_from_rays(
    rays: Float[Tensor, "n 3"],
    height: int,
    width: int,
    cameras: list[SphereCamera],
    rig_R_cams: Float[Tensor, "n_src 3 3"],
    image_hw: list[tuple[int, int]],
) -> SphereView:
    """The sampling plan of a ``height x width`` view whose pixel rays (row-major, rig frame) are ``rays``.

    Each ray takes the source that sees it closest to that source's optical axis (+Z), among the sources whose field
    holds it.
    """
    n_src: int = len(cameras)
    grids: list[Tensor] = []
    scores: list[Tensor] = []
    for index in range(n_src):
        cam_rays: Float[Tensor, "n 3"] = rays @ rig_R_cams[index].to(rays.dtype)
        pixels, valid = cameras[index].project(cam_rays)
        img_h, img_w = image_hw[index]
        grid: Float[Tensor, "n 2"] = torch.stack([(pixels[:, 0] + 0.5) / img_w * 2 - 1, (pixels[:, 1] + 0.5) / img_h * 2 - 1], dim=1)
        grids.append(grid.nan_to_num(0.0).reshape(height, width, 2))
        scores.append(torch.where(valid, _normalize(cam_rays)[:, 2], torch.full_like(cam_rays[:, 2], -math.inf)))
    stacked: Float[Tensor, "n_src n"] = torch.stack(scores)
    best: Tensor = stacked.argmax(dim=0)
    any_valid: Bool[Tensor, " n"] = torch.isfinite(stacked).any(dim=0)
    source: Bool[Tensor, "n_src n"] = (torch.arange(n_src, device=rays.device)[:, None] == best[None]) & any_valid[None]
    return SphereView(torch.stack(grids).float(), source.float().reshape(n_src, height, width))


def sample_sphere(images: list[Float[Tensor, "c h_src w_src"]], view: SphereView) -> Float[Tensor, "c h w"]:
    """Render a planned view from source images (float, channels first); pixels outside every field are 0."""
    out: Tensor | None = None
    for index, image in enumerate(images):
        sampled: Float[Tensor, "c h w"] = F.grid_sample(
            image[None], view.grids[index][None].to(image.dtype), mode="bilinear", padding_mode="border", align_corners=False
        )[0]
        part: Float[Tensor, "c h w"] = sampled * view.source[index][None].to(image.dtype)
        out = part if out is None else out + part
    assert out is not None
    return out


__all__ = (
    "EquirectCamera",
    "Kb4Camera",
    "MeiCamera",
    "PinholeCamera",
    "SphereCamera",
    "SphereView",
    "cube_face_rotations",
    "look_at",
    "pixel_grid",
    "sample_sphere",
    "view_from_rays",
)
