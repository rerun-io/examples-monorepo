"""Batched CUDA lens inversion, virtual cameras and hand crop sampling."""

import os
from dataclasses import dataclass

import numpy as np
import torch
import warp as wp
from jaxtyping import Float32, Float64, UInt8
from torch import Tensor

from egohand_inference.camera.camera_models import _PerspectiveCamera, _project_rational_ray
from posekit.cameras import CalibratedCamera
from posekit.models.egohand_crops import _compiled_normalize_crops, _normalize_crops
from posekit.models.egohand_geometry import lens_model
from posekit.predictions import HandDetections


@wp.func
def _fisheye_project(ray: wp.vec3d, intrinsic: wp.vec4d, coefficients: wp.array(dtype=wp.float64), radial_count: int):
    radius = wp.sqrt(ray[0] * ray[0] + ray[1] * ray[1])
    theta = wp.atan2(radius, ray[2])
    power = theta * theta
    radial = wp.float64(1.0)
    for i in range(radial_count):
        radial += coefficients[i] * power
        power *= theta * theta
    factor = wp.float64(0.0)
    if radius > wp.float64(0.0):
        factor = theta * radial / radius
    x, y = ray[0] * factor, ray[1] * factor
    r2 = x * x + y * y
    if radial_count == 6:
        p0, p1 = coefficients[6], coefficients[7]
        xd = x + wp.float64(2.0) * p1 * x * y + p0 * (r2 + wp.float64(2.0) * x * x) + coefficients[8] * r2 + coefficients[9] * r2 * r2
        yd = y + wp.float64(2.0) * p0 * x * y + p1 * (r2 + wp.float64(2.0) * y * y) + coefficients[10] * r2 + coefficients[11] * r2 * r2
        x, y = xd, yd
    if ray[2] <= wp.float64(0.0):
        return wp.vec2d(wp.float64(wp.nan), wp.float64(wp.nan))
    return wp.vec2d(x * intrinsic[0] + intrinsic[2], y * intrinsic[1] + intrinsic[3])


@wp.func
def _project(ray: wp.vec3d, intrinsic: wp.vec4d, coefficients: wp.array(dtype=wp.float64), tilt: wp.vec4d, radius_bound: wp.float64, radial_count: int):
    if radial_count == 0:
        return _project_rational_ray(ray, intrinsic, coefficients, tilt, radius_bound)
    return _fisheye_project(ray, intrinsic, coefficients, radial_count)


@wp.func
def _inverse(pixel: wp.vec2d, intrinsic: wp.vec4d, coefficients: wp.array(dtype=wp.float64), tilt: wp.vec4d, radius_bound: wp.float64, radial_count: int):
    x = (pixel[0] - intrinsic[2]) / intrinsic[0]
    y = (pixel[1] - intrinsic[3]) / intrinsic[1]
    r2 = x * x + y * y
    if radial_count == 0 and wp.isfinite(radius_bound) and r2 >= radius_bound:
        scale = wp.sqrt(wp.float64(0.9) * radius_bound / r2)
        x *= scale
        y *= scale
    epsilon = wp.float64(1e-5)
    error = wp.float64(wp.inf)
    for _ in range(50):
        ray = wp.vec3d(x, y, wp.float64(1.0))
        projected = _project(ray, intrinsic, coefficients, tilt, radius_bound, radial_count)
        residual = projected - pixel
        error = wp.max(wp.abs(residual[0]), wp.abs(residual[1]))
        if not wp.isfinite(error) or error < wp.float64(1e-8):
            break
        dx = (_project(ray + wp.vec3d(epsilon, wp.float64(0.0), wp.float64(0.0)), intrinsic, coefficients, tilt, radius_bound, radial_count) - projected) / epsilon
        dy = (_project(ray + wp.vec3d(wp.float64(0.0), epsilon, wp.float64(0.0)), intrinsic, coefficients, tilt, radius_bound, radial_count) - projected) / epsilon
        determinant = dx[0] * dy[1] - dy[0] * dx[1]
        if not wp.isfinite(determinant) or determinant <= wp.float64(0.0):
            break
        step_x = (dy[1] * residual[0] - dy[0] * residual[1]) / determinant
        step_y = (dx[0] * residual[1] - dx[1] * residual[0]) / determinant
        factor = wp.float64(1.0)
        for _ in range(20):
            candidate_x, candidate_y = x - factor * step_x, y - factor * step_y
            trial = _project(wp.vec3d(candidate_x, candidate_y, wp.float64(1.0)), intrinsic, coefficients, tilt, radius_bound, radial_count) - pixel
            trial_error = wp.max(wp.abs(trial[0]), wp.abs(trial[1]))
            if wp.isfinite(trial_error) and trial_error < error:
                x, y = candidate_x, candidate_y
                break
            factor *= wp.float64(0.5)
    check = _project(wp.vec3d(x, y, wp.float64(1.0)), intrinsic, coefficients, tilt, radius_bound, radial_count) - pixel
    if not wp.isfinite(check[0]) or not wp.isfinite(check[1]) or wp.max(wp.abs(check[0]), wp.abs(check[1])) > wp.float64(1e-6):
        return wp.vec3d(wp.float64(wp.nan), wp.float64(wp.nan), wp.float64(wp.nan))
    if radial_count > 0:
        # The released inverse rejects radial coordinates beyond the polynomial
        # at pi/2, even when a non-monotonic polynomial admits another solution.
        # Accepting that solution expands edge crops and changes predicted depth.
        theta = wp.atan(wp.sqrt(x * x + y * y))
        half_pi = wp.float64(1.5707963267948966)
        radial = wp.float64(1.0)
        maximum_radial = wp.float64(1.0)
        power = theta * theta
        maximum_power = half_pi * half_pi
        for i in range(radial_count):
            radial += coefficients[i] * power
            maximum_radial += coefficients[i] * maximum_power
            power *= theta * theta
            maximum_power *= half_pi * half_pi
        if theta * radial >= half_pi * maximum_radial:
            return wp.vec3d(wp.float64(wp.nan), wp.float64(wp.nan), wp.float64(wp.nan))
    return wp.normalize(wp.vec3d(x, y, wp.float64(1.0)))


@wp.kernel(enable_backward=False)
def _unproject_pixels(pixels: wp.array2d(dtype=wp.vec2d), intrinsic: wp.vec4d, coefficients: wp.array(dtype=wp.float64), tilt: wp.vec4d,
                      radius_bound: wp.float64, radial_count: int, rays: wp.array2d(dtype=wp.vec3d)):
    hand, point = wp.tid()
    rays[hand, point] = _inverse(pixels[hand, point], intrinsic, coefficients, tilt, radius_bound, radial_count)


@wp.kernel(enable_backward=False)
def _remap(focal: wp.array(dtype=wp.float64), rotation: wp.array(dtype=wp.mat33d), mirrored: wp.array(dtype=wp.bool),
           intrinsic: wp.vec4d, coefficients: wp.array(dtype=wp.float64), tilt: wp.vec4d, radius_bound: wp.float64, radial_count: int,
           pixels: wp.array3d(dtype=wp.vec2)):
    hand, row, col = wp.tid()
    u = col
    if mirrored[hand]:
        u = 255 - col
    ray = rotation[hand] @ wp.vec3d((wp.float64(u) - wp.float64(127.5)) / wp.float64(focal[hand]),
                                   (wp.float64(row) - wp.float64(127.5)) / wp.float64(focal[hand]), wp.float64(1.0))
    uv = _project(ray, intrinsic, coefficients, tilt, radius_bound, radial_count)
    if wp.isfinite(uv[0]) and wp.isfinite(uv[1]):
        pixels[hand, row, col] = wp.vec2(wp.float32(uv[0]), wp.float32(uv[1]))
    else:
        pixels[hand, row, col] = wp.vec2(-1.0, -1.0)


@wp.kernel(enable_backward=False)
def _project_joints(points: wp.array2d(dtype=wp.vec3), intrinsic: wp.vec4d, coefficients: wp.array(dtype=wp.float64), tilt: wp.vec4d,
                    radius_bound: wp.float64, radial_count: int, pixels: wp.array2d(dtype=wp.vec2)):
    hand, joint = wp.tid()
    p = points[hand, joint]
    uv = _project(wp.vec3d(wp.float64(p[0]), wp.float64(p[1]), wp.float64(p[2])), intrinsic, coefficients, tilt, radius_bound, radial_count)
    pixels[hand, joint] = wp.vec2(wp.float32(uv[0]), wp.float32(uv[1]))


def _sample_perimeter(boxes: Float32[Tensor, "n 4"], edge: Float64[Tensor, "1025"], padding: float) -> Float64[Tensor, "n 4101 2"]:
    center = ((boxes[:, :2] + boxes[:, 2:]) / 2.0).double()
    size = ((boxes[:, 2:] - boxes[:, :2]).amax(-1).double() * (1.0 + padding)).float().double()
    low, high = center - size[:, None] / 2.0, center + size[:, None] / 2.0
    x = low[:, 0, None] + size[:, None] * edge
    y = low[:, 1, None] + size[:, None] * edge
    return torch.cat((center[:, None], torch.stack((x, low[:, 1, None].expand_as(x)), -1),
                      torch.stack((x, high[:, 1, None].expand_as(x)), -1),
                      torch.stack((low[:, 0, None].expand_as(y), y), -1),
                      torch.stack((high[:, 0, None].expand_as(y), y), -1)), dim=1)


def _virtual_camera(rays: Float64[Tensor, "n 4101 3"]) -> tuple[Float64[Tensor, "n"], Float64[Tensor, "n 3 3"]]:
    center = rays[:, 0]
    x, y, z = center.unbind(-1)
    denominator = 1.0 + z
    rotation = torch.stack((1.0 - x * x / denominator, -x * y / denominator, x,
                            -x * y / denominator, 1.0 - y * y / denominator, y, -x, -y, z), dim=-1).reshape(-1, 3, 3)
    perimeter = rays[:, 1:] @ rotation
    normalized = perimeter[..., :2] / perimeter[..., 2:]
    extent = torch.where(torch.isfinite(normalized), normalized.abs(), 0.0).amax((1, 2))
    return 128.0 / extent, rotation


_sample_perimeter_compiled = torch.compile(_sample_perimeter, fullgraph=True, dynamic=True)
_virtual_camera_compiled = torch.compile(_virtual_camera, fullgraph=True, dynamic=True)


@dataclass(frozen=True, slots=True)
class PreparedHandCrops:
    """Model crops and their virtual cameras, all on the frame's CUDA device."""

    images: Float32[Tensor, "n 3 256 256"]
    focal: Float64[Tensor, "n"]
    rotation: Float64[Tensor, "n 3 3"]


class CudaHandCropper:
    """One camera's immutable lens data shared by inversion, crops and projection."""

    def __init__(self, camera: CalibratedCamera, device: torch.device) -> None:
        model = lens_model(camera)
        self.device = device
        self.radial_count = 6 if camera.lens == "FISHEYE624" else 4 if camera.lens == "KB4" else 0
        coefficients = np.pad(camera.distortion, (0, 14 - len(camera.distortion)))
        self.intrinsic = wp.vec4d(camera.intrinsics[0, 0], camera.intrinsics[1, 1], camera.intrinsics[0, 2], camera.intrinsics[1, 2])
        self.coefficients = wp.array(coefficients, dtype=wp.float64, device=str(device))
        self.tilt = wp.vec4d(*np.r_[np.sin(coefficients[-2:]), np.cos(coefficients[-2:])])
        self.radius_bound = wp.float64(model._max_radius_squared if isinstance(model, _PerspectiveCamera) else np.inf)
        self.edge = torch.linspace(0.0, 1.0, 1025, device=device, dtype=torch.float64)

    def unproject(self, pixels: Float64[Tensor, "n p 2"]) -> Float64[Tensor, "n p 3"]:
        """Invert the physical lens on the GPU, retaining the reference pixel tolerance."""
        rays = torch.empty((*pixels.shape[:-1], 3), device=pixels.device, dtype=torch.float64)
        wp.launch(_unproject_pixels, dim=pixels.shape[:2], inputs=[wp.from_torch(pixels.contiguous(), dtype=wp.vec2d), self.intrinsic,
                  self.coefficients, self.tilt, self.radius_bound, self.radial_count], outputs=[wp.from_torch(rays, dtype=wp.vec3d)],
                  device=str(self.device), stream=wp.stream_from_torch(), record_tape=False)
        return rays

    def geometry(self, boxes: Float32[Tensor, "n 4"], padding: float) -> tuple[Float64[Tensor, "n"], Float64[Tensor, "n 3 3"]]:
        """Fit the same 4,100-point perimeter and shortest optical-axis rotation."""
        sample = _sample_perimeter if os.environ.get("PIXI_DEV_MODE") == "1" else _sample_perimeter_compiled
        virtual = _virtual_camera if os.environ.get("PIXI_DEV_MODE") == "1" else _virtual_camera_compiled
        pixels = sample(boxes, self.edge, padding)
        return virtual(self.unproject(pixels))

    def maps(self, focal: Float64[Tensor, "n"], rotation: Float64[Tensor, "n 3 3"], mirrored: Tensor) -> Float32[Tensor, "n 256 256 2"]:
        """Project every virtual pixel in one batch, mirroring right-hand crops."""
        maps = torch.empty((len(focal), 256, 256, 2), device=focal.device)
        wp.launch(_remap, dim=(len(focal), 256, 256), inputs=[wp.from_torch(focal), wp.from_torch(rotation, dtype=wp.mat33d),
                  wp.from_torch(mirrored, dtype=wp.bool), self.intrinsic, self.coefficients, self.tilt, self.radius_bound, self.radial_count],
                  outputs=[wp.from_torch(maps, dtype=wp.vec2)], device=str(self.device), stream=wp.stream_from_torch(), record_tape=False)
        return maps

    def project(self, joints: Float32[Tensor, "n 25 3"]) -> Float32[Tensor, "n 25 2"]:
        """Project source-camera joints using the same lens as the crop sampler."""
        pixels = torch.empty((*joints.shape[:-1], 2), device=joints.device)
        wp.launch(_project_joints, dim=joints.shape[:2], inputs=[wp.from_torch(joints.contiguous(), dtype=wp.vec3), self.intrinsic,
                  self.coefficients, self.tilt, self.radius_bound, self.radial_count], outputs=[wp.from_torch(pixels, dtype=wp.vec2)],
                  device=str(self.device), stream=wp.stream_from_torch(), record_tape=False)
        return pixels

    def prepare(self, frames: UInt8[Tensor, "b h w 3"], detections: HandDetections, padding: float) -> PreparedHandCrops:
        """Create normalized, anatomically mirrored crops without host geometry."""
        focal, rotation = self.geometry(detections.boxes.xyxy, padding)
        mirrored = torch.tensor([side == "right" for side in detections.sides], device=frames.device)
        maps = self.maps(focal, rotation, mirrored)
        normalize = _normalize_crops if os.environ.get("PIXI_DEV_MODE") == "1" else _compiled_normalize_crops
        images = normalize(frames, detections.boxes.frame_indices, maps)
        return PreparedHandCrops(images, focal, rotation)
