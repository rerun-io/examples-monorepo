"""Batched CUDA lens inversion, virtual cameras and hand crop sampling."""

from dataclasses import dataclass

import numpy as np
import torch
import warp as wp
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from torch import Tensor

from egohand_inference.camera.camera_models import _PerspectiveCamera
from posekit.cameras import CalibratedCamera
from posekit.models.egohand_capture import CapturedGraph, GraphPool
from posekit.models.egohand_crops import normalize_crops
from posekit.models.egohand_geometry import lens_model
from posekit.models.egohand_lens_kernels import project_joints, remap, unproject_pixels
from posekit.ops.compile import compile_unless_dev
from posekit.predictions import HandDetections


@compile_unless_dev
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


@compile_unless_dev
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


@dataclass(frozen=True, slots=True)
class PreparedHandCrops:
    """Model crops and their virtual cameras, all on the frame's CUDA device."""

    images: Float32[Tensor, "n 3 256 256"]
    """ImageNet-normalized virtual-camera crops."""
    focal: Float64[Tensor, "n"]
    """Isotropic virtual focal lengths in pixels."""
    rotation: Float64[Tensor, "n 3 3"]
    """Virtual-camera to source-camera rotations."""


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
        # Captured crops by frame shape, hand count and padding; boxes and the side sequence are replay inputs.
        self.graphs: dict[tuple[tuple[int, ...], int, float], CapturedGraph[PreparedHandCrops]] = {}
        # One static frames buffer per frame shape, shared by every hand count's graph.
        self.frame_buffers: dict[tuple[int, ...], UInt8[Tensor, "b h w 3"]] = {}
        self.pool: GraphPool = torch.cuda.graph_pool_handle()
        self.streams: dict[int, wp.Stream] = {}

    def _stream(self) -> wp.Stream:
        """Warp's wrapper of the current Torch stream: the scoped capture stream, or one retained per stream."""
        current: torch.cuda.Stream = torch.cuda.current_stream(self.device)
        scoped: wp.Stream = wp.get_stream(str(self.device))
        if scoped.cuda_stream == current.cuda_stream:
            return scoped
        if current.cuda_stream not in self.streams:
            self.streams[current.cuda_stream] = wp.stream_from_torch(current)
        return self.streams[current.cuda_stream]

    def unproject(self, pixels: Float64[Tensor, "n p 2"]) -> Float64[Tensor, "n p 3"]:
        """Invert the physical lens on the GPU, retaining the reference pixel tolerance."""
        rays = torch.empty((*pixels.shape[:-1], 3), device=pixels.device, dtype=torch.float64)
        wp.launch(unproject_pixels, dim=pixels.shape[:2], inputs=[wp.from_torch(pixels.contiguous(), dtype=wp.vec2d), self.intrinsic,
                  self.coefficients, self.tilt, self.radius_bound, self.radial_count], outputs=[wp.from_torch(rays, dtype=wp.vec3d)],
                  device=str(self.device), stream=self._stream(), record_tape=False)
        return rays

    def geometry(self, boxes: Float32[Tensor, "n 4"], padding: float) -> tuple[Float64[Tensor, "n"], Float64[Tensor, "n 3 3"]]:
        """Fit the same 4,100-point perimeter and shortest optical-axis rotation."""
        pixels = _sample_perimeter(boxes, self.edge, padding)
        return _virtual_camera(self.unproject(pixels))

    def maps(self, focal: Float64[Tensor, "n"], rotation: Float64[Tensor, "n 3 3"], mirrored: Tensor) -> Float32[Tensor, "n 256 256 2"]:
        """Project every virtual pixel in one batch, mirroring right-hand crops."""
        maps = torch.empty((len(focal), 256, 256, 2), device=focal.device)
        wp.launch(remap, dim=(len(focal), 256, 256), inputs=[wp.from_torch(focal), wp.from_torch(rotation, dtype=wp.mat33d),
                  wp.from_torch(mirrored, dtype=wp.bool), self.intrinsic, self.coefficients, self.tilt, self.radius_bound, self.radial_count],
                  outputs=[wp.from_torch(maps, dtype=wp.vec2)], device=str(self.device), stream=self._stream(), record_tape=False)
        return maps

    def project(self, joints: Float32[Tensor, "n 25 3"]) -> Float32[Tensor, "n 25 2"]:
        """Project source-camera joints using the same lens as the crop sampler."""
        pixels = torch.empty((*joints.shape[:-1], 2), device=joints.device)
        wp.launch(project_joints, dim=joints.shape[:2], inputs=[wp.from_torch(joints.contiguous(), dtype=wp.vec3), self.intrinsic,
                  self.coefficients, self.tilt, self.radius_bound, self.radial_count], outputs=[wp.from_torch(pixels, dtype=wp.vec2)],
                  device=str(self.device), stream=self._stream(), record_tape=False)
        return pixels

    def prepare(self, frames: UInt8[Tensor, "b h w 3"], detections: HandDetections, padding: float) -> PreparedHandCrops:
        """Create normalized, anatomically mirrored crops without host geometry."""
        mirrored: Bool[Tensor, "n"] = torch.tensor([side == "right" for side in detections.sides], device=frames.device)
        key: tuple[tuple[int, ...], int, float] = (tuple(frames.shape), len(detections.sides), padding)
        if key not in self.graphs:
            if key[0] not in self.frame_buffers:
                self.frame_buffers[key[0]] = frames.clone()
            inputs: tuple[Tensor, ...] = (
                self.frame_buffers[key[0]], detections.boxes.xyxy.clone(), detections.boxes.frame_indices.clone(), mirrored.clone(),
            )
            self.graphs[key] = CapturedGraph(lambda: self._prepare(inputs[0], inputs[1], inputs[2], inputs[3], padding), inputs, pool=self.pool, keep=(self.coefficients, self.edge))
        output: PreparedHandCrops = self.graphs[key].replay(frames, detections.boxes.xyxy, detections.boxes.frame_indices, mirrored)
        return PreparedHandCrops(output.images.clone(), output.focal.clone(), output.rotation.clone())

    def _prepare(
        self, frames: UInt8[Tensor, "b h w 3"], boxes: Float32[Tensor, "n 4"], frame_indices: Int64[Tensor, "n"],
        mirrored: Bool[Tensor, "n"], padding: float,
    ) -> PreparedHandCrops:
        """Capture-safe crop body with all changing inputs in device storage."""
        focal, rotation = self.geometry(boxes, padding)
        maps = self.maps(focal, rotation, mirrored)
        images = normalize_crops(frames, frame_indices, maps)
        return PreparedHandCrops(images, focal, rotation)
