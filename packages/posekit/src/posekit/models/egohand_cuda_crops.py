"""Batched CUDA lens inversion, virtual cameras and hand crop sampling."""

import os
from dataclasses import dataclass

import numpy as np
import torch
import warp as wp
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from torch import Tensor

from egohand_inference.camera.camera_models import _PerspectiveCamera
from posekit.cameras import CalibratedCamera
from posekit.models.egohand_capture import capture_hand_graph
from posekit.models.egohand_crops import _compiled_normalize_crops, _normalize_crops
from posekit.models.egohand_geometry import lens_model
from posekit.models.egohand_lens_kernels import _project_joints, _remap, _unproject_pixels
from posekit.predictions import HandDetections, HandSide


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
    """ImageNet-normalized virtual-camera crops."""
    focal: Float64[Tensor, "n"]
    """Isotropic virtual focal lengths in pixels."""
    rotation: Float64[Tensor, "n 3 3"]
    """Virtual-camera to source-camera rotations."""


@dataclass(frozen=True, slots=True)
class _CapturedCrops:
    """Resources retained until all captured Warp and Torch work is destroyed."""

    graph: torch.cuda.CUDAGraph
    """Captured crop operations."""
    frames: UInt8[Tensor, "b h w 3"]
    """Replay input image storage."""
    boxes: Float32[Tensor, "n 4"]
    """Replay input detector boxes."""
    frame_indices: Int64[Tensor, "n"]
    """Replay source frame indices."""
    mirrored: Bool[Tensor, "n"]
    """Anatomical right-hand mask retained by the remap kernel."""
    output: PreparedHandCrops
    """Captured output buffers, copied before each public return."""


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
        self.graphs: dict[tuple[tuple[int, ...], tuple[HandSide, ...], float], _CapturedCrops] = {}
        self.streams: dict[int, wp.Stream] = {}

    def __del__(self) -> None:
        """Release captured GPU work before its external Warp resources."""
        for state in getattr(self, "graphs", {}).values():
            state.graph.reset()

    def _stream(self) -> wp.Stream:
        """Retain Warp's stream wrapper while a Torch graph records its kernels."""
        current: torch.cuda.Stream = torch.cuda.current_stream(self.device)
        if current.cuda_stream not in self.streams:
            self.streams[current.cuda_stream] = wp.stream_from_torch(current)
        return self.streams[current.cuda_stream]

    def unproject(self, pixels: Float64[Tensor, "n p 2"]) -> Float64[Tensor, "n p 3"]:
        """Invert the physical lens on the GPU, retaining the reference pixel tolerance."""
        rays = torch.empty((*pixels.shape[:-1], 3), device=pixels.device, dtype=torch.float64)
        wp.launch(_unproject_pixels, dim=pixels.shape[:2], inputs=[wp.from_torch(pixels.contiguous(), dtype=wp.vec2d), self.intrinsic,
                  self.coefficients, self.tilt, self.radius_bound, self.radial_count], outputs=[wp.from_torch(rays, dtype=wp.vec3d)],
                  device=str(self.device), stream=self._stream(), record_tape=False)
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
                  outputs=[wp.from_torch(maps, dtype=wp.vec2)], device=str(self.device), stream=self._stream(), record_tape=False)
        return maps

    def project(self, joints: Float32[Tensor, "n 25 3"]) -> Float32[Tensor, "n 25 2"]:
        """Project source-camera joints using the same lens as the crop sampler."""
        pixels = torch.empty((*joints.shape[:-1], 2), device=joints.device)
        wp.launch(_project_joints, dim=joints.shape[:2], inputs=[wp.from_torch(joints.contiguous(), dtype=wp.vec3), self.intrinsic,
                  self.coefficients, self.tilt, self.radius_bound, self.radial_count], outputs=[wp.from_torch(pixels, dtype=wp.vec2)],
                  device=str(self.device), stream=self._stream(), record_tape=False)
        return pixels

    def prepare(self, frames: UInt8[Tensor, "b h w 3"], detections: HandDetections, padding: float) -> PreparedHandCrops:
        """Create normalized, anatomically mirrored crops without host geometry."""
        key: tuple[tuple[int, ...], tuple[HandSide, ...], float] = (tuple(frames.shape), detections.sides, padding)
        if key not in self.graphs:
            static_frames: UInt8[Tensor, "b h w 3"] = frames.clone()
            static_boxes: Float32[Tensor, "n 4"] = detections.boxes.xyxy.clone()
            static_indices: Int64[Tensor, "n"] = detections.boxes.frame_indices.clone()
            mirrored: Bool[Tensor, "n"] = torch.tensor([side == "right" for side in detections.sides], device=frames.device)
            stream: torch.cuda.Stream = torch.cuda.Stream()
            warp_stream: wp.Stream = wp.stream_from_torch(stream)
            self.streams[stream.cuda_stream] = warp_stream
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), wp.ScopedStream(warp_stream):
                for _ in range(3):
                    self._prepare(static_frames, static_boxes, static_indices, mirrored, padding)
            torch.cuda.current_stream().wait_stream(stream)
            graph: torch.cuda.CUDAGraph = torch.cuda.CUDAGraph()
            with capture_hand_graph(graph, stream, warp_stream):
                output: PreparedHandCrops = self._prepare(static_frames, static_boxes, static_indices, mirrored, padding)
            self.graphs[key] = _CapturedCrops(graph, static_frames, static_boxes, static_indices, mirrored, output)
        state: _CapturedCrops = self.graphs[key]
        state.frames.copy_(frames)
        state.boxes.copy_(detections.boxes.xyxy)
        state.frame_indices.copy_(detections.boxes.frame_indices)
        state.graph.replay()
        return PreparedHandCrops(state.output.images.clone(), state.output.focal.clone(), state.output.rotation.clone())

    def _prepare(
        self, frames: UInt8[Tensor, "b h w 3"], boxes: Float32[Tensor, "n 4"], frame_indices: Int64[Tensor, "n"],
        mirrored: Bool[Tensor, "n"], padding: float,
    ) -> PreparedHandCrops:
        """Capture-safe crop body with all changing inputs in device storage."""
        focal, rotation = self.geometry(boxes, padding)
        maps = self.maps(focal, rotation, mirrored)
        normalize = _normalize_crops if os.environ.get("PIXI_DEV_MODE") == "1" else _compiled_normalize_crops
        images = normalize(frames, frame_indices, maps)
        return PreparedHandCrops(images, focal, rotation)
