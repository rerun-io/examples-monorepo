"""Torch helpers for batched COCO-133 pose preprocessing and decode."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Float32, Int, UInt8
from numpy import ndarray
from torch import Tensor

from sapiens2_pose.api.metadata import get_sapiens_metainfo
from sapiens2_pose.api.runtime import DEFAULT_MODEL_SIZE, ModelSize
from sapiens2_pose.sapiens_lite.pose import MODEL_SPECS

TORCH_UINT8 = torch.__dict__["uint8"]
YoloxRawOutput = Float[Tensor, "batch anchors fields"]
POSE_INPUT_SHAPE = (3, 1024, 768)
RTMLIB_POSE_INPUT_SIZE = (192, 256)
YOLOX_DEFAULT_MAX_DETECTIONS = 1024


@dataclass(frozen=True, slots=True)
class DecodedYoloxHeadOutput:
    """Decoded YOLOX head outputs before score thresholding and NMS."""

    boxes: Float[Tensor, "batch anchors 4"]
    """Decoded image-space boxes in detector-input coordinates."""
    scores: Float[Tensor, "batch anchors classes"]
    """Class confidence scores aligned with ``boxes``."""


YoloxDetectorOutput = YoloxRawOutput | DecodedYoloxHeadOutput


@dataclass(frozen=True, slots=True)
class PoseInputBatch:
    """Batched pose inputs and bbox decode metadata."""

    inputs: Float[Tensor, "n 3 h w"]
    """Normalized pose crops ready for the selected pose model."""
    frame_indices: Int[Tensor, "n"]
    """Source frame index for each pose crop."""
    bboxes_xyxy: Float[Tensor, "n 4"]
    """Source person boxes in image-space ``xyxy`` order."""
    bbox_centers: Float[Tensor, "n 2"]
    """Image-space bbox centers used to map model outputs back to the source image."""
    bbox_scales: Float[Tensor, "n 2"]
    """Width and height of the padded crop in source-image pixels."""
    input_size: tuple[int, int]
    """Pose model crop size as ``(width, height)``."""


SapiensPoseInputBatch = PoseInputBatch
RtmlibPoseInputBatch = PoseInputBatch


def preprocess_yolox_detector_torch(frames_rgb: UInt8[Tensor, "batch h w 3"], *, model_input_size: tuple[int, int]) -> tuple[Float[Tensor, "batch 3 det_h det_w"], Float[Tensor, "batch"]]:
    """Preprocess RGB frames for the RTMLib YOLOX detector.

    The operation stays in torch so CUDA video frames can flow directly into the TensorRT detector. It matches RTMLib's
    preprocessing convention: RGB to BGR, top-left letterbox padding with value 114, and no mean/std normalization.

    Args:
        frames_rgb: RGB video frames with shape ``(batch, height, width, 3)``.
        model_input_size: Detector input size as ``(height, width)``.

    Returns:
        A tuple of detector inputs with shape ``(batch, 3, det_h, det_w)`` and per-frame resize ratios.

    Raises:
        ValueError: If ``frames_rgb`` is not a uint8 NHWC RGB tensor.
    """
    if frames_rgb.dtype != TORCH_UINT8 or frames_rgb.ndim != 4 or int(frames_rgb.shape[-1]) != 3:
        raise ValueError("YOLOX preprocessing expects uint8 RGB frames with shape (batch, height, width, 3).")
    in_h, in_w, out_h, out_w = int(frames_rgb.shape[1]), int(frames_rgb.shape[2]), int(model_input_size[0]), int(model_input_size[1])
    ratio = min(out_h / in_h, out_w / in_w)
    resized_h, resized_w = int(in_h * ratio), int(in_w * ratio)
    resized = F.interpolate(frames_rgb.flip((-1,)).permute(0, 3, 1, 2).float(), size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    detector_inputs = torch.full((int(frames_rgb.shape[0]), 3, out_h, out_w), 114.0, dtype=torch.float32, device=frames_rgb.device)
    detector_inputs[:, :, :resized_h, :resized_w] = resized
    return detector_inputs.contiguous(), torch.full((int(frames_rgb.shape[0]),), ratio, dtype=torch.float32, device=frames_rgb.device)


def prepare_sapiens_pose_inputs_torch(frames_rgb: UInt8[Tensor, "batch h w 3"], *, frame_indices: Int[Tensor, "n"], bboxes_xyxy: Float[Tensor, "n 4"], input_size: tuple[int, int] = (768, 1024), padding: float = 1.25, dtype: torch.dtype = torch.float16) -> SapiensPoseInputBatch:
    """Generate batched Sapiens pose crops from GPU frames and person boxes.

    Args:
        frames_rgb: RGB video frames with shape ``(batch, height, width, 3)``.
        frame_indices: Source frame index for each person crop.
        bboxes_xyxy: Person boxes in image-space ``xyxy`` order.
        input_size: Sapiens crop size as ``(width, height)``.
        padding: Bbox padding multiplier used before cropping.
        dtype: Output tensor dtype for TensorRT/PyTorch pose inference.

    Returns:
        Pose input batch with normalized crops and decode metadata.
    """
    return _prepare_pose_inputs_torch(frames_rgb, frame_indices=frame_indices, bboxes_xyxy=bboxes_xyxy, input_size=input_size, padding=padding, dtype=dtype, batch_type=SapiensPoseInputBatch, cv2_affine=False, bgr_input=False)


def prepare_rtmlib_pose_inputs_torch(frames_rgb: UInt8[Tensor, "batch h w 3"], *, frame_indices: Int[Tensor, "n"], bboxes_xyxy: Float[Tensor, "n 4"], input_size: tuple[int, int] = RTMLIB_POSE_INPUT_SIZE, padding: float = 1.25, dtype: torch.dtype = torch.float16) -> RtmlibPoseInputBatch:
    """Generate batched RTMLib pose crops from GPU RGB frames.

    Args:
        frames_rgb: RGB video frames with shape ``(batch, height, width, 3)``.
        frame_indices: Source frame index for each person crop.
        bboxes_xyxy: Person boxes in image-space ``xyxy`` order.
        input_size: RTMLib crop size as ``(width, height)``.
        padding: Bbox padding multiplier used before cropping.
        dtype: Output tensor dtype for TensorRT/PyTorch pose inference.

    Returns:
        Pose input batch with BGR-normalized crops and decode metadata.
    """
    return _prepare_pose_inputs_torch(frames_rgb, frame_indices=frame_indices, bboxes_xyxy=bboxes_xyxy, input_size=input_size, padding=padding, dtype=dtype, batch_type=RtmlibPoseInputBatch, cv2_affine=True, bgr_input=True)


def _prepare_pose_inputs_torch(frames_rgb: UInt8[Tensor, "batch h w 3"], *, frame_indices: Int[Tensor, "n"], bboxes_xyxy: Float[Tensor, "n 4"], input_size: tuple[int, int], padding: float, dtype: torch.dtype, batch_type: type[PoseInputBatch], cv2_affine: bool, bgr_input: bool) -> PoseInputBatch:
    if frames_rgb.dtype != TORCH_UINT8 or frames_rgb.ndim != 4 or int(frames_rgb.shape[-1]) != 3:
        raise ValueError("Pose crop preprocessing expects uint8 RGB frames with shape (batch, height, width, 3).")
    frame_indices = frame_indices.to(device=frames_rgb.device, dtype=torch.long).reshape(-1)
    bboxes = bboxes_xyxy.to(device=frames_rgb.device, dtype=torch.float32).reshape(-1, 4)
    out_w, out_h = int(input_size[0]), int(input_size[1])
    if int(bboxes.shape[0]) == 0:
        return batch_type(torch.empty((0, 3, out_h, out_w), dtype=dtype, device=frames_rgb.device), frame_indices, bboxes, torch.empty((0, 2), dtype=torch.float32, device=frames_rgb.device), torch.empty((0, 2), dtype=torch.float32, device=frames_rgb.device), input_size)
    centers, scales = _bbox_xyxy_to_sapiens_center_scale_torch(bboxes, input_size=input_size, padding=padding)
    yy, xx = _sapiens_crop_meshgrid_torch(out_h, out_w, centers.device, centers.dtype)
    src_x = (xx[None] - float(out_w) * 0.5) * scales[:, 0, None, None] / float(out_w) + centers[:, 0, None, None] if cv2_affine else xx[None] * scales[:, 0, None, None] / float(out_w - 1) + centers[:, 0, None, None] - 0.5 * scales[:, 0, None, None]
    src_y = (yy[None] - float(out_h) * 0.5) * scales[:, 1, None, None] / float(out_h) + centers[:, 1, None, None] if cv2_affine else yy[None] * scales[:, 1, None, None] / float(out_h - 1) + centers[:, 1, None, None] - 0.5 * scales[:, 1, None, None]
    grid = torch.stack([src_x / float(int(frames_rgb.shape[2]) - 1) * 2.0 - 1.0, src_y / float(int(frames_rgb.shape[1]) - 1) * 2.0 - 1.0], dim=-1).contiguous()
    with torch.backends.cudnn.flags(enabled=False):
        crops = F.grid_sample((frames_rgb[frame_indices].flip((-1,)) if bgr_input else frames_rgb[frame_indices]).permute(0, 3, 1, 2).contiguous().float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    mean, std = _sapiens_normalization_tensors_torch(frames_rgb.device)
    return batch_type(((crops - mean) / std).to(dtype=dtype).contiguous(), frame_indices, bboxes, centers, scales, input_size)


def _bbox_xyxy_to_sapiens_center_scale_torch(bboxes_xyxy: Float[Tensor, "n 4"], *, input_size: tuple[int, int], padding: float) -> tuple[Float[Tensor, "n 2"], Float[Tensor, "n 2"]]:
    centers = (bboxes_xyxy[:, 0:2] + bboxes_xyxy[:, 2:4]) * 0.5
    scales = (bboxes_xyxy[:, 2:4] - bboxes_xyxy[:, 0:2]) * float(padding)
    aspect, widths, heights = float(input_size[0]) / float(input_size[1]), scales[:, 0], scales[:, 1]
    return centers.contiguous(), torch.where((widths > heights * aspect)[:, None], torch.stack([widths, widths / aspect], dim=1), torch.stack([heights * aspect, heights], dim=1)).contiguous()


@lru_cache(maxsize=8)
def _sapiens_normalization_tensors_torch(device: torch.device) -> tuple[Float[Tensor, "1 3 1 1"], Float[Tensor, "1 3 1 1"]]:
    return torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32, device=device).view(1, 3, 1, 1), torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32, device=device).view(1, 3, 1, 1)


@lru_cache(maxsize=8)
def _sapiens_crop_meshgrid_torch(output_height: int, output_width: int, device: torch.device, dtype: torch.dtype) -> tuple[Float[Tensor, "out_h out_w"], Float[Tensor, "out_h out_w"]]:
    grid_y: Float[Tensor, "out_h out_w"]
    grid_x: Float[Tensor, "out_h out_w"]
    grid_y, grid_x = torch.meshgrid(torch.linspace(0.0, float(output_height - 1), output_height, dtype=dtype, device=device), torch.linspace(0.0, float(output_width - 1), output_width, dtype=dtype, device=device), indexing="ij")
    return grid_y, grid_x


def decode_sapiens_heatmaps_to_coco133_torch(heatmaps: Tensor, batch: SapiensPoseInputBatch, *, model_size: ModelSize = DEFAULT_MODEL_SIZE) -> tuple[Float32[ndarray, "n 133 2"], Float32[ndarray, "n 133"]]:
    """Decode Sapiens heatmaps into image-space COCO-133 keypoints.

    Args:
        heatmaps: Sapiens heatmaps with shape ``(n, 308, heatmap_h, heatmap_w)``.
        batch: Sapiens pose input batch containing crop centers and scales.
        model_size: Sapiens model size used to select heatmap/input metadata.

    Returns:
        A tuple containing image-space COCO-133 keypoints with shape ``(n, 133, 2)`` and confidence scores with shape
        ``(n, 133)``.

    Raises:
        ValueError: If ``heatmaps`` does not have the expected Sapiens channel layout.
    """
    if heatmaps.ndim != 4 or int(heatmaps.shape[1]) != 308:
        raise ValueError(f"Expected Sapiens heatmaps with shape (n, 308, h, w), got {tuple(heatmaps.shape)}.")
    spec, device, num_instances = MODEL_SPECS[model_size], heatmaps.device, int(heatmaps.shape[0])
    coco_indices, sapiens_indices = _coco133_projection_index_tensors(device)
    keypoints_input, mapped_scores = _decode_udp_heatmaps_torch(heatmaps[:, sapiens_indices].float(), input_size=(int(spec.input_size[0]), int(spec.input_size[1])), heatmap_size=(int(spec.heatmap_size[0]), int(spec.heatmap_size[1])), blur_kernel_size=11)
    centers, scales = batch.bbox_centers.to(device=device, dtype=torch.float32), batch.bbox_scales.to(device=device, dtype=torch.float32)
    keypoints_image = keypoints_input / _float32_vector_torch(batch.input_size, device) * scales[:, None, :] + centers[:, None, :] - 0.5 * scales[:, None, :]
    keypoints_coco = torch.full((num_instances, 133, 2), float("nan"), dtype=torch.float32, device=device)
    scores_coco = torch.zeros((num_instances, 133), dtype=torch.float32, device=device)
    keypoints_coco[:, coco_indices], scores_coco[:, coco_indices] = keypoints_image, mapped_scores
    return keypoints_coco.cpu().numpy().astype(np.float32, copy=False), scores_coco.cpu().numpy().astype(np.float32, copy=False)


def decode_rtmlib_simcc_to_coco133_torch(simcc_x: Tensor, simcc_y: Tensor, batch: RtmlibPoseInputBatch) -> tuple[Float32[ndarray, "n 133 2"], Float32[ndarray, "n 133"]]:
    """Decode RTMLib RTMW SimCC outputs into image-space COCO-133 keypoints.

    Args:
        simcc_x: RTMW x-axis SimCC logits with shape ``(n, 133, bins_x)``.
        simcc_y: RTMW y-axis SimCC logits with shape ``(n, 133, bins_y)``.
        batch: RTMLib pose input batch containing crop centers and scales.

    Returns:
        A tuple containing image-space COCO-133 keypoints with shape ``(n, 133, 2)`` and confidence scores with shape
        ``(n, 133)``.

    Raises:
        ValueError: If either SimCC tensor does not have the expected COCO-133 layout.
    """
    if simcc_x.ndim != 3 or simcc_y.ndim != 3 or int(simcc_x.shape[1]) != 133 or int(simcc_y.shape[1]) != 133:
        raise ValueError(f"Expected RTMLib SimCC outputs with shape (n, 133, bins), got {tuple(simcc_x.shape)} and {tuple(simcc_y.shape)}.")
    x_scores, x_locs = simcc_x.float().max(dim=-1)
    y_scores, y_locs = simcc_y.float().max(dim=-1)
    scores = 0.5 * (x_scores + y_scores)
    locs = torch.where((scores > 0.0)[:, :, None], torch.stack([x_locs.float(), y_locs.float()], dim=-1), torch.full((*scores.shape, 2), -1.0, dtype=torch.float32, device=simcc_x.device))
    centers, scales = batch.bbox_centers.to(device=simcc_x.device, dtype=torch.float32), batch.bbox_scales.to(device=simcc_x.device, dtype=torch.float32)
    keypoints_image = (locs / 2.0) / _float32_vector_torch(batch.input_size, simcc_x.device) * scales[:, None, :] + centers[:, None, :] - 0.5 * scales[:, None, :]
    return keypoints_image.cpu().numpy().astype(np.float32, copy=False), scores.cpu().numpy().astype(np.float32, copy=False)


def _decode_udp_heatmaps_torch(heatmaps: Float[Tensor, "n k h w"], *, input_size: tuple[int, int], heatmap_size: tuple[int, int], blur_kernel_size: int) -> tuple[Float[Tensor, "n k 2"], Float[Tensor, "n k"]]:
    num_instances, num_keypoints, heatmap_height, heatmap_width = int(heatmaps.shape[0]), int(heatmaps.shape[1]), int(heatmaps.shape[2]), int(heatmaps.shape[3])
    scores, flat_indices = heatmaps.reshape(num_instances, num_keypoints, -1).max(dim=-1)
    keypoints = torch.stack([(flat_indices % heatmap_width).float(), torch.div(flat_indices, heatmap_width, rounding_mode="floor").float()], dim=-1)
    keypoints = torch.where((scores > 0.0)[:, :, None], keypoints, torch.full_like(keypoints, -1.0))
    original_max = heatmaps.reshape(num_instances * num_keypoints, -1).amax(dim=1).clamp_min(1e-12)
    blurred = _gaussian_blur_heatmaps_torch(heatmaps, blur_kernel_size)
    blurred = blurred * (original_max / blurred.reshape(num_instances * num_keypoints, -1).amax(dim=1).clamp_min(1e-12)).reshape(num_instances, num_keypoints, 1, 1)
    padded = F.pad(blurred.clamp(1e-3, 50.0).log().reshape(num_instances * num_keypoints, 1, heatmap_height, heatmap_width), (1, 1, 1, 1), mode="replicate").reshape(num_instances, num_keypoints, heatmap_height + 2, heatmap_width + 2)
    x_idx, y_idx = keypoints[..., 0].round().long().clamp(0, heatmap_width - 1) + 1, keypoints[..., 1].round().long().clamp(0, heatmap_height - 1) + 1
    instance_idx, keypoint_idx = _long_arange_torch(num_instances, heatmaps.device)[:, None], _long_arange_torch(num_keypoints, heatmaps.device)[None, :]
    center, ix1, iy1 = padded[instance_idx, keypoint_idx, y_idx, x_idx], padded[instance_idx, keypoint_idx, y_idx, x_idx + 1], padded[instance_idx, keypoint_idx, y_idx + 1, x_idx]
    ix1y1, ix1_y1_, ix1_, iy1_ = padded[instance_idx, keypoint_idx, y_idx + 1, x_idx + 1], padded[instance_idx, keypoint_idx, y_idx - 1, x_idx - 1], padded[instance_idx, keypoint_idx, y_idx, x_idx - 1], padded[instance_idx, keypoint_idx, y_idx - 1, x_idx]
    derivative = torch.stack([0.5 * (ix1 - ix1_), 0.5 * (iy1 - iy1_)], dim=-1).unsqueeze(-1)
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + center + center - ix1_ - iy1_ + ix1_y1_)
    hessian = torch.stack([ix1 - 2.0 * center + ix1_, dxy, dxy, iy1 - 2.0 * center + iy1_], dim=-1).reshape(num_instances, num_keypoints, 2, 2)
    offsets = torch.linalg.solve(hessian + torch.finfo(torch.float32).eps * _eye2_torch(heatmaps.device), derivative).squeeze(-1)
    refined = torch.where(torch.isfinite(offsets).all(dim=-1, keepdim=True), keypoints - offsets, keypoints)
    return refined * _float32_vector_torch((float(input_size[0]) / float(heatmap_size[0] - 1), float(input_size[1]) / float(heatmap_size[1] - 1)), heatmaps.device), scores


def _gaussian_blur_heatmaps_torch(heatmaps: Float[Tensor, "n k h w"], kernel_size: int) -> Float[Tensor, "n k h w"]:
    border, n, k, h, w = (kernel_size - 1) // 2, int(heatmaps.shape[0]), int(heatmaps.shape[1]), int(heatmaps.shape[2]), int(heatmaps.shape[3])
    kernel = _gaussian_kernel1d_torch(kernel_size, device=heatmaps.device)
    padded = F.pad(heatmaps.reshape(n * k, 1, h, w), (border, border, border, border), mode="constant", value=0.0)
    return F.conv2d(F.conv2d(padded, kernel.reshape(1, 1, 1, kernel_size), padding=(0, border)), kernel.reshape(1, 1, kernel_size, 1), padding=(border, 0))[:, :, border:-border, border:-border].reshape(n, k, h, w)


@lru_cache(maxsize=8)
def _gaussian_kernel1d_np(kernel_size: int) -> Float32[ndarray, "kernel"]:
    import cv2

    return cv2.getGaussianKernel(kernel_size, 0).astype(np.float32).reshape(kernel_size)


@lru_cache(maxsize=16)
def _gaussian_kernel1d_torch(kernel_size: int, *, device: torch.device) -> Float[Tensor, "kernel"]:
    return torch.as_tensor(_gaussian_kernel1d_np(kernel_size), dtype=torch.float32, device=device)


@lru_cache(maxsize=1)
def _coco133_projection_indices_np() -> tuple[Int[ndarray, "mapped"], Int[ndarray, "mapped"]]:
    mapping = {int(k): int(v) for k, v in get_sapiens_metainfo()["coco_wholebody_to_goliath_mapping"].items()}
    pairs = sorted((coco_idx, sapiens_idx) for coco_idx, sapiens_idx in mapping.items() if 0 <= coco_idx < 133)
    return np.asarray([p[0] for p in pairs], dtype=np.int64), np.asarray([p[1] for p in pairs], dtype=np.int64)


@lru_cache(maxsize=8)
def _coco133_projection_index_tensors(device: torch.device) -> tuple[Tensor, Tensor]:
    coco_indices, sapiens_indices = _coco133_projection_indices_np()
    return torch.as_tensor(coco_indices, dtype=torch.long, device=device), torch.as_tensor(sapiens_indices, dtype=torch.long, device=device)


@lru_cache(maxsize=16)
def _float32_vector_torch(values: tuple[float, ...] | tuple[int, ...], device: torch.device) -> Float[Tensor, "n"]:
    return torch.tensor(tuple(float(v) for v in values), dtype=torch.float32, device=device)


@lru_cache(maxsize=16)
def _long_arange_torch(length: int, device: torch.device) -> Tensor:
    return torch.arange(length, dtype=torch.long, device=device)


@lru_cache(maxsize=8)
def _eye2_torch(device: torch.device) -> Float[Tensor, "1 1 2 2"]:
    return torch.eye(2, dtype=torch.float32, device=device).reshape(1, 1, 2, 2)


class TensorRtYoloxDetectorRunner:
    """Callable wrapper for a static-batch RTMLib YOLOX TensorRT engine."""

    def __init__(self, engine_path: Path, *, input_name: str, output_name: str, score_output_name: str | None = None, extra_output_names: tuple[str, ...] = (), max_detections: int = YOLOX_DEFAULT_MAX_DETECTIONS, model_input_size: tuple[int, int], static_batch_size: int) -> None:
        """Initialize a static-batch YOLOX TensorRT runner.

        Args:
            engine_path: Path to the TensorRT detector engine.
            input_name: TensorRT input tensor name.
            output_name: TensorRT decoded-box or raw-output tensor name.
            score_output_name: Optional decoded-score tensor name.
            extra_output_names: Additional output tensor names to bind.
            max_detections: Maximum decoded detections expected from the engine. Kept for API symmetry.
            model_input_size: Detector input size as ``(height, width)``.
            static_batch_size: Static batch size baked into the engine.
        """
        from wilor_nano.api.tensorrt_runtime import _StaticTensorRtRunner

        del max_detections
        self._runner = _StaticTensorRtRunner(engine_path, input_name=input_name, output_names=(output_name, *(() if score_output_name is None else (score_output_name,)), *extra_output_names), static_input_shape=(3, int(model_input_size[0]), int(model_input_size[1])), static_batch_size=static_batch_size)
        self._output_name, self._score_output_name = output_name, score_output_name

    def __call__(self, inputs: Float[Tensor, "batch 3 det_h det_w"]) -> YoloxDetectorOutput:
        """Run YOLOX TensorRT inference.

        Args:
            inputs: CUDA float32 detector inputs with shape ``(batch, 3, det_h, det_w)``.

        Returns:
            Raw YOLOX output or decoded boxes and scores, depending on the engine outputs.

        Raises:
            ValueError: If ``inputs`` is not CUDA float32.
        """
        if inputs.device.type != "cuda" or inputs.dtype != torch.float32:
            raise ValueError("YOLOX TensorRT expects CUDA float32 NCHW inputs.")
        batch_size, outputs = self._runner.run(inputs)
        return DecodedYoloxHeadOutput(outputs[self._output_name][:batch_size], outputs[self._score_output_name][:batch_size]) if self._score_output_name is not None else outputs[self._output_name][:batch_size]


class TensorRtPoseRunner:
    """Callable wrapper for a static-batch pose TensorRT engine."""

    def __init__(self, engine_path: Path, *, input_name: str, output_names: tuple[str, ...], static_input_shape: tuple[int, int, int], static_batch_size: int, convert_input_dtype: bool) -> None:
        """Initialize a static-batch pose TensorRT runner.

        Args:
            engine_path: Path to the TensorRT pose engine.
            input_name: TensorRT input tensor name.
            output_names: TensorRT output tensor names.
            static_input_shape: Static input shape without batch as ``(channels, height, width)``.
            static_batch_size: Static batch size baked into the engine.
            convert_input_dtype: Whether to cast incoming tensors to the engine input dtype.
        """
        from wilor_nano.api.tensorrt_runtime import _StaticTensorRtRunner

        self._runner = _StaticTensorRtRunner(engine_path, input_name=input_name, output_names=output_names, static_input_shape=static_input_shape, static_batch_size=static_batch_size)
        self._output_names, self._convert_input_dtype = output_names, convert_input_dtype
        input_dtype: Any = self._runner._engine.get_tensor_dtype(self._runner._input_name)
        self._input_dtype: torch.dtype = torch.float16 if input_dtype == self._runner._trt.DataType.HALF else torch.float32

    def __call__(self, inputs: Float[Tensor, "batch 3 h w"]) -> tuple[Tensor, ...]:
        """Run pose TensorRT inference.

        Args:
            inputs: CUDA NCHW pose inputs.

        Returns:
            Output tensors sliced back to the unpadded runtime batch size.

        Raises:
            ValueError: If device or dtype does not match the engine requirements.
        """
        if inputs.device.type != "cuda" or inputs.dtype not in (torch.float16, torch.float32):
            raise ValueError("Pose TensorRT expects CUDA float16 or float32 NCHW inputs.")
        if inputs.dtype != self._input_dtype and self._convert_input_dtype:
            inputs = inputs.to(dtype=self._input_dtype)
        elif inputs.dtype != self._input_dtype:
            raise ValueError(f"Pose TensorRT expects {self._input_dtype} inputs.")
        batch_size, outputs = self._runner.run(inputs)
        return tuple(outputs[name][:batch_size] for name in self._output_names)


def TensorRtSapiensPoseRunner(engine_path: Path, *, input_name: str, output_name: str, static_batch_size: int) -> TensorRtPoseRunner:
    """Create a Sapiens heatmap TensorRT runner.

    Args:
        engine_path: Path to the Sapiens TensorRT engine.
        input_name: TensorRT input tensor name.
        output_name: TensorRT heatmap output tensor name.
        static_batch_size: Static batch size baked into the engine.

    Returns:
        Configured static-batch pose runner.
    """
    return TensorRtPoseRunner(engine_path, input_name=input_name, output_names=(output_name,), static_input_shape=POSE_INPUT_SHAPE, static_batch_size=static_batch_size, convert_input_dtype=False)


def TensorRtRtmlibPoseRunner(engine_path: Path, *, input_name: str, simcc_x_output_name: str, simcc_y_output_name: str, model_input_size: tuple[int, int], static_batch_size: int) -> TensorRtPoseRunner:
    """Create an RTMLib RTMW SimCC TensorRT runner.

    Args:
        engine_path: Path to the RTMW TensorRT engine.
        input_name: TensorRT input tensor name.
        simcc_x_output_name: TensorRT SimCC-x output tensor name.
        simcc_y_output_name: TensorRT SimCC-y output tensor name.
        model_input_size: RTMW pose input size as ``(width, height)``.
        static_batch_size: Static batch size baked into the engine.

    Returns:
        Configured static-batch pose runner.
    """
    return TensorRtPoseRunner(engine_path, input_name=input_name, output_names=(simcc_x_output_name, simcc_y_output_name), static_input_shape=(3, int(model_input_size[1]), int(model_input_size[0])), static_batch_size=static_batch_size, convert_input_dtype=True)


def postprocess_yolox_outputs_torch(outputs: YoloxDetectorOutput, *, resize_ratios: Float[Tensor, "batch"], model_input_size: tuple[int, int], score_thr: float = 0.3, nms_thr: float = 0.45) -> list[Float[Tensor, "n 4"]]:
    """Postprocess batched YOLOX outputs into per-frame person boxes.

    Args:
        outputs: Raw or decoded YOLOX TensorRT outputs.
        resize_ratios: Per-frame letterbox resize ratios returned by preprocessing.
        model_input_size: Detector input size as ``(height, width)``.
        score_thr: Minimum person score.
        nms_thr: Non-maximum suppression threshold.

    Returns:
        One image-space ``xyxy`` box tensor per input frame.
    """
    from torchvision.ops import nms

    boxes_xyxy, person_scores = _yolox_boxes_and_scores(outputs, model_input_size=model_input_size)
    frame_boxes = []
    for idx in range(int(boxes_xyxy.shape[0])):
        boxes, scores = boxes_xyxy[idx] / resize_ratios[idx], person_scores[idx]
        mask = scores > float(score_thr)
        boxes, scores = boxes[mask], scores[mask]
        frame_boxes.append(torch.empty((0, 4), dtype=torch.float32, device=boxes_xyxy.device) if int(boxes.shape[0]) == 0 else boxes[nms(boxes, scores, float(nms_thr))].contiguous())
    return frame_boxes


def _yolox_boxes_and_scores(outputs: YoloxDetectorOutput, *, model_input_size: tuple[int, int]) -> tuple[Tensor, Tensor]:
    if isinstance(outputs, DecodedYoloxHeadOutput):
        if outputs.boxes.ndim != 3 or outputs.scores.ndim != 3 or int(outputs.boxes.shape[-1]) != 4 or int(outputs.scores.shape[-1]) < 1:
            raise ValueError(f"Expected decoded YOLOX boxes (batch, anchors, 4) and scores (batch, anchors, classes), got {tuple(outputs.boxes.shape)} and {tuple(outputs.scores.shape)}.")
        return outputs.boxes.float(), outputs.scores[..., 0].float()
    if outputs.ndim != 3 or int(outputs.shape[-1]) < 5:
        raise ValueError(f"Expected YOLOX output shape (batch, anchors, fields>=5), got {tuple(outputs.shape)}.")
    if int(outputs.shape[-1]) == 5:
        return outputs[..., :4].float(), outputs[..., 4].float()
    grid, stride = _yolox_grid_stride(int(model_input_size[0]), int(model_input_size[1]), outputs.device)
    pred = outputs.float().clone()
    pred[..., :2], pred[..., 2:4] = (pred[..., :2] + grid) * stride, torch.exp(pred[..., 2:4]) * stride
    boxes = torch.empty_like(pred[..., :4])
    boxes[..., 0:2], boxes[..., 2:4] = pred[..., :2] - pred[..., 2:4] * 0.5, pred[..., :2] + pred[..., 2:4] * 0.5
    return boxes, (pred[..., 4:5] * pred[..., 5:])[..., 0]


@lru_cache(maxsize=8)
def _yolox_grid_stride(height: int, width: int, device: torch.device) -> tuple[Tensor, Tensor]:
    grids, strides = [], []
    for stride in (8, 16, 32):
        yy, xx = torch.meshgrid(torch.arange(height // stride, dtype=torch.float32, device=device), torch.arange(width // stride, dtype=torch.float32, device=device), indexing="ij")
        grid = torch.stack((xx, yy), dim=2).reshape(1, -1, 2)
        grids.append(grid)
        strides.append(torch.full((*grid.shape[:2], 1), float(stride), dtype=torch.float32, device=device))
    return torch.cat(grids, dim=1), torch.cat(strides, dim=1)
