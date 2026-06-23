"""GPU-only TensorRT WiLor video pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int, UInt8
from numpy import ndarray
from torch import Tensor

from wilor_nano.api.tensorrt_runtime import (
    FULL_WILOR_OUTPUT_NAMES,
    TensorRtFullWilorRunner,
    TensorRtRawDetectorRunner,
    WiLorOutput,
)
from wilor_nano.torch_image_patch import generate_rgb_image_patches_torch
from wilor_nano.utils import utils

TORCH_UINT8: torch.dtype = torch.__dict__["uint8"]
RgbFrames = UInt8[Tensor, "batch h w 3"]


class WilorPreds(TypedDict):
    pred_cam: Float[ndarray, "1 3"]
    global_orient: Float[ndarray, "1 1 3"]
    hand_pose: Float[ndarray, "1 15 3"]
    betas: Float[ndarray, "1 10"]
    pred_keypoints_3d: Float[ndarray, "1 21 3"]
    pred_vertices: Float[ndarray, "1 778 3"]
    pred_cam_t_full: Float[ndarray, "1 3"]
    scaled_focal_length: float
    pred_keypoints_2d: Float[ndarray, "1 21 2"]


class Detection(TypedDict, total=False):
    hand_bbox: list[float]
    is_right: int
    wilor_preds: NotRequired[WilorPreds]


@dataclass(slots=True)
class DetectionExtraction:
    """Detector outputs split into Python logging records and tensor crop inputs."""

    detections: list[Detection]
    """Mutable per-frame output records returned by the public API."""
    boxes_xyxy: Float[Tensor, "n 4"]
    """Scaled full-image detector boxes in xyxy pixel coordinates."""
    is_right_tensor: Int[Tensor, "n"]
    """Detector hand labels, where 1 means right hand and 0 means left hand."""


@dataclass(slots=True)
class _FrameRecord:
    detections: list[Detection]
    center: Float[ndarray, "n 2"]
    scale: Float[ndarray, "n 2"]
    img_size: Int[ndarray, "2"]
    is_rights: list[int]
    patch_start: int
    patch_end: int


@dataclass
class WilorPipelineConfig:
    detector_engine_path: Path
    """Machine-local raw detector TensorRT engine."""
    wilor_engine_path: Path
    """Machine-local full post-crop WiLor TensorRT engine."""
    detector_static_batch_size: int
    """Static batch size baked into the detector engine."""
    wilor_static_batch_size: int
    """Static batch size baked into the full WiLor engine."""
    device: torch.device
    """CUDA device used for TensorRT execution."""
    dtype: torch.dtype = torch.float16
    """Input dtype for full WiLor TensorRT execution."""
    focal_length: float = 5000.0
    """Focal length used by WiLor camera conversion."""
    detector_max_det: int = 300
    """Maximum detector boxes retained per frame."""


class WiLorHandPose3dEstimationPipeline:
    """Batched CUDA/TensorRT video pipeline kept separate from the original PyTorch path."""

    IMAGE_SIZE: int = 256
    DETECTOR_SHAPE: tuple[int, int] = (512, 416)

    def __init__(
        self,
        config: WilorPipelineConfig,
        *,
        detector_runner_factory: Callable[..., TensorRtRawDetectorRunner] = TensorRtRawDetectorRunner,
        wilor_runner_factory: Callable[..., TensorRtFullWilorRunner] = TensorRtFullWilorRunner,
    ) -> None:
        """Create the TensorRT detector and full-WiLor runners.

        Args:
            config: TensorRT engine paths, static batch sizes, CUDA device, and
                model constants for the optimized video path.
            detector_runner_factory: Factory used to create the raw detector
                runner. Tests pass fakes here to avoid loading TensorRT engines.
            wilor_runner_factory: Factory used to create the full WiLor runner.
                Tests pass fakes here to avoid loading TensorRT engines.

        Raises:
            ValueError: If the requested pipeline device/dtype is not CUDA
                float16, which is the only supported crop input contract for
                this TensorRT path.
        """
        if config.device.type != "cuda" or config.dtype != torch.float16:
            raise ValueError("The TensorRT WiLor pipeline requires CUDA float16 crop tensors.")
        self.device: torch.device = config.device
        self.dtype: torch.dtype = config.dtype
        self.focal_length: float = config.focal_length
        self.detector_max_det: int = config.detector_max_det
        self.detector = detector_runner_factory(
            config.detector_engine_path, static_batch_size=config.detector_static_batch_size, device="cuda"
        )
        self.wilor = wilor_runner_factory(
            config.wilor_engine_path, static_batch_size=config.wilor_static_batch_size, device="cuda"
        )

    @torch.no_grad()
    def predict_batch_rgb_tensor(
        self,
        frames_rgb: RgbFrames,
        *,
        hand_conf: float = 0.3,
        rescale_factor: float = 2.5,
        detector_batch_size: int,
        wilor_batch_size: int,
    ) -> list[list[Detection]]:
        """Run batched hand detection, crop generation, and WiLor pose inference.

        Args:
            frames_rgb: CUDA RGB frames as a ``UInt8[Tensor, "batch h w 3"]``
                tensor in NHWC layout.
            hand_conf: Detector confidence threshold passed to Ultralytics NMS.
            rescale_factor: Multiplier applied to detector box width/height
                before generating square WiLor crops.
            detector_batch_size: Number of frames sent to the detector runner at
                a time. Must be no larger than the detector engine's static
                batch size.
            wilor_batch_size: Number of hand crops sent to the full-WiLor runner
                at a time. Must be no larger than the WiLor engine's static
                batch size.

        Returns:
            Frame-major detections. The outer list has one item per input frame;
            each detection contains ``hand_bbox``, ``is_right``, and, when a hand
            was detected, ``wilor_preds`` NumPy outputs.

        Raises:
            ValueError: If frames are not CUDA ``uint8`` RGB tensors or either
                runtime batch size is non-positive.
        """
        if int(frames_rgb.shape[0]) == 0:
            return []
        if frames_rgb.device.type != self.device.type or frames_rgb.dtype != TORCH_UINT8:
            raise ValueError("TRT video path expects CUDA uint8 RGB frames.")
        if detector_batch_size <= 0 or wilor_batch_size <= 0:
            raise ValueError("Batch sizes must be positive.")

        image_hw: tuple[int, int] = (int(frames_rgb.shape[1]), int(frames_rgb.shape[2]))
        detections: list[DetectionExtraction] = []
        for start in range(0, int(frames_rgb.shape[0]), detector_batch_size):
            frame_chunk: RgbFrames = frames_rgb[start : start + detector_batch_size]
            raw_predictions: Tensor = self.detector(self._preprocess_detector(frame_chunk))
            detections.extend(self._postprocess_detector(raw_predictions, image_hw, hand_conf=hand_conf))

        records: list[_FrameRecord]
        patches: Float[Tensor, "n 256 256 3"] | None
        records, patches = self._crop_frames(frames_rgb, detections, rescale_factor=rescale_factor)
        if patches is not None and int(patches.shape[0]) > 0:
            self._attach_wilor_outputs(records, self._run_wilor(patches, wilor_batch_size=wilor_batch_size))
        return [record.detections for record in records]

    def _preprocess_detector(self, frames_rgb: RgbFrames) -> Float[Tensor, "batch channels height width"]:
        """Resize and letterbox RGB frames into the detector's static input shape.

        Args:
            frames_rgb: RGB frames as ``UInt8[Tensor, "batch h w 3"]`` in NHWC
                layout.

        Returns:
            BGR detector input as ``Float[Tensor, "batch 3 512 416"]`` with
            values normalized to ``0..1`` and padded with the YOLO-style
            ``114 / 255`` fill value.
        """
        input_height: int = int(frames_rgb.shape[1])
        input_width: int = int(frames_rgb.shape[2])
        output_height: int = self.DETECTOR_SHAPE[0]
        output_width: int = self.DETECTOR_SHAPE[1]
        resize_ratio: float = min(output_height / input_height, output_width / input_width)
        resized_width: int = round(input_width * resize_ratio)
        resized_height: int = round(input_height * resize_ratio)
        pad_width: int = output_width - resized_width
        pad_height: int = output_height - resized_height
        left: int = round(pad_width / 2.0 - 0.1)
        right: int = round(pad_width / 2.0 + 0.1)
        top: int = round(pad_height / 2.0 - 0.1)
        bottom: int = round(pad_height / 2.0 + 0.1)
        frames_bgr: Float[Tensor, "batch 3 h w"] = frames_rgb.flip((-1,)).permute(0, 3, 1, 2).float() / 255.0
        resized: Float[Tensor, "batch channels resized_h resized_w"] = F.interpolate(
            frames_bgr,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )
        return F.pad(resized, (left, right, top, bottom), value=114.0 / 255.0).contiguous()

    def _postprocess_detector(
        self,
        raw_predictions: Tensor,
        image_hw: tuple[int, int],
        *,
        hand_conf: float,
    ) -> list[DetectionExtraction]:
        """Apply detector NMS and keep both logging records and tensor boxes.

        Args:
            raw_predictions: Raw detector output tensor from TensorRT. The tensor
                is expected to stay on CUDA until NMS/scaling finishes.
            image_hw: Original image size as ``(height, width)`` so detector
                boxes can be scaled back out of the letterboxed detector space.
            hand_conf: Confidence threshold passed to Ultralytics NMS.

        Returns:
            One ``DetectionExtraction`` per input frame. Each extraction carries
            Python ``Detection`` records for final output plus
            ``Float[Tensor, "n 4"]`` boxes and ``Int[Tensor, "n"]`` handedness
            labels for the GPU crop path.
        """
        from ultralytics.utils import nms, ops

        predictions: list[Tensor] = cast(
            list[Tensor],
            nms.non_max_suppression(
                raw_predictions.float(), hand_conf, 0.7, None, False, max_det=self.detector_max_det, nc=2
            ),
        )
        output: list[DetectionExtraction] = []
        for prediction in predictions:
            detections: list[Detection] = []
            boxes: Float[Tensor, "detections fields"] = prediction.clone()
            num_detections: int = int(boxes.shape[0])
            if num_detections > 0:
                boxes[:, :4] = cast(Tensor, ops.scale_boxes(self.DETECTOR_SHAPE, boxes[:, :4], image_hw))
            boxes_xyxy: Float[Tensor, "detections 4"] = boxes[:, :4].contiguous()
            is_right_tensor: Int[Tensor, "detections"] = boxes[:, 5].to(dtype=torch.long).contiguous()
            if num_detections > 0:
                boxes_for_output: Float[ndarray, "detections 6"] = (
                    boxes[:, :6].detach().cpu().numpy().astype(np.float32)
                )
                for box in boxes_for_output:
                    hand_bbox: list[float] = box[:4].tolist()
                    is_right: int = int(box[5])
                    detections.append({"hand_bbox": hand_bbox, "is_right": is_right})
            output.append(
                DetectionExtraction(detections=detections, boxes_xyxy=boxes_xyxy, is_right_tensor=is_right_tensor)
            )
        return output

    def _crop_frames(
        self,
        frames_rgb: RgbFrames,
        detections: list[DetectionExtraction],
        *,
        rescale_factor: float,
    ) -> tuple[list[_FrameRecord], Float[Tensor, "n 256 256 3"] | None]:
        """Generate batched RGB hand crops and bookkeeping records.

        Args:
            frames_rgb: Source RGB frames as ``UInt8[Tensor, "batch h w 3"]``.
            detections: Per-frame detector outputs from ``_postprocess_detector``.
            rescale_factor: Multiplier applied to box width/height before taking
                the square crop side length.

        Returns:
            A tuple containing frame records and optional crops. Records preserve
            the crop-to-frame mapping and NumPy metadata needed by final output
            projection. Crops are ``Float[Tensor, "n 256 256 3"]`` in NHWC
            layout, rounded to image-like values and cast to the pipeline dtype.
            Returns ``None`` for crops when no hands were detected.
        """
        records: list[_FrameRecord] = []
        frame_indices_for_crop: list[Int[Tensor, "n"]] = []
        centers_for_crop: list[Float[Tensor, "n 2"]] = []
        box_sizes_for_crop: list[Float[Tensor, "n"]] = []
        flips_for_crop: list[Bool[Tensor, "n"]] = []
        patch_cursor: int = 0
        img_size: Int[ndarray, "2"] = np.array([int(frames_rgb.shape[2]), int(frames_rgb.shape[1])])
        for frame_idx, detection in enumerate(detections):
            bboxes: Float[Tensor, "n 4"] = detection.boxes_xyxy.to(device=self.device, dtype=torch.float32)
            is_right_tensor: Int[Tensor, "n"] = detection.is_right_tensor.to(device=self.device, dtype=torch.long)
            num_detections: int = int(bboxes.shape[0])
            center_tensor: Float[Tensor, "n 2"] = (bboxes[:, 2:4] + bboxes[:, 0:2]) / 2.0
            scale_tensor: Float[Tensor, "n 2"] = rescale_factor * (bboxes[:, 2:4] - bboxes[:, 0:2])
            center: Float[ndarray, "n 2"] = center_tensor.detach().cpu().numpy().astype(np.float32)
            scale: Float[ndarray, "n 2"] = scale_tensor.detach().cpu().numpy().astype(np.float32)
            is_rights: list[int] = [int(value) for value in is_right_tensor.detach().cpu().tolist()]
            patch_start: int = patch_cursor
            if num_detections > 0:
                frame_indices_for_frame: Int[Tensor, "n"] = is_right_tensor * 0 + int(frame_idx)
                frame_indices_for_crop.append(frame_indices_for_frame)
                centers_for_crop.append(center_tensor)
                box_sizes_for_crop.append(scale_tensor.max(dim=1).values)
                flips_for_crop.append(is_right_tensor == 0)
                patch_cursor += num_detections
            records.append(
                _FrameRecord(detection.detections, center, scale, img_size, is_rights, patch_start, patch_cursor)
            )
        if len(frame_indices_for_crop) == 0:
            return records, None

        all_frame_indices: Int[Tensor, "n"] = torch.cat(frame_indices_for_crop)
        centers_xy: Float[Tensor, "n 2"] = torch.cat(centers_for_crop)
        box_sizes: Float[Tensor, "n"] = torch.cat(box_sizes_for_crop)
        flips: Bool[Tensor, "n"] = torch.cat(flips_for_crop)
        patches: Float[Tensor, "n 256 256 3"] = generate_rgb_image_patches_torch(
            frames_rgb,
            frame_indices=all_frame_indices,
            centers_xy=centers_xy,
            box_sizes=box_sizes,
            flips=flips,
            patch_size=self.IMAGE_SIZE,
        )
        return records, patches.round().clamp(0, 255).to(dtype=self.dtype).contiguous()

    def _run_wilor(self, patches: Float[Tensor, "n 256 256 3"], *, wilor_batch_size: int) -> dict[str, ndarray]:
        """Run the full WiLor TensorRT engine over generated hand crops.

        Args:
            patches: WiLor crop tensor as ``Float[Tensor, "n 256 256 3"]`` in
                NHWC layout. In normal use this is CUDA float16.
            wilor_batch_size: Maximum crop count sent to TensorRT per call.

        Returns:
            Dictionary keyed by ``FULL_WILOR_OUTPUT_NAMES``. Values are CPU
            NumPy arrays concatenated across all TensorRT chunks.
        """
        chunks: dict[str, list[ndarray]] = {name: [] for name in FULL_WILOR_OUTPUT_NAMES}
        for start in range(0, int(patches.shape[0]), wilor_batch_size):
            raw: WiLorOutput = self.wilor(patches[start : start + wilor_batch_size])
            raw_dict: dict[str, Tensor] = cast(dict[str, Tensor], raw)
            for name in FULL_WILOR_OUTPUT_NAMES:
                chunks[name].append(raw_dict[name].cpu().float().numpy())
        return {
            name: values[0] if len(values) == 1 else np.concatenate(values, axis=0) for name, values in chunks.items()
        }

    def _attach_wilor_outputs(self, records: list[_FrameRecord], wilor_output: dict[str, ndarray]) -> None:
        """Attach camera-corrected WiLor predictions back to frame detections.

        Args:
            records: Frame records returned by ``_crop_frames``. The method
                mutates each record's ``detections`` list in place by inserting
                ``wilor_preds`` for every crop-backed detection.
            wilor_output: CPU NumPy outputs from ``_run_wilor`` keyed by
                ``FULL_WILOR_OUTPUT_NAMES``.

        Returns:
            None. The output detections are updated in place.
        """
        total_patches: int = sum(record.patch_end - record.patch_start for record in records)
        if total_patches == 0:
            return
        centers: Float[ndarray, "n 2"] = np.empty((total_patches, 2), dtype=np.float32)
        box_sizes: Float[ndarray, "n"] = np.empty((total_patches,), dtype=np.float32)
        img_sizes: Int[ndarray, "n 2"] = np.empty((total_patches, 2), dtype=np.int64)
        is_rights: Int[ndarray, "n"] = np.empty((total_patches,), dtype=np.int64)
        assignments: list[tuple[_FrameRecord, int, int]] = []
        for record in records:
            patch_slice: slice = slice(record.patch_start, record.patch_end)
            if record.patch_start == record.patch_end:
                continue
            centers[patch_slice] = record.center
            box_sizes[patch_slice] = record.scale.max(axis=1)
            img_sizes[patch_slice] = record.img_size[None]
            is_rights[patch_slice] = np.asarray(record.is_rights, dtype=np.int64)
            assignments.extend(
                (record, local_idx, patch_idx)
                for local_idx, patch_idx in enumerate(range(record.patch_start, record.patch_end))
            )

        pred_cam: Float[ndarray, "n 3"] = wilor_output["pred_cam"].copy()
        pred_cam[:, 1] = (2 * is_rights - 1).astype(pred_cam.dtype) * pred_cam[:, 1]
        global_orient: Float[ndarray, "n 1 3"] = wilor_output["global_orient"].copy()
        hand_pose: Float[ndarray, "n 15 3"] = wilor_output["hand_pose"].copy()
        keypoints_3d: Float[ndarray, "n 21 3"] = wilor_output["pred_keypoints_3d"].copy()
        vertices: Float[ndarray, "n 778 3"] = wilor_output["pred_vertices"].copy()
        left_mask: Bool[ndarray, "n"] = is_rights == 0
        keypoints_3d[left_mask, :, 0] = -keypoints_3d[left_mask, :, 0]
        vertices[left_mask, :, 0] = -vertices[left_mask, :, 0]
        global_orient[left_mask, :, 1:3] = -global_orient[left_mask, :, 1:3]
        hand_pose[left_mask, :, 1:3] = -hand_pose[left_mask, :, 1:3]

        scaled_focal_lengths: Float[ndarray, "n"] = self.focal_length / self.IMAGE_SIZE * img_sizes.max(axis=1)
        pred_cam_t_full: Float[ndarray, "n 3"] = utils.cam_crop_to_full(
            pred_cam, centers, box_sizes, img_sizes, scaled_focal_lengths
        )
        keypoints_2d: Float[ndarray, "n 21 2"] = utils.perspective_projection(
            keypoints_3d,
            translation=pred_cam_t_full,
            focal_length=np.stack([scaled_focal_lengths, scaled_focal_lengths], axis=-1),
            camera_center=img_sizes / 2,
        )
        for record, local_idx, patch_idx in assignments:
            record.detections[local_idx]["wilor_preds"] = {
                "global_orient": global_orient[[patch_idx]],
                "hand_pose": hand_pose[[patch_idx]],
                "betas": wilor_output["betas"][[patch_idx]],
                "pred_cam": pred_cam[[patch_idx]],
                "pred_keypoints_3d": keypoints_3d[[patch_idx]],
                "pred_vertices": vertices[[patch_idx]],
                "pred_cam_t_full": pred_cam_t_full[[patch_idx]],
                "scaled_focal_length": float(scaled_focal_lengths[patch_idx]),
                "pred_keypoints_2d": keypoints_2d[[patch_idx]],
            }
