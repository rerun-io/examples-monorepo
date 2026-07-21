"""YOLOX-HumanArt person detector on the posekit runtime abstraction.

The ONNX artifact is the OpenMMLab SDK deploy zip (the rtmlib source) with its
baked-in NMS stripped (:func:`posekit.artifacts.strip_detector_nms`), so ONNX
Runtime and TensorRT share one static-shape graph and thresholding/NMS runs on
GPU with torchvision.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from jaxtyping import Float, UInt8
from torch import Tensor

from posekit.artifacts import fetch_openmmlab_onnx, strip_detector_nms
from posekit.models.base import PersonDetector
from posekit.ops.letterbox import letterbox_frames
from posekit.ops.yolox import decode_yolox_head_outputs
from posekit.predictions import BoxDetections
from posekit.runtimes import OnnxBackendConfig, OnnxOrTrtBackendConfig, TensorRuntime, create_runtime_from_onnx
from posekit.runtimes.base import run_chunked

YoloxVariant = Literal["yolox-tiny-humanart", "yolox-m-humanart", "yolox-x-humanart"]

YOLOX_ONNX_ZIP_URLS: dict[YoloxVariant, str] = {
    "yolox-tiny-humanart": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
    "yolox-m-humanart": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
    "yolox-x-humanart": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",
}


@dataclass(frozen=True, slots=True)
class YoloxDetectorConfig:
    """YOLOX-HumanArt person detector configuration."""

    variant: YoloxVariant = "yolox-m-humanart"
    """Model size/checkpoint to run."""
    backend: OnnxOrTrtBackendConfig = field(default_factory=OnnxBackendConfig)
    """Accelerated backend running the detector graph."""
    score_thr: float = 0.5
    """Minimum person score."""
    nms_thr: float = 0.45
    """Non-maximum suppression IoU threshold."""

    def setup(self) -> "YoloxDetector":
        """Fetch artifacts, build/load the runtime, and return a ready detector."""
        return YoloxDetector(self)


class YoloxDetector(PersonDetector):
    """Batched GPU YOLOX person detector over an ONNX/TensorRT runtime."""

    def __init__(self, config: YoloxDetectorConfig) -> None:
        """Fetch the (NMS-stripped) ONNX artifact and create the backend runtime.

        Args:
            config: Detector variant, backend, and postprocessing thresholds.
        """
        self.config: YoloxDetectorConfig = config
        onnx_path: Path = strip_detector_nms(fetch_openmmlab_onnx(YOLOX_ONNX_ZIP_URLS[config.variant]))
        self.runtime: TensorRuntime = create_runtime_from_onnx(onnx_path, config.backend)
        input_shape: tuple[int, ...] = self.runtime.spec.inputs[0].shape
        self.input_size: tuple[int, int] = (int(input_shape[1]), int(input_shape[2]))
        """Detector input size as ``(height, width)``."""
        boxes_spec, scores_spec = self.runtime.spec.outputs
        if boxes_spec.shape[-1] != 4:
            boxes_spec, scores_spec = scores_spec, boxes_spec
        self._boxes_name: str = boxes_spec.name
        self._scores_name: str = scores_spec.name
        self._num_anchors: int = int(boxes_spec.shape[0])

    @torch.inference_mode()
    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"]) -> BoxDetections:
        """Detect persons across a frame batch.

        Args:
            frames_rgb: uint8 RGB NHWC frame batch on the inference device.

        Returns:
            Flattened, NMS-filtered detections across the batch.
        """
        inputs, ratios = letterbox_frames(frames_rgb, output_size=self.input_size, pad_value=114.0, bgr=True)
        outputs: dict[str, Tensor] = run_chunked(self.runtime, {self.runtime.spec.inputs[0].name: inputs})
        boxes: Float[Tensor, "b anchors 4"] = outputs[self._boxes_name].float()
        scores: Tensor = outputs[self._scores_name].float()
        if int(scores.shape[-1]) == self._num_anchors:
            # ONNX NonMaxSuppression score layout (batch, classes, anchors).
            scores = scores.transpose(1, 2)
        return decode_yolox_head_outputs(
            boxes,
            scores[..., 0],
            resize_ratios=ratios,
            score_thr=self.config.score_thr,
            nms_thr=self.config.nms_thr,
        )
