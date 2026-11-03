"""YOLOX-HumanArt person detector on the posekit runtime abstraction.

The ONNX artifact is the OpenMMLab SDK deploy zip (the rtmlib source) with its
baked-in NMS stripped (:func:`posekit.artifacts.strip_detector_nms`), so ONNX
Runtime and TensorRT share one static-shape graph and thresholding/NMS runs on
GPU with torchvision.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
import onnx
import torch
from jaxtyping import Float, UInt8
from onnx import numpy_helper
from torch import Tensor
from trtkit.trt_builder import onnx_content_hash

from posekit.artifacts import DEFAULT_ONNX_CACHE_DIR, fetch_openmmlab_onnx, strip_detector_nms
from posekit.models.base import PersonDetector
from posekit.ops.letterbox import letterbox_frames
from posekit.ops.yolox import decode_yolox_head_outputs
from posekit.predictions import BoxDetections
from posekit.runtimes import (
    OnnxBackendConfig,
    OnnxOrTrtBackendConfig,
    TensorRuntime,
    create_runtime_from_onnx,
    run_chunked,
)

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

    onnx_transform: Callable[[Path], Path] | None = None
    """Optional graph transform applied after detector NMS removal, before runtime creation."""

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
        if config.onnx_transform is not None:
            onnx_path = config.onnx_transform(onnx_path)
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


def dynamic_yolox(source: Path) -> Path:
    """Make only the five reviewed YOLOX-m batch reshape constants dynamic."""
    directory = Path(os.environ.get("POSEKIT_ONNX_CACHE", str(DEFAULT_ONNX_CACHE_DIR))).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    digest: str = onnx_content_hash(source, cache_dir=directory)[:16]
    dest: Path = directory / f"{source.stem}_{digest}_dynamic_batch_v1.onnx"
    if dest.exists():
        return dest
    model = onnx.load(source)
    constants = {node.output[0]: next(a.t for a in node.attribute if a.name == "value") for node in model.graph.node if node.op_type == "Constant"}
    constants.update({value.name: value for value in model.graph.initializer})
    changed: set[str] = set()
    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue
        proto = constants[node.input[1]]
        shape = numpy_helper.to_array(proto).copy()
        if shape.ndim != 1 or shape[0] not in (0, 1):
            raise ValueError(f"Unreviewed YOLOX reshape: {node.name}")
        shape[0] = 0
        proto.CopyFrom(numpy_helper.from_array(shape.astype(np.int64), name=proto.name))
        changed.add(node.input[1])
    if len(changed) != 5:
        raise ValueError("Expected five YOLOX batch constants")
    for value in [*model.graph.input, *model.graph.output]:
        value.type.tensor_type.shape.dim[0].ClearField("dim_value")
        value.type.tensor_type.shape.dim[0].dim_param = "batch"
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    partial = dest.with_name(f".{dest.name}.{uuid4().hex}.tmp")
    try:
        onnx.save(model, str(partial))
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)
    return dest

