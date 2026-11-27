"""The zoo `body` preset (YOLOX-m + RTMPose-m) on ONNX Runtime's CPU provider against TensorRT on the GPU.

The CPU run is the engine-free path (`auto` off NVIDIA); TensorRT is what `auto` runs on an NVIDIA GPU. Same ONNX
artifacts, same pre/post code; the differences are the engine's reduced-precision tactics.
"""

from pathlib import Path
from typing import Final

import pytest
import torch
from jaxtyping import Bool, Float, UInt8
from torch import Tensor

from posekit.models import PersonDetector, RtmPoseConfig, TopDownPose2d, YoloxDetectorConfig
from posekit.predictions import BoxDetections, Keypoints2d
from posekit.runtimes import OnnxBackendConfig, ResolvedBackendConfig, TensorRtBackendConfig, tensorrt_available
from posekit.zoo import PRESETS

VIDEO: Final[Path] = Path(__file__).parents[1] / "assets" / "examples" / "gsi_workbench_left_front.mp4"
"""A committed example clip; the test reads four of its frames."""

MAX_BOX_DIFFERENCE_PX: Final[float] = 1.0
"""Largest box-corner difference; measured 0.27 px (RTX 5090, 2026-11-27)."""

CONFIDENT_SCORE: Final[float] = 0.3
"""Keypoints the model scores above this are compared; below it the SimCC peak is flat and argmax may jump."""

MAX_CONFIDENT_KEYPOINT_DIFFERENCE_PX: Final[float] = 6.0
"""Largest confident-keypoint difference; measured 4.0 px (RTX 5090, 2026-11-27)."""


def _run(frames: UInt8[Tensor, "n h w 3"], backend: ResolvedBackendConfig) -> tuple[BoxDetections, Keypoints2d]:
    configs: tuple[YoloxDetectorConfig, RtmPoseConfig] = PRESETS["body"].with_backend(backend)
    detector: PersonDetector = configs[0].setup()
    pose: TopDownPose2d = configs[1].setup()
    with torch.inference_mode():
        boxes: BoxDetections = detector(frames)
        keypoints: Keypoints2d = pose(frames, boxes)
    return boxes, keypoints


@pytest.mark.integration
@pytest.mark.skipif(not tensorrt_available(), reason="TensorRT is the reference; needs a CUDA device and the tensorrt bindings")
def test_the_body_preset_on_onnx_runtime_cpu_matches_tensorrt() -> None:
    """Same boxes within a pixel, and the same keypoints wherever the model is confident.

    Rows are compared in order: the shared decode keeps each frame's detections in score order (`batched_nms`), and at
    this clip's well-separated scores reduced precision does not reorder them (a reordering would show as a box
    difference of tens of pixels).
    """
    from torchcodec.decoders import VideoDecoder

    frames: UInt8[Tensor, "n h w 3"] = VideoDecoder(str(VIDEO), device="cpu", dimension_order="NHWC").get_frames_at(indices=[0, 30, 60, 90]).data
    cpu: tuple[BoxDetections, Keypoints2d] = _run(frames, OnnxBackendConfig(device="cpu"))
    gpu: tuple[BoxDetections, Keypoints2d] = _run(frames.cuda(), TensorRtBackendConfig())
    assert cpu[0].xyxy.device.type == "cpu"
    assert cpu[0].xyxy.shape[0] == gpu[0].xyxy.shape[0] > 0
    assert float((cpu[0].xyxy - gpu[0].xyxy.cpu()).abs().max()) <= MAX_BOX_DIFFERENCE_PX
    confident: Bool[Tensor, "k j"] = gpu[1].scores.cpu() > CONFIDENT_SCORE
    distances_px: Float[Tensor, "k j"] = torch.linalg.vector_norm(cpu[1].xy - gpu[1].xy.cpu(), dim=-1)
    assert bool(confident.any())
    assert float(distances_px[confident].max()) <= MAX_CONFIDENT_KEYPOINT_DIFFERENCE_PX
