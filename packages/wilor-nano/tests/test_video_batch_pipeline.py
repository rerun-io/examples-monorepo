from __future__ import annotations

import torch
from jaxtyping import Float, UInt8

TORCH_UINT8: torch.dtype = torch.__dict__["uint8"]


def test_trt_detector_preprocess_commits_to_static_shape() -> None:
    from wilor_nano.pipelines.wilor_hand_pose3d_estimation_pipeline_trt import WiLorHandPose3dEstimationPipeline

    pipe: WiLorHandPose3dEstimationPipeline = WiLorHandPose3dEstimationPipeline.__new__(
        WiLorHandPose3dEstimationPipeline
    )
    pipe.device = torch.device("cpu")
    frames: UInt8[torch.Tensor, "batch=2 h=8 w=16 3"] = 255 * torch.ones((2, 8, 16, 3), dtype=TORCH_UINT8)

    detector_input: Float[torch.Tensor, "batch=2 channels=3 height=512 width=416"] = pipe._preprocess_detector(frames)

    assert detector_input.shape == (2, 3, 512, 416)
    assert detector_input.dtype == torch.float32
    assert torch.all(detector_input >= 0.0)
    assert torch.all(detector_input <= 1.0)


def test_trt_torch_crop_path_keeps_patches_as_tensor() -> None:
    from wilor_nano.pipelines.wilor_hand_pose3d_estimation_pipeline_trt import (
        DetectionExtraction,
        WiLorHandPose3dEstimationPipeline,
    )

    pipe: WiLorHandPose3dEstimationPipeline = WiLorHandPose3dEstimationPipeline.__new__(
        WiLorHandPose3dEstimationPipeline
    )
    pipe.device = torch.device("cpu")
    pipe.dtype = torch.float32
    rgb: UInt8[torch.Tensor, "batch=1 h=64 w=64 3"] = torch.zeros((1, 64, 64, 3), dtype=TORCH_UINT8)
    rgb[:, :, :, 0] = 255
    detection: DetectionExtraction = DetectionExtraction(
        detections=[{"hand_bbox": [16.0, 16.0, 48.0, 48.0], "is_right": 1}],
        boxes_xyxy=torch.tensor([[16.0, 16.0, 48.0, 48.0]], dtype=torch.float32),
        is_right_tensor=torch.tensor([1], dtype=torch.long),
    )

    records, patches = pipe._crop_frames(rgb, [detection], rescale_factor=1.0)

    assert len(records) == 1
    assert records[0].patch_start == 0
    assert records[0].patch_end == 1
    assert patches is not None
    assert patches.shape == (1, 256, 256, 3)
    assert patches.dtype == torch.float32
    assert int(patches[0, 128, 128, 0].item()) == 255


def test_trt_torch_crop_path_handles_empty_tensor_detections() -> None:
    from wilor_nano.pipelines.wilor_hand_pose3d_estimation_pipeline_trt import (
        DetectionExtraction,
        WiLorHandPose3dEstimationPipeline,
    )

    pipe: WiLorHandPose3dEstimationPipeline = WiLorHandPose3dEstimationPipeline.__new__(
        WiLorHandPose3dEstimationPipeline
    )
    pipe.device = torch.device("cpu")
    pipe.dtype = torch.float32
    rgb: UInt8[torch.Tensor, "batch=1 h=64 w=64 3"] = torch.zeros((1, 64, 64, 3), dtype=TORCH_UINT8)
    detection: DetectionExtraction = DetectionExtraction(
        detections=[],
        boxes_xyxy=torch.empty((0, 4), dtype=torch.float32),
        is_right_tensor=torch.empty((0,), dtype=torch.long),
    )

    records, patches = pipe._crop_frames(rgb, [detection], rescale_factor=1.0)

    assert len(records) == 1
    assert records[0].patch_start == 0
    assert records[0].patch_end == 0
    assert records[0].center.shape == (0, 2)
    assert records[0].scale.shape == (0, 2)
    assert patches is None
