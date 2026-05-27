from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from numpy import ndarray

TORCH_UINT8: torch.dtype = torch.__dict__["uint8"]


def test_trt_config_builds_pipeline_config(monkeypatch: Any) -> None:
    from simplecv.rerun_log_utils import RerunTyroConfig

    import wilor_nano.api.wilor_inference_trt as wilor_trt

    monkeypatch.setattr(wilor_trt, "get_torch_device", lambda: torch.device("cuda"))
    monkeypatch.setattr(wilor_trt, "get_torch_dtype", lambda _device: torch.float16)

    config = wilor_trt.BatchedTensorRtVideoConfig(
        rr_config=RerunTyroConfig(headless=True),
        detector_engine_path=Path("/tmp/detector.trt"),
        wilor_engine_path=Path("/tmp/wilor.trt"),
    )
    pipeline_config = config.to_pipeline_config()

    assert pipeline_config.detector_engine_path == Path("/tmp/detector.trt")
    assert pipeline_config.wilor_engine_path == Path("/tmp/wilor.trt")
    assert pipeline_config.detector_static_batch_size == 110
    assert pipeline_config.wilor_static_batch_size == 224


def test_collect_columnar_video_detections_groups_by_side() -> None:
    from wilor_nano.api.wilor_inference_trt import _group_outputs
    from wilor_nano.pipelines.wilor_hand_pose3d_estimation_pipeline_trt import Detection, WilorPreds

    points2d: Float[ndarray, "1 21 2"] = np.zeros((1, 21, 2), dtype=np.float32)
    points3d: Float[ndarray, "1 21 3"] = np.ones((1, 21, 3), dtype=np.float32)

    def preds(
        keypoints2d: Float[ndarray, "1 21 2"],
        keypoints3d: Float[ndarray, "1 21 3"],
    ) -> WilorPreds:
        return {
            "pred_cam": np.zeros((1, 3), dtype=np.float32),
            "global_orient": np.zeros((1, 1, 3), dtype=np.float32),
            "hand_pose": np.zeros((1, 15, 3), dtype=np.float32),
            "betas": np.zeros((1, 10), dtype=np.float32),
            "pred_keypoints_3d": keypoints3d,
            "pred_vertices": np.zeros((1, 778, 3), dtype=np.float32),
            "pred_cam_t_full": np.zeros((1, 3), dtype=np.float32),
            "scaled_focal_length": 5000.0,
            "pred_keypoints_2d": keypoints2d,
        }

    frame_outputs: list[list[Detection]] = [
        [{"hand_bbox": [1.0, 2.0, 5.0, 8.0], "is_right": 1, "wilor_preds": preds(points2d, points3d)}],
        [{"hand_bbox": [10.0, 20.0, 30.0, 40.0], "is_right": 0, "wilor_preds": preds(points2d + 3.0, points3d * 2.0)}],
    ]
    timestamps_ns: Int[ndarray, "num_frames=2"] = np.array([100, 200], dtype=np.int64)

    columnar = _group_outputs(frame_outputs, timestamps_ns)

    assert columnar["right"][0] == [100]
    assert columnar["right"][1] == [[1.0, 2.0, 5.0, 8.0]]
    assert columnar["left"][0] == [200]
    np.testing.assert_allclose(columnar["left"][3][0], np.full((21, 3), 2.0, dtype=np.float32))


def test_run_batched_tensorrt_video_uses_video_only_pipeline(monkeypatch: Any) -> None:
    from simplecv.rerun_log_utils import RerunTyroConfig

    import wilor_nano.api.wilor_inference_trt as wilor_trt

    calls: list[tuple[str, Any]] = []

    class FakeFrameBatch:
        def __init__(self, data: UInt8[torch.Tensor, "batch h w 3"]) -> None:
            self.data: UInt8[torch.Tensor, "batch h w 3"] = data

    class FakeDecoder:
        def __init__(self, path: Path, *, dimension_order: str, device: str) -> None:
            calls.append(("decoder", (path, dimension_order, device)))
            self.frames: UInt8[torch.Tensor, "batch h w 3"] = torch.stack(
                [torch.zeros((8, 8, 3), dtype=TORCH_UINT8), torch.ones((8, 8, 3), dtype=TORCH_UINT8)]
            )

        def get_frames_in_range(self, start: int, stop: int, step: int = 1) -> FakeFrameBatch:
            del step
            return FakeFrameBatch(self.frames[start:stop])

    class FakePipeline:
        def __init__(self, config: Any) -> None:
            calls.append(("config", config))

        def predict_batch_rgb_tensor(self, frames: UInt8[torch.Tensor, "batch h w 3"], **kwargs: Any) -> list[list[wilor_trt.Detection]]:
            calls.append(("predict", (frames.shape[0], frames.dtype, kwargs)))
            return [[] for _frame in frames]

    monkeypatch.setattr(wilor_trt, "get_torch_device", lambda: torch.device("cuda"))
    monkeypatch.setattr(wilor_trt, "get_torch_dtype", lambda _device: torch.float16)
    monkeypatch.setattr(wilor_trt, "set_annotation_context", lambda: None)
    monkeypatch.setattr(wilor_trt, "_video_timestamps_ns", lambda *_args, **_kwargs: np.array([0, 1], dtype=np.int64))
    monkeypatch.setattr(wilor_trt, "log_video_detections_columnar", lambda outputs, _timestamps: calls.append(("log", len(outputs))))
    monkeypatch.setattr(wilor_trt, "VideoDecoder", FakeDecoder)
    monkeypatch.setattr(wilor_trt, "WiLorHandPose3dEstimationPipeline", FakePipeline)

    config = wilor_trt.BatchedTensorRtVideoConfig(
        rr_config=RerunTyroConfig(headless=True),
        video_path=Path("video.mp4"),
        detector_batch_size=2,
        detector_static_batch_size=2,
    )
    wilor_trt.run_batched_tensorrt_video(config)

    assert calls[0] == ("decoder", (Path("video.mp4"), "NHWC", "cuda"))
    assert calls[1][0] == "config"
    assert calls[2] == ("predict", (2, TORCH_UINT8, {"detector_batch_size": 2, "wilor_batch_size": 224}))
    assert calls[-1] == ("log", 2)
