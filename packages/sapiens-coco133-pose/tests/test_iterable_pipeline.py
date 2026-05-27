from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from jaxtyping import Float32, UInt8
from numpy import ndarray
from sapiens2_pose.api.metadata import get_sapiens_metainfo
from sapiens2_pose.api.runtime import PoseEstimation

from sapiens_coco133_pose.api import iterable as iterable_module
from sapiens_coco133_pose.api.iterable import DEFAULT_ITERABLE_RRD_PATH, IterableVideoPoseConfig, estimate_frame_pose_coco133, iter_video_pose_coco133


def test_estimate_frame_pose_coco133_detects_boxes_and_projects_sapiens_keypoints() -> None:
    expected_image_rgb: UInt8[ndarray, "h w 3"] = np.zeros((24, 32, 3), dtype=np.uint8)
    expected_bboxes: Float32[ndarray, "n 4"] = np.asarray([[2.0, 3.0, 20.0, 22.0]], dtype=np.float32)
    meta: dict[str, Any] = get_sapiens_metainfo()
    mapping: dict[int, int] = {int(k): int(v) for k, v in meta["coco_wholebody_to_goliath_mapping"].items()}
    coco_idx: int = min(mapping)
    sapiens_idx: int = mapping[coco_idx]
    native_keypoints: Float32[ndarray, "sapiens_k 2"] = np.zeros((308, 2), dtype=np.float32)
    native_scores: Float32[ndarray, "sapiens_k"] = np.zeros((308,), dtype=np.float32)
    native_keypoints[sapiens_idx] = np.asarray([9.0, 10.0], dtype=np.float32)
    native_scores[sapiens_idx] = 0.75
    detector_calls: list[UInt8[ndarray, "h w 3"]] = []

    def fake_detector(input_image: UInt8[ndarray, "h w 3"]) -> Float32[ndarray, "n 4"]:
        detector_calls.append(input_image)
        return expected_bboxes

    def fake_pose_estimator(
        image_rgb: UInt8[ndarray, "h w 3"],
        bboxes: Float32[ndarray, "n 4"],
        **_kwargs: object,
    ) -> PoseEstimation:
        assert image_rgb is expected_image_rgb
        np.testing.assert_allclose(bboxes, expected_bboxes)
        return PoseEstimation(bboxes=bboxes, keypoints=[native_keypoints], scores=[native_scores])

    frame = estimate_frame_pose_coco133(
        expected_image_rgb,
        frame_index=7,
        timestamp_ns=1234,
        detector=fake_detector,
        native_pose_estimator=fake_pose_estimator,
        device="cpu",
    )

    assert detector_calls == [expected_image_rgb]
    assert frame.frame_index == 7
    assert frame.timestamp_ns == 1234
    np.testing.assert_allclose(frame.bboxes, expected_bboxes)
    assert frame.keypoints.shape == (1, 133, 2)
    assert frame.scores.shape == (1, 133)
    np.testing.assert_allclose(frame.keypoints[0, coco_idx], np.asarray([9.0, 10.0], dtype=np.float32))
    assert frame.scores[0, coco_idx] == np.float32(0.75)


def test_estimate_frame_pose_coco133_can_use_rtmlib_direct_coco133_keypoints() -> None:
    expected_image_rgb: UInt8[ndarray, "h w 3"] = np.zeros((24, 32, 3), dtype=np.uint8)
    expected_bboxes: Float32[ndarray, "n 4"] = np.asarray([[2.0, 3.0, 20.0, 22.0]], dtype=np.float32)
    expected_keypoints: Float32[ndarray, "n coco_k 2"] = np.zeros((1, 133, 2), dtype=np.float32)
    expected_scores: Float32[ndarray, "n coco_k"] = np.zeros((1, 133), dtype=np.float32)
    expected_keypoints[0, 17] = np.asarray([12.0, 14.0], dtype=np.float32)
    expected_scores[0, 17] = 0.91

    def fake_detector(_input_image: UInt8[ndarray, "h w 3"]) -> Float32[ndarray, "n 4"]:
        return expected_bboxes

    def fake_rtmlib_pose_estimator(
        image_rgb: UInt8[ndarray, "h w 3"],
        bboxes: Float32[ndarray, "n 4"],
    ) -> tuple[Float32[ndarray, "n coco_k 2"], Float32[ndarray, "n coco_k"]]:
        assert image_rgb is expected_image_rgb
        np.testing.assert_allclose(bboxes, expected_bboxes)
        return expected_keypoints, expected_scores

    frame = estimate_frame_pose_coco133(
        expected_image_rgb,
        frame_index=9,
        timestamp_ns=5678,
        detector=fake_detector,
        rtmlib_pose_estimator=fake_rtmlib_pose_estimator,
        pose_backend="rtmlib",
        device="cpu",
    )

    assert frame.frame_index == 9
    assert frame.timestamp_ns == 5678
    np.testing.assert_allclose(frame.bboxes, expected_bboxes)
    np.testing.assert_allclose(frame.keypoints, expected_keypoints)
    np.testing.assert_allclose(frame.scores, expected_scores)


def test_iter_video_pose_coco133_uses_rtmlib_progress_defaults(monkeypatch: Any, tmp_path: Path) -> None:
    import tqdm.auto as tqdm_auto
    from torchcodec import decoders as torchcodec_decoders

    progress_calls: list[tuple[int | None, str | None]] = []

    class FakeVideoDecoder:
        """Small TorchCodec-compatible decoder used to keep the iterable API test in-process."""

        metadata: SimpleNamespace = SimpleNamespace(num_frames=3, average_fps=30.0)

        def __init__(self, video_path: Path, *, dimension_order: str, device: str) -> None:
            assert video_path == tmp_path / "input.mp4"
            assert dimension_order == "NHWC"
            assert device == "cpu"

        def get_frames_in_range(self, start: int, stop: int) -> SimpleNamespace:
            del stop
            frame_rgb: UInt8[torch.Tensor, "1 h w 3"] = torch.full((1, 8, 10, 3), start, dtype=torch.uint8)
            return SimpleNamespace(data=frame_rgb)

    def fake_tqdm(iterable: Any, *, total: int | None = None, desc: str | None = None) -> Any:
        progress_calls.append((total, desc))
        return iterable

    def fake_detector(_image_rgb: UInt8[ndarray, "h w 3"]) -> Float32[ndarray, "n 4"]:
        return np.empty((0, 4), dtype=np.float32)

    def fake_rtmlib_pose_estimator(
        image_rgb: UInt8[ndarray, "h w 3"],
        bboxes: Float32[ndarray, "n 4"],
    ) -> tuple[Float32[ndarray, "n coco_k 2"], Float32[ndarray, "n coco_k"]]:
        del image_rgb, bboxes
        raise AssertionError("Pose estimator should not run when the detector returns no boxes.")

    monkeypatch.setattr(torchcodec_decoders, "VideoDecoder", FakeVideoDecoder)
    monkeypatch.setattr(tqdm_auto, "tqdm", fake_tqdm)

    config = IterableVideoPoseConfig(video_path=tmp_path / "input.mp4", max_frames=2, pose_backend="rtmlib")
    frames: list[Any] = list(iter_video_pose_coco133(config, detector=fake_detector, rtmlib_pose_estimator=fake_rtmlib_pose_estimator))

    assert config.rrd_path == DEFAULT_ITERABLE_RRD_PATH
    assert config.show_progress is True
    assert [frame.frame_index for frame in frames] == [0, 1]
    assert progress_calls == [(2, "rtmlib coco133 iterable frames")]


def test_iter_video_pose_coco133_allows_independent_rtmlib_pose_runtime_config(monkeypatch: Any, tmp_path: Path) -> None:
    from torchcodec import decoders as torchcodec_decoders

    constructed_pose_configs: list[Any] = []

    class FakeVideoDecoder:
        """Small TorchCodec-compatible decoder for config plumbing tests."""

        metadata: SimpleNamespace = SimpleNamespace(num_frames=1, average_fps=30.0)

        def __init__(self, video_path: Path, *, dimension_order: str, device: str) -> None:
            assert video_path == tmp_path / "input.mp4"
            assert dimension_order == "NHWC"
            assert device == "cpu"

        def get_frames_in_range(self, start: int, stop: int) -> SimpleNamespace:
            del start, stop
            frame_rgb: UInt8[torch.Tensor, "1 h w 3"] = torch.zeros((1, 8, 10, 3), dtype=torch.uint8)
            return SimpleNamespace(data=frame_rgb)

    class FakeRtmlibPoseEstimator:
        """Records the RTMLib pose config without constructing real ONNX sessions."""

        def __init__(self, config: object) -> None:
            constructed_pose_configs.append(config)

        def __call__(
            self,
            _image_rgb: UInt8[ndarray, "h w 3"],
            _bboxes: Float32[ndarray, "n 4"],
        ) -> tuple[Float32[ndarray, "n coco_k 2"], Float32[ndarray, "n coco_k"]]:
            return np.empty((0, 133, 2), dtype=np.float32), np.empty((0, 133), dtype=np.float32)

    def fake_detector(_image_rgb: UInt8[ndarray, "h w 3"]) -> Float32[ndarray, "n 4"]:
        return np.empty((0, 4), dtype=np.float32)

    monkeypatch.setattr(torchcodec_decoders, "VideoDecoder", FakeVideoDecoder)
    monkeypatch.setattr(iterable_module, "RtmlibCoco133PoseEstimator", FakeRtmlibPoseEstimator)

    config = IterableVideoPoseConfig(
        video_path=tmp_path / "input.mp4",
        max_frames=1,
        pose_backend="rtmlib",
        detector_backend="onnxruntime",
        detector_device="cuda",
        rtmlib_pose_backend="opencv",
        rtmlib_pose_device="cpu",
        show_progress=False,
    )
    frames: list[Any] = list(iter_video_pose_coco133(config, detector=fake_detector))

    assert [frame.frame_index for frame in frames] == [0]
    assert len(constructed_pose_configs) == 1
    assert constructed_pose_configs[0].backend == "opencv"
    assert constructed_pose_configs[0].device == "cpu"
