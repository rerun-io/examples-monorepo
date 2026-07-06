# ruff: noqa: I001

from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import rerun.experimental as rrx
import torch
import tyro
from sapiens2_pose.api.coco133_gpu import (
    DecodedYoloxHeadOutput,
    RtmlibPoseInputBatch,
    SapiensPoseInputBatch,
    _gaussian_blur_heatmaps_torch,
    decode_rtmlib_simcc_to_coco133_torch,
    decode_sapiens_heatmaps_to_coco133_torch,
    postprocess_yolox_outputs_torch,
    prepare_rtmlib_pose_inputs_torch,
    prepare_sapiens_pose_inputs_torch,
    preprocess_yolox_detector_torch,
)
from sapiens2_pose.api.metadata import get_sapiens_metainfo
from simplecv.rerun_log_utils import RerunTyroConfig

from sapiens_coco133_pose.api import batched_tensorrt as batched_module
from sapiens_coco133_pose.api.artifact import POINTS_CONFIDENCE_COMPONENT, Coco133PoseFrame
from sapiens_coco133_pose.api.batched_tensorrt import (
    BatchedTensorRtCoco133Pipeline,
    BatchedTensorRtPoseConfig,
    BatchedTensorRtVideoConfig,
    RtmlibTensorRtPoseConfig,
    SapiensTensorRtPoseConfig,
    TensorRtDetectorConfig,
    run_batched_tensorrt_video_coco133,
)


def _headless_rr_config() -> RerunTyroConfig:
    return RerunTyroConfig(application_id="sapiens_coco133_pose_test", headless=True)


def test_preprocess_yolox_detector_torch_matches_rtmlib_top_left_bgr_letterbox() -> None:
    frames_rgb: torch.Tensor = torch.zeros((1, 2, 4, 3), dtype=torch.uint8)
    frames_rgb[0, 0, 0] = torch.tensor([1, 2, 3], dtype=torch.uint8)

    detector_inputs, resize_ratios = preprocess_yolox_detector_torch(frames_rgb, model_input_size=(4, 4))

    assert tuple(detector_inputs.shape) == (1, 3, 4, 4)
    assert detector_inputs.dtype == torch.float32
    assert resize_ratios.tolist() == [1.0]
    torch.testing.assert_close(detector_inputs[0, :, 0, 0], torch.tensor([3.0, 2.0, 1.0]))
    torch.testing.assert_close(detector_inputs[0, :, 3, 0], torch.tensor([114.0, 114.0, 114.0]))


def test_prepare_sapiens_pose_inputs_torch_uses_udp_bbox_scale_and_normalizes_rgb() -> None:
    frames_rgb: torch.Tensor = torch.zeros((1, 20, 10, 3), dtype=torch.uint8)
    bboxes: torch.Tensor = torch.tensor([[0.0, 0.0, 9.0, 19.0]], dtype=torch.float32)
    frame_indices: torch.Tensor = torch.tensor([0], dtype=torch.long)

    batch = prepare_sapiens_pose_inputs_torch(
        frames_rgb,
        frame_indices=frame_indices,
        bboxes_xyxy=bboxes,
        dtype=torch.float32,
    )

    assert tuple(batch.inputs.shape) == (1, 3, 1024, 768)
    assert batch.inputs.dtype == torch.float32
    np.testing.assert_allclose(batch.bbox_centers.cpu().numpy(), np.asarray([[4.5, 9.5]], dtype=np.float32))
    np.testing.assert_allclose(batch.bbox_scales.cpu().numpy(), np.asarray([[17.8125, 23.75]], dtype=np.float32))
    torch.testing.assert_close(batch.inputs[0, :, 512, 384], torch.tensor([-2.1179, -2.0357, -1.8044]), rtol=1e-4, atol=1e-4)


def test_prepare_rtmlib_pose_inputs_torch_matches_rtmlib_affine_bgr_normalization() -> None:
    from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs, top_down_affine

    frames_rgb: torch.Tensor = torch.arange(1 * 12 * 10 * 3, dtype=torch.uint8).reshape(1, 12, 10, 3)
    bboxes: torch.Tensor = torch.tensor([[1.0, 2.0, 8.0, 10.0]], dtype=torch.float32)
    frame_indices: torch.Tensor = torch.tensor([0], dtype=torch.long)

    batch = prepare_rtmlib_pose_inputs_torch(
        frames_rgb,
        frame_indices=frame_indices,
        bboxes_xyxy=bboxes,
        dtype=torch.float32,
    )

    image_bgr: np.ndarray = frames_rgb[0].numpy()[..., ::-1].copy()
    expected_center, expected_scale = bbox_xyxy2cs(bboxes[0].numpy(), padding=1.25)
    top_down_affine_any: Any = top_down_affine
    expected_crop, expected_scale = top_down_affine_any((192, 256), expected_scale, expected_center, image_bgr)
    expected_input: np.ndarray = ((expected_crop - np.asarray([123.675, 116.28, 103.53], dtype=np.float32)) / np.asarray([58.395, 57.12, 57.375], dtype=np.float32)).transpose(2, 0, 1)

    assert tuple(batch.inputs.shape) == (1, 3, 256, 192)
    np.testing.assert_allclose(batch.bbox_centers.cpu().numpy(), expected_center.reshape(1, 2), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(batch.bbox_scales.cpu().numpy(), expected_scale.reshape(1, 2), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(batch.inputs.cpu().numpy()[0, :, 64:192:64, 48:144:48], expected_input[:, 64:192:64, 48:144:48], rtol=2e-2, atol=2e-2)


def test_postprocess_yolox_outputs_torch_accepts_decoded_head_boxes_and_scores() -> None:
    boxes: torch.Tensor = torch.tensor(
        [
            [
                [10.0, 20.0, 50.0, 80.0],
                [12.0, 22.0, 52.0, 82.0],
                [100.0, 120.0, 140.0, 160.0],
            ]
        ],
        dtype=torch.float32,
    )
    scores: torch.Tensor = torch.zeros((1, 3, 80), dtype=torch.float32)
    scores[0, 0, 0] = 0.95
    scores[0, 1, 0] = 0.90
    scores[0, 2, 0] = 0.05
    outputs = DecodedYoloxHeadOutput(boxes=boxes, scores=scores)

    frame_boxes = postprocess_yolox_outputs_torch(
        outputs,
        resize_ratios=torch.tensor([0.5], dtype=torch.float32),
        model_input_size=(640, 640),
        score_thr=0.3,
        nms_thr=0.45,
    )

    assert len(frame_boxes) == 1
    torch.testing.assert_close(frame_boxes[0], torch.tensor([[20.0, 40.0, 100.0, 160.0]], dtype=torch.float32))


def test_batched_video_config_drops_default_label_output_for_decoded_head(tmp_path: Path) -> None:
    config = BatchedTensorRtVideoConfig(
        rr_config=_headless_rr_config(),
        video_path=tmp_path / "input.mp4",
        detector=TensorRtDetectorConfig(engine_path=tmp_path / "detector.trt"),
        pose=SapiensTensorRtPoseConfig(engine_path=tmp_path / "pose.trt"),
    )

    assert config.detector.extra_output_names == ()
    assert config.to_pipeline_config().detector.extra_output_names == ()
    assert config.show_progress is True


def test_batched_video_config_defaults_to_minimal_sapiens_trt_preset(tmp_path: Path) -> None:
    config = BatchedTensorRtVideoConfig(rr_config=_headless_rr_config(), video_path=tmp_path / "input.mp4")

    assert config.detector.engine_path.name == "yolox_humanart_head_b8_fp16.trt"
    assert config.pose.engine_path.name == "sapiens2_0.4b_b8_fp16.trt"
    assert config.runtime.detector_batch_size == 8
    assert config.detector.static_batch_size == 8
    assert config.runtime.effective_pose_batch_size(config.pose) == 8
    assert config.pose.static_batch_size == 8
    assert config.detector.output_name == "1436"
    assert config.detector.score_output_name == "1438"
    assert config.detector.extra_output_names == ()
    assert isinstance(config.pose, SapiensTensorRtPoseConfig)
    assert config.pose.input_name == "inputs"
    assert config.pose.output_name == "heatmaps"
    assert config.pose.backend == "sapiens"
    assert config.show_progress is True


def test_batched_video_config_defaults_to_minimal_rtmlib_trt_preset(tmp_path: Path) -> None:
    config = BatchedTensorRtVideoConfig(
        rr_config=_headless_rr_config(),
        video_path=tmp_path / "input.mp4",
        pose=RtmlibTensorRtPoseConfig(),
    )

    assert config.detector.engine_path.name == "yolox_humanart_head_b8_fp16.trt"
    assert isinstance(config.pose, RtmlibTensorRtPoseConfig)
    assert config.pose.engine_path.name == "rtmw_256x192_b32_fp16.trt"
    assert config.pose.input_name == "input"
    assert config.pose.simcc_x_output_name == "simcc_x"
    assert config.pose.simcc_y_output_name == "simcc_y"
    assert config.runtime.effective_pose_batch_size(config.pose) == 32
    assert config.pose.static_batch_size == 32
    assert config.pose.backend == "rtmlib"
    assert config.show_progress is True


def test_static_tensorrt_engine_metadata_is_not_user_config() -> None:
    assert tuple(TensorRtDetectorConfig.__dataclass_fields__) == ("engine_path",)
    assert "input_name" not in TensorRtDetectorConfig.__dataclass_fields__
    assert "output_name" not in TensorRtDetectorConfig.__dataclass_fields__
    assert "static_batch_size" not in TensorRtDetectorConfig.__dataclass_fields__
    assert "input_size" not in TensorRtDetectorConfig.__dataclass_fields__
    assert "max_detections" not in TensorRtDetectorConfig.__dataclass_fields__
    assert tuple(SapiensTensorRtPoseConfig.__dataclass_fields__) == ("engine_path",)
    assert "static_batch_size" not in SapiensTensorRtPoseConfig.__dataclass_fields__
    assert "model_size" not in SapiensTensorRtPoseConfig.__dataclass_fields__
    assert tuple(RtmlibTensorRtPoseConfig.__dataclass_fields__) == ("engine_path",)
    assert "static_batch_size" not in RtmlibTensorRtPoseConfig.__dataclass_fields__
    assert "input_size" not in RtmlibTensorRtPoseConfig.__dataclass_fields__
    assert "simcc_x_output_name" not in RtmlibTensorRtPoseConfig.__dataclass_fields__
    assert "simcc_y_output_name" not in RtmlibTensorRtPoseConfig.__dataclass_fields__


def test_batched_tensorrt_public_exports_hide_low_level_gpu_helpers() -> None:
    public_names: set[str] = set(batched_module.__all__)

    assert {
        "BatchedTensorRtVideoConfig",
        "BatchedTensorRtCoco133Pipeline",
        "TensorRtDetectorConfig",
        "SapiensTensorRtPoseConfig",
        "RtmlibTensorRtPoseConfig",
        "TensorRtRuntimeConfig",
        "run_batched_tensorrt_video_coco133",
    } <= public_names
    assert "preprocess_yolox_detector_torch" not in public_names
    assert "prepare_sapiens_pose_inputs_torch" not in public_names
    assert "TensorRtYoloxDetectorRunner" not in public_names
    assert "_gaussian_blur_heatmaps_torch" not in public_names


def test_batched_video_cli_hides_static_tensorrt_engine_metadata(capsys: Any) -> None:
    with suppress(SystemExit):
        tyro.cli(BatchedTensorRtVideoConfig, args=["--help"])

    help_text = capsys.readouterr().out

    assert "--detector.engine-path" in help_text
    assert "--runtime.detector-batch-size" in help_text
    assert "--runtime.pose-batch-size" in help_text
    assert "--detector.static-batch-size" not in help_text
    assert "--detector.input-size" not in help_text
    assert "--detector.max-detections" not in help_text
    assert "--static-batch-size" not in help_text
    assert "--input-size" not in help_text


def test_batched_video_cli_accepts_minimal_sapiens_trt_command(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(rr, "spawn", lambda **_kwargs: None)

    config = tyro.cli(BatchedTensorRtVideoConfig, args=["--video-path", str(tmp_path / "input.mp4")])

    assert config.video_path == tmp_path / "input.mp4"
    assert config.rr_config.headless is False
    assert isinstance(config.pose, SapiensTensorRtPoseConfig)
    assert config.to_pipeline_config().dtype is torch.float16
    assert config.to_pipeline_config().execution_device == "cuda"
    assert config.to_pipeline_config().device == torch.device("cuda")


def test_batched_video_cli_accepts_rtmlib_pose_subcommand(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(rr, "spawn", lambda **_kwargs: None)

    config = tyro.cli(BatchedTensorRtVideoConfig, args=["--video-path", str(tmp_path / "input.mp4"), "rtmlib"])

    assert config.video_path == tmp_path / "input.mp4"
    assert isinstance(config.pose, RtmlibTensorRtPoseConfig)
    assert config.runtime.effective_pose_batch_size(config.pose) == 32


def test_run_batched_video_logs_to_rr_config_recording_without_writing_rrd(monkeypatch: Any, tmp_path: Path) -> None:
    frame = Coco133PoseFrame(0, 0, None, np.empty((0, 4), np.float32), np.empty((0, 133, 2), np.float32), np.empty((0, 133), np.float32))
    events: list[str] = []

    class FakePredictor:
        def predict_frames(self) -> list[Coco133PoseFrame]:
            events.append("predict")
            return [frame]

    def fake_create_predictor(_config: BatchedTensorRtVideoConfig) -> FakePredictor:
        return FakePredictor()

    def fake_log_video_pose_frames(frames: list[Coco133PoseFrame], *, keypoint_threshold: float, recording: rr.RecordingStream, video_path: Path | None) -> rr.RecordingStream:
        assert frames == [frame]
        assert keypoint_threshold == 0.3
        assert video_path == tmp_path / "input.mp4"
        events.append("log-viewer")
        return recording

    monkeypatch.setattr(batched_module, "create_batched_tensorrt_video_frame_predictor", fake_create_predictor)
    monkeypatch.setattr(batched_module, "log_video_pose_frames", fake_log_video_pose_frames)

    summary = run_batched_tensorrt_video_coco133(BatchedTensorRtVideoConfig(rr_config=_headless_rr_config(), video_path=tmp_path / "input.mp4"))

    assert events == ["predict", "log-viewer"]
    assert summary.output_path is None
    assert summary.frame_count == 1
    assert summary.person_count == 0


def test_run_batched_video_rrd_contains_keypoint_confidence_components(monkeypatch: Any, tmp_path: Path) -> None:
    bboxes: np.ndarray = np.asarray([[1.0, 2.0, 10.0, 14.0]], dtype=np.float32)
    keypoints: np.ndarray = np.full((1, 133, 2), np.nan, dtype=np.float32)
    scores: np.ndarray = np.zeros((1, 133), dtype=np.float32)
    keypoints[0, 0] = np.asarray([3.0, 4.0], dtype=np.float32)
    scores[0, 0] = 0.91
    frame = Coco133PoseFrame(0, 0, None, bboxes, keypoints, scores)
    rrd_path = tmp_path / "video_trt.rrd"

    class FakePredictor:
        def predict_frames(self) -> list[Coco133PoseFrame]:
            return [frame]

    monkeypatch.setattr(batched_module, "create_batched_tensorrt_video_frame_predictor", lambda _config: FakePredictor())

    summary = run_batched_tensorrt_video_coco133(
        BatchedTensorRtVideoConfig(
            rr_config=RerunTyroConfig(application_id="sapiens_coco133_pose_trt_confidence_test", save=rrd_path, headless=True),
            video_path=tmp_path / "input.mp4",
            log_video_media=False,
        )
    )

    assert summary.output_path == rrd_path
    reader = rrx.RrdReader(str(rrd_path))
    recordings: list[Any] = list(reader.recordings())
    keypoint_rows: list[dict[str, list[Any]]] = [
        chunk.to_record_batch().to_pydict()
        for chunk in reader.stream(store=recordings[0]).to_chunks()
        if str(chunk.entity_path) == "/video/person_0/keypoints"
    ]
    assert keypoint_rows
    np.testing.assert_allclose(np.asarray(keypoint_rows[0][POINTS_CONFIDENCE_COMPONENT][0], dtype=np.float32)[:1], scores[0, :1])


def test_decode_sapiens_heatmaps_to_coco133_torch_matches_numpy_udp_decoder() -> None:
    from sapiens2_pose.api.runtime import DEFAULT_MODEL_SIZE
    from sapiens2_pose.sapiens_lite.pose import MODEL_SPECS, UDPHeatmap

    meta: dict[str, Any] = get_sapiens_metainfo()
    mapping: dict[int, int] = {int(k): int(v) for k, v in meta["coco_wholebody_to_goliath_mapping"].items()}
    coco_idx: int = min(mapping)
    sapiens_idx: int = mapping[coco_idx]
    heatmaps_np: np.ndarray = np.zeros((1, 308, 256, 192), dtype=np.float32)
    heatmaps_np[0, sapiens_idx, 80, 64] = 10.0
    heatmaps: torch.Tensor = torch.from_numpy(heatmaps_np)
    batch = SapiensPoseInputBatch(
        inputs=torch.empty((1, 3, 1024, 768), dtype=torch.float32),
        frame_indices=torch.tensor([0], dtype=torch.long),
        bboxes_xyxy=torch.tensor([[0.0, 0.0, 768.0, 1024.0]], dtype=torch.float32),
        bbox_centers=torch.tensor([[384.0, 512.0]], dtype=torch.float32),
        bbox_scales=torch.tensor([[768.0, 1024.0]], dtype=torch.float32),
        input_size=(768, 1024),
    )

    keypoints_torch, scores_torch = decode_sapiens_heatmaps_to_coco133_torch(heatmaps, batch)

    spec = MODEL_SPECS[DEFAULT_MODEL_SIZE]
    expected_input_keypoints, expected_scores = UDPHeatmap(input_size=spec.input_size, heatmap_size=spec.heatmap_size, sigma=spec.sigma).decode(heatmaps_np[0, sapiens_idx : sapiens_idx + 1])
    expected_image_keypoint: np.ndarray = expected_input_keypoints[0, 0] / np.asarray(batch.input_size, dtype=np.float32) * batch.bbox_scales.numpy()[0] + batch.bbox_centers.numpy()[0] - 0.5 * batch.bbox_scales.numpy()[0]

    assert keypoints_torch.shape == (1, 133, 2)
    assert scores_torch.shape == (1, 133)
    np.testing.assert_allclose(keypoints_torch[0, coco_idx], expected_image_keypoint, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(scores_torch[0, coco_idx], expected_scores[0, 0], rtol=1e-6, atol=1e-6)


def test_decode_rtmlib_simcc_to_coco133_torch_maps_simcc_peaks_to_image_space() -> None:
    simcc_x: torch.Tensor = torch.zeros((1, 133, 384), dtype=torch.float32)
    simcc_y: torch.Tensor = torch.zeros((1, 133, 512), dtype=torch.float32)
    simcc_x[0, 5, 40] = 0.8
    simcc_y[0, 5, 80] = 0.6
    batch = RtmlibPoseInputBatch(
        inputs=torch.empty((1, 3, 256, 192), dtype=torch.float32),
        frame_indices=torch.tensor([0], dtype=torch.long),
        bboxes_xyxy=torch.tensor([[0.0, 0.0, 96.0, 128.0]], dtype=torch.float32),
        bbox_centers=torch.tensor([[48.0, 64.0]], dtype=torch.float32),
        bbox_scales=torch.tensor([[96.0, 128.0]], dtype=torch.float32),
        input_size=(192, 256),
    )

    keypoints, scores = decode_rtmlib_simcc_to_coco133_torch(simcc_x, simcc_y, batch)

    expected_keypoint: np.ndarray = np.asarray([10.0, 20.0], dtype=np.float32)
    np.testing.assert_allclose(keypoints[0, 5], expected_keypoint, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(scores[0, 5], np.float32(0.7), rtol=1e-6, atol=1e-6)


def test_gaussian_blur_heatmaps_torch_matches_full_2d_convolution() -> None:
    heatmaps: torch.Tensor = torch.arange(2 * 3 * 7 * 9, dtype=torch.float32).reshape(2, 3, 7, 9)
    kernel_size: int = 5
    border: int = (kernel_size - 1) // 2
    kernel_1d: torch.Tensor = torch.as_tensor(__import__("cv2").getGaussianKernel(kernel_size, 0).reshape(kernel_size), dtype=torch.float32)
    kernel_2d: torch.Tensor = (kernel_1d[:, None] * kernel_1d[None, :]).reshape(1, 1, kernel_size, kernel_size)
    flat_heatmaps: torch.Tensor = heatmaps.reshape(6, 1, 7, 9)
    zero_padded: torch.Tensor = torch.nn.functional.pad(flat_heatmaps, (border, border, border, border), mode="constant", value=0.0)
    blurred_padded: torch.Tensor = torch.nn.functional.conv2d(zero_padded, kernel_2d, padding=border)
    expected: torch.Tensor = blurred_padded[:, :, border:-border, border:-border].reshape_as(heatmaps)

    actual = _gaussian_blur_heatmaps_torch(heatmaps, kernel_size)

    torch.testing.assert_close(actual, expected)


def test_batched_pipeline_preserves_empty_frames_and_person_rows_with_fake_rtmlib_runners() -> None:
    config = BatchedTensorRtPoseConfig(
        detector=TensorRtDetectorConfig(engine_path=Path("detector.trt")),
        pose=RtmlibTensorRtPoseConfig(engine_path=Path("pose.trt")),
    )
    object.__setattr__(config, "execution_device", "cpu")
    object.__setattr__(config, "dtype", torch.float32)

    class FakeDetectorRunner:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.frame_cursor: int = 0

        def __call__(self, inputs: torch.Tensor) -> DecodedYoloxHeadOutput:
            batch_size: int = int(inputs.shape[0])
            boxes: torch.Tensor = torch.zeros((batch_size, 2, 4), dtype=torch.float32)
            scores: torch.Tensor = torch.zeros((batch_size, 2, 80), dtype=torch.float32)
            detections: dict[int, list[list[float]]] = {
                0: [[0.0, 0.0, 2.0, 2.0]],
                1: [],
                2: [[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.5, 3.5]],
            }
            for local_idx in range(batch_size):
                for anchor_idx, box in enumerate(detections[self.frame_cursor + local_idx]):
                    boxes[local_idx, anchor_idx] = torch.tensor(box, dtype=torch.float32)
                    scores[local_idx, anchor_idx, 0] = 0.9 - 0.1 * anchor_idx
            self.frame_cursor += batch_size
            return DecodedYoloxHeadOutput(boxes, scores)

    class FakeRtmlibPoseRunner:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __call__(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            simcc_x: torch.Tensor = torch.zeros((int(inputs.shape[0]), 133, 384), dtype=torch.float32)
            simcc_y: torch.Tensor = torch.zeros((int(inputs.shape[0]), 133, 512), dtype=torch.float32)
            simcc_x[:, 5, 40] = 0.8
            simcc_y[:, 5, 80] = 0.6
            return simcc_x, simcc_y

    pipeline = BatchedTensorRtCoco133Pipeline(config, detector_runner_factory=FakeDetectorRunner, pose_runner_factory=FakeRtmlibPoseRunner)
    frames = pipeline.predict_batch_rgb_tensor(torch.zeros((3, 640, 640, 3), dtype=torch.uint8), frame_start_index=100, timestamps_ns=np.asarray([10, 20, 30], dtype=np.int64), detector_batch_size=2, pose_batch_size=2)

    assert [frame.frame_index for frame in frames] == [100, 101, 102]
    assert [frame.timestamp_ns for frame in frames] == [10, 20, 30]
    assert [int(frame.bboxes.shape[0]) for frame in frames] == [1, 0, 2]
    np.testing.assert_allclose(frames[0].bboxes, np.asarray([[0.0, 0.0, 2.0, 2.0]], dtype=np.float32))
    np.testing.assert_allclose(frames[1].keypoints, np.empty((0, 133, 2), dtype=np.float32))
    np.testing.assert_allclose(frames[2].bboxes, np.asarray([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.5, 3.5]], dtype=np.float32))
    np.testing.assert_allclose(frames[2].scores[:, 5], np.asarray([0.7, 0.7], dtype=np.float32), rtol=1e-6, atol=1e-6)


def test_batched_pipeline_direct_call_uses_runtime_detector_score_threshold_default() -> None:
    config = BatchedTensorRtPoseConfig(
        detector=TensorRtDetectorConfig(engine_path=Path("detector.trt")),
        pose=RtmlibTensorRtPoseConfig(engine_path=Path("pose.trt")),
    )
    object.__setattr__(config, "execution_device", "cpu")
    object.__setattr__(config, "dtype", torch.float32)

    class FakeDetectorRunner:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __call__(self, inputs: torch.Tensor) -> DecodedYoloxHeadOutput:
            boxes: torch.Tensor = torch.zeros((int(inputs.shape[0]), 1, 4), dtype=torch.float32)
            scores: torch.Tensor = torch.zeros((int(inputs.shape[0]), 1, 80), dtype=torch.float32)
            boxes[:, 0] = torch.tensor([0.0, 0.0, 2.0, 2.0], dtype=torch.float32)
            scores[:, 0, 0] = 0.5
            return DecodedYoloxHeadOutput(boxes, scores)

    class FakeRtmlibPoseRunner:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __call__(self, _inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            raise AssertionError("Pose should not run when the direct-call detector default threshold rejects the box.")

    pipeline = BatchedTensorRtCoco133Pipeline(
        config,
        detector_runner_factory=FakeDetectorRunner,
        pose_runner_factory=FakeRtmlibPoseRunner,
    )
    frames = pipeline.predict_batch_rgb_tensor(
        torch.zeros((1, 640, 640, 3), dtype=torch.uint8),
        detector_batch_size=1,
        pose_batch_size=1,
    )

    assert len(frames) == 1
    assert frames[0].bboxes.shape == (0, 4)
