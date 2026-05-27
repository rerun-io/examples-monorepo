from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tyro
from sapiens2_pose.api.pose_artifact import PoseArtifactComparisonMetrics

from sapiens_coco133_pose.api import benchmark as benchmark_module
from sapiens_coco133_pose.api.artifact import Coco133PoseFrame
from sapiens_coco133_pose.api.batched_tensorrt import SapiensTensorRtPoseConfig, TensorRtDetectorConfig
from sapiens_coco133_pose.api.benchmark import PoseBenchmarkConfig, run_pose_benchmark


def test_pose_benchmark_cli_hides_static_tensorrt_engine_metadata(capsys: Any) -> None:
    with suppress(SystemExit):
        tyro.cli(run_pose_benchmark, args=["--help"])

    help_text = capsys.readouterr().out

    assert "--config.detector.engine-path" in help_text
    assert "--config.runtime.detector-batch-size" in help_text
    assert "--config.runtime.pose-batch-size" in help_text
    assert "--config.detector.static-batch-size" not in help_text
    assert "--config.detector.input-size" not in help_text
    assert "--config.detector.max-detections" not in help_text
    assert "--static-batch-size" not in help_text
    assert "--input-size" not in help_text


def test_run_pose_benchmark_times_inference_before_artifact_writes(monkeypatch: Any, tmp_path: Path) -> None:
    frame = Coco133PoseFrame(
        frame_index=0,
        timestamp_ns=0,
        image_rgb=None,
        bboxes=np.empty((0, 4), dtype=np.float32),
        keypoints=np.empty((0, 133, 2), dtype=np.float32),
        scores=np.empty((0, 133), dtype=np.float32),
    )
    events: list[str] = []

    def fake_iter_video_pose_coco133(_config: object) -> list[Coco133PoseFrame]:
        events.append("iterable_inference")
        return [frame]

    class FakeBatchedPredictor:
        """Fake initialized predictor used to verify benchmark timing boundaries."""

        def predict_frames(self) -> list[Coco133PoseFrame]:
            events.append("batched_inference")
            return [frame]

    def fake_create_batched_tensorrt_video_frame_predictor(_config: object) -> FakeBatchedPredictor:
        events.append("batched_init")
        return FakeBatchedPredictor()

    def fake_write_video_pose_rrd(_frames: list[Coco133PoseFrame], *, rrd_path: Path, keypoint_threshold: float) -> None:
        del keypoint_threshold
        events.append(f"rrd:{rrd_path.name}")

    def fake_compare_video_pose_rrds(baseline_rrd_path: Path, candidate_rrd_path: Path, _config: object) -> PoseArtifactComparisonMetrics:
        events.append(f"compare-rrd:{baseline_rrd_path.name}:{candidate_rrd_path.name}")
        return PoseArtifactComparisonMetrics(
            max_relative_difference=0.0,
            mean_relative_difference=0.0,
            max_absolute_difference=0.0,
            mean_absolute_difference=0.0,
            compared_values=0,
            max_keypoint_bbox_fraction=0.0,
            mean_keypoint_bbox_fraction=0.0,
            compared_keypoints=0,
        )

    monkeypatch.setattr(benchmark_module, "iter_video_pose_coco133", fake_iter_video_pose_coco133)
    monkeypatch.setattr(benchmark_module, "create_batched_tensorrt_video_frame_predictor", fake_create_batched_tensorrt_video_frame_predictor)
    monkeypatch.setattr(benchmark_module, "write_video_pose_rrd", fake_write_video_pose_rrd)
    monkeypatch.setattr(benchmark_module, "compare_video_pose_rrds", fake_compare_video_pose_rrds)
    monkeypatch.setattr(benchmark_module, "_synchronize_cuda", lambda: None)
    monkeypatch.setattr(benchmark_module, "_release_iterable_gpu_caches", lambda: None)

    output_dir = tmp_path / "outputs"
    run_pose_benchmark(
        PoseBenchmarkConfig(
            video_path=tmp_path / "video.mp4",
            detector=TensorRtDetectorConfig(engine_path=tmp_path / "detector.trt"),
            pose=SapiensTensorRtPoseConfig(engine_path=tmp_path / "pose.trt"),
            output_dir=output_dir,
            max_frames=1,
        )
    )

    assert events[:2] == ["iterable_inference", "rrd:iterable_golden.rrd"]
    assert "batched_init" in events
    assert "batched_inference" in events
    assert events.index("batched_init") < events.index("batched_inference")
    assert events.index("batched_inference") < events.index("rrd:batched_tensorrt.rrd")
    assert all(not event.startswith("save:") for event in events)
    assert list(output_dir.glob("*.npz")) == []
    assert events.index("rrd:iterable_golden.rrd") < events.index("compare-rrd:iterable_golden.rrd:batched_tensorrt.rrd")
    assert events.index("rrd:batched_tensorrt.rrd") < events.index("compare-rrd:iterable_golden.rrd:batched_tensorrt.rrd")


def test_release_iterable_gpu_caches_uses_public_sapiens_runtime_api(monkeypatch: Any) -> None:
    import sapiens2_pose.api.runtime as sapiens_runtime

    events: list[str] = []

    monkeypatch.setattr(sapiens_runtime, "clear_pose_model_cache", lambda: events.append("clear-sapiens-cache"))
    monkeypatch.setattr(benchmark_module.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(benchmark_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(benchmark_module.torch.cuda, "empty_cache", lambda: events.append("empty-cuda-cache"))

    benchmark_module._release_iterable_gpu_caches()

    assert events == ["clear-sapiens-cache", "gc", "empty-cuda-cache"]


def test_run_pose_benchmark_rejects_empty_iterable_runs(monkeypatch: Any, tmp_path: Path) -> None:
    events: list[str] = []

    def fake_iter_video_pose_coco133(_config: object) -> list[Coco133PoseFrame]:
        events.append("iterable_inference")
        return []

    def fake_create_batched_tensorrt_video_frame_predictor(_config: object) -> object:
        events.append("batched_init")
        raise AssertionError("Batched path should not run after an empty iterable benchmark.")

    monkeypatch.setattr(benchmark_module, "iter_video_pose_coco133", fake_iter_video_pose_coco133)
    monkeypatch.setattr(benchmark_module, "create_batched_tensorrt_video_frame_predictor", fake_create_batched_tensorrt_video_frame_predictor)
    monkeypatch.setattr(benchmark_module, "_synchronize_cuda", lambda: None)

    with pytest.raises(ValueError, match="Iterable benchmark produced no frames"):
        run_pose_benchmark(PoseBenchmarkConfig(video_path=tmp_path / "video.mp4", output_dir=tmp_path / "outputs"))

    assert events == ["iterable_inference"]
