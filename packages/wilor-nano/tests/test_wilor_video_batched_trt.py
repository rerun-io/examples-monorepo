from __future__ import annotations

import tomllib
from pathlib import Path

import pytest


def test_batched_tensorrt_video_config_maps_to_fast_wilor_config(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    from simplecv.rerun_log_utils import RerunTyroConfig

    import wilor_nano.api.wilor_inference_trt as wilor_trt

    monkeypatch.setattr(wilor_trt, "get_torch_device", lambda: torch.device("cuda"))
    monkeypatch.setattr(wilor_trt, "get_torch_dtype", lambda _device: torch.float16)
    config: wilor_trt.BatchedTensorRtVideoConfig = wilor_trt.BatchedTensorRtVideoConfig(
        rr_config=RerunTyroConfig(headless=True),
        video_path=Path("assets/video.mp4"),
        detector_engine_path=Path("/tmp/detector.trt"),
        wilor_engine_path=Path("/tmp/wilor.trt"),
    )

    pipeline_config = config.to_pipeline_config()

    assert pipeline_config.detector_engine_path == Path("/tmp/detector.trt")
    assert pipeline_config.wilor_engine_path == Path("/tmp/wilor.trt")
    assert pipeline_config.detector_static_batch_size == 110
    assert pipeline_config.wilor_static_batch_size == 224
    assert pipeline_config.device == torch.device("cuda")


def test_batched_tensorrt_video_config_rejects_batches_larger_than_static_engine() -> None:
    from simplecv.rerun_log_utils import RerunTyroConfig

    from wilor_nano.api.wilor_inference_trt import BatchedTensorRtVideoConfig

    with pytest.raises(ValueError, match="detector_batch_size"):
        BatchedTensorRtVideoConfig(
            rr_config=RerunTyroConfig(headless=True),
            detector_batch_size=111,
            detector_static_batch_size=110,
        )


def test_pixi_tasks_keep_generic_wilor_entrypoints() -> None:
    repo_root: Path = Path(__file__).resolve().parents[3]
    pixi_toml: dict[str, object] = tomllib.loads((repo_root / "pixi.toml").read_text())
    feature_section = pixi_toml["feature"]
    assert isinstance(feature_section, dict)
    wilor_section = feature_section["wilor"]
    assert isinstance(wilor_section, dict)
    tasks = wilor_section["tasks"]
    assert isinstance(tasks, dict)

    assert set(tasks) == {
        "image-example",
        "video-example",
        "video-trt",
        "export-onnx",
        "build-trt",
        "compare-reference",
    }
    assert all(not task_name.startswith("wilor-") for task_name in tasks)
    assert tasks["image-example"]["cmd"] == "python tools/wilor_inference.py --image-path assets/img.png"
    assert tasks["video-example"]["cmd"] == "python tools/wilor_inference.py --video-path assets/video.mp4"
    assert tasks["video-trt"]["cmd"] == "python tools/wilor_inference_trt.py"
    assert tasks["export-onnx"]["cmd"] == "python tools/conversion/export_wilor_onnx.py"
    assert tasks["build-trt"]["cmd"] == "python tools/conversion/build_wilor_tensorrt.py"
    assert tasks["compare-reference"]["cmd"] == "python tools/compare_rrd_reference.py --reference-rrd tests/reference_data/wilor_video_30.rrd --candidate-rrd /tmp/wilor_candidate.rrd --index video_time --rtol 0.01 --atol 0.25"


def test_original_wilor_inference_config_stays_minimal() -> None:
    from dataclasses import fields

    from wilor_nano.api.wilor_inference import WilorConfig

    field_names: set[str] = {field.name for field in fields(WilorConfig)}

    assert field_names == {"rr_config", "image_path", "video_path", "max_frames"}
