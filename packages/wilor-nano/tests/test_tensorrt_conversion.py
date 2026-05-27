from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


class _FakeFullWilor(nn.Module):
    def forward(self, img_patches: Tensor) -> dict[str, Tensor]:
        batch_size: int = int(img_patches.shape[0])
        return {
            "global_orient": torch.zeros((batch_size, 1, 3), dtype=img_patches.dtype),
            "hand_pose": torch.zeros((batch_size, 15, 3), dtype=img_patches.dtype),
            "betas": torch.zeros((batch_size, 10), dtype=img_patches.dtype),
            "pred_cam": torch.zeros((batch_size, 3), dtype=img_patches.dtype),
            "pred_keypoints_3d": torch.zeros((batch_size, 21, 3), dtype=img_patches.dtype),
            "pred_vertices": torch.zeros((batch_size, 778, 3), dtype=img_patches.dtype),
        }


class _FakeDetectorModel(nn.Module):
    def forward(self, images: Tensor) -> Tensor:
        return torch.zeros((int(images.shape[0]), 6, 10), dtype=images.dtype)


class _FakeDetector:
    def __init__(self) -> None:
        self.model: nn.Module = _FakeDetectorModel()


class _FakePipeline:
    def __init__(self, **_kwargs: Any) -> None:
        self.wilor_model: nn.Module = _FakeFullWilor()
        self.hand_detector: _FakeDetector = _FakeDetector()


def test_full_wilor_onnx_export_contract(tmp_path: Path) -> None:
    from wilor_nano.api.tensorrt_conversion import WiLorOnnxExportConfig, WiLorTensorRtArtifactConfig, export_wilor_onnx

    calls: list[dict[str, Any]] = []

    def fake_export_fn(model: nn.Module, args: tuple[Tensor, ...], output_path: Path, **kwargs: Any) -> None:
        calls.append({"model": model, "args": args, "output_path": output_path, **kwargs})

    onnx_path: Path = tmp_path / "wilor_full.onnx"
    summary = export_wilor_onnx(
        WiLorOnnxExportConfig(
            artifact=WiLorTensorRtArtifactConfig(target="full_postcrop", onnx_path=onnx_path, batch_size=224),
            device="cpu",
            dtype="float32",
        ),
        pipeline_factory=_FakePipeline,
        export_fn=fake_export_fn,
    )

    assert summary.onnx_path == onnx_path
    assert summary.input_shape == (224, 256, 256, 3)
    assert summary.output_names == ("global_orient", "hand_pose", "betas", "pred_cam", "pred_keypoints_3d", "pred_vertices")
    assert calls[0]["input_names"] == ["img_patches"]
    assert calls[0]["output_names"] == list(summary.output_names)
    assert tuple(int(dim) for dim in calls[0]["args"][0].shape) == summary.input_shape


def test_raw_detector_onnx_export_contract(tmp_path: Path) -> None:
    from wilor_nano.api.tensorrt_conversion import WiLorOnnxExportConfig, WiLorTensorRtArtifactConfig, export_wilor_onnx

    calls: list[dict[str, Any]] = []

    def fake_export_fn(model: nn.Module, args: tuple[Tensor, ...], output_path: Path, **kwargs: Any) -> None:
        calls.append({"model": model, "args": args, "output_path": output_path, **kwargs})

    onnx_path: Path = tmp_path / "detector.onnx"
    summary = export_wilor_onnx(
        WiLorOnnxExportConfig(
            artifact=WiLorTensorRtArtifactConfig(target="detector_raw", onnx_path=onnx_path, batch_size=110),
            device="cpu",
            dtype="float32",
        ),
        pipeline_factory=_FakePipeline,
        export_fn=fake_export_fn,
    )

    assert summary.input_shape == (110, 3, 512, 416)
    assert summary.output_names == ("output0",)
    assert calls[0]["input_names"] == ["images"]
    assert calls[0]["output_names"] == ["output0"]
    assert tuple(int(dim) for dim in calls[0]["args"][0].shape) == summary.input_shape


def test_tensorrt_build_manifest_marks_engine_machine_local(tmp_path: Path) -> None:
    from wilor_nano.api.tensorrt_conversion import TensorRtBuildConfig, WiLorTensorRtArtifactConfig

    onnx_path: Path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake onnx")
    config = TensorRtBuildConfig(
        artifact=WiLorTensorRtArtifactConfig(target="full_postcrop", onnx_path=onnx_path, batch_size=224),
        engine_path=tmp_path / "model.trt",
        precision="fp16",
        allow_tf32=False,
    )

    manifest: dict[str, object] = config.to_manifest(tensorrt_version="10.13.3.9", cuda_device_name="test gpu")

    assert manifest["target"] == "full_postcrop"
    assert manifest["precision"] == "fp16"
    assert manifest["portable_engine"] is False
    assert manifest["rebuild_from_onnx_on_target_machine"] is True
    assert manifest["batch_profile"] == {"min": 224, "optimal": 224, "max": 224}
    assert manifest["allow_tf32"] is False


def test_default_conversion_artifacts_live_under_pretrained_models() -> None:
    from wilor_nano.api.tensorrt_conversion import DEFAULT_DETECTOR_ENGINE_PATH, DEFAULT_FULL_WILOR_ONNX_PATH

    assert "pretrained_models/tensorrt" in DEFAULT_FULL_WILOR_ONNX_PATH.as_posix()
    assert "pretrained_models/tensorrt" in DEFAULT_DETECTOR_ENGINE_PATH.as_posix()
    assert not DEFAULT_FULL_WILOR_ONNX_PATH.as_posix().startswith("/tmp/")
    assert not DEFAULT_DETECTOR_ENGINE_PATH.as_posix().startswith("/tmp/")
