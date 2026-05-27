from pathlib import Path
from typing import Any

import torch
from sapiens2_pose.api.coco133_tensorrt_conversion import (
    SapiensCoco133PoseOnnxExportConfig,
    TensorRtEngineBuildConfig,
    export_sapiens_coco133_pose_onnx,
)


def test_export_sapiens_coco133_pose_onnx_uses_static_fp16_batch(tmp_path: Path) -> None:
    checkpoint_path: Path = tmp_path / "sapiens.safetensors"
    onnx_path: Path = tmp_path / "sapiens_b8_fp16.onnx"
    checkpoint_path.write_bytes(b"checkpoint")
    calls: list[dict[str, Any]] = []

    def fake_loader(size: str, checkpoint: str | Path, device: str = "cuda") -> torch.nn.Module:
        assert size == "0.4B"
        assert Path(checkpoint) == checkpoint_path
        assert device == "cpu"
        return torch.nn.Identity()

    def fake_export(_model: torch.nn.Module, args: tuple[torch.Tensor], f: str | Path, **kwargs: Any) -> None:
        calls.append({"inputs": args[0], "path": Path(f), **kwargs})
        Path(f).write_bytes(b"onnx")

    summary = export_sapiens_coco133_pose_onnx(
        SapiensCoco133PoseOnnxExportConfig(
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path,
            batch_size=8,
            device="cpu",
        ),
        model_loader=fake_loader,
        export_fn=fake_export,
    )

    assert summary.onnx_path == onnx_path
    assert summary.input_shape == (8, 3, 1024, 768)
    assert summary.output_shape == (8, 308, 256, 192)
    assert calls[0]["inputs"].dtype == torch.float16
    assert calls[0]["input_names"] == ["inputs"]
    assert calls[0]["output_names"] == ["heatmaps"]


def test_tensorrt_engine_build_manifest_records_static_batch_and_fp16(tmp_path: Path) -> None:
    onnx_path: Path = tmp_path / "model.onnx"
    engine_path: Path = tmp_path / "model.trt"
    onnx_path.write_bytes(b"onnx bytes")

    manifest = TensorRtEngineBuildConfig(
        target="pose",
        onnx_path=onnx_path,
        engine_path=engine_path,
        input_name="inputs",
        output_names=("heatmaps",),
        input_shape=(3, 1024, 768),
        batch_size=8,
    ).to_manifest(tensorrt_version="10.13.3.9", cuda_device_name="NVIDIA")

    assert manifest["target"] == "pose"
    assert manifest["precision"] == "fp16"
    assert manifest["batch_profile"] == {"min": 8, "optimal": 8, "max": 8}
    assert manifest["model_io"] == {
        "input_name": "inputs",
        "input_shape": [8, 3, 1024, 768],
        "output_names": ["heatmaps"],
    }
