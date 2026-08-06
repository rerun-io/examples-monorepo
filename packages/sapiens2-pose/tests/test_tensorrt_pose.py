from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from jaxtyping import Float32, UInt8
from numpy import ndarray

from sapiens2_pose.api.pose_artifact import PosePredictionArtifact
from sapiens2_pose.api.tensorrt_pose import (
    ExportableRMSNorm,
    SapiensPoseOnnxExportConfig,
    TensorRtBuildConfig,
    estimate_sapiens_pose_tensorrt,
    estimate_sapiens_pose_with_heatmap_runner,
    export_sapiens_pose_onnx,
    make_sapiens_pose_onnx_exportable,
)
from sapiens2_pose.sapiens_lite.backbones.sapiens2 import RopePositionEmbedding


def test_export_sapiens_pose_onnx_uses_static_batch_one_graph(tmp_path: Path) -> None:
    checkpoint_path: Path = tmp_path / "sapiens2_0.4b_pose.safetensors"
    onnx_path: Path = tmp_path / "sapiens2_0.4b_pose.onnx"
    checkpoint_path.write_bytes(b"placeholder checkpoint")
    calls: list[dict[str, Any]] = []

    class TinyPoseModel(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            heatmaps: torch.Tensor = torch.zeros((1, 308, 256, 192), dtype=inputs.dtype, device=inputs.device)
            return heatmaps

    def fake_loader(size: str, checkpoint: str | Path, device: str = "cuda") -> TinyPoseModel:
        assert size == "0.4B"
        assert Path(checkpoint) == checkpoint_path
        assert device == "cpu"
        return TinyPoseModel()

    def fake_export(
        model: torch.nn.Module,
        args: tuple[torch.Tensor],
        f: str | Path,
        **kwargs: Any,
    ) -> None:
        dummy_inputs: torch.Tensor = args[0]
        calls.append({"dummy_inputs": dummy_inputs, "path": Path(f), **kwargs})
        Path(f).write_bytes(b"onnx")

    export_summary = export_sapiens_pose_onnx(
        SapiensPoseOnnxExportConfig(
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path,
            model_size="0.4B",
            device="cpu",
        ),
        model_loader=fake_loader,
        export_fn=fake_export,
    )

    assert export_summary.onnx_path == onnx_path
    assert export_summary.input_shape == (1, 3, 1024, 768)
    assert export_summary.output_shape == (1, 308, 256, 192)
    assert calls[0]["path"].parent == onnx_path.parent
    assert calls[0]["path"].name.startswith(onnx_path.name + ".part")
    assert onnx_path.exists()
    assert calls[0]["dummy_inputs"].shape == (1, 3, 1024, 768)
    assert calls[0]["input_names"] == ["inputs"]
    assert calls[0]["output_names"] == ["heatmaps"]
    assert "dynamic_axes" not in calls[0]


def test_tensorrt_build_config_writes_bf16_static_batch_manifest(tmp_path: Path) -> None:
    onnx_path: Path = tmp_path / "model.onnx"
    engine_path: Path = tmp_path / "model.trt"
    onnx_path.write_bytes(b"onnx bytes")

    config: TensorRtBuildConfig = TensorRtBuildConfig(
        onnx_path=onnx_path,
        engine_path=engine_path,
        model_size="0.4B",
    )
    manifest: dict[str, object] = config.to_manifest(tensorrt_version="10.13.3.9", cuda_device_name="NVIDIA GeForce RTX 5090")

    assert manifest["model_size"] == "0.4B"
    assert manifest["precision"] == "bf16"
    assert manifest["portable_engine"] is False
    assert manifest["rebuild_from_onnx_on_target_machine"] is True
    assert manifest["runtime_recommendation"] == "cuda_graph_replay"
    assert manifest["onnx_sha256"] == "c1650a9961b22b18b7919016db749eb46e2f14ad9da48dd02aa586fe8a335978"
    assert manifest["batch_profile"] == {"min": 1, "optimal": 1, "max": 1}
    assert manifest["batch_profile_preset"] == "static-b1"
    assert manifest["builder_optimization_level"] == 3
    assert manifest["workspace_gib"] == 24.0
    assert manifest["model_io"] == {
        "input_name": "inputs",
        "output_name": "heatmaps",
        "input_shape": [1, 3, 1024, 768],
        "output_shape": [1, 308, 256, 192],
    }
    assert manifest["preprocessing"] == {
        "color_order_before_normalize": "RGB",
        "mean": [123.675, 116.28, 103.53],
        "std": [58.395, 57.12, 57.375],
    }
    assert manifest["decode"] == {"codec": "UDPHeatmap", "input_size": [768, 1024], "heatmap_size": [192, 256], "sigma": 6.0}


def test_make_sapiens_pose_onnx_exportable_replaces_rmsnorm_with_equivalent_math() -> None:
    model: torch.nn.Sequential = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.RMSNorm(4, eps=1e-6))
    inputs: torch.Tensor = torch.randn((2, 3, 4), dtype=torch.float32)
    with torch.no_grad():
        expected: torch.Tensor = model(inputs)

    converted: torch.nn.Module = make_sapiens_pose_onnx_exportable(model)
    converted_sequential: torch.nn.Sequential = cast(torch.nn.Sequential, converted)
    with torch.no_grad():
        actual: torch.Tensor = converted(inputs)

    assert isinstance(converted_sequential[1], ExportableRMSNorm)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_make_sapiens_pose_onnx_exportable_promotes_rope_periods_to_float32() -> None:
    rope: RopePositionEmbedding = RopePositionEmbedding(embed_dim=16, num_heads=1, dtype=torch.bfloat16)
    model: torch.nn.Sequential = torch.nn.Sequential(rope)

    converted: torch.nn.Module = make_sapiens_pose_onnx_exportable(model)
    converted_sequential: torch.nn.Sequential = cast(torch.nn.Sequential, converted)
    converted_rope: RopePositionEmbedding = cast(RopePositionEmbedding, converted_sequential[0])

    assert converted_rope.dtype == torch.float32
    assert converted_rope.periods.dtype == torch.float32


def test_estimate_sapiens_pose_with_heatmap_runner_uses_common_decode_path() -> None:
    image_rgb: UInt8[ndarray, "h w 3"] = np.zeros((64, 48, 3), dtype=np.uint8)
    bboxes: Float32[ndarray, "n 4"] = np.asarray([[0.0, 0.0, 47.0, 63.0]], dtype=np.float32)
    captured_shapes: list[tuple[int, ...]] = []

    def fake_heatmap_runner(inputs: torch.Tensor) -> Float32[ndarray, "n k h w"]:
        captured_shapes.append(tuple(int(dim) for dim in inputs.shape))
        heatmaps: Float32[ndarray, "n k h w"] = np.zeros((1, 308, 256, 192), dtype=np.float32)
        heatmaps[:, :, 128, 96] = 1.0
        return heatmaps

    artifact: PosePredictionArtifact = estimate_sapiens_pose_with_heatmap_runner(
        image_rgb,
        bboxes,
        model_size="0.4B",
        device="cpu",
        heatmap_runner=fake_heatmap_runner,
    )

    assert captured_shapes == [(1, 3, 1024, 768)]
    assert artifact.bboxes.shape == (1, 4)
    assert artifact.keypoints.shape == (1, 308, 2)
    assert artifact.scores.shape == (1, 308)
    assert np.all(np.isfinite(artifact.keypoints))
    assert np.all(artifact.scores > 0.0)


def test_estimate_sapiens_pose_tensorrt_runs_multiple_boxes_as_static_batch_one_calls() -> None:
    image_rgb: UInt8[ndarray, "h w 3"] = np.zeros((64, 48, 3), dtype=np.uint8)
    bboxes: Float32[ndarray, "n 4"] = np.asarray([[0.0, 0.0, 47.0, 63.0], [4.0, 5.0, 40.0, 55.0]], dtype=np.float32)
    captured_shapes: list[tuple[int, ...]] = []

    def fake_heatmap_runner(inputs: torch.Tensor) -> Float32[ndarray, "n k h w"]:
        captured_shapes.append(tuple(int(dim) for dim in inputs.shape))
        heatmaps: Float32[ndarray, "n k h w"] = np.zeros((1, 308, 256, 192), dtype=np.float32)
        heatmaps[:, :, 128, 96] = 1.0
        return heatmaps

    artifact: PosePredictionArtifact = estimate_sapiens_pose_tensorrt(
        image_rgb,
        bboxes,
        engine_path=Path("/tmp/static-b1.trt"),
        model_size="0.4B",
        device="cpu",
        heatmap_runner=fake_heatmap_runner,
    )

    assert captured_shapes == [(1, 3, 1024, 768), (1, 3, 1024, 768)]
    assert artifact.bboxes.shape == (2, 4)
    assert artifact.keypoints.shape == (2, 308, 2)
    assert artifact.scores.shape == (2, 308)
