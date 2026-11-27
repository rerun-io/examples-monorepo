"""The ONNX-artifact models default to ``auto``: TensorRT on an NVIDIA GPU, ONNX Runtime on the CPU elsewhere."""

from pathlib import Path

import pytest
import tyro

from posekit.models import DetectorConfig, Pose2dConfig
from posekit.models.detnet import DetNetHandDetectorConfig
from posekit.models.egohand import EgoHandDetectorConfig
from posekit.models.rtmpose import RtmPoseConfig
from posekit.models.yolox import YoloxDetectorConfig
from posekit.runtimes import AutoBackendConfig, OnnxBackendConfig, TensorRtBackendConfig


@pytest.mark.parametrize("config", [YoloxDetectorConfig(), RtmPoseConfig(), DetNetHandDetectorConfig(), EgoHandDetectorConfig()], ids=type)
def test_the_onnx_models_default_to_auto(config: YoloxDetectorConfig | RtmPoseConfig | DetNetHandDetectorConfig | EgoHandDetectorConfig) -> None:
    assert isinstance(config.backend, AutoBackendConfig)


def test_the_cli_picks_auto_unless_told_otherwise() -> None:
    default: object = tyro.cli(DetectorConfig, args=["yolox"])
    on_cpu: object = tyro.cli(DetectorConfig, args=["yolox", "onnx", "--backend.device", "cpu"])
    pose: object = tyro.cli(Pose2dConfig, args=["rtmpose", "tensorrt"])
    assert isinstance(default, YoloxDetectorConfig) and isinstance(default.backend, AutoBackendConfig)
    assert isinstance(on_cpu, YoloxDetectorConfig) and on_cpu.backend == OnnxBackendConfig(device="cpu")
    assert isinstance(pose, RtmPoseConfig) and isinstance(pose.backend, TensorRtBackendConfig)


def test_the_fp16_trunk_resolves_auto_before_it_asks_for_tensorrt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``auto`` without TensorRT is ONNX Runtime, which cannot take the TensorRT-only FP16 trunk."""
    monkeypatch.setattr("trtkit.backends.tensorrt_available", lambda: False)
    weights: Path = tmp_path / "hand_detector.onnx"
    weights.write_bytes(b"")
    with pytest.raises(ValueError, match="FP16 detector trunk requires a TensorRT backend"):
        EgoHandDetectorConfig(weights, fp16_trunk=True).setup()
