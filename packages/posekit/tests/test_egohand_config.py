"""Fast hand roles remain selectable through the ordinary Tyro unions."""

from pathlib import Path

import pytest
import tyro

pytest.importorskip("warp", reason="EgoHand configs require posekit-hands")

from trtkit import TensorRtBackendConfig

from posekit.models import HandDetectorConfig, HandPose3dConfig
from posekit.models.egohand import EgoHandConfig, EgoHandDetectorConfig
from posekit.models.egohand_fast import EgoHandFastConfig, fast_egohand_configs


def test_fast_role_subcommands_select_the_calibrated_config() -> None:
    detector = tyro.cli(HandDetectorConfig, args=["egohand-fast", "tensorrt"])
    pose = tyro.cli(HandPose3dConfig, args=["egohand-fast"])
    assert isinstance(detector, EgoHandDetectorConfig)
    assert isinstance(detector.backend, TensorRtBackendConfig)
    assert detector.gpu_preprocessing and detector.fp16_trunk
    assert isinstance(pose, EgoHandFastConfig)
    assert pose == fast_egohand_configs()[1] == EgoHandFastConfig()
    assert pose.fp8_tail_blocks == 8
    assert isinstance(tyro.cli(HandPose3dConfig, args=["egohand"]), EgoHandConfig)


def test_fast_role_quantizes_the_asset_tree_bf16_backbone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With an asset tree, the fast role starts from its dinov3-bf16.onnx instead of exporting the 3 GB DINOv3 again."""
    from posekit.models import egohand_fast
    from posekit.models.egohand_quantization import load_calibration

    (tmp_path / "dinov3-bf16.onnx").write_bytes(b"bf16")
    sources: list[Path] = []

    def export_missing(_weights: Path, _soma: Path, onnx: Path, **__: object) -> None:
        if not onnx.is_file():
            raise AssertionError(f"exported the bf16 DINOv3 to {onnx} although the asset tree has one")

    def quantize(source: Path, output: Path, *_: object, **__: object) -> None:
        sources.append(source)
        output.write_bytes(b"quantized")

    monkeypatch.setattr(egohand_fast, "export_backbone_once", export_missing)
    monkeypatch.setattr(egohand_fast, "quantize_backbone", quantize)
    _, config = fast_egohand_configs(tmp_path)
    quantized = egohand_fast.export_quantized_backbone_once(config, tmp_path / "weights", tmp_path / "soma", load_calibration())
    assert sources == [tmp_path / "dinov3-bf16.onnx"]
    assert quantized.parent == tmp_path / "engines" and quantized.is_file()
