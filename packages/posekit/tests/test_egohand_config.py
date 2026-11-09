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
    detector: object = tyro.cli(HandDetectorConfig, args=["egohand-fast", "tensorrt"])
    pose: object = tyro.cli(HandPose3dConfig, args=["egohand-fast"])
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
    quantized: Path = egohand_fast.export_quantized_backbone_once(config, tmp_path / "weights", tmp_path / "soma", load_calibration())
    assert sources == [tmp_path / "dinov3-bf16.onnx"]
    assert quantized.parent == tmp_path / "engines" and quantized.is_file()


@pytest.mark.parametrize("preset,precision,quantized_detector", [("egohand-catalog", "nvfp4", True), ("egohand-catalog-bf16", "bf16", False)])
def test_catalog_cli_selects_measured_paired_roles(preset: str, precision: str, quantized_detector: bool) -> None:
    detector: object = tyro.cli(HandDetectorConfig, args=[preset, "tensorrt"])
    pose: object = tyro.cli(HandPose3dConfig, args=[preset])
    assert isinstance(detector, EgoHandDetectorConfig)
    assert isinstance(pose, EgoHandFastConfig)
    assert (detector.fp8_calibration is not None) == quantized_detector
    assert pose.backbone_precision == precision and pose.batch_size == 2


def test_cached_benchmark_can_run_the_public_catalog_preset() -> None:
    from posekit.apis.hands_cached_bench import Config

    config: Config = tyro.cli(Config, args=["--model", "catalog", "--precision", "bf16", "--no-measure-timing"])
    assert config.model == "catalog" and config.precision == "bf16" and not config.measure_timing


def test_detector_rejects_two_precision_choices(tmp_path: Path) -> None:
    """The FP8 detector keeps its other convolutions in FP16, so an FP16 trunk on top is a contradiction."""
    with pytest.raises(ValueError, match="not both"):
        EgoHandDetectorConfig(fp16_trunk=True, fp8_calibration=tmp_path / "detector.rrd")
    assert EgoHandDetectorConfig(fp8_calibration=tmp_path / "detector.rrd").fp8_calibration is not None
