"""Fast hand roles remain selectable through the ordinary Tyro unions."""

from dataclasses import replace
from pathlib import Path

import pytest
import tyro

pytest.importorskip("warp", reason="EgoHand configs require posekit-hands")

from trtkit import TensorRtBackendConfig

from posekit.models import HandDetectorConfig, HandPose3dConfig
from posekit.models.egohand import EgoHandConfig, EgoHandDetectorConfig
from posekit.models.egohand_fast import EgoHandFastConfig, EgoHandStack


def test_fast_role_subcommands_select_the_calibrated_config() -> None:
    detector = tyro.cli(HandDetectorConfig, args=["egohand-fast", "tensorrt"])
    pose = tyro.cli(HandPose3dConfig, args=["egohand-fast"])
    assert isinstance(detector, EgoHandDetectorConfig)
    assert detector == EgoHandStack.fast().detector
    assert isinstance(pose, EgoHandFastConfig)
    assert pose == EgoHandStack.fast().pose == EgoHandFastConfig()
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
    config = EgoHandStack.fast(tmp_path).pose
    quantized = egohand_fast.export_quantized_backbone_once(config, tmp_path / "weights", tmp_path / "soma", load_calibration())
    assert sources == [tmp_path / "dinov3-bf16.onnx"]
    assert quantized.parent == tmp_path / "engines" and quantized.is_file()


@pytest.mark.parametrize("preset,precision,quantized_detector", [("egohand-fast", "nvfp4", True), ("egohand-fast-bf16", "bf16", False)])
def test_fast_cli_selects_measured_paired_roles(preset: str, precision: str, quantized_detector: bool) -> None:
    detector: object = tyro.cli(HandDetectorConfig, args=[preset, "tensorrt"])
    pose: object = tyro.cli(HandPose3dConfig, args=[preset])
    assert isinstance(detector, EgoHandDetectorConfig)
    assert isinstance(pose, EgoHandFastConfig)
    assert (detector.fp8_calibration is not None) == quantized_detector
    assert pose.backbone_precision == precision and pose.batch_size == 2


def test_fast_preset_is_the_measured_configuration() -> None:
    from posekit.models.egohand_fast import FP8_TAIL_BLOCKS
    from posekit.models.egohand_quantization import load_calibration

    stack = EgoHandStack.fast()
    detector, pose = stack.detector, stack.pose
    assert isinstance(detector.backend, TensorRtBackendConfig)
    assert detector.backend.max_batch_size == detector.backend.opt_batch_size == 1
    assert detector.gpu_preprocessing and not detector.fp16_trunk
    assert detector.fp8_calibration is not None and detector.fp8_calibration.is_file()
    assert pose.backbone_precision == "nvfp4" and FP8_TAIL_BLOCKS == 24
    assert pose.batch_size == pose.backbone_backend.max_batch_size == pose.backbone_backend.opt_batch_size == 2
    assert pose.calibration is None and load_calibration().segment == "egohand-catalog-calibration-v1"


def test_detector_rejects_two_precision_choices(tmp_path: Path) -> None:
    """The FP8 detector keeps its other convolutions in FP16, so an FP16 trunk on top is a contradiction."""
    with pytest.raises(ValueError, match="not both"):
        EgoHandDetectorConfig(fp16_trunk=True, fp8_calibration=tmp_path / "detector.rrd")
    assert EgoHandDetectorConfig(fp8_calibration=tmp_path / "detector.rrd").fp8_calibration is not None


def test_nvfp4_preset_points_older_gpus_to_the_bf16_preset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On a GPU without FP8/NVFP4 (here an Ampere RTX 3060, compute capability 8.6) both roles name the BF16 preset."""
    import torch

    from posekit.models.egohand import EgoHandDetector
    from posekit.models.egohand_fast import FastEgoHandPredictor

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (8, 6))
    stack = EgoHandStack.fast()
    detector, pose = stack.detector, stack.pose
    with pytest.raises(ValueError, match='backbone_precision="bf16"'):
        EgoHandDetector(replace(detector, weights=tmp_path / "hand_detector.onnx"))
    with pytest.raises(ValueError, match='backbone_precision="bf16"'):
        FastEgoHandPredictor(pose)


def test_every_stack_name_is_one_cli_subcommand_for_both_roles() -> None:
    """The CLI unions come from EGOHAND_STACKS, so a name means the same stack in every tool."""
    from posekit.models import EGOHAND_STACKS

    for name, preset in EGOHAND_STACKS.items():
        stack: EgoHandStack = preset(None)
        assert tyro.cli(HandDetectorConfig, args=[name]) == stack.detector
        assert tyro.cli(HandPose3dConfig, args=[name]) == stack.pose
    assert EgoHandStack.reference() == EgoHandStack(EgoHandDetectorConfig(), EgoHandConfig())
