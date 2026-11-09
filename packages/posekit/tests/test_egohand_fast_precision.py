"""The fast executor must also support GPUs without FP8 or NVFP4."""

from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("egohand_inference", reason="requires posekit-hands")

from trtkit.onnx_qdq import QdqPlan

from posekit.models.egohand_fast import EgoHandFastConfig, export_quantized_backbone_once
from posekit.models.egohand_quantization import EgoHandCalibration, load_calibration


def test_bf16_fast_export_never_loads_or_quantizes_calibration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import posekit.models.egohand_fast as fast

    calls: list[Path] = []

    def export(weights: Path, soma: Path, onnx: Path, **kwargs: object) -> None:
        calls.append(onnx)

    monkeypatch.setattr(fast, "export_backbone_once", export)
    config: EgoHandFastConfig = EgoHandFastConfig(backbone_precision="bf16", backbone_source_onnx=tmp_path / "bf16.onnx")
    assert fast.export_fast_backbone_once(config, tmp_path / "weights", tmp_path / "soma") == config.backbone_source_onnx
    assert calls == [config.backbone_source_onnx]


def test_fp8_fast_export_uses_all_blocks_and_distinct_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import posekit.models.egohand_fast as fast

    tails: list[int] = []
    monkeypatch.setattr(fast, "export_backbone_once", lambda *args, **kwargs: None)
    monkeypatch.setattr(fast, "quantize_backbone", lambda *args, fp8_tail_blocks, fp8_prefix_blocks: tails.append(fp8_tail_blocks))
    monkeypatch.setattr(fast, "build_once", lambda path, build: build())
    config: EgoHandFastConfig = EgoHandFastConfig(backbone_precision="fp8", backbone_onnx=tmp_path / "backbone.onnx")
    fp8: Path = fast.export_fast_backbone_once(config, tmp_path / "weights", tmp_path / "soma")
    nvfp4: Path = export_quantized_backbone_once(replace(config, backbone_precision="nvfp4"), tmp_path / "weights", tmp_path / "soma", load_calibration())
    assert tails == [32, 8]
    assert fp8 != nvfp4


def test_nvfp4_can_protect_early_and_final_blocks(tmp_path: Path) -> None:
    from trtkit.onnx_qdq import Fp8Linear, Nvfp4Linear

    from posekit.models.egohand_onnx import backbone_qdq_plan
    from posekit.models.egohand_quantization import calibrated_backbone_path

    calibration: EgoHandCalibration = load_calibration()
    plan: QdqPlan = backbone_qdq_plan(calibration, fp8_prefix_blocks=4, fp8_tail_blocks=8)
    assert sum(isinstance(value, Fp8Linear) for value in plan.modules.values()) == 60
    assert sum(isinstance(value, Nvfp4Linear) for value in plan.modules.values()) == 100
    for name, precision in plan.modules.items():
        block: int = int(name.split('.')[2])
        assert isinstance(precision, Fp8Linear) == (block < 4 or block >= 24)
    source: Path = tmp_path / 'backbone.onnx'
    assert calibrated_backbone_path(source, calibration, fp8_prefix_blocks=4, fp8_tail_blocks=8) != calibrated_backbone_path(source, calibration, fp8_tail_blocks=8)


def test_catalog_presets_match_measured_batch_one_precision() -> None:
    from trtkit import TensorRtBackendConfig

    from posekit.models.egohand_fast import catalog_egohand_configs

    detector, pose = catalog_egohand_configs()
    assert isinstance(detector.backend, TensorRtBackendConfig)
    assert detector.backend.max_batch_size == detector.backend.opt_batch_size == 1
    assert detector.gpu_preprocessing and not detector.fp16_trunk
    assert detector.fp8_scope == "projections" and detector.fp8_calibration is not None
    assert detector.fp8_calibration.is_file()
    assert pose.backbone_precision == "nvfp4" and pose.fp8_tail_blocks == 24 and pose.fp8_prefix_blocks == 0
    assert pose.batch_size == pose.backbone_backend.max_batch_size == pose.backbone_backend.opt_batch_size == 2
    assert pose.calibration is not None and pose.calibration.is_file()

    detector, pose = catalog_egohand_configs(backbone_precision="bf16")
    assert detector.fp16_trunk and detector.fp8_calibration is None
    assert pose.backbone_precision == "bf16" and pose.calibration is None
