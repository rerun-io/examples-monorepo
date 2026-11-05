"""Calibration must reject evaluation overlap before any model is loaded."""

from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest


def test_body_calibration_rejects_same_source_frame() -> None:
    from posekit.models.sam3d_body_quantization import BodyCalibrationSplit, CalibrationFrame

    with pytest.raises(ValueError, match="indices"):
        BodyCalibrationSplit([CalibrationFrame("segment/phone", 5, "a" * 64)], [CalibrationFrame("segment/phone", 5, "b" * 64)])


def test_body_calibration_rejects_duplicate_pixels_across_sources() -> None:
    from posekit.models.sam3d_body_quantization import BodyCalibrationSplit, CalibrationFrame

    with pytest.raises(ValueError, match="decoded"):
        BodyCalibrationSplit([CalibrationFrame("segment/phone", 5, "a" * 64)], [CalibrationFrame("other/phone", 9, "a" * 64)])


def test_body_calibration_allows_disjoint_sources_and_pixels() -> None:
    from posekit.models.sam3d_body_quantization import BodyCalibrationSplit, CalibrationFrame

    split = BodyCalibrationSplit([CalibrationFrame("segment/phone", 5, "a" * 64)], [CalibrationFrame("segment/phone", 9, "b" * 64)])
    assert len(split.calibration) == len(split.evaluation) == 1


def test_committed_calibration_controls_artifact_identity() -> None:
    from posekit.models.sam3d_body import SAM3D_BODY_REVISION
    from posekit.models.sam3d_body_quantization import body_backbone_path, load_body_calibration

    calibration = load_body_calibration()
    assert calibration.revision == SAM3D_BODY_REVISION
    assert len(calibration.split.evaluation) >= 100
    base = Path("backbone.onnx")
    original = body_backbone_path(base, calibration, 8)
    assert body_backbone_path(original, calibration, 8) == original
    changed_ranges = dict(calibration.input_amax)
    name = next(iter(changed_ranges))
    changed_ranges[name] *= 1.01
    assert body_backbone_path(base, replace(calibration, input_amax=changed_ranges), 8) != original
    assert body_backbone_path(base, calibration, 4) != original


def test_fast_config_selects_calibrated_gpu_pipeline_by_default() -> None:
    from posekit.models.sam3d_body_fast import Sam3dBodyFastConfig

    config = Sam3dBodyFastConfig()
    assert config.gpu_preprocessing
    assert config.backbone_onnx is not None
    assert config.decoder_onnx_dir is not None


def test_body_precision_plan_can_preserve_outliers_without_changing_other_modules() -> None:
    from trtkit.onnx_qdq import Fp8Linear, KeepPrecision, Nvfp4Linear

    from posekit.models.sam3d_body_quantization import BodyPrecisionPlan, body_qdq_plan, load_body_calibration

    calibration = load_body_calibration()
    plan = body_qdq_plan(calibration, 8, BodyPrecisionPlan(
        linears={"encoder.blocks.22.mlp.w3": "bf16", "encoder.blocks.0.attn.qkv": "fp8"},
        attention={22: "bf16"},
    ))
    assert isinstance(plan.modules["encoder.blocks.22.mlp.w3"], KeepPrecision)
    assert isinstance(plan.modules["encoder.blocks.0.attn.qkv"], Fp8Linear)
    assert isinstance(plan.modules["encoder.blocks.0.mlp.w3"], Nvfp4Linear)
    assert isinstance(plan.modules["encoder.blocks.31.mlp.w3"], Fp8Linear)
    assert isinstance(plan.attention[22], KeepPrecision)


def test_body_precision_plan_rejects_misspelled_modules() -> None:
    from posekit.models.sam3d_body_quantization import BodyPrecisionPlan, body_qdq_plan, load_body_calibration

    with pytest.raises(ValueError, match="Unknown"):
        body_qdq_plan(load_body_calibration(), 8, BodyPrecisionPlan(linears={"encoder.block.22.mlp.w3": "bf16"}))


def test_body_artifact_identity_includes_complete_plan_and_source_version() -> None:
    from posekit.models.sam3d_body_quantization import BodyPrecisionPlan, body_backbone_path, load_body_calibration

    calibration = load_body_calibration()
    base = Path("backbone.onnx")
    kept = BodyPrecisionPlan(linears={"encoder.blocks.22.mlp.w3": "bf16"})
    fp8 = BodyPrecisionPlan(linears={"encoder.blocks.22.mlp.w3": "fp8"})
    original = body_backbone_path(base, calibration, 8, plan=kept, source_version="v3")
    assert body_backbone_path(original, calibration, 8, plan=kept, source_version="v3") == original
    assert body_backbone_path(base, calibration, 8, plan=fp8, source_version="v3") != original
    assert body_backbone_path(base, calibration, 8, plan=kept, source_version="v4") != original
    assert body_backbone_path(base, calibration, 8, plan=BodyPrecisionPlan(attention={22: "bf16"}), source_version="v3") != original
    reordered = BodyPrecisionPlan(linears={"encoder.blocks.0.mlp.w3": "fp8", "encoder.blocks.22.mlp.w3": "bf16"})
    reversed_order = BodyPrecisionPlan(linears=dict(reversed(list(reordered.linears.items()))))
    assert body_backbone_path(base, calibration, 8, plan=reordered) == body_backbone_path(base, calibration, 8, plan=reversed_order)


def test_fast_config_accepts_separate_body_and_hand_precision_plans() -> None:
    from posekit.models.sam3d_body_fast import Sam3dBodyFastConfig
    from posekit.models.sam3d_body_quantization import BodyPrecisionPlan

    body = BodyPrecisionPlan(linears={"encoder.blocks.22.mlp.w3": "bf16"})
    hand = BodyPrecisionPlan(attention={31: "bf16"})
    config = Sam3dBodyFastConfig(backbone_plan=body, hand_backbone_plan=hand)
    assert config.backbone_plan == body
    assert config.hand_backbone_plan == hand


def test_cpu_crop_ablation_rejects_a_hand_precision_plan_it_cannot_apply() -> None:
    """The shared-backbone ablation must not silently ignore a requested hand plan."""
    from posekit.models.sam3d_body_fast import Sam3dBodyFastConfig
    from posekit.models.sam3d_body_quantization import BodyPrecisionPlan

    with pytest.raises(ValueError, match="hand.*GPU"):
        Sam3dBodyFastConfig(gpu_preprocessing=False, hand_backbone_plan=BodyPrecisionPlan(attention={31: "bf16"}))


def test_compiled_torch_backbone_rejects_a_simultaneous_tensorrt_backbone() -> None:
    """A requested Torch optimization must not silently disappear behind TensorRT."""
    from posekit.models.sam3d_body_fast import Sam3dBodyFastConfig

    with pytest.raises(ValueError, match="compile_torch_backbone.*backbone_onnx"):
        Sam3dBodyFastConfig(compile_torch_backbone=True)


@pytest.mark.parametrize("backbone_onnx,precision", [(None, "nvfp4"), (Path("backbone.onnx"), "bf16")])
def test_unquantized_backbones_reject_precision_plans_they_cannot_apply(backbone_onnx: Path | None, precision: Literal["nvfp4", "bf16"]) -> None:
    """An explicit module precision must not be silently ignored by an unquantized path."""
    from posekit.models.sam3d_body_fast import Sam3dBodyFastConfig
    from posekit.models.sam3d_body_quantization import BodyPrecisionPlan

    with pytest.raises(ValueError, match="precision plans require"):
        Sam3dBodyFastConfig(
            backbone_onnx=backbone_onnx, backbone_precision=precision,
            hand_backbone_plan=BodyPrecisionPlan(attention={31: "fp8"}),
        )


def test_development_split_excludes_calibration_and_acceptance_pixels() -> None:
    from posekit.models.sam3d_body_quantization import CalibrationFrame, load_body_calibration, validate_development_split

    calibration = load_body_calibration()
    for reserved in (calibration.split.calibration[0], calibration.split.evaluation[0]):
        with pytest.raises(ValueError, match="indices"):
            validate_development_split(calibration, [replace(reserved, sha256="different")])
        with pytest.raises(ValueError, match="decoded"):
            validate_development_split(calibration, [CalibrationFrame("new/source", 0, reserved.sha256)])
    validate_development_split(calibration, [CalibrationFrame("new/source", 0, "different")])


def test_accuracy_policy_keeps_stress_sources_out_of_acceptance() -> None:
    """The declared stress source is reported, while every real source must pass."""
    from posekit.apis.body_accuracy import AccuracyBand, acceptance_passes

    good = AccuracyBand(1.0, 2.0, 100.0, 3.0, 8.0, True)
    bad = AccuracyBand(4.0, 9.0, 100.0, 3.0, 8.0, False)
    bands = {"person": {"pixels": good}, "empty": {"pixels": bad}}
    assert acceptance_passes(bands, ("empty",))
    assert not acceptance_passes(bands, ())
    assert not acceptance_passes({"person": {"pixels": bad}, "empty": {"pixels": good}}, ("empty",))
    assert not acceptance_passes({"empty": {"pixels": good}}, ("empty",))


def test_prior_development_report_cannot_be_relabelled_as_acceptance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from serde.json import to_json

    from posekit.apis import body_accuracy
    from posekit.models.sam3d_body_quantization import CalibrationFrame, load_body_calibration

    def unexpected_catalog(*args: object, **kwargs: object) -> None:
        pytest.fail("Development report must be rejected before catalog access")

    monkeypatch.setattr(body_accuracy, "CatalogClient", unexpected_catalog)
    report_path = tmp_path / "development.json"
    report = body_accuracy.BodyAccuracyReport(
        [CalibrationFrame("development", 0, "d" * 64)], 0, {}, False, [], {}, split="development",
    )
    report_path.write_text(to_json(report))
    config = body_accuracy.Config(calibration=tmp_path / "calibration.rrd", output=tmp_path / "output.json", evaluation_report=report_path)
    with pytest.raises(ValueError, match="development_frames"):
        body_accuracy.evaluation_inputs(config, load_body_calibration())
