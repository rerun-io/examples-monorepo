"""The released fast calibration has a strict, auditable evaluation split."""

from dataclasses import replace

import pytest
from serde import SerdeError
from serde.json import from_json, to_json

from posekit.models.egohand_quantization import EgoHandCalibration, load_calibration


def test_calibration_has_complete_ranges_and_a_disjoint_split() -> None:
    """The five-dataset catalog record: 200 frames, 243 crops; its frames were hash-checked against every evaluation frame."""
    calibration = load_calibration()
    assert calibration.segment == "egohand-catalog-calibration-v1"
    assert calibration.n_crops == 243
    assert len(calibration.input_amax) == 160
    assert len(calibration.attention_amax) == 96
    assert calibration.frame_indices == list(range(200)) and not calibration.excluded_mount_indices
    assert len(set(calibration.frame_sha256)) == len(calibration.frame_indices)


def test_calibration_satisfies_the_runtime_shared_range_contract() -> None:
    from posekit.models.dinov3_qdq import DinoActivationRanges

    assert isinstance(load_calibration(), DinoActivationRanges)


def test_calibration_rejects_bad_ranges_and_unknown_fields() -> None:
    calibration = load_calibration()
    with pytest.raises(ValueError, match="finite and positive"):
        replace(calibration, input_amax={**calibration.input_amax, "encoder.blocks.0.attn.qkv": float("nan")})
    with pytest.raises(SerdeError):
        from_json(EgoHandCalibration, to_json(calibration)[:-1] + ',"unknown":true}')


def test_changed_calibration_cannot_reuse_an_old_backbone(tmp_path) -> None:
    from posekit.models.egohand_quantization import calibrated_backbone_path

    calibration = load_calibration()
    updated = replace(calibration, input_amax={**calibration.input_amax, "encoder.blocks.0.attn.qkv": 100.0})
    original = calibrated_backbone_path(tmp_path / "backbone.onnx", calibration)
    assert original == calibrated_backbone_path(tmp_path / "backbone.onnx", load_calibration())
    assert original != calibrated_backbone_path(tmp_path / "backbone.onnx", updated)
    assert original != calibrated_backbone_path(tmp_path / "backbone.onnx", calibration, fp8_tail_blocks=4)
    with pytest.raises(ValueError, match="0.*32"):
        calibrated_backbone_path(tmp_path / "backbone.onnx", calibration, fp8_tail_blocks=33)
