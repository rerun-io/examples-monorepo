"""Calibration must reject evaluation overlap before any model is loaded."""

from dataclasses import replace
from pathlib import Path

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
