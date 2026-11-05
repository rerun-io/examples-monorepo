"""The released fast calibration has a strict, auditable evaluation split."""

from dataclasses import replace

import pytest
from serde import SerdeError
from serde.json import from_json, to_json

from posekit.models.egohand_quantization import EgoHandCalibration, load_calibration


def test_calibration_has_complete_ranges_and_a_disjoint_split() -> None:
    calibration = load_calibration()
    assert calibration.n_crops == 138
    assert len(calibration.input_amax) == 160
    assert len(calibration.attention_amax) == 96
    assert not set(calibration.frame_indices).intersection(calibration.excluded_mount_indices)
    assert len(calibration.frame_sha256) == len(calibration.frame_indices)
    assert calibration.frame_indices[0] == 2200
    assert calibration.frame_indices[-1] == 8150


def test_calibration_satisfies_the_runtime_shared_range_contract() -> None:
    from posekit.models.egohand_onnx import DinoActivationRanges

    assert isinstance(load_calibration(), DinoActivationRanges)


def test_calibration_rejects_bad_ranges_and_unknown_fields() -> None:
    calibration = load_calibration()
    with pytest.raises(ValueError, match="finite and positive"):
        replace(calibration, input_amax={**calibration.input_amax, "encoder.blocks.0.attn.qkv": float("nan")})
    with pytest.raises(SerdeError):
        from_json(EgoHandCalibration, to_json(calibration)[:-1] + ',"unknown":true}')


def test_measurement_rejects_evaluation_frames_before_loading_models() -> None:
    """A caller cannot accidentally use a mount-fit or benchmark frame to calibrate."""
    import hashlib

    import numpy as np
    import torch

    from posekit.cameras import CalibratedCamera
    from posekit.models.egohand_quantization import measure_calibration

    frame = torch.zeros((8, 8, 3), dtype=torch.uint8)
    camera = CalibratedCamera(np.eye(3), np.empty(0), 8, 8, "PINHOLE")
    with pytest.raises(ValueError, match="mount-fit"):
        measure_calibration([(7, frame)], camera, segment="example", excluded_mount_indices=[7])
    digest = hashlib.sha256(frame.numpy().tobytes()).hexdigest()
    with pytest.raises(ValueError, match="evaluation frame"):
        measure_calibration([(7, frame)], camera, segment="example", excluded_frame_sha256=[digest])


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
