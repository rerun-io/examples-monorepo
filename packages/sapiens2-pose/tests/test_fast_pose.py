"""Exact CPU checks for the fast Sapiens2 head."""

import pytest

# ruff: noqa: E402

torch = pytest.importorskip("torch")
from torch import nn

from sapiens2_pose.api.fast_pose import DeferredHead, FusedPolyphase


def test_polyphase_matches_transpose_convolution() -> None:
    torch.manual_seed(7)
    original = nn.ConvTranspose2d(3, 5, 4, stride=2, padding=1, bias=False)
    inputs = torch.randn(3, 3, 7, 6)
    torch.testing.assert_close(FusedPolyphase(original)(inputs), original(inputs), atol=1e-5, rtol=1e-5)


def test_deferred_head_matches_original_heatmaps() -> None:
    torch.manual_seed(8)
    head = nn.Module()
    head.deconv_layers = nn.Sequential(
        nn.ConvTranspose2d(3, 5, 4, 2, 1, bias=False),
        nn.InstanceNorm2d(5),
        nn.SiLU(),
        nn.ConvTranspose2d(5, 4, 4, 2, 1, bias=False),
        nn.InstanceNorm2d(4),
        nn.SiLU(),
    )
    head.conv_layers = nn.Sequential(nn.Conv2d(4, 4, 1), nn.InstanceNorm2d(4), nn.SiLU())
    head.conv_pose = nn.Conv2d(4, 2, 1)
    inputs = torch.randn(3, 3, 7, 6)
    expected = head.conv_pose(head.conv_layers(head.deconv_layers(inputs)))
    torch.testing.assert_close(DeferredHead(head)(inputs), expected, atol=1e-5, rtol=1e-5)


def test_nvfp4_grid_blocks_and_scale_floor() -> None:
    from sapiens2_pose.api.nvfp4 import bf16_round_up, quant_nvfp4_mse, round_e2m1_codes

    values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, -6.0])
    assert round_e2m1_codes(values).tolist() == [0, 0, 1, 2, 2, 4, 4, 6, 6, 7, 15]
    weights = torch.cat([torch.zeros(2, 16), torch.ones(2, 16)], dim=1)
    packed, scales, _ = quant_nvfp4_mse(weights)
    assert packed.shape == (2, 16)
    assert scales.shape == (2, 2)
    assert float(scales.float().min()) >= 2**-9
    assert torch.all(scales[:, 0].float() < scales[:, 1].float())
    assert bf16_round_up(1.001) == 1.0078125
    assert bf16_round_up(1.0) == 1.0


def test_calibration_is_strict_and_complete() -> None:
    import json

    from serde import SerdeError
    from serde.json import from_json

    from sapiens2_pose.api.fast_calibration import FastPoseCalibration, calibration_path, load_calibration

    calibration = load_calibration()
    assert calibration.input_hw == (768, 576)
    assert len(calibration.attention_amax) == 72
    data = json.loads(calibration_path().read_text())
    data["unexpected"] = 1
    with pytest.raises(SerdeError):
        from_json(FastPoseCalibration, json.dumps(data))


def test_polyphase_export_keeps_batch_dynamic(tmp_path) -> None:
    """The exact head must export without specializing its depth-to-space batch."""
    onnx = pytest.importorskip("onnx")
    from trtkit import export_onnx

    head = FusedPolyphase(nn.ConvTranspose2d(3, 5, 4, 2, 1, bias=False)).eval()
    path = tmp_path / "head.onnx"
    export_onnx(head, (torch.randn(3, 3, 7, 6),), path, input_names=["inputs"], output_names=["heatmaps"], dynamic_batch_max=16)
    graph = onnx.load(path).graph
    assert graph.input[0].type.tensor_type.shape.dim[0].dim_param
    assert graph.output[0].type.tensor_type.shape.dim[0].dim_param
