"""Capture-safe geometry against the released conversion and MHR math."""

from pathlib import Path

import pytest
import torch

roma = pytest.importorskip("roma", reason="SAM body geometry requires roma")


@pytest.mark.parametrize("batch", [1, 3])
def test_constant_sparse_linear_matches_values_and_gradients(batch: int) -> None:
    from posekit.models.sam3d_body_mhr import ConstantSparseLinear

    torch.manual_seed(52)
    weights = torch.randn(29, 41)
    weights *= torch.rand_like(weights) < 0.08
    weights[4] = 0
    values = torch.randn(batch, 41, requires_grad=True)
    expected = torch.nn.functional.linear(values, weights)
    actual = ConstantSparseLinear(weights)(values)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    expected_gradient = torch.autograd.grad(expected.square().sum(), values)[0]
    actual_gradient = torch.autograd.grad(actual.square().sum(), values)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-5, rtol=1e-6)


@pytest.mark.parametrize("convention", ["xyz", "ZYX", "XZY"])
def test_euler_conversion_matches_roma(convention: str) -> None:
    from posekit.models.sam3d_body_geometry import rotmat_to_euler

    torch.manual_seed(7)
    rotations = roma.rotvec_to_rotmat(torch.randn(100, 3))
    actual = rotmat_to_euler(convention, rotations)
    expected = roma.rotmat_to_euler(convention, rotations)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0.0)


def test_compact_pose_conversion_matches_released_math() -> None:
    from posekit.models.sam3d_body_geometry import CompactPose
    from sam3d_body.models.modules.mhr_utils import compact_cont_to_model_params_body, compact_cont_to_model_params_hand

    torch.manual_seed(11)
    for kind, width, reference in (("body", 260, compact_cont_to_model_params_body), ("hand", 54, compact_cont_to_model_params_hand)):
        values = torch.randn(8, width)
        actual = CompactPose(kind, torch.device("cpu"))(values)
        torch.testing.assert_close(actual, reference(values), atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("twoway", [False, True])
@pytest.mark.parametrize("skip_first_pe", [False, True])
def test_export_decoder_mask_matches_released_layer(twoway: bool, skip_first_pe: bool) -> None:
    pytest.importorskip("trtkit", reason="SAM decoder export requires trtkit")
    from posekit.models.sam3d_body_onnx import ExportDecoderLayer
    from sam3d_body.models.modules.transformer import TransformerDecoderLayer

    torch.manual_seed(23)
    layer = TransformerDecoderLayer(
        token_dims=32, context_dims=40, num_heads=2, head_dims=8, mlp_dims=48,
        repeat_pe=True, skip_first_pe=skip_first_pe, enable_twoway=twoway,
    ).eval()
    tokens, context = torch.randn(2, 7, 32), torch.randn(2, 11, 40)
    token_pe, context_pe = torch.randn_like(tokens), torch.randn_like(context)
    mask = torch.tensor([[1., 1., 0., 1., 0., 0., 1.], [1., 0., 0., 1., 1., 0., 1.]])
    with torch.inference_mode():
        expected = layer(tokens, context, token_pe, context_pe, mask)
        actual = ExportDecoderLayer(layer)(tokens, context, token_pe, context_pe, mask)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, atol=0.0, rtol=0.0)


@pytest.mark.integration
def test_dynamic_mhr_matches_released_torchscript() -> None:
    from huggingface_hub import try_to_load_from_cache

    from posekit.models.sam3d_body import SAM3D_BODY_REVISION
    from posekit.models.sam3d_body_mhr import BodySkinning

    path = try_to_load_from_cache("facebook/sam-3d-body-dinov3", "assets/mhr_model.pt", revision=SAM3D_BODY_REVISION)
    if not torch.cuda.is_available() or not isinstance(path, str) or not Path(path).is_file():
        pytest.skip("requires CUDA and the cached SAM body assets/mhr_model.pt")
    reference = torch.jit.load(path, map_location="cuda")
    fast = BodySkinning(reference)
    indices = torch.arange(0, 18439, 41, device="cuda")
    selected = BodySkinning(reference, indices)
    torch.manual_seed(4)
    with torch.inference_mode():
        identity = torch.randn(2, 45, device="cuda") * 0.2
        pose = torch.randn(2, 204, device="cuda") * 0.03
        expression = torch.zeros(2, 72, device="cuda")
        expected = reference(identity, pose, expression)
        actual = fast(identity, pose, expression)
        torch.testing.assert_close(actual[0], expected[0], atol=0.003, rtol=0.0)
        torch.testing.assert_close(actual[1], expected[1], atol=0.0003, rtol=0.0)
        sparse = selected(identity, pose, expression)
        torch.testing.assert_close(sparse.vertices, expected[0][:, indices], atol=0.003, rtol=0.0)
        torch.testing.assert_close(sparse.state, expected[1], atol=0.0003, rtol=0.0)
