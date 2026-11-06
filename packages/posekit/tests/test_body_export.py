"""The SAM export preserves its stored BF16 residual and boundary rounding."""

import pytest
import torch

pytest.importorskip("yacs", reason="SAM body exports require the posekit-body environment")


def test_body_export_preserves_native_bf16_features() -> None:
    from dinov3.models.vision_transformer import DinoVisionTransformer

    from posekit.models.sam3d_body_onnx import ExportBodyBackbone

    torch.manual_seed(42)
    encoder = DinoVisionTransformer(
        img_size=32, patch_size=16, embed_dim=32, depth=2, num_heads=2, ffn_ratio=2.0,
        norm_layer="layernormbf16", ffn_layer="swiglu", n_storage_tokens=4,
        layerscale_init=1.0, pos_embed_rope_dtype="fp32",
    ).eval()
    encoder.init_weights()
    encoder.bfloat16()
    images = torch.randn(2, 3, 32, 32).bfloat16()
    with torch.inference_mode():
        expected = encoder.get_intermediate_layers(images, n=1, reshape=True, norm=True)[-1]
        exported = ExportBodyBackbone(encoder)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = exported(images.float())
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert encoder.cls_token.dtype == torch.bfloat16
    assert encoder.norm.weight.dtype == torch.bfloat16
    assert encoder.get_parameter("blocks.0.attn.qkv.weight").dtype == torch.float32


def test_body_runtime_adapter_accepts_bf16_features() -> None:
    """The stored-BF16 export's output satisfies the adapter's checked contract."""
    from trtkit import TensorSpec, TorchRuntime

    from posekit.models.sam3d_body_onnx import TensorRtBodyBackbone

    projection = torch.nn.Conv2d(3, 1280, kernel_size=1)
    assert projection.bias is not None
    with torch.no_grad():
        projection.weight.zero_()
        projection.bias.fill_(1.25)
    runtime = TorchRuntime(
        torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((32, 32)), projection),
        input_specs=[TensorSpec("images", (3, 512, 512), torch.float32)],
        output_specs=[TensorSpec("features", (1280, 32, 32), torch.bfloat16)],
        max_batch_size=1,
    )
    actual = TensorRtBodyBackbone(runtime)(torch.zeros(1, 3, 512, 512))
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, torch.full((1, 1280, 32, 32), 1.25, dtype=torch.bfloat16), rtol=0.0, atol=0.0)
