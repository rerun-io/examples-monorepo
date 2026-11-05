"""SAM body exports through the shared strongly typed TensorRT and QDQ recipe."""

from pathlib import Path

import onnx
import torch
from jaxtyping import Float, Float32
from torch import Tensor, nn
from trtkit import TensorRtBackendConfig, export_onnx
from trtkit.artifacts import build_once
from trtkit.onnx_qdq import quantize_qdq, save_with_external_data

from egohand_dinov3.models.vision_transformer import DinoVisionTransformer
from posekit.models.sam3d_body_quantization import (
    BodyBackbonePrecision,
    BodyCalibration,
    BodyPrecisionPlan,
    body_backbone_path,
    body_qdq_plan,
)
from posekit.runtimes import TensorRuntime, create_runtime_from_onnx, run_chunked
from sam3d_body.models.backbones.dinov3 import Dinov3Backbone
from sam3d_body.models.modules.transformer import TransformerDecoderLayer


class ExportBodyBackbone(nn.Module):
    """Keep native BF16 rounding and named FP32 linear weights for QDQ rewriting."""

    def __init__(self, encoder: DinoVisionTransformer) -> None:
        super().__init__()
        self.encoder: DinoVisionTransformer = encoder
        for module in self.encoder.modules():
            if isinstance(module, nn.Linear):
                module.float()

    def forward(self, images: Float32[Tensor, "b 3 h w"]) -> Float[Tensor, "b c ph pw"]:
        """Round input and final norm exactly as the stored-BF16 native backbone does."""
        features: Tensor | tuple[Tensor, ...] = self.encoder.get_intermediate_layers(images.to(torch.bfloat16), n=1, reshape=True, norm=True)[-1]
        assert isinstance(features, Tensor)
        return features.to(torch.bfloat16)


def build_body_backbone(
    backbone: nn.Module, source_path: Path, quantized_base: Path, calibration: BodyCalibration | None,
    backend: TensorRtBackendConfig, fp8_tail_blocks: int, precision: BodyBackbonePrecision = "nvfp4",
    *, plan: BodyPrecisionPlan | None = None,
) -> TensorRuntime:
    """Export the actual SAM weights at 512 pixels and quantize with their own measured ranges."""
    def export() -> None:
        if not isinstance(backbone, Dinov3Backbone):
            raise TypeError("SAM body export requires the released DINOv3 backbone.")
        images: Tensor = torch.zeros((2, 3, 512, 512), device="cuda")
        with torch.inference_mode():
            expected: Tensor = backbone(images.to(torch.bfloat16)).clone()
            exported: ExportBodyBackbone = ExportBodyBackbone(backbone.encoder)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                torch.testing.assert_close(exported(images), expected, atol=0.0, rtol=0.0)
        export_onnx(
            exported, (images,), source_path,
            input_names=["images"], output_names=["features"], compute_dtype=torch.bfloat16,
            dynamic_batch_max=backend.max_batch_size,
        )

    if precision == "bf16":
        build_once(source_path, export)
        return create_runtime_from_onnx(source_path, backend)
    if calibration is None:
        raise ValueError("The NVFP4 backbone requires measured calibration ranges.")
    quantized: Path = body_backbone_path(quantized_base, calibration, fp8_tail_blocks, plan=plan, source_version=source_path.name)

    def quantize() -> None:
        build_once(source_path, export)
        save_with_external_data(quantize_qdq(onnx.load(str(source_path)), body_qdq_plan(calibration, fp8_tail_blocks, plan)), quantized)

    build_once(quantized, quantize)
    return create_runtime_from_onnx(quantized, backend)


class TensorRtBodyBackbone(nn.Module):
    """The released 512-pixel feature-map contract over a GPU TensorRT runtime."""

    def __init__(self, runtime: TensorRuntime) -> None:
        super().__init__()
        self.runtime: TensorRuntime = runtime

    def forward(self, images: Tensor, extra_embed: Tensor | None = None) -> Float[Tensor, "b 1280 32 32"]:
        """Return the spatial tokens, preserving the model's ray-conditioning interface."""
        if extra_embed is not None:
            raise ValueError("SAM DINOv3 applies ray conditioning after the backbone.")
        return run_chunked(self.runtime, {"images": images.float()})["features"]


class TensorRtDecoderLayer(nn.Module):
    """One learned decoder layer, with masked padding to a fixed 148-token profile."""

    def __init__(self, runtime: TensorRuntime) -> None:
        super().__init__()
        self.runtime: TensorRuntime = runtime

    def forward(
        self, x: Tensor, context: Tensor, x_pe: Tensor, context_pe: Tensor, x_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        count: int = x.shape[1]
        if count > 148:
            raise ValueError("The full SAM body path has at most four keypoint prompts (148 tokens).")
        if x_mask is None:
            x_mask = x.new_ones(x.shape[:2])
        padding: int = 148 - count
        values: dict[str, Tensor] = {
            "tokens": torch.nn.functional.pad(x, (0, 0, 0, padding)),
            "context": context,
            "token_pe": torch.nn.functional.pad(x_pe, (0, 0, 0, padding)),
            "context_pe": context_pe.expand(len(x), -1, -1).contiguous(),
            "mask": torch.nn.functional.pad(x_mask, (0, padding)),
        }
        output = run_chunked(self.runtime, values)
        return output["tokens_out"][:, :count], output["context_out"]


class ExportDecoderLayer(nn.Module):
    """Released layer math with a functional attention-mask diagonal for ONNX export."""

    def __init__(self, layer: TransformerDecoderLayer) -> None:
        super().__init__()
        self.layer: TransformerDecoderLayer = layer

    def forward(self, x: Tensor, context: Tensor, x_pe: Tensor, context_pe: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        layer: TransformerDecoderLayer = self.layer
        if layer.repeat_pe:
            x_pe = layer.ln_pe_1(x_pe)
            context_pe = layer.ln_pe_2(context_pe)
        value: Tensor = layer.ln1(x)
        query: Tensor = value + x_pe if layer.repeat_pe and not layer.skip_first_pe else value
        valid: Tensor = mask > 0
        attention_mask: Tensor = (valid[:, :, None] & valid[:, None, :]) | torch.eye(x.shape[1], device=x.device, dtype=torch.bool)[None]
        x = x + layer.self_attn(q=query, k=query, v=value, attn_mask=attention_mask)
        query = layer.ln2_1(x)
        value = layer.ln2_2(context)
        key: Tensor = value
        if layer.repeat_pe:
            query = query + x_pe
            key = key + context_pe
        x = x + layer.cross_attn(q=query, k=key, v=value)
        x = layer.ffn(layer.ln3(x), identity=x)
        if layer.enable_twoway:
            query = layer.ln4_1(context)
            value = layer.ln4_2(x)
            key = value
            if layer.repeat_pe:
                query = query + context_pe
                key = key + x_pe
            context = context + layer.cross_attn_2(q=query, k=key, v=value, attn_mask=valid[:, None, :].expand(-1, context.shape[1], -1))
        return x, context


def build_decoder_layer(
    layer: TransformerDecoderLayer, path: Path, backend: TensorRtBackendConfig,
) -> TensorRtDecoderLayer:
    """Keep MHR outside TensorRT; export the learned attention and MLP of one layer."""
    def export() -> None:
        export_onnx(
            ExportDecoderLayer(layer).eval(),
            (torch.zeros((2, 148, 1024), device="cuda"), torch.zeros((2, 1024, 1280), device="cuda"),
             torch.zeros((2, 148, 1024), device="cuda"), torch.zeros((2, 1024, 1280), device="cuda"),
             torch.ones((2, 148), device="cuda")),
            path, input_names=["tokens", "context", "token_pe", "context_pe", "mask"],
            output_names=["tokens_out", "context_out"], compute_dtype=torch.bfloat16,
            dynamic_batch_max=backend.max_batch_size,
        )
    build_once(path, export)
    return TensorRtDecoderLayer(create_runtime_from_onnx(path, backend))
