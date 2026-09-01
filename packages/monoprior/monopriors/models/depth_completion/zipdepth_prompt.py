"""Prompt-conditioned metric ZipDepth model.

The wrapper keeps the vendored ZipDepth-base tensors unchanged, injects a
normalized metric prompt into all four decoder stages, and returns depth in the
same units as the prompt (metres for the public API).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import torch
import torch.nn.functional as F
from jaxtyping import Float, Float32, UInt8
from torch import Tensor, nn
from trtkit import (
    BackendConfig,
    OnnxBackendConfig,
    TensorRtBackendConfig,
    TensorRuntime,
    TensorSpec,
    TorchBackendConfig,
    TorchRuntime,
    create_runtime_from_onnx,
)

from monopriors.models.depth_completion.base_completion_depth import (
    DEPTH_OUTPUT_NAME,
    IMAGE_INPUT_NAME,
    PROMPT_INPUT_NAME,
    BaseCompletionPredictor,
)
from monopriors.third_party.zipdepth.architecture import EncoderOutput, FeaturePyramid, ZipDepth, create_model
from monopriors.third_party.zipdepth.model_utils import StateDict, strip_state_dict_prefixes

PROMPT_HEIGHT: int = 192
PROMPT_WIDTH: int = 256
MIN_PROMPT_DEPTH_M: float = 0.1
MAX_PROMPT_DEPTH_M: float = 4.0
MIN_PROMPT_SPAN_M: float = 1.0e-6


def _make_prompt_branch(channels: int) -> nn.Sequential:
    """Build one PromptDA-style depthwise-separable prompt branch.

    Args:
        channels: Decoder-stage channel count.

    Returns:
        A branch mapping float normalized prompt depth with shape
        ``(batch, 1, height, width)`` to float features with shape
        ``(batch, channels, height, width)``. The last pointwise convolution is
        exactly zero-initialized, so the branch initially contributes zero.
    """
    input_conv: nn.Conv2d = nn.Conv2d(1, channels, kernel_size=3, padding=1)
    depthwise_conv: nn.Conv2d = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
    final_conv: nn.Conv2d = nn.Conv2d(channels, channels, kernel_size=1)
    branch: nn.Sequential = nn.Sequential(
        input_conv,
        nn.ReLU(inplace=True),
        depthwise_conv,
        nn.ReLU(inplace=True),
        final_conv,
    )
    nn.init.zeros_(final_conv.weight)
    if final_conv.bias is not None:
        nn.init.zeros_(final_conv.bias)
    return branch


class ZipDepthPrompt(nn.Module):
    """ZipDepth-base with PromptDA-style multi-scale metric conditioning."""

    def __init__(self, backbone: ZipDepth | None = None) -> None:
        super().__init__()
        self.backbone: ZipDepth = create_model(variant="base") if backbone is None else backbone
        if self.backbone.variant != "base":
            raise ValueError(f"ZipDepthPrompt requires the base variant, got {self.backbone.variant!r}")
        if not self.backbone.decoder.convex_up.use_unfold:
            raise ValueError("ZipDepthPrompt currently requires ZipDepth's unfold-based convex head")
        self.prompt_branches: nn.ModuleList = nn.ModuleList(
            [_make_prompt_branch(channels) for channels in (288, 192, 144, 96)]
        )

    def forward(
        self,
        image: UInt8[Tensor, "b 3 h w"] | Float[Tensor, "b 3 h w"],
        prompt_depth: Float32[Tensor, "b 1 192 256"],
    ) -> Float[Tensor, "b 1 h w"]:
        """Predict metric depth from RGB and a low-resolution metric prompt.

        Confidence-invalid prompt pixels must be zeroed by the caller. This
        method also rejects non-finite values and depths outside 0.1--4.0 m
        when computing the per-image prompt range.

        Args:
            image: UInt8 RGB or float RGB tensor with shape
                ``(batch, 3, height, width)``. UInt8 values are scaled to
                ``[0, 1]``; float values are already expected in that range.
            prompt_depth: Float32 metric prompt in metres with shape
                ``(batch, 1, 192, 256)``. Invalid pixels are zero.

        Returns:
            Float metric depth in metres with shape
                ``(batch, 1, height, width)``.
        """
        metric_depth_b1hw: Float[Tensor, "b 1 h w"]
        metric_depth_b1hw, _, _ = self.forward_with_range(image, prompt_depth)
        return metric_depth_b1hw

    def forward_with_range(
        self,
        image: UInt8[Tensor, "b 3 h w"] | Float[Tensor, "b 3 h w"],
        prompt_depth: Float32[Tensor, "b 1 192 256"],
    ) -> tuple[
        Float[Tensor, "b 1 h w"],
        Float32[Tensor, "b 1 1 1"],
        Float32[Tensor, "b 1 1 1"],
    ]:
        """Predict metric depth and expose the prompt range for TRT export.

        The two scalar outputs force TensorRT to materialize the input-derived
        reductions. Normal callers use :meth:`forward` and receive depth only.
        """
        if image.dtype == torch.uint8:
            image_b3hw: Float[Tensor, "b 3 h w"] = image.to(dtype=self.backbone.mean.dtype) / 255.0
        else:
            image_b3hw = image.to(dtype=self.backbone.mean.dtype)

        valid_b1hw: torch.Tensor = (
            torch.isfinite(prompt_depth)
            & (prompt_depth >= MIN_PROMPT_DEPTH_M)
            & (prompt_depth <= MAX_PROMPT_DEPTH_M)
        )
        valid_any_b111: torch.Tensor = valid_b1hw.any(dim=(1, 2, 3), keepdim=True)
        masked_min_b111: Float32[Tensor, "b 1 1 1"] = torch.where(
            valid_b1hw, prompt_depth, torch.full_like(prompt_depth, torch.inf)
        ).amin(dim=(1, 2, 3), keepdim=True)
        masked_max_b111: Float32[Tensor, "b 1 1 1"] = torch.where(
            valid_b1hw, prompt_depth, torch.full_like(prompt_depth, -torch.inf)
        ).amax(dim=(1, 2, 3), keepdim=True)
        min_depth_b111: Float32[Tensor, "b 1 1 1"] = torch.where(
            valid_any_b111, masked_min_b111, torch.full_like(masked_min_b111, MIN_PROMPT_DEPTH_M)
        )
        max_depth_b111: Float32[Tensor, "b 1 1 1"] = torch.where(
            valid_any_b111, masked_max_b111, torch.full_like(masked_max_b111, MAX_PROMPT_DEPTH_M)
        )
        span_b111: Float32[Tensor, "b 1 1 1"] = (max_depth_b111 - min_depth_b111).clamp_min(MIN_PROMPT_SPAN_M)
        normalized_prompt_b1hw: Float32[Tensor, "b 1 192 256"] = torch.where(
            valid_b1hw,
            (prompt_depth - min_depth_b111) / span_b111,
            torch.zeros_like(prompt_depth),
        )

        normalized_image_b3hw: Float[Tensor, "b 3 h w"] = (image_b3hw - self.backbone.mean) / self.backbone.std
        encoder_output: EncoderOutput = self.backbone.encoder(normalized_image_b3hw)
        stem_half_bchw: Float[Tensor, "b c_half h_half w_half"] = encoder_output[0]
        encoder_features: FeaturePyramid = encoder_output[1]
        c1_bchw: Float[Tensor, "b c1 h_quarter w_quarter"] = encoder_features[0]
        c2_bchw: Float[Tensor, "b c2 h_eighth w_eighth"] = encoder_features[1]
        c3_bchw: Float[Tensor, "b c3 h_sixteenth w_sixteenth"] = encoder_features[2]
        c4_bchw: Float[Tensor, "b c4 h_thirtysecond w_thirtysecond"] = encoder_features[3]

        decoder = self.backbone.decoder
        f4_bchw: Float[Tensor, "b 288 h_thirtysecond w_thirtysecond"] = decoder.proj4(c4_bchw)
        prompt4_b1hw: Float[Tensor, "b 1 h_thirtysecond w_thirtysecond"] = F.interpolate(
            normalized_prompt_b1hw, size=f4_bchw.shape[-2:], mode="bilinear", align_corners=False
        ).to(dtype=f4_bchw.dtype)
        f4_bchw = f4_bchw + self.prompt_branches[0](prompt4_b1hw)

        f3_bchw: Float[Tensor, "b 192 h_sixteenth w_sixteenth"] = decoder.fuse3(c3_bchw, f4_bchw)
        prompt3_b1hw: Float[Tensor, "b 1 h_sixteenth w_sixteenth"] = F.interpolate(
            normalized_prompt_b1hw, size=f3_bchw.shape[-2:], mode="bilinear", align_corners=False
        ).to(dtype=f3_bchw.dtype)
        f3_bchw = f3_bchw + self.prompt_branches[1](prompt3_b1hw)

        f2_bchw: Float[Tensor, "b 144 h_eighth w_eighth"] = decoder.fuse2(c2_bchw, f3_bchw)
        prompt2_b1hw: Float[Tensor, "b 1 h_eighth w_eighth"] = F.interpolate(
            normalized_prompt_b1hw, size=f2_bchw.shape[-2:], mode="bilinear", align_corners=False
        ).to(dtype=f2_bchw.dtype)
        f2_bchw = f2_bchw + self.prompt_branches[2](prompt2_b1hw)

        f1_bchw: Float[Tensor, "b 96 h_quarter w_quarter"] = decoder.fuse1(c1_bchw, f2_bchw)
        prompt1_b1hw: Float[Tensor, "b 1 h_quarter w_quarter"] = F.interpolate(
            normalized_prompt_b1hw, size=f1_bchw.shape[-2:], mode="bilinear", align_corners=False
        ).to(dtype=f1_bchw.dtype)
        f1_bchw = f1_bchw + self.prompt_branches[3](prompt1_b1hw)

        f_half_bchw: Float[Tensor, "b 32 h_half w_half"] = decoder.fuse_half(stem_half_bchw, f1_bchw)
        depth_logits_half_b1hw: Float[Tensor, "b 1 h_half w_half"] = decoder.head_half(f_half_bchw)
        depth_logits_b1hw: Float[Tensor, "b 1 h w"] = decoder.convex_up._forward_unfold(f_half_bchw, depth_logits_half_b1hw)
        normalized_depth_b1hw: Float[Tensor, "b 1 h w"] = torch.sigmoid(depth_logits_b1hw)
        metric_depth_b1hw: Float[Tensor, "b 1 h w"] = normalized_depth_b1hw * span_b111 + min_depth_b111
        return metric_depth_b1hw, min_depth_b111, max_depth_b111

    def fuse_for_inference(self) -> Self:
        """Fuse the released ZipDepth backbone in place and enter eval mode."""
        self.backbone.fuse_for_inference()
        self.eval()
        return self


def load_zipdepth_prompt(checkpoint: Path) -> ZipDepthPrompt:
    """Load a complete released ZipDepth-base checkpoint into the wrapper.

    Args:
        checkpoint: Bare ZipDepth state dict or trainer checkpoint containing
            ``model_state_dict``. DDP and ``torch.compile`` prefixes are
            accepted.

    Returns:
        A trainable prompted model. Every released tensor is loaded strictly;
        prompt-branch tensors retain their zero-output initialization.
    """
    checkpoint_data: dict[str, Any] = torch.load(checkpoint, map_location="cpu", weights_only=True)
    raw_state_dict: StateDict = checkpoint_data.get("model_state_dict", checkpoint_data)
    state_dict: StateDict = strip_state_dict_prefixes(raw_state_dict)
    if any(key.startswith("backbone.") for key in state_dict):
        prompted_model: ZipDepthPrompt = ZipDepthPrompt()
        prompted_model.load_state_dict(state_dict, strict=True)
        return prompted_model
    backbone: ZipDepth = create_model(variant="base")
    backbone.load_state_dict(state_dict, strict=True)
    return ZipDepthPrompt(backbone)


@dataclass(frozen=True, slots=True)
class ZipDepthPromptConfig:
    """ZipDepth-PromptDA model and runtime configuration."""

    checkpoint: Path = Path()
    """Prompted model checkpoint; an empty path is only a CLI placeholder."""
    backend: BackendConfig = field(default_factory=TorchBackendConfig)
    """Backend running the network: torch, ONNX Runtime CUDA, or TensorRT."""
    image_height: int = 768
    """Static network image height; must be divisible by 32."""
    image_width: int = 1024
    """Static network image width; must be divisible by 32."""

    def setup(self) -> "ZipDepthPromptPredictor":
        """Build the selected runtime and return a ready predictor."""
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"ZipDepth-PromptDA checkpoint is not a file: {self.checkpoint}")
        image_hw: tuple[int, int] = (self.image_height, self.image_width)
        image_spec = TensorSpec(IMAGE_INPUT_NAME, (3, *image_hw), torch.float32)
        prompt_spec = TensorSpec(PROMPT_INPUT_NAME, (1, PROMPT_HEIGHT, PROMPT_WIDTH), torch.float32)
        depth_spec = TensorSpec(DEPTH_OUTPUT_NAME, (1, *image_hw), torch.float32)
        if isinstance(self.backend, TorchBackendConfig):
            model: nn.Module = load_zipdepth_prompt(self.checkpoint).cuda().fuse_for_inference()
            runtime: TensorRuntime = TorchRuntime(
                model,
                input_specs=(image_spec, prompt_spec),
                output_specs=(depth_spec,),
                max_batch_size=self.backend.max_batch_size,
                autocast_dtype=self.backend.autocast_dtype,
            )
        else:
            from monopriors.models.depth_completion.zipdepth_prompt_export import export_zipdepth_prompt_onnx

            export_batch_size: int
            if isinstance(self.backend, TensorRtBackendConfig):
                export_batch_size = self.backend.opt_batch_size
            else:
                onnx_backend: OnnxBackendConfig = self.backend
                export_batch_size = onnx_backend.max_batch_size
            onnx_path: Path = export_zipdepth_prompt_onnx(
                checkpoint=self.checkpoint,
                image_hw=image_hw,
                batch_size=export_batch_size,
            )
            runtime = create_runtime_from_onnx(onnx_path, self.backend)
        return ZipDepthPromptPredictor(runtime=runtime)


class ZipDepthPromptPredictor(BaseCompletionPredictor[TensorRuntime]):
    """ZipDepth-PromptDA through the shared batched GPU tensor contract."""


__all__ = (
    "MAX_PROMPT_DEPTH_M",
    "MIN_PROMPT_DEPTH_M",
    "PROMPT_HEIGHT",
    "PROMPT_WIDTH",
    "ZipDepthPrompt",
    "ZipDepthPromptConfig",
    "ZipDepthPromptPredictor",
    "load_zipdepth_prompt",
)
