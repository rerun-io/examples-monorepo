"""Prompt-conditioned metric ZipDepth model.

The wrapper keeps the vendored ZipDepth-base tensors unchanged, injects a
normalized metric prompt into all four decoder stages, and returns depth in the
same units as the prompt (metres for the public API).

The head can only reach depths inside the range it is rescaled into, which by
default is the prompt's own ``[min, max]``. ``ZipDepthPrompt(range_margin=...)``
widens that range by a fraction of the prompt span on each side, which ultrawide
captures need: their prompt covers the image centre, so the periphery is often
nearer or farther than anything the prompt saw.
"""

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Self

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Float32, UInt8
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
    MAX_METRIC_DEPTH_M,
    MIN_METRIC_DEPTH_M,
    PROMPT_DEPTH_HW,
    PROMPT_INPUT_NAME,
    BaseCompletionPredictor,
)
from monopriors.models.zipdepth_checkpoint import (
    load_zipdepth_checkpoint,
    range_margin_from_checkpoint,
    state_dict_from_checkpoint,
)
from monopriors.third_party.zipdepth.architecture import EncoderOutput, FeaturePyramid, ZipDepth, create_model
from monopriors.third_party.zipdepth.model_utils import StateDict

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


def _inject_prompt(
    feature_bchw: Float[Tensor, "b c h w"],
    normalized_prompt_bchw: Float32[Tensor, "b 1 192 256"],
    branch: nn.Module,
) -> Float[Tensor, "b c h w"]:
    """Resize and inject one normalized prompt into a decoder feature."""
    feature_hw: tuple[int, int] = (int(feature_bchw.shape[-2]), int(feature_bchw.shape[-1]))
    prompt_bchw: Float32[Tensor, "b 1 prompt_h prompt_w"] = normalized_prompt_bchw
    if tuple(int(dim) for dim in prompt_bchw.shape[-2:]) != feature_hw:
        prompt_bchw = F.interpolate(prompt_bchw, size=feature_hw, mode="bilinear", align_corners=False)
    return feature_bchw + branch(prompt_bchw.to(dtype=feature_bchw.dtype))


class ZipDepthPrompt(nn.Module):
    """ZipDepth-base with PromptDA-style multi-scale metric conditioning.

    The head is a sigmoid rescaled into a per-image depth range derived from the
    prompt. ``range_margin`` widens that output range symmetrically, so predicted
    depth is no longer trapped inside the prompt's own ``[min, max]``. The prompt
    *normalization* fed to the four decoder injection branches deliberately keeps
    the original ``[min, max]``: the prompt itself does not change, only the range
    the head may reach, and holding the conditioning fixed keeps a margin change
    from shifting the features a trained model already relies on.
    """

    def __init__(self, backbone: ZipDepth | None = None, *, range_margin: float = 0.0) -> None:
        """Build the prompted wrapper.

        Args:
            backbone: ZipDepth-base network; ``None`` creates a fresh one.
            range_margin: Fraction of the prompt span by which the head's output
                range is widened past the prompt's own minimum and maximum. Zero
                reproduces the prompt-bounded head exactly, emitting none of the
                widening operations.
        """
        super().__init__()
        self.backbone: ZipDepth = create_model(variant="base") if backbone is None else backbone
        if self.backbone.variant != "base":
            raise ValueError(f"ZipDepthPrompt requires the base variant, got {self.backbone.variant!r}")
        if not self.backbone.decoder.convex_up.use_unfold:
            raise ValueError("ZipDepthPrompt currently requires ZipDepth's unfold-based convex head")
        if not isfinite(range_margin) or range_margin < 0.0:
            raise ValueError(f"range_margin must be finite and non-negative, got {range_margin}")
        self.range_margin: float = range_margin
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
        metric_depth_bchw: Float[Tensor, "b 1 h w"]
        metric_depth_bchw, _, _ = self.forward_with_range(image, prompt_depth)
        return metric_depth_bchw

    def forward_with_range(
        self,
        image: UInt8[Tensor, "b 3 h w"] | Float[Tensor, "b 3 h w"],
        prompt_depth: Float32[Tensor, "b 1 192 256"],
    ) -> tuple[
        Float[Tensor, "b 1 h w"],
        Float32[Tensor, "b 1 1 1"],
        Float32[Tensor, "b 1 1 1"],
    ]:
        """Predict metric depth and expose the head's output range for TRT export.

        The two scalar outputs force TensorRT to materialize the input-derived
        reductions. Normal callers use :meth:`forward` and receive depth only.

        Returns:
            Metric depth, and the minimum and maximum the head can reach. Those
            two already include ``range_margin``, so ``sigmoid(logits) * (max -
            min) + min`` reproduces depth from captured logits.
        """
        if image.dtype == torch.uint8:
            image_bchw: Float[Tensor, "b 3 h w"] = image.to(dtype=self.backbone.mean.dtype) / 255.0
        else:
            image_bchw = image.to(dtype=self.backbone.mean.dtype)

        valid_bchw: Bool[Tensor, "b 1 192 256"] = (
            torch.isfinite(prompt_depth)
            & (prompt_depth >= MIN_METRIC_DEPTH_M)
            & (prompt_depth <= MAX_METRIC_DEPTH_M)
        )
        valid_any_b: Bool[Tensor, "b 1 1 1"] = valid_bchw.any(dim=(1, 2, 3), keepdim=True)
        masked_min_b: Float32[Tensor, "b 1 1 1"] = torch.where(
            valid_bchw, prompt_depth, torch.inf
        ).amin(dim=(1, 2, 3), keepdim=True)
        masked_max_b: Float32[Tensor, "b 1 1 1"] = torch.where(
            valid_bchw, prompt_depth, -torch.inf
        ).amax(dim=(1, 2, 3), keepdim=True)
        min_depth_b: Float32[Tensor, "b 1 1 1"] = torch.where(
            valid_any_b, masked_min_b, MIN_METRIC_DEPTH_M
        )
        max_depth_b: Float32[Tensor, "b 1 1 1"] = torch.where(
            valid_any_b, masked_max_b, MAX_METRIC_DEPTH_M
        )
        span_b: Float32[Tensor, "b 1 1 1"] = (max_depth_b - min_depth_b).clamp_min(MIN_PROMPT_SPAN_M)
        normalized_prompt_bchw: Float32[Tensor, "b 1 192 256"] = torch.where(
            valid_bchw,
            (prompt_depth - min_depth_b) / span_b,
            0.0,
        )

        # The head may leave the prompt's own range by ``range_margin`` spans on each
        # side, still clipped to the shared validity window. At zero margin these
        # operations are skipped entirely, so torch and the exported fp16 graph are
        # unchanged. Prompt normalization above keeps the un-widened range.
        min_output_b: Float32[Tensor, "b 1 1 1"] = min_depth_b
        max_output_b: Float32[Tensor, "b 1 1 1"] = max_depth_b
        span_output_b: Float32[Tensor, "b 1 1 1"] = span_b
        if self.range_margin > 0.0:
            min_output_b = (min_depth_b - self.range_margin * span_b).clamp_min(MIN_METRIC_DEPTH_M)
            max_output_b = (max_depth_b + self.range_margin * span_b).clamp_max(MAX_METRIC_DEPTH_M)
            span_output_b = (max_output_b - min_output_b).clamp_min(MIN_PROMPT_SPAN_M)

        normalized_image_bchw: Float[Tensor, "b 3 h w"] = (image_bchw - self.backbone.mean).div_(self.backbone.std)
        encoder_output: EncoderOutput = self.backbone.encoder(normalized_image_bchw)
        stem_half_bchw: Float[Tensor, "b c_half h_half w_half"] = encoder_output[0]
        encoder_features: FeaturePyramid = encoder_output[1]
        c1_bchw: Float[Tensor, "b c1 h_quarter w_quarter"] = encoder_features[0]
        c2_bchw: Float[Tensor, "b c2 h_eighth w_eighth"] = encoder_features[1]
        c3_bchw: Float[Tensor, "b c3 h_sixteenth w_sixteenth"] = encoder_features[2]
        c4_bchw: Float[Tensor, "b c4 h_thirtysecond w_thirtysecond"] = encoder_features[3]

        decoder = self.backbone.decoder
        f4_bchw: Float[Tensor, "b 288 h_thirtysecond w_thirtysecond"] = decoder.proj4(c4_bchw)
        f4_bchw = _inject_prompt(f4_bchw, normalized_prompt_bchw, self.prompt_branches[0])

        f3_bchw: Float[Tensor, "b 192 h_sixteenth w_sixteenth"] = decoder.fuse3(c3_bchw, f4_bchw)
        f3_bchw = _inject_prompt(f3_bchw, normalized_prompt_bchw, self.prompt_branches[1])

        f2_bchw: Float[Tensor, "b 144 h_eighth w_eighth"] = decoder.fuse2(c2_bchw, f3_bchw)
        f2_bchw = _inject_prompt(f2_bchw, normalized_prompt_bchw, self.prompt_branches[2])

        f1_bchw: Float[Tensor, "b 96 h_quarter w_quarter"] = decoder.fuse1(c1_bchw, f2_bchw)
        f1_bchw = _inject_prompt(f1_bchw, normalized_prompt_bchw, self.prompt_branches[3])

        f_half_bchw: Float[Tensor, "b 32 h_half w_half"] = decoder.fuse_half(stem_half_bchw, f1_bchw)
        depth_logits_half_bchw: Float[Tensor, "b 1 h_half w_half"] = decoder.head_half(f_half_bchw)
        depth_logits_bchw: Float[Tensor, "b 1 h w"] = decoder.convex_up._forward_unfold(f_half_bchw, depth_logits_half_bchw)
        normalized_depth_bchw: Float[Tensor, "b 1 h w"] = torch.sigmoid(depth_logits_bchw)
        metric_depth_bchw: Float[Tensor, "b 1 h w"] = normalized_depth_bchw * span_output_b + min_output_b
        return metric_depth_bchw, min_output_b, max_output_b

    def fuse_for_inference(self) -> Self:
        """Fuse the released ZipDepth backbone in place and enter eval mode."""
        self.backbone.fuse_for_inference()
        self.eval()
        return self


def load_zipdepth_prompt(checkpoint: Path, range_margin: float | None = None) -> ZipDepthPrompt:
    """Load a complete released ZipDepth-base checkpoint into the wrapper.

    Args:
        checkpoint: Bare ZipDepth state dict or trainer checkpoint containing
            ``model_state_dict``. DDP and ``torch.compile`` prefixes are
            accepted.
        range_margin: Output-range margin override. ``None`` takes the margin
            the training run recorded in the checkpoint, which is ``0.0`` for
            bare state dicts and for checkpoints written before the margin
            existed.

    Returns:
        A trainable prompted model. Every released tensor is loaded strictly;
        prompt-branch tensors retain their zero-output initialization.
    """
    loaded: dict[str, object] = load_zipdepth_checkpoint(checkpoint)
    state_dict: StateDict = state_dict_from_checkpoint(loaded, checkpoint)
    margin: float = range_margin if range_margin is not None else range_margin_from_checkpoint(loaded, checkpoint)
    if any(key.startswith("backbone.") for key in state_dict):
        prompted_model: ZipDepthPrompt = ZipDepthPrompt(range_margin=margin)
        prompted_model.load_state_dict(state_dict, strict=True)
        return prompted_model
    backbone: ZipDepth = create_model(variant="base")
    backbone.load_state_dict(state_dict, strict=True)
    return ZipDepthPrompt(backbone, range_margin=margin)


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
        prompt_spec = TensorSpec(PROMPT_INPUT_NAME, (1, *PROMPT_DEPTH_HW), torch.float32)
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


class ZipDepthPromptPredictor(BaseCompletionPredictor):
    """ZipDepth-PromptDA through the shared batched GPU tensor contract."""


__all__ = (
    "ZipDepthPrompt",
    "ZipDepthPromptConfig",
    "ZipDepthPromptPredictor",
    "load_zipdepth_prompt",
)
