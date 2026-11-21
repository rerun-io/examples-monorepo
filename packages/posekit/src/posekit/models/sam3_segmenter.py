"""SAM3 open-vocabulary promptable instance segmentation (text prompts).

Loads ``facebook/sam3`` through transformers as a weights + ``nn.Module``
source (docs/design.md §4) and replaces the CPU image processor with GPU
preprocessing: SAM3's pipeline is a plain non-aspect-preserving bilinear
resize to 1008x1008 plus 0.5/0.5 normalization, so CUDA frames go straight
into the model without host copies. The text features of a prompt are computed
once and reused.

``backend="tensorrt"`` runs the model as two strongly typed fp16 TensorRT
engines with fp32 I/O: the vision trunk (ViT + FPN neck, about three quarters
of a call) and the heads (DETR encoder/decoder, scoring, mask decoder; one
engine per prompt token count). Post-processing stays in torch, in FP32.
TensorRT picks fp16-accumulating kernels for fp16 convolutions and GEMMs; the
ViT's patch embedding is exported as an fp32 conv on fp16-rounded operands,
which TensorRT accumulates in fp32 like cuDNN does.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int64, UInt8
from torch import Tensor, nn

from posekit.artifacts import DEFAULT_ONNX_CACHE_DIR
from posekit.models.base import PromptableSegmenter, SegmentationPrompts
from posekit.predictions import BoxDetections, validate_frames_rgb

ENGINE_BATCH: int = 2
"""Images per engine call: the two lenses of a 360 camera. Smaller calls are zero-padded, larger ones chunked."""
EXPORT_VERSION: int = 2
"""Bump when an exported graph changes; it names the cached ONNX files."""
HEAD_OUTPUTS: tuple[str, ...] = ("pred_masks", "pred_boxes", "pred_logits", "presence_logits")
"""The model outputs post-processing reads."""


@dataclass(frozen=True, slots=True)
class Sam3SegmenterConfig:
    """SAM3 text-prompted instance segmentation configuration."""

    checkpoint: str = "facebook/sam3"
    """HF checkpoint accepted by ``Sam3Model.from_pretrained``."""
    device: str = "cuda"
    """Inference device."""
    score_threshold: float = 0.5
    """Minimum instance confidence kept in the output."""
    mask_threshold: float = 0.5
    """Probability threshold binarizing the output masks."""
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"
    """Autocast dtype of the torch backend (tensor cores); post-processing stays in FP32. On an RTX 5090 with clip B lens
    frames a two-image call takes 300 ms in float32, 120 ms in bfloat16 and 118 ms in float16, whose 10-bit mantissa
    keeps the masks closer to FP32 (union IoU 0.9993 against 0.9966)."""
    backend: Literal["torch", "tensorrt"] = "torch"
    """``torch``: the eager model under ``dtype`` autocast. ``tensorrt``: fp16 TensorRT engines for the trunk and the
    heads (static batch :data:`ENGINE_BATCH`, exported and built once per machine; ``dtype`` is not used)."""

    def setup(self) -> "Sam3Segmenter":
        """Download weights and return a ready segmenter."""
        return Sam3Segmenter(self)


class _Fp32PatchConv(nn.Module):
    """The ViT patch embedding as torch autocast runs it (fp16 operands, fp32 accumulation, fp16 output), spelled in
    fp32 so the exported graph keeps the accumulation in fp32.

    TensorRT accumulates fp16 convs and GEMMs in fp16. In the SAM 3 trunk that moved the FPN levels 4x further from the
    torch fp16 trunk than torch fp16 is from FP32 (relative L2 0.050 against 0.011 on the coarsest level) and flipped
    person scores near the 0.5 threshold. The patch embedding caused most of it; in fp32 it costs nothing (trunk 54 ms)
    and brings the drift to 0.015. The MLPs cause most of the rest: as fp32-accumulating GEMMs they reach 0.009 but
    cost +12 ms, and this torch spelling of them builds as fp32 GEMMs (140 ms), so they stay fp16.
    """

    def __init__(self, conv: nn.Conv2d) -> None:
        super().__init__()
        self.conv: nn.Conv2d = conv

    @property
    def weight(self) -> Tensor:
        """The conv's weight (``Sam3ViTPatchEmbeddings`` casts its input to the weight's dtype)."""
        return self.conv.weight

    def forward(self, pixel_values: Tensor) -> Tensor:
        conv: nn.Conv2d = self.conv
        with torch.autocast(device_type=pixel_values.device.type, enabled=False):
            bias: Tensor | None = None if conv.bias is None else conv.bias.half().float()
            return F.conv2d(pixel_values.half().float(), conv.weight.half().float(), bias, conv.stride, conv.padding).half()


class _VisionTrunk(nn.Module):
    """The ViT and its FPN neck: the three levels SAM 3's heads read (``Sam3Model.forward`` drops the 0.5x level).

    For the export the patch embedding runs as :class:`_Fp32PatchConv`.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model: Any = model

    def forward(self, pixel_values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        patch: Any = self.model.vision_encoder.backbone.embeddings.patch_embeddings
        conv: nn.Conv2d = patch.projection
        patch.projection = _Fp32PatchConv(conv)
        try:
            levels: tuple[Tensor, ...] = self.model.vision_encoder(pixel_values).fpn_hidden_states
        finally:
            patch.projection = conv
        return levels[0], levels[1], levels[2]


class _Heads(nn.Module):
    """DETR encoder and decoder, scoring and mask decoder on the trunk's FPN levels and the prompt's text features."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model: Any = model

    def forward(self, fpn0: Tensor, fpn1: Tensor, fpn2: Tensor, text: Tensor, text_mask: Tensor) -> tuple[Tensor, ...]:
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput

        neck: Any = self.model.vision_encoder.neck
        # Sam3Model.forward never reads the 0.5x level; None stands in for it
        hidden: Any = (fpn0, fpn1, fpn2, None)
        position: Any = (*(neck.position_encoding(h.shape, h.device, h.dtype) for h in hidden[:3]), None)
        pooled: Any = text
        outputs: Any = self.model(
            vision_embeds=Sam3VisionEncoderOutput(fpn_hidden_states=hidden, fpn_position_encoding=position),
            text_embeds=BaseModelOutputWithPooling(pooler_output=pooled),
            attention_mask=text_mask,
        )
        return tuple(outputs[key] for key in HEAD_OUTPUTS)


class Sam3Segmenter(PromptableSegmenter):
    """Batched GPU SAM3 segmenter returning detections with instance masks.

    SAM3 is concept-prompted: it segments *every* instance matching the text
    prompt, so box/point prompts are not supported by this implementation
    (SAM2 image mode is the planned box/point ``PromptableSegmenter``).
    """

    def __init__(self, config: Sam3SegmenterConfig) -> None:
        """Load the SAM3 model and tokenizer (and, for ``backend="tensorrt"``, the trunk engine).

        Args:
            config: Checkpoint, device, thresholds and backend.
        """
        from transformers.models.sam3 import Sam3Model, Sam3Processor

        self.config: Sam3SegmenterConfig = config
        self.model = Sam3Model.from_pretrained(config.checkpoint).to(config.device).eval()
        self.processor = Sam3Processor.from_pretrained(config.checkpoint)
        size = self.processor.image_processor.size
        self.input_hw: tuple[int, int] = (int(size["height"]), int(size["width"]))
        self.text_features: dict[str, tuple[Float[Tensor, "1 t c"], Int64[Tensor, "1 t"]]] = {}
        """Per prompt: the projected text features (computed once, in FP32) and their attention mask."""
        self.engine_name: str = f"{config.checkpoint.replace('/', '--')}_{self.input_hw[0]}x{self.input_hw[1]}"
        self.trunk: Any = None
        """The vision trunk engine (``backend="tensorrt"``)."""
        self.heads: dict[int, Any] = {}
        """Heads engines by prompt token count (``backend="tensorrt"``)."""
        if config.backend == "tensorrt":
            example: dict[str, Tensor] = {"pixel_values": torch.zeros((ENGINE_BATCH, 3, *self.input_hw), device=config.device)}
            self.trunk = self.engine(_VisionTrunk(self.model), f"{self.engine_name}_trunk", example, ["fpn0", "fpn1", "fpn2"])
            self.model.vision_encoder.backbone = None  # the engine replaces it; the heads keep the neck's position encoding

    def engine(self, module: nn.Module, name: str, example: dict[str, Tensor], output_names: list[str]) -> Any:
        """A TensorRT runtime for one SAM 3 part: exported once to an fp16-autocast ONNX graph with fp32 I/O at static batch
        :data:`ENGINE_BATCH`, built once per machine (trtkit's engine cache)."""
        from trtkit import TensorRtRuntime, TrtBuildConfig, ensure_engine, export_onnx

        onnx_path: Path = DEFAULT_ONNX_CACHE_DIR / f"{name}_b{ENGINE_BATCH}_fp16_v{EXPORT_VERSION}.onnx"
        if not onnx_path.exists():
            print(f"[posekit] exporting SAM 3 to ONNX (one-time): {onnx_path.name}")
            export_onnx(
                module.eval(), tuple(example.values()), onnx_path, input_names=list(example), output_names=output_names, compute_dtype=torch.float16
            )
        return TensorRtRuntime(ensure_engine(onnx_path, TrtBuildConfig(max_batch_size=ENGINE_BATCH, opt_batch_size=ENGINE_BATCH)))

    def text(self, prompt: str) -> tuple[Float[Tensor, "1 t c"], Int64[Tensor, "1 t"]]:
        """The prompt's projected text features and attention mask; the text encoder runs once per prompt."""
        if prompt not in self.text_features:
            tokens = self.processor.tokenizer([prompt], return_tensors="pt").to(self.config.device)
            features: Tensor = self.model.get_text_features(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]).pooler_output
            self.text_features[prompt] = (features.float(), tokens["attention_mask"])
        return self.text_features[prompt]

    def prepare(self, prompt: str) -> None:
        """Encode a prompt and, for ``backend="tensorrt"``, load (export and build on first use) its heads engine, so the
        first call does not pay for it. Calls with an unprepared prompt prepare it themselves."""
        features, text_mask = self.text(prompt)
        tokens: int = int(features.shape[1])
        if self.trunk is None or tokens in self.heads:
            return
        example: dict[str, Tensor] = {
            spec.name: torch.zeros((ENGINE_BATCH, *spec.shape), device=self.config.device) for spec in self.trunk.spec.outputs
        }
        example["text"] = features.expand(ENGINE_BATCH, -1, -1).contiguous()
        example["text_mask"] = text_mask.expand(ENGINE_BATCH, -1).contiguous()
        self.heads[tokens] = self.engine(_Heads(self.model), f"{self.engine_name}_heads_t{tokens}", example, list(HEAD_OUTPUTS))

    def run_engines(self, pixel_values: Float[Tensor, "b 3 in_h in_w"], text: Float[Tensor, "b t c"], text_mask: Int64[Tensor, "b t"]) -> Any:
        """The trunk and the prompt's heads engine (:meth:`prepare` loaded it)."""
        from transformers.models.sam3.modeling_sam3 import Sam3ImageSegmentationOutput
        from trtkit import run_chunked

        inputs: dict[str, Tensor] = {
            **run_chunked(self.trunk, {"pixel_values": pixel_values}),
            "text": text.contiguous(),
            "text_mask": text_mask.contiguous(),
        }
        heads: dict[str, Any] = run_chunked(self.heads[int(text.shape[1])], inputs)
        return Sam3ImageSegmentationOutput(**heads)

    @torch.inference_mode()
    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"], prompts: SegmentationPrompts) -> BoxDetections:
        """Segment every instance matching the text prompt in each frame.

        Args:
            frames_rgb: uint8 RGB NHWC frame batch on the inference device.
            prompts: Must carry ``text``; box/point prompts are rejected.

        Returns:
            Detections with binary ``masks`` at frame resolution; ``track_ids``
            stays ``None`` (text prompts carry no identities).

        Raises:
            ValueError: If ``prompts.text`` is missing or box/point prompts are given.
        """
        if prompts.text is None:
            raise ValueError("Sam3Segmenter requires a text prompt (prompts.text).")
        if prompts.boxes_xyxy is not None or prompts.points_xy is not None:
            raise ValueError("Sam3Segmenter is concept-prompted; box/point prompts are not supported.")
        validate_frames_rgb(frames_rgb)
        batch_size: int = int(frames_rgb.shape[0])
        frame_h: int = int(frames_rgb.shape[1])
        frame_w: int = int(frames_rgb.shape[2])

        # GPU replication of Sam3ImageProcessor: bilinear resize to 1008x1008 on
        # [0, 1] floats (antialias matches PIL), then (x - 0.5) / 0.5.
        chw: Float[Tensor, "b 3 h w"] = frames_rgb.permute(0, 3, 1, 2).float() / 255.0
        resized: Float[Tensor, "b 3 in_h in_w"] = F.interpolate(chw, size=self.input_hw, mode="bilinear", align_corners=False, antialias=True)
        pixel_values: Float[Tensor, "b 3 in_h in_w"] = (resized - 0.5) / 0.5

        self.prepare(prompts.text)
        features, text_mask = self.text(prompts.text)
        text: Float[Tensor, "b t c"] = features.expand(batch_size, -1, -1)
        attention_mask: Int64[Tensor, "b t"] = text_mask.expand(batch_size, -1)
        if self.trunk is not None:
            outputs = self.run_engines(pixel_values, text, attention_mask)
        else:
            from transformers.modeling_outputs import BaseModelOutputWithPooling

            dtype: torch.dtype = getattr(torch, self.config.dtype)
            with torch.autocast(device_type=pixel_values.device.type, dtype=dtype, enabled=dtype != torch.float32):
                outputs = self.model(
                    pixel_values=pixel_values, text_embeds=BaseModelOutputWithPooling(pooler_output=text), attention_mask=attention_mask
                )
        for key, value in outputs.items():
            if isinstance(value, Tensor) and value.is_floating_point():
                outputs[key] = value.float()
        per_frame: list[dict[str, Tensor]] = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.config.score_threshold,
            mask_threshold=self.config.mask_threshold,
            target_sizes=[(frame_h, frame_w)] * batch_size,
        )
        device: torch.device = frames_rgb.device
        xyxy_rows: list[Tensor] = []
        score_rows: list[Tensor] = []
        index_rows: list[Tensor] = []
        mask_rows: list[Tensor] = []
        for frame_idx, results in enumerate(per_frame):
            num: int = int(results["scores"].shape[0])
            if num == 0:
                continue
            xyxy_rows.append(results["boxes"].to(device=device, dtype=torch.float32))
            score_rows.append(results["scores"].to(device=device, dtype=torch.float32))
            index_rows.append(torch.full((num,), frame_idx, dtype=torch.long, device=device))
            mask_rows.append(results["masks"].to(device=device) > self.config.mask_threshold)
        if not xyxy_rows:
            return BoxDetections.empty(device, mask_hw=(frame_h, frame_w))
        masks: Bool[Tensor, "n h w"] = torch.cat(mask_rows, dim=0)
        return BoxDetections(
            xyxy=torch.cat(xyxy_rows, dim=0),
            scores=torch.cat(score_rows, dim=0),
            frame_indices=torch.cat(index_rows, dim=0),
            masks=masks,
        )


__all__ = ("Sam3Segmenter", "Sam3SegmenterConfig")
