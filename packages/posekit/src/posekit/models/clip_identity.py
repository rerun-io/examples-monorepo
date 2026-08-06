"""CLIP appearance-embedding identity encoder (re-ID role).

Loads the laion2b ViT-B/32 weights (the checkpoint mamma's tracker uses via
open_clip) from HuggingFace transformers as a plain ``nn.Module`` — the
transformers-as-weights-source pattern from docs/design.md §4 — and replaces
the PIL/CPU preprocessing with the GPU crop path: a center-square crop per box
(matching resize-shortest-side + center-crop semantics) resampled straight to
224x224 with ``grid_sample``.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Float, Float32, UInt8
from torch import Tensor

from posekit.models.base import IdentityEncoder
from posekit.ops.crops import CropBatch, CropSpec, crop_frames
from posekit.predictions import BoxDetections
from posekit.runtimes import TensorRuntime, TensorSpec, TorchRuntime, run_chunked

# OpenAI-CLIP normalization constants scaled to the 0-255 crops posekit produces.
CLIP_MEAN: tuple[float, float, float] = (0.48145466 * 255.0, 0.4578275 * 255.0, 0.40821073 * 255.0)
CLIP_STD: tuple[float, float, float] = (0.26862954 * 255.0, 0.26130258 * 255.0, 0.27577711 * 255.0)


class _ClipVisionEmbed(torch.nn.Module):
    """Adapter returning the projected image embedding as a plain tensor."""

    def __init__(self, vision_model: torch.nn.Module) -> None:
        super().__init__()
        self.vision_model = vision_model

    def forward(self, pixel_values: Float[Tensor, "n 3 224 224"]) -> Float[Tensor, "n embed_dim"]:
        return self.vision_model(pixel_values=pixel_values).image_embeds


@dataclass(frozen=True, slots=True)
class ClipIdentityConfig:
    """CLIP ViT-B/32 identity-encoder configuration (torch backend)."""

    model_id: str = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    """HF checkpoint providing the vision tower + projection weights."""
    device: str = "cuda"
    """Inference device."""
    autocast: bool = True
    """Run under bfloat16 autocast (matches mamma's tracker-level autocast)."""
    max_batch_size: int = 64
    """Largest crop batch a single runtime call may submit."""

    def setup(self) -> "ClipIdentity":
        """Download weights and return a ready encoder."""
        return ClipIdentity(self)


class ClipIdentity(IdentityEncoder):
    """Batched GPU CLIP crop encoder for instance re-identification."""

    def __init__(self, config: ClipIdentityConfig) -> None:
        """Load the vision tower and build the torch runtime.

        Args:
            config: Checkpoint, device, and batching options.
        """
        from transformers import CLIPVisionModelWithProjection

        self.config: ClipIdentityConfig = config
        vision = CLIPVisionModelWithProjection.from_pretrained(config.model_id).to(config.device)
        self.embed_dim = int(vision.config.projection_dim)
        image_size_raw = vision.config.image_size
        crop_side: int = int(image_size_raw[0]) if isinstance(image_size_raw, list | tuple) else int(image_size_raw or 224)
        self.runtime: TensorRuntime = TorchRuntime(
            _ClipVisionEmbed(vision),
            input_specs=(TensorSpec("pixel_values", (3, crop_side, crop_side), torch.float32),),
            output_specs=(TensorSpec("image_embeds", (self.embed_dim,), torch.float32),),
            max_batch_size=config.max_batch_size,
            autocast_dtype=torch.bfloat16 if config.autocast else None,
        )
        self.crop_spec: CropSpec = CropSpec(
            input_size=(crop_side, crop_side),
            padding=1.0,
            align="cv2",
            bgr=False,
            mean_rgb=CLIP_MEAN,
            std_rgb=CLIP_STD,
        )

    @torch.inference_mode()
    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"], detections: BoxDetections) -> Float32[Tensor, "n embed_dim"]:
        """Embed every detection crop.

        Args:
            frames_rgb: uint8 RGB NHWC frame batch on the inference device.
            detections: Instance boxes referencing ``frames_rgb`` by index.

        Returns:
            One unnormalized embedding row per detection.
        """
        if detections.num_detections == 0:
            return torch.empty((0, self.embed_dim), dtype=torch.float32, device=frames_rgb.device)
        # Center square with side = min(w, h): the GPU equivalent of PIL
        # resize-shortest-side-to-224 followed by a 224 center crop.
        centers: Float[Tensor, "n 2"] = (detections.xyxy[:, 0:2] + detections.xyxy[:, 2:4]) * 0.5
        sides: Float[Tensor, "n 1"] = (detections.xyxy[:, 2:4] - detections.xyxy[:, 0:2]).amin(dim=1, keepdim=True)
        square_xyxy: Float[Tensor, "n 4"] = torch.cat([centers - sides * 0.5, centers + sides * 0.5], dim=1)
        crop_batch: CropBatch = crop_frames(
            frames_rgb, frame_indices=detections.frame_indices, bboxes_xyxy=square_xyxy, spec=self.crop_spec, dtype=torch.float32
        )
        outputs: dict[str, Tensor] = run_chunked(self.runtime, {"pixel_values": crop_batch.inputs})
        return outputs["image_embeds"].float()
