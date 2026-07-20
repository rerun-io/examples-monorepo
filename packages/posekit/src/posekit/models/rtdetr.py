"""RT-DETRv2 NMS-free person detector (transformers weights source).

Loads ``PekingU/rtdetr_v2_*`` checkpoints as plain modules (docs/design.md §4)
and keeps pre/post on GPU: preprocessing is a bilinear 640x640 resize plus
1/255 rescale (no normalization), decode is sigmoid over the query logits and
a cxcywh->xyxy rescale — no NMS anywhere, queries are one-to-one with
instances. The person class id is read from ``config.id2label``, never
hardcoded.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Float, UInt8
from torch import Tensor

from posekit.models.base import PersonDetector
from posekit.ops.crops import _float32_vector
from posekit.predictions import BoxDetections, validate_frames_rgb
from posekit.runtimes import TensorRuntime
from posekit.runtimes.base import TensorSpec, run_chunked
from posekit.runtimes.torch_runtime import TorchRuntime


class _RtDetrHeads(torch.nn.Module):
    """Adapter returning the raw query logits and normalized cxcywh boxes."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: Float[Tensor, "b 3 in_h in_w"]) -> tuple[Float[Tensor, "b q c"], Float[Tensor, "b q 4"]]:
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits, outputs.pred_boxes


@dataclass(frozen=True, slots=True)
class RtDetrDetectorConfig:
    """RT-DETRv2 person detector configuration (torch backend)."""

    model_id: str = "PekingU/rtdetr_v2_r50vd"
    """HF checkpoint providing weights, input size, and the label map."""
    device: str = "cuda"
    """Inference device."""
    score_threshold: float = 0.5
    """Minimum per-query person probability kept in the output."""
    label: str = "person"
    """Class name looked up in ``config.id2label`` to select detections."""
    autocast: bool = True
    """Run under bfloat16 autocast."""
    max_batch_size: int = 32
    """Largest frame batch a single runtime call may submit."""

    def setup(self) -> "RtDetrDetector":
        """Download weights and return a ready detector."""
        return RtDetrDetector(self)


class RtDetrDetector(PersonDetector):
    """Batched GPU RT-DETRv2 detector (query-based, NMS-free)."""

    def __init__(self, config: RtDetrDetectorConfig) -> None:
        """Load the checkpoint and build the torch runtime.

        Args:
            config: Checkpoint, device, and thresholds.

        Raises:
            ValueError: If ``config.label`` is not in the checkpoint's label map.
        """
        from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

        self.config: RtDetrDetectorConfig = config
        model = RTDetrV2ForObjectDetection.from_pretrained(config.model_id).to(config.device).eval()
        id2label: dict[int, str] = {int(idx): str(name) for idx, name in (model.config.id2label or {}).items()}
        label_ids: list[int] = [idx for idx, name in id2label.items() if name == config.label]
        if not label_ids:
            raise ValueError(f"Label {config.label!r} not found in {config.model_id} id2label ({len(id2label)} classes).")
        self.class_id: int = label_ids[0]
        processor = AutoImageProcessor.from_pretrained(config.model_id)
        self.input_hw: tuple[int, int] = (int(processor.size["height"]), int(processor.size["width"]))
        num_queries: int = int(model.config.num_queries)
        num_labels: int = len(id2label)
        self.runtime: TensorRuntime = TorchRuntime(
            _RtDetrHeads(model),
            input_specs=(TensorSpec("pixel_values", (3, self.input_hw[0], self.input_hw[1]), torch.float32),),
            output_specs=(
                TensorSpec("logits", (num_queries, num_labels), torch.float32),
                TensorSpec("pred_boxes", (num_queries, 4), torch.float32),
            ),
            max_batch_size=config.max_batch_size,
            autocast_dtype=torch.bfloat16 if config.autocast else None,
        )

    @torch.inference_mode()
    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"]) -> BoxDetections:
        """Detect instances across a frame batch.

        Args:
            frames_rgb: uint8 RGB NHWC frame batch on the inference device.

        Returns:
            Flattened detections across the batch.
        """
        validate_frames_rgb(frames_rgb)
        frame_h: int = int(frames_rgb.shape[1])
        frame_w: int = int(frames_rgb.shape[2])
        chw: Float[Tensor, "b 3 h w"] = frames_rgb.permute(0, 3, 1, 2).float() / 255.0
        pixel_values: Float[Tensor, "b 3 in_h in_w"] = F.interpolate(chw, size=self.input_hw, mode="bilinear", align_corners=False)
        outputs: dict[str, Tensor] = run_chunked(self.runtime, {"pixel_values": pixel_values})
        person_scores: Float[Tensor, "b q"] = outputs["logits"].sigmoid()[:, :, self.class_id]
        boxes_cxcywh: Float[Tensor, "b q 4"] = outputs["pred_boxes"]
        keep: Tensor = person_scores >= self.config.score_threshold
        frame_indices, query_indices = torch.where(keep)
        kept: Float[Tensor, "n 4"] = boxes_cxcywh[frame_indices, query_indices]
        half_wh: Float[Tensor, "n 2"] = kept[:, 2:4] * 0.5
        scale: Float[Tensor, "4"] = _float32_vector((float(frame_w), float(frame_h), float(frame_w), float(frame_h)), frames_rgb.device)
        xyxy: Float[Tensor, "n 4"] = torch.cat([kept[:, 0:2] - half_wh, kept[:, 0:2] + half_wh], dim=1) * scale
        return BoxDetections(xyxy=xyxy.contiguous(), scores=person_scores[frame_indices, query_indices].contiguous(), frame_indices=frame_indices)


__all__ = ("RtDetrDetector", "RtDetrDetectorConfig")
