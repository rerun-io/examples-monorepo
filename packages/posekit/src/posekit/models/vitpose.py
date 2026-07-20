"""ViTPose top-down 2D pose estimator (transformers weights source).

The reference transformers-as-source pose adapter (docs/design.md §4): HF
carries the weights and the plain ``pixel_values -> heatmaps`` module, while
posekit supplies the GPU crop path and decode — the HF image processor
(numpy/scipy-bound) is never used at inference time. HF's ViTPose pipeline is
UDP end to end (unbiased ``get_warp_matrix`` crops, DARK-UDP decode), so this
adapter uses posekit's ``"udp"`` crop align and :func:`decode_udp_heatmaps`.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Float, UInt8
from torch import Tensor

from posekit.models.base import TopDownPose2d
from posekit.ops.crops import IMAGENET_MEAN_255, IMAGENET_STD_255, CropBatch, CropSpec, crop_coords_to_image, crop_frames
from posekit.ops.decode import decode_udp_heatmaps
from posekit.predictions import BoxDetections, Keypoints2d
from posekit.runtimes import TensorRuntime
from posekit.runtimes.base import TensorSpec, run_chunked
from posekit.runtimes.torch_runtime import TorchRuntime
from posekit.skeletons import COCO_17


class _VitPoseHeatmaps(torch.nn.Module):
    """Adapter returning the raw heatmap tensor for a fixed dataset index."""

    def __init__(self, model: torch.nn.Module, dataset_index: int) -> None:
        super().__init__()
        self.model = model
        self.dataset_index = dataset_index

    def forward(self, pixel_values: Float[Tensor, "n 3 crop_h crop_w"]) -> Float[Tensor, "n k hm_h hm_w"]:
        index: Tensor = torch.full((int(pixel_values.shape[0]),), self.dataset_index, dtype=torch.long, device=pixel_values.device)
        return self.model(pixel_values=pixel_values, dataset_index=index).heatmaps


@dataclass(frozen=True, slots=True)
class VitPoseConfig:
    """ViTPose top-down 2D pose configuration (torch backend)."""

    model_id: str = "usyd-community/vitpose-base-simple"
    """HF checkpoint providing the backbone + head weights (COCO-17 head)."""
    dataset_index: int = 0
    """Expert index for ViTPose+ MoE checkpoints (selects the skeleton); 0 for simple models."""
    device: str = "cuda"
    """Inference device."""
    padding: float = 1.25
    """Bbox padding multiplier before cropping."""
    autocast: bool = True
    """Run under bfloat16 autocast."""
    max_batch_size: int = 32
    """Largest crop batch a single runtime call may submit."""

    def setup(self) -> "VitPose2d":
        """Download weights and return a ready estimator."""
        return VitPose2d(self)


class VitPose2d(TopDownPose2d):
    """Batched GPU ViTPose estimator over a torch runtime."""

    def __init__(self, config: VitPoseConfig) -> None:
        """Load the checkpoint and build the torch runtime.

        Args:
            config: Checkpoint, device, and crop options.
        """
        from transformers import AutoImageProcessor, VitPoseForPoseEstimation

        self.config: VitPoseConfig = config
        self.skeleton = COCO_17
        model = VitPoseForPoseEstimation.from_pretrained(config.model_id).to(config.device).eval()
        processor = AutoImageProcessor.from_pretrained(config.model_id)
        crop_h: int = int(processor.size["height"])
        crop_w: int = int(processor.size["width"])
        num_keypoints: int = self.skeleton.num_keypoints
        with torch.inference_mode():
            probe: Tensor = model(
                pixel_values=torch.zeros((1, 3, crop_h, crop_w), device=config.device),
                dataset_index=torch.zeros((1,), dtype=torch.long, device=config.device),
            ).heatmaps
        self.heatmap_size: tuple[int, int] = (int(probe.shape[3]), int(probe.shape[2]))
        self.runtime: TensorRuntime = TorchRuntime(
            _VitPoseHeatmaps(model, config.dataset_index),
            input_specs=(TensorSpec("pixel_values", (3, crop_h, crop_w), torch.float32),),
            output_specs=(TensorSpec("heatmaps", (num_keypoints, self.heatmap_size[1], self.heatmap_size[0]), torch.float32),),
            max_batch_size=config.max_batch_size,
            autocast_dtype=torch.bfloat16 if config.autocast else None,
        )
        # ImageNet constants in RGB order (ViTPose consumes RGB crops).
        self.crop_spec: CropSpec = CropSpec(
            input_size=(crop_w, crop_h),
            padding=config.padding,
            align="udp",
            bgr=False,
            mean_rgb=IMAGENET_MEAN_255,
            std_rgb=IMAGENET_STD_255,
        )

    @torch.inference_mode()
    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"], detections: BoxDetections) -> Keypoints2d:
        """Estimate keypoints for every detection.

        Args:
            frames_rgb: uint8 RGB NHWC frame batch on the inference device.
            detections: Instance boxes referencing ``frames_rgb`` by index.

        Returns:
            Image-space keypoints, one instance per detection row.
        """
        crop_batch: CropBatch = crop_frames(
            frames_rgb, frame_indices=detections.frame_indices, bboxes_xyxy=detections.xyxy, spec=self.crop_spec, dtype=torch.float32
        )
        if int(crop_batch.inputs.shape[0]) == 0:
            return Keypoints2d.empty(self.skeleton, frames_rgb.device)
        outputs: dict[str, Tensor] = run_chunked(self.runtime, {"pixel_values": crop_batch.inputs})
        xy_crop, scores = decode_udp_heatmaps(outputs["heatmaps"], input_size=self.crop_spec.input_size, heatmap_size=self.heatmap_size)
        xy_image: Float[Tensor, "n k 2"] = crop_coords_to_image(
            xy_crop, centers=crop_batch.centers, scales=crop_batch.scales, input_size=self.crop_spec.input_size
        )
        return Keypoints2d(xy_image, scores, detections.frame_indices, self.skeleton)


__all__ = ("VitPose2d", "VitPoseConfig")
