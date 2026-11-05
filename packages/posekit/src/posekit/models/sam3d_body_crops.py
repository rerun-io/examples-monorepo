"""GPU affine crops with the released SAM body box and interpolation rules."""

import torch
import warp as wp
from jaxtyping import Bool, Float32, Int64, UInt8
from torch import Tensor

from posekit.models.sam3d_body_crop_kernels import affine_sample, crop_geometry


class BodyCropper:
    """Sample body or hand boxes directly from decoded GPU frames."""

    def __init__(self, device: torch.device) -> None:
        wp.init()
        self.device: torch.device = device
        self.streams: dict[int, wp.Stream] = {}
        self.frame_constants: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}

    def prepare(
        self, frames: UInt8[Tensor, "b h w 3"], boxes: Float32[Tensor, "n 4"],
        frame_indices: Int64[Tensor, "n"], *, padding: float,
        masks: Float32[Tensor, "n h w"] | None = None, scores: Float32[Tensor, "n"] | None = None,
        camera: Float32[Tensor, "3 3"] | None = None, mirrored: Bool[Tensor, "n"] | None = None,
    ) -> dict[str, Tensor]:
        """Create the model's tensor batch; no frame pixels or box values leave CUDA."""
        count: int = len(boxes)
        height, width = frames.shape[1:3]
        shape: tuple[int, int] = (height, width)
        if shape not in self.frame_constants:
            focal: float = (height**2 + width**2) ** 0.5
            default_camera: Tensor = torch.tensor([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], device=frames.device)
            image_size: Tensor = torch.tensor([width, height], dtype=torch.float32, device=frames.device)
            self.frame_constants[shape] = (default_camera, image_size)
        default_camera, image_size = self.frame_constants[shape]
        centers = torch.empty((count, 2), device=frames.device)
        scales = torch.empty_like(centers)
        affine = torch.empty((count, 2, 3), device=frames.device)
        inverse = torch.empty((count, 4), device=frames.device, dtype=torch.float64)
        images = torch.empty((count, 3, 512, 512), device=frames.device)
        output_masks = torch.empty((count, 1, 512, 512), device=frames.device)
        if mirrored is None:
            mirrored = torch.zeros(count, dtype=torch.bool, device=frames.device)
        current = torch.cuda.current_stream(self.device)
        if current.cuda_stream not in self.streams:
            self.streams[current.cuda_stream] = wp.stream_from_torch(current)
        stream: wp.Stream = self.streams[current.cuda_stream]
        wp.launch(crop_geometry, dim=count, inputs=[wp.from_torch(boxes), padding],
                  outputs=[wp.from_torch(centers), wp.from_torch(scales), wp.from_torch(affine), wp.from_torch(inverse)],
                  device=str(self.device), stream=stream, record_tape=False)
        empty_masks = torch.empty((count, 0, 0), device=frames.device) if masks is None else masks
        wp.launch(affine_sample, dim=(count, 512, 512), inputs=[wp.from_torch(frames), wp.from_torch(frame_indices),
                  wp.from_torch(inverse), wp.from_torch(mirrored), wp.from_torch(empty_masks), masks is not None],
                  outputs=[wp.from_torch(images), wp.from_torch(output_masks)],
                  device=str(self.device), stream=stream, record_tape=False)
        if camera is None:
            camera = default_camera
        mask_scores = torch.zeros(count, device=frames.device) if masks is None else scores
        if mask_scores is None:
            mask_scores = torch.ones(count, device=frames.device)
        return {
            "img": images[None], "bbox_center": centers[None], "bbox_scale": scales[None], "affine_trans": affine[None],
            "img_size": torch.full((1, count, 2), 512.0, device=frames.device),
            "ori_img_size": image_size.expand(1, count, 2),
            "bbox": boxes[None], "mask": output_masks[None], "mask_score": mask_scores[None],
            "person_valid": torch.ones((1, count), device=frames.device), "cam_int": camera[None],
        }
