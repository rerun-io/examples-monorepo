"""open_clip ViT-B-32 (laion2b_s34b_b79k) person-crop encoder in posekit's ``IdentityEncoder`` role.

The crop is the detection box truncated to whole pixels, preprocessed by open_clip on the host (PIL), exactly as the
original tracker did; posekit's own ``ClipIdentity`` crops on the GPU and loads the transformers checkpoint instead.
"""

from typing import Any

import numpy as np
import torch
from jaxtyping import Float32, UInt8
from numpy import ndarray
from PIL import Image
from posekit.models import IdentityEncoder
from posekit.predictions import BoxDetections
from torch import Tensor


class OpenClipIdentity(IdentityEncoder):
    embed_dim: int = 512

    def __init__(self, device: str = "cuda") -> None:
        import open_clip

        created: Any = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")  # loose upstream stubs
        self.model: Any = created[0].to(device).eval()
        self.preprocess: Any = created[2]
        self.device: str = device

    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"], detections: BoxDetections) -> Float32[Tensor, "n 512"]:
        """Unnormalised CLIP features of every detection crop, on the host."""
        if detections.num_detections == 0:
            return torch.zeros((0, self.embed_dim), dtype=torch.float32)
        frames: UInt8[ndarray, "b h w 3"] = frames_rgb.cpu().numpy()
        boxes: Float32[ndarray, "n 4"] = detections.xyxy_numpy()
        crops: list[Tensor] = []
        for box, index in zip(boxes, detections.frame_indices.tolist(), strict=True):
            height, width = frames.shape[1:3]
            x1, y1, x2, y2 = max(0, int(box[0])), max(0, int(box[1])), min(width, int(box[2])), min(height, int(box[3]))
            crops.append(self.preprocess(Image.fromarray(np.ascontiguousarray(frames[index, y1:y2, x1:x2]))))
        with torch.inference_mode():
            features: Tensor = self.model.encode_image(torch.stack(crops).to(self.device))
        return features.detach().cpu().float()
