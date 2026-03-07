from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from jaxtyping import Float, UInt8
from omnidata_tools.torch.modules.midas.dpt_depth import DPTDepthModel

from monopriors.models.surface_normal.base_normal_model import (
    BaseNormalPredictor,
    SurfaceNormalPrediction,
)


class OmniNormalPredictor(BaseNormalPredictor):
    def __init__(self, device: Literal["cpu", "cuda"], omnidata_pretrained_weights_path: Path) -> None:
        self.device = device
        self.model = self._load_model(omnidata_pretrained_weights_path)
        self.image_size = 384

    def _load_model(self, omnidata_pretrained_weights_path: Path):
        model = DPTDepthModel(backbone="vitb_rn50_384", num_channels=3)  # DPT Hybrid
        omnidata_pretrained_weights_path = omnidata_pretrained_weights_path / "omnidata_dpt_normal_v2.ckpt"
        assert omnidata_pretrained_weights_path.exists(), "Weights not found"
        map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device("cpu")
        checkpoint = torch.load(omnidata_pretrained_weights_path, map_location=map_location)

        if "state_dict" in checkpoint:
            state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(self.device)
        return model

    def __call__(
        self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float[np.ndarray, "3 3"] | None
    ) -> SurfaceNormalPrediction:
        rgb = rgb.astype(np.float32) / 255.0
        img_3hw = torch.from_numpy(rgb).permute(2, 0, 1)
        _, H, W = img_3hw.shape
        # omnidata normal model expects 384x384 images
        img_3hw = TF.resize(img_3hw, (self.image_size, self.image_size), antialias=None)
        img_b3hw = img_3hw.unsqueeze(0).to(self.device)

        # return normal_b3hw
        normal_b3hw = self.model(img_b3hw).clamp(min=0, max=1)
        normal_3hw = rearrange(normal_b3hw, "1 c h w -> c h w")
        # normal_3hw = normal_b3hw.squeeze(0)
        # reshape back to original size
        if normal_3hw.shape[1] != H and normal_3hw.shape[2] != W:
            normal_3hw = TF.resize(normal_3hw, (H, W), antialias=None)

        # generates normal that is between 0-1 in the OpenCV Format (RDF)
        # no confidence map is returned from omnidata model
        normal_pred = SurfaceNormalPrediction(
            normal_hw3=rearrange(normal_3hw, "c h w -> h w c"),
            confidence_hw1=np.ones((H, W, 1)),
        )
        return normal_pred
