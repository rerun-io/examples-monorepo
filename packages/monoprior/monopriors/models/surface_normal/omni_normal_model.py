from __future__ import annotations

from dataclasses import dataclass
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
    BaseNormalPredictorConfig,
    SurfaceNormalPrediction,
)

OMNIDATA_NORMAL_CHECKPOINT_NAME: str = "omnidata_dpt_normal_v2.ckpt"
OMNIDATA_NORMAL_CHECKPOINT_URL: str = "https://zenodo.org/records/10447888/files/omnidata_dpt_normal_v2.ckpt?download=1"
OMNIDATA_DEFAULT_WEIGHTS_PATH: Path = Path("pretrained_models")


@dataclass
class OmniNormalConfig(BaseNormalPredictorConfig):
    """Configuration for OmniData surface-normal prediction."""

    omnidata_pretrained_weights_path: Path = OMNIDATA_DEFAULT_WEIGHTS_PATH
    """Directory containing ``omnidata_dpt_normal_v2.ckpt``."""

    def setup(self, device: Literal["cpu", "cuda"]) -> OmniNormalPredictor:
        """Build the configured OmniData predictor after validating its manually fetched checkpoint."""
        checkpoint_path: Path = self.omnidata_pretrained_weights_path / OMNIDATA_NORMAL_CHECKPOINT_NAME
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"OmniData normal weights not found at {checkpoint_path}. "
                "Run omnidata_tools/torch/tools/download_surface_normal_models.sh from the upstream EPFL-VILAB/omnidata repository, "
                f"or download {OMNIDATA_NORMAL_CHECKPOINT_URL} to that path. "
                "Pass --omnidata-pretrained-weights-path if the checkpoint is in another directory."
            )
        return OmniNormalPredictor(device=device, checkpoint=checkpoint_path)


class OmniNormalPredictor(BaseNormalPredictor[DPTDepthModel]):
    def __init__(self, device: Literal["cpu", "cuda"], checkpoint: Path) -> None:
        self.device = device
        self.model = self._load_model(checkpoint)
        self.image_size = 384

    def _load_model(self, checkpoint: Path) -> DPTDepthModel:
        model: DPTDepthModel = DPTDepthModel(backbone="vitb_rn50_384", num_channels=3)  # DPT Hybrid
        map_location = (lambda storage, _loc: storage.cuda()) if torch.cuda.is_available() else torch.device("cpu")
        checkpoint_data = torch.load(checkpoint, map_location=map_location)

        if "state_dict" in checkpoint_data:
            state_dict = {}
            for k, v in checkpoint_data["state_dict"].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint_data

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
        img_3hw = TF.resize(img_3hw, [self.image_size, self.image_size], antialias=None)
        img_b3hw = img_3hw.unsqueeze(0).to(self.device)

        # return normal_b3hw
        normal_b3hw = self.model(img_b3hw).clamp(min=0, max=1)
        normal_3hw = rearrange(normal_b3hw, "1 c h w -> c h w")
        # normal_3hw = normal_b3hw.squeeze(0)
        # reshape back to original size
        if normal_3hw.shape[1] != H or normal_3hw.shape[2] != W:
            normal_3hw = TF.resize(normal_3hw, [H, W], antialias=None)

        # generates normal that is between 0-1 in the OpenCV Format (RDF)
        # no confidence map is returned from omnidata model
        normal_pred = SurfaceNormalPrediction(
            normal_hw3=rearrange(normal_3hw, "c h w -> h w c"),
            confidence_hw1=np.ones((H, W, 1)),
        )
        return normal_pred
