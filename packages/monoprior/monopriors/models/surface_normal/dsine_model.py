import os
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float, UInt8
from torchvision import transforms

from monopriors.models.surface_normal.base_normal_model import (
    BaseNormalPredictor,
    SurfaceNormalPrediction,
)
from monopriors.third_party.dsine.dsine import DSINE
from monopriors.third_party.dsine.dsine_kappa import DSINE_v02_kappa
from monopriors.third_party.dsine.utils.utils import get_intrins_from_fov, pad_input


class DSineNormalPredictor(BaseNormalPredictor):
    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        model_type: Literal["dsine", "dsine_kappa"] = "dsine_kappa",
    ):
        self.device = device
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.model = self.load_model(model_type=model_type)

    def load_model(self, model_type: Literal["dsine", "dsine_kappa"]):
        state_dict = self._load_state_dict(local_file_path="checkpoints/dsine_kappa.pt", model_type=model_type)
        model = DSINE() if model_type == "dsine" else DSINE_v02_kappa()

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model = model.to(self.device)
        model.pixel_coords = model.pixel_coords.to(self.device)

        return model

    def _load_state_dict(
        self,
        local_file_path: str | None = None,
        model_type: Literal["dsine", "dsine_kappa"] = "dsine_kappa",
    ):
        if local_file_path is not None and os.path.exists(local_file_path):
            # Load state_dict from local file
            state_dict = torch.load(local_file_path, map_location=torch.device("cpu"))
        else:
            # Load state_dict from the default URL
            file_name = "dsine.pt" if model_type == "dsine" else "dsine_kappa.pt"
            url = (
                "https://huggingface.co/camenduru/DSINE/resolve/main/dsine.pt"
                if model_type == "dsine"
                else "https://huggingface.co/pablovela5620/dsine_kappa/resolve/main/dsine_kappa.pt"
            )
            state_dict = torch.hub.load_state_dict_from_url(url, file_name=file_name, map_location=torch.device("cpu"))

        return state_dict["model"]

    def __call__(
        self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float[np.ndarray, "3 3"] | None
    ) -> SurfaceNormalPrediction:
        # preprocess the input image
        rgb: Float[np.ndarray, "h w 3"] = rgb.astype(np.float32) / 255.0
        rgb: Float[torch.Tensor, "1 c h w"] = torch.from_numpy(rearrange(rgb, "h w c -> 1 c h w")).to(self.device)

        _, _, h, w = rgb.shape

        # zero-pad the input image so that both the width and height are multiples of 32
        left, right, top, bottom = pad_input(h, w)
        rgb = F.pad(rgb, (left, right, top, bottom), mode="constant", value=0.0)
        rgb = self.transform(rgb)

        if K_33 is None:
            K_33: Float[torch.Tensor, "3 3"] = get_intrins_from_fov(new_fov=60.0, H=h, W=w, device=self.device)
            K_b33: Float[torch.Tensor, "b 3 3"] = rearrange(K_33, "r c -> 1 r c")
        else:
            K_b33 = torch.from_numpy(rearrange(K_33, "r c -> 1 r c")).to(self.device)

        # add padding to intrinsics
        K_b33[:, 0, 2] += left
        K_b33[:, 1, 2] += top

        with torch.no_grad():
            normal_list: list[torch.Tensor] = self.model(rgb, intrins=K_b33)
            # last value in the list is the normal map
            normal_bchw: Float[torch.Tensor, "b 3 _ _"] | Float[torch.Tensor, "b 4 _ _"] = normal_list[-1]
            # undo padding
            normal_bchw: Float[torch.Tensor, "b 3 _ _"] | Float[torch.Tensor, "b 4 _ _"] = normal_bchw[
                :, :, top : top + h, left : left + w
            ]

            normal_b3hw: Float[torch.Tensor, "b 3 h w"]
            conf_b1hw: Float[torch.Tensor, "b 1 h w"] | None

            normal_b3hw, conf_b1hw = normal_bchw[:, :3, :, :], normal_bchw[:, 3:, :, :]
            conf_b1hw = None if conf_b1hw.size(1) == 0 else conf_b1hw

        confidence_hw1: np.ndarray = (
            rearrange(conf_b1hw, "1 c h w -> h w c").numpy(force=True)
            if conf_b1hw is not None
            else np.ones((h, w, 1), dtype=np.float32)
        )

        normal_pred = SurfaceNormalPrediction(
            normal_hw3=rearrange(normal_b3hw, "1 c h w -> h w c").numpy(force=True),
            confidence_hw1=confidence_hw1,
        )

        return normal_pred
