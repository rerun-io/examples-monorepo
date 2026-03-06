from typing import Literal
import torch
import numpy as np
from jaxtyping import Float, UInt8, Float32
from timeit import default_timer as timer
from monopriors.depth_utils import estimate_intrinsics, depth_to_disparity
from monopriors.relative_depth_models.base_relative_depth import (
    RelativeDepthPrediction,
    BaseRelativePredictor,
)
from einops import rearrange
import cv2
from typing import TypedDict


class Metric3DPredDict(TypedDict):
    prediction: torch.Tensor
    predictions_list: list[torch.Tensor]
    confidence: torch.Tensor
    confidence_list: list[torch.Tensor]
    pred_logit: torch.Tensor
    prediction_normal: torch.Tensor
    normal_out_list: list[torch.Tensor]
    low_resolution_init: torch.Tensor


class Metric3DRelativePredictor(BaseRelativePredictor):
    def __init__(
        self,
        device: Literal["cpu", "cuda"],
    ):
        super().__init__()
        self.device = device
        print("Loading Relative Metric3D model...")
        start = timer()
        self.model = (
            torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True)
            .to(device)
            .eval()
        )
        print(f"Metric3D model loaded. Time: {timer() - start:.2f}s")

        # only for VIT models
        self.input_height: int = 616
        self.input_width: int = 1064

        self.padding_value: list[float] = [123.675, 116.28, 103.53]

    def __call__(
        self,
        rgb: UInt8[np.ndarray, "h w 3"],
        K_33: Float[np.ndarray, "3 3"] | None,
    ) -> RelativeDepthPrediction:
        h, w, _ = rgb.shape
        scale: float = min(self.input_height / h, self.input_width / w)
        rgb_rescaled: UInt8[np.ndarray, "_ _ 3"] = cv2.resize(
            rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR
        )

        k33_rescaled: Float[np.ndarray, "3 3"] | None = (
            K_33 * scale if K_33 is not None else None
        )

        rgb_padded: UInt8[np.ndarray, "616 1064 3"]
        pad_info: list[int]
        rgb_padded, pad_info = self.pad_rgb(rgb_rescaled)

        ##### normalize
        mean: Float32[torch.Tensor, "c 1 1"] = rearrange(
            torch.tensor([123.675, 116.28, 103.53]), "c -> c 1 1"
        )
        std: Float32[torch.Tensor, "c 1 1"] = rearrange(
            torch.tensor([58.395, 57.12, 57.375]), "c -> c 1 1"
        )
        rgb_tensor_hwc: UInt8[torch.Tensor, "616 1064 c"] = torch.from_numpy(rgb_padded)
        rgb_tensor_bchw: Float32[torch.Tensor, "b c 616 1064"] = rearrange(
            rgb_tensor_hwc.float(), "h w c -> 1 c h w"
        )
        rgb_tensor_bchw: Float32[torch.Tensor, "b c 616 1064"] = (
            rgb_tensor_bchw - mean
        ) / std
        rgb_tensor_bchw: Float32[torch.Tensor, "b c 616 1064"] = rgb_tensor_bchw.to(
            self.device
        )

        with torch.no_grad():
            pred_depth_b1hw: Float32[torch.Tensor, "b 1 616 1064"]
            confidence_b1hw: Float32[torch.Tensor, "b 1 616 1064"]
            output_dict: Metric3DPredDict
            pred_depth_b1hw, confidence_b1hw, output_dict = self.model.inference(
                {"input": rgb_tensor_bchw}
            )

        # un pad
        pred_depth_hw: Float32[torch.Tensor, "616 1064"] = rearrange(
            pred_depth_b1hw, "1 1 h w -> h w"
        )
        pred_depth_hw: Float32[torch.Tensor, "_ _"] = pred_depth_hw[
            pad_info[0] : pred_depth_hw.shape[0] - pad_info[1],
            pad_info[2] : pred_depth_hw.shape[1] - pad_info[3],
        ]

        # # upsample to original size
        pred_depth_bchw = torch.nn.functional.interpolate(
            rearrange(pred_depth_hw, "h w -> 1 1 h w"), (h, w), mode="bilinear"
        )
        pred_depth_hw = rearrange(pred_depth_bchw, "1 1 h w -> h w")

        # confidence_b1hw: Float32[torch.Tensor, "b 1 h w"] = (
        #     torch.nn.functional.interpolate(confidence_b1hw, (h, w), mode="bilinear")
        # )
        # conf_tensor_hw: Float32[torch.Tensor, "h w"] = rearrange(
        #     confidence_b1hw, "1 1 h w -> h w"
        # )
        # conf_hw: Float32[np.ndarray, "h w"] = conf_tensor_hw.numpy(force=True)

        # #### de-canonical transform
        # 1000.0 is the focal length of canonical camera
        if k33_rescaled is not None:
            canonical_to_real_scale = k33_rescaled[0, 0] / 1000.0
            # now the depth is metric
            pred_depth_metric_hw = pred_depth_hw * canonical_to_real_scale
            pred_depth_metric_hw = torch.clamp(pred_depth_metric_hw, 0, 300)
        else:
            K_33 = estimate_intrinsics(rgb.shape[0], rgb.shape[1])

        pred_depth_hw = pred_depth_hw.numpy(force=True)
        disparity = depth_to_disparity(pred_depth_hw, focal_length=int(K_33[0, 0]))

        relative_pred = RelativeDepthPrediction(
            disparity=disparity,
            depth=pred_depth_hw,
            # TODO add confidence
            confidence=np.ones_like(pred_depth_hw),
            K_33=K_33,
        )

        return relative_pred

    def pad_rgb(
        self, rgb_resized: UInt8[np.ndarray, "_ _ 3"]
    ) -> tuple[UInt8[np.ndarray, "616 1064 3"], list[int]]:
        h_new, w_new, _ = rgb_resized.shape
        pad_h = self.input_height - h_new
        pad_w = self.input_width - w_new
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb_padded: UInt8[np.ndarray, "616 1064 3"] = cv2.copyMakeBorder(
            rgb_resized,
            pad_h_half,
            pad_h - pad_h_half,
            pad_w_half,
            pad_w - pad_w_half,
            cv2.BORDER_CONSTANT,
            value=self.padding_value,
        )
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
        return rgb_padded, pad_info
