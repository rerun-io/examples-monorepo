from typing import Literal
import torch
from torch import Tensor
import numpy as np
from numpy import ndarray
from jaxtyping import UInt16, UInt8, Float32
from timeit import default_timer as timer
from monopriors.depth_completion_models.base_completion_depth import (
    BaseCompletionPredictor,
    CompletionDepthPrediction,
)
from monopriors.third_party.promptda.promptda import PromptDA
import cv2
from einops import rearrange


def ensure_multiple_of(x: float, multiple_of: int = 14) -> int:
    return int(x // multiple_of * multiple_of)


NAME_TO_HFNAME: dict[str, str] = {
    "large": "depth-anything/prompt-depth-anything-vitl",
    "small": "depth-anything/prompt-depth-anything-vits",
    "small-transparent": "depth-anything/prompt-depth-anything-vits-transparent",
}


class PromptDAPredictor(BaseCompletionPredictor):
    def __init__(
        self,
        device: Literal["cpu", "cuda", "mps"],
        model_type: Literal["large"] = "large",
        max_size: int = 1008,
    ) -> None:
        super().__init__()
        self.device = device
        print("Loading Prompt Depth Anything model...")
        model_name: str = NAME_TO_HFNAME[model_type]
        start = timer()
        self.model = PromptDA.from_pretrained(model_name).to(device).eval()
        print(f"Prompt Depth Anything model loaded. Time: {timer() - start:.2f}s")
        self.max_size: int = max_size

    def __call__(
        self,
        rgb: UInt8[ndarray, "h w 3"],
        prompt_depth: UInt16[ndarray, "192 256"],
    ) -> CompletionDepthPrediction:
        rgb_hw3: UInt8[ndarray, "h w 3"] = rgb.copy()
        original_height, original_width, _ = rgb_hw3.shape

        rgb_b3hw: Float32[Tensor, "1 3 h w"] = self.preprocess_rgb(rgb_hw3)
        prompt_depth = self.preprocess_depths(prompt_depth)

        rgb_b3hw, prompt_depth = rgb_b3hw.to(self.device), prompt_depth.to(self.device)
        depth_pred: Float32[Tensor, "1 1 h w"] = self.model.predict(
            rgb_b3hw, prompt_depth
        )

        # convert predicted depth to numpy array and UInt16 mm and resize to match the image size
        depth_pred_np = depth_pred.squeeze().cpu().numpy()
        depth_pred_mm: UInt16[np.ndarray, "h w"] = (depth_pred_np * 1000).astype(
            np.uint16
        )
        depth_pred_resized: UInt16[np.ndarray, "h w"] = cv2.resize(
            depth_pred_mm,
            (original_width, original_height),
        )
        completion_depth_prediction: CompletionDepthPrediction = (
            CompletionDepthPrediction(
                depth_mm=depth_pred_resized,
                confidence=np.ones_like(depth_pred_resized, dtype=np.float32),
            )
        )
        return completion_depth_prediction

    def preprocess_rgb(
        self, rgb_hw3: UInt8[ndarray, "h w 3"]
    ) -> Float32[Tensor, "1 3 h w"]:
        """
        Preprocesses an RGB image by resizing if needed and normalizing pixel values.

        Args:
            rgb_hw3 (ndarray): Input RGB image with shape (H, W, 3) and uint8 dtype.

        Returns:
            Tensor: Preprocessed RGB image as torch.Tensor with shape (1, 3, H, W)
                   and values normalized to [0,1] range.

        Notes:
            - If image size exceeds self.max_size, resizes while maintaining aspect ratio
            - Converts HWC format to BCHW (batch, channels, height, width)
            - Converts uint8 values to float32 and normalizes to [0,1] range
        """
        if max(rgb_hw3.shape) > self.max_size:
            h, w = rgb_hw3.shape[:2]
            scale: float = self.max_size / max(h, w)
            tar_h: int = ensure_multiple_of(x=h * scale)
            tar_w: int = ensure_multiple_of(x=w * scale)
            rgb_hw3 = cv2.resize(rgb_hw3, (tar_w, tar_h), interpolation=cv2.INTER_AREA)

        rgb_b3hw: Float32[ndarray, "1 3 h w"] = rearrange(
            rgb_hw3, "h w c -> 1 c h w"
        ).astype(np.float32)
        rgb_b3hw: Float32[Tensor, "1 3 h w"] = torch.from_numpy(rgb_b3hw) / 255.0
        return rgb_b3hw

    def preprocess_depths(
        self,
        prompt_depth: UInt16[ndarray, "192 256"],
    ) -> Float32[Tensor, "1 1 192 256"]:
        """
        Preprocesses depth data by converting it to meters and rearranging dimensions.

        Args:
            prompt_depth (np.ndarray): Input depth image array in millimeters.
                Shape: (192, 256), dtype: uint16

        Returns:
            torch.Tensor: Preprocessed depth tensor in meters.
                Shape: (1, 1, 192, 256), dtype: float32

        Note:
            - Input depth values are expected to be in millimeters
            - Output depth values are converted to meters (divided by 1000)
            - Adds batch and channel dimensions to the input array
        """
        # convert depth data to tensor
        prompt_depth: Float32[ndarray, "192 256"] = prompt_depth.astype(np.float32)
        prompt_depth: Float32[ndarray, "1 1 192 256"] = rearrange(
            prompt_depth, "h w -> 1 1 h w"
        )
        # convert to meters from mm
        prompt_depth: Float32[Tensor, "1 1 192 256"] = (
            torch.from_numpy(prompt_depth) / 1000
        )
        return prompt_depth
