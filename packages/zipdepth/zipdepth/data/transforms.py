from collections.abc import Mapping
from typing import NotRequired, TypeAlias, TypedDict

import albumentations as A
import cv2
import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float32, Num, UInt8
from numpy import ndarray
from torch import Tensor
from torch.nn import functional as F

cv2.setNumThreads(0)

ImageArray: TypeAlias = Num[ndarray, "h w 3"]
DepthArray: TypeAlias = Num[ndarray, "h w"]


class TransformResult(TypedDict):
    """Typed aligned outputs returned by Albumentations."""

    image: ImageArray
    depth: NotRequired[DepthArray]
    replay: NotRequired[Mapping[str, object]]


def resize_rgb_and_depth_exact(
    image_chw: UInt8[Tensor, "3 h w"],
    depth_hw: Num[Tensor, "h w"] | None,
    resize_hw: tuple[int, int],
) -> tuple[UInt8[Tensor, "3 out_h out_w"], Num[Tensor, "out_h out_w"] | None]:
    """Resize RGB with antialiased bilinear and aligned depth with nearest-exact."""
    resized_image_bchw: Float32[Tensor, "1 3 out_h out_w"] = F.interpolate(
        image_chw.unsqueeze(0).to(dtype=torch.float32),
        size=resize_hw,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    resized_image_chw: UInt8[Tensor, "3 out_h out_w"] = torch.round(resized_image_bchw.squeeze(0)).clamp_(0.0, 255.0).to(
        dtype=torch.uint8
    )
    if depth_hw is None:
        return resized_image_chw, None
    resized_depth_bchw: Num[Tensor, "1 1 out_h out_w"] = F.interpolate(
        depth_hw.unsqueeze(0).unsqueeze(0),
        size=resize_hw,
        mode="nearest-exact",
    )
    return resized_image_chw, resized_depth_bchw.squeeze(0).squeeze(0)


def _replay_applied_horizontal_flip(replay: Mapping[str, object]) -> bool:
    """Return whether an Albumentations replay applied a horizontal flip."""
    transforms: object = replay.get("transforms", [])
    if not isinstance(transforms, list):
        return False
    transform: object
    for transform in transforms:
        if not isinstance(transform, Mapping):
            continue
        class_name_value: object = transform.get("__class_fullname__", "")
        class_name: str = class_name_value if isinstance(class_name_value, str) else ""
        if class_name.endswith("HorizontalFlip") and bool(transform.get("applied", False)):
            return True
        if _replay_applied_horizontal_flip(transform):
            return True
    return False


class AlbumentationsWrapper:
    """Wrapper: transform(image, depth) -> (image, depth)"""

    def __init__(self, transform: A.Compose, resize_hw: tuple[int, int] | None = None) -> None:
        self.transform: A.Compose = transform
        self.resize_hw: tuple[int, int] | None = resize_hw

    def apply(self, image: ImageArray, depth: DepthArray | None = None) -> TransformResult:
        """Apply the configured Torch resize and Albumentations operations."""
        if image.dtype not in [np.uint8, np.float32]:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

        if depth is not None:
            if depth.ndim == 3:
                depth = depth.squeeze()
            image_height: int = image.shape[0]
            image_width: int = image.shape[1]
            if depth.shape[:2] != (image_height, image_width):
                depth = cv2.resize(depth, (image_width, image_height), interpolation=cv2.INTER_NEAREST_EXACT)

        if self.resize_hw is not None:
            if image.dtype != np.uint8:
                raise ValueError("Torch resize requires a uint8 image")
            image_chw: UInt8[Tensor, "3 h w"] = torch.from_numpy(rearrange(np.ascontiguousarray(image), "h w c -> c h w"))
            depth_tensor_hw: Num[Tensor, "h w"] | None = torch.from_numpy(np.ascontiguousarray(depth)) if depth is not None else None
            resized: tuple[UInt8[Tensor, "3 out_h out_w"], Num[Tensor, "out_h out_w"] | None] = resize_rgb_and_depth_exact(
                image_chw, depth_tensor_hw, self.resize_hw
            )
            resized_image_chw: UInt8[Tensor, "3 out_h out_w"] = resized[0]
            image = rearrange(resized_image_chw.numpy(), "c h w -> h w c")
            resized_depth_hw: Num[Tensor, "out_h out_w"] | None = resized[1]
            depth = resized_depth_hw.numpy() if resized_depth_hw is not None else None

        if depth is not None:
            result: TransformResult = self.transform(image=image, depth=depth)
            return result
        return self.transform(image=image)

    def __call__(self, image: ImageArray, depth: DepthArray | None = None) -> tuple[ImageArray, DepthArray | None]:
        result: TransformResult = self.apply(image, depth)
        if depth is not None:
            return result["image"], result["depth"]
        return result["image"], None

    def apply_with_flip(
        self,
        image: UInt8[ndarray, "h w 3"],
        depth: Float32[ndarray, "h w"],
    ) -> tuple[UInt8[ndarray, "out_h out_w 3"], Float32[ndarray, "out_h out_w"], bool]:
        """Apply aligned transforms and report whether they mirrored the inputs."""
        if not isinstance(self.transform, A.ReplayCompose):
            raise TypeError("flip reporting requires ReplayCompose")
        result: TransformResult = self.apply(image, depth)
        replay: Mapping[str, object] | None = result.get("replay")
        if not isinstance(replay, Mapping):
            raise RuntimeError("ReplayCompose did not return augmentation metadata")
        transformed_image_hwc: UInt8[ndarray, "out_h out_w 3"] = np.asarray(result["image"], dtype=np.uint8)
        transformed_depth_hw: Float32[ndarray, "out_h out_w"] = np.asarray(result["depth"], dtype=np.float32)
        return transformed_image_hwc, transformed_depth_hw, _replay_applied_horizontal_flip(replay)

    def __repr__(self) -> str:
        return repr(self.transform)


def get_train_transforms(height: int = 512, width: int = 512) -> AlbumentationsWrapper:
    """
    Training transforms: random crop/resize + horizontal flip + mild color jitter.
    No aggressive augmentation — with 14M+ training images, variance comes from data diversity.
    """
    transforms = A.ReplayCompose(
        [
            A.OneOf(
                [
                    A.Compose([
                        A.SmallestMaxSize(max_size=max(height, width), interpolation=cv2.INTER_LINEAR),
                        A.RandomCrop(height=height, width=width),
                    ]),
                    A.Resize(height=height, width=width, interpolation=cv2.INTER_LINEAR),
                ],
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.ChannelShuffle(p=0.05),
            A.ToGray(num_output_channels=3, p=0.05),
        ],
        additional_targets={"depth": "mask"},
    )
    return AlbumentationsWrapper(transforms)
