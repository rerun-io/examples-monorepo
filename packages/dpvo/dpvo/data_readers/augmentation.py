"""Data augmentation pipeline for RGBD video training clips.

Provides random spatial transforms (scale + centre crop) and colour jitter
that are jointly applied to aligned image/depth/pose/intrinsic tuples.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from jaxtyping import Float32
from torch import Tensor


class RGBDAugmentor:
    """Apply spatial and colour augmentations to RGBD video clips.

    The augmentor performs two independent stages:

    1. **Colour transform** -- random brightness, contrast, saturation,
       hue jitter; random grayscale; and random inversion applied uniformly
       across all frames (50 % probability).
    2. **Spatial transform** -- random uniform scale followed by a
       deterministic centre crop to ``crop_size``. Intrinsics are adjusted
       accordingly.
    """

    def __init__(self, crop_size: list[int]) -> None:
        """Initialise the augmentor.

        Args:
            crop_size: Target ``[height, width]`` after spatial augmentation.
        """
        self.crop_size: list[int] = crop_size
        self.augcolor: transforms.Compose = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2/3.14),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomInvert(p=0.1),
            transforms.ToTensor()])

        self.max_scale: float = 0.5

    def spatial_transform(
        self,
        images: Float32[Tensor, "n 3 h w"],
        depths: Float32[Tensor, "n h w"],
        poses: Float32[Tensor, "n 7"],
        intrinsics: Float32[Tensor, "n 4"],
    ) -> tuple[Float32[Tensor, "n 3 crop_h crop_w"], Float32[Tensor, "n 7"], Float32[Tensor, "n crop_h crop_w"], Float32[Tensor, "n 4"]]:
        """Randomly scale and centre-crop images, depths, and intrinsics.

        With 80 % probability a random scale in ``[1, 2^max_scale]`` is
        applied via bicubic interpolation (images) / nearest interpolation
        (depths). A deterministic centre crop to ``self.crop_size`` follows.
        Intrinsics are scaled and shifted to match.

        Args:
            images: RGB images ``(n, 3, h, w)`` in ``[0, 255]`` float.
            depths: Per-pixel depth maps ``(n, h, w)``.
            poses: Camera poses ``(n, 7)`` -- unchanged by this transform.
            intrinsics: Camera intrinsics ``(n, 4)`` as ``[fx, fy, cx, cy]``.

        Returns:
            Tuple of ``(images, poses, depths, intrinsics)`` after cropping.
        """
        ht: int = images.shape[2]
        wd: int = images.shape[3]

        max_scale: float = self.max_scale
        _min_scale: float = np.log2(np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd)))

        scale: float = 1
        if np.random.rand() < 0.8:
            scale = 2 ** np.random.uniform(0.0, max_scale)

        intrinsics = scale * intrinsics

        ht1: int = int(scale * ht)
        wd1: int = int(scale * wd)

        depths = depths.unsqueeze(dim=1)

        images = F.interpolate(images, (ht1, wd1), mode='bicubic', align_corners=False)
        depths = F.interpolate(depths, (ht1, wd1), recompute_scale_factor=False)

        # always perform center crop (TODO: try non-center crops)
        y0: int = (images.shape[2] - self.crop_size[0]) // 2
        x0: int = (images.shape[3] - self.crop_size[1]) // 2

        intrinsics = intrinsics - torch.tensor([0.0, 0.0, x0, y0])
        images = images[:, :, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        depths = depths[:, :, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        depths = depths.squeeze(dim=1)
        return images, poses, depths, intrinsics

    def color_transform(self, images: Float32[Tensor, "n 3 h w"]) -> Float32[Tensor, "n 3 h w"]:
        """Apply colour jitter uniformly across all frames in a clip.

        Frames are concatenated along the width axis so that the same
        random jitter parameters are shared, then split back.

        Args:
            images: RGB images ``(n, 3, h, w)`` in ``[0, 255]`` float.

        Returns:
            Colour-augmented images with the same shape.
        """
        num: int = images.shape[0]
        ch: int = images.shape[1]
        ht: int = images.shape[2]
        wd: int = images.shape[3]
        images = images.permute(1, 2, 3, 0).reshape(ch, ht, wd*num)
        images = 255 * self.augcolor(images[[2,1,0]] / 255.0)
        return images[[2,1,0]].reshape(ch, ht, wd, num).permute(3,0,1,2).contiguous()

    def __call__(
        self,
        images: Float32[Tensor, "n 3 h w"],
        poses: Float32[Tensor, "n 7"],
        depths: Float32[Tensor, "n h w"],
        intrinsics: Float32[Tensor, "n 4"],
    ) -> tuple[Float32[Tensor, "n 3 crop_h crop_w"], Float32[Tensor, "n 7"], Float32[Tensor, "n crop_h crop_w"], Float32[Tensor, "n 4"]]:
        """Apply colour (50 % chance) then spatial augmentation.

        Args:
            images: RGB images ``(n, 3, h, w)``.
            poses: Camera poses ``(n, 7)``.
            depths: Depth maps ``(n, h, w)``.
            intrinsics: Intrinsics ``(n, 4)``.

        Returns:
            Augmented ``(images, poses, depths, intrinsics)`` tuple.
        """
        if np.random.rand() < 0.5:
            images = self.color_transform(images)

        return self.spatial_transform(images, depths, poses, intrinsics)
