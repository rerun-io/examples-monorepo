import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from jaxtyping import Float32


class RGBDAugmentor:
    """ perform augmentation on RGB-D video """

    def __init__(self, crop_size: list[int]) -> None:
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
        images: Float32[torch.Tensor, "n 3 h w"],
        depths: Float32[torch.Tensor, "n h w"],
        poses: Float32[torch.Tensor, "n 7"],
        intrinsics: Float32[torch.Tensor, "n 4"],
    ) -> tuple[Float32[torch.Tensor, "n 3 crop_h crop_w"], Float32[torch.Tensor, "n 7"], Float32[torch.Tensor, "n crop_h crop_w"], Float32[torch.Tensor, "n 4"]]:
        """ cropping and resizing """
        ht: int = images.shape[2]
        wd: int = images.shape[3]

        max_scale: float = self.max_scale
        min_scale: float = np.log2(np.maximum(
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

    def color_transform(self, images: Float32[torch.Tensor, "n 3 h w"]) -> Float32[torch.Tensor, "n 3 h w"]:
        """ color jittering """
        num: int = images.shape[0]
        ch: int = images.shape[1]
        ht: int = images.shape[2]
        wd: int = images.shape[3]
        images = images.permute(1, 2, 3, 0).reshape(ch, ht, wd*num)
        images = 255 * self.augcolor(images[[2,1,0]] / 255.0)
        return images[[2,1,0]].reshape(ch, ht, wd, num).permute(3,0,1,2).contiguous()

    def __call__(
        self,
        images: Float32[torch.Tensor, "n 3 h w"],
        poses: Float32[torch.Tensor, "n 7"],
        depths: Float32[torch.Tensor, "n h w"],
        intrinsics: Float32[torch.Tensor, "n 4"],
    ) -> tuple[Float32[torch.Tensor, "n 3 crop_h crop_w"], Float32[torch.Tensor, "n 7"], Float32[torch.Tensor, "n crop_h crop_w"], Float32[torch.Tensor, "n 4"]]:
        if np.random.rand() < 0.5:
            images = self.color_transform(images)

        return self.spatial_transform(images, depths, poses, intrinsics)
