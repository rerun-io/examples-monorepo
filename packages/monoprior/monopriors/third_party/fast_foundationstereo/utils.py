"""Input padding and stereo sampling utilities."""

from collections.abc import Sequence

import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


class InputPadder:
    """Pad 4D image tensors to dimensions divisible by a requested factor."""

    def __init__(self, dims: Sequence[int], mode: str = "sintel", divis_by: int = 8, force_square: bool = False) -> None:
        """Compute replicate padding for an image shape.

        Args:
            dims: Input tensor dimensions ending in ``(height, width)``.
            mode: ``"sintel"`` for symmetric vertical padding; any other value pads only below.
            divis_by: Required height and width divisor.
            force_square: Whether to pad both dimensions to the next divisible square size.
        """
        self.ht: int = dims[-2]
        self.wd: int = dims[-1]
        if force_square:
            max_side: int = max(self.ht, self.wd)
            pad_ht: int = ((max_side // divis_by) + 1) * divis_by - self.ht
            pad_wd: int = ((max_side // divis_by) + 1) * divis_by - self.wd
        else:
            pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
            pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == "sintel":
            self._pad: list[int] = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs_bchw: Float[Tensor, "b channels h w"]) -> list[Float[Tensor, "b channels h_pad w_pad"]]:
        """Replicate-pad one or more floating-point image tensors.

        Args:
            *inputs_bchw: Floating-point tensors with shape ``(batch, channels, height, width)``.

        Returns:
            Floating-point tensors with shape ``(batch, channels, padded_height, padded_width)``.
        """
        assert all(input_bchw.ndim == 4 for input_bchw in inputs_bchw)
        padded_inputs: list[Float[Tensor, "b channels h_pad w_pad"]] = [
            F.pad(input_bchw, self._pad, mode="replicate") for input_bchw in inputs_bchw
        ]
        return padded_inputs

    def unpad(self, x_bchw: Float[Tensor, "b channels h_pad w_pad"]) -> Float[Tensor, "b channels h w"]:
        """Remove this instance's padding from a floating-point tensor.

        Args:
            x_bchw: Padded tensor with shape ``(batch, channels, padded_height, padded_width)``.

        Returns:
            Unpadded tensor with shape ``(batch, channels, height, width)``.
        """
        assert x_bchw.ndim == 4
        height: int = x_bchw.shape[-2]
        width: int = x_bchw.shape[-1]
        crop: list[int] = [self._pad[2], height - self._pad[3], self._pad[0], width - self._pad[1]]
        unpadded_bchw: Float[Tensor, "b channels h w"] = x_bchw[..., crop[0] : crop[1], crop[2] : crop[3]]
        return unpadded_bchw


def bilinear_sampler(
    image_nchw: Float[Tensor, "samples channels h w"],
    coordinates_nhk2: Float[Tensor, "samples sample_h sample_w 2"],
) -> Float[Tensor, "samples channels sample_h sample_w"]:
    """Sample a floating-point tensor with pixel-space horizontal coordinates.

    Args:
        image_nchw: Features with shape ``(samples, channels, height, width)``.
        coordinates_nhk2: Grid with shape ``(samples, sample_height, sample_width, 2)``; x is in pixel space and y is normalized.

    Returns:
        Samples with shape ``(samples, channels, sample_height, sample_width)``.
    """
    width: int = image_nchw.shape[-1]
    coordinates_nhk2[..., 0] = 2 * coordinates_nhk2[..., 0] / (width - 1) - 1
    sampled_nchk: Float[Tensor, "samples channels sample_h sample_w"] = F.grid_sample(
        image_nchw,
        coordinates_nhk2,
        align_corners=True,
    )
    return sampled_nchk
