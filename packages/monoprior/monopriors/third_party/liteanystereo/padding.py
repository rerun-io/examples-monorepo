"""Input padding utilities for LiteAnyStereo V2 inference."""

from collections.abc import Sequence

import torch.nn.functional as F
from jaxtyping import Float32
from torch import Tensor


class InputPadder:
    """Pad 4D image tensors to dimensions divisible by a requested factor."""

    def __init__(self, dims: Sequence[int], mode: str = "sintel", divis_by: int = 8) -> None:
        """Compute symmetric padding for an image shape.

        Args:
            dims: Input tensor dimensions ending in ``(height, width)``.
            mode: ``"sintel"`` for symmetric vertical padding; any other value pads only below.
            divis_by: Required height and width divisor.
        """
        self.ht: int = dims[-2]
        self.wd: int = dims[-1]
        pad_ht: int = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd: int = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == "sintel":
            self._pad: list[int] = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs_bchw: Float32[Tensor, "b c h w"]) -> list[Float32[Tensor, "b c h_pad w_pad"]]:
        """Replicate-pad one or more float32 image tensors.

        Args:
            *inputs_bchw: Float32 tensors with shape ``(batch, channels, height, width)``.

        Returns:
            Float32 tensors with shape ``(batch, channels, padded_height, padded_width)``.
        """
        assert all(input_bchw.ndim == 4 for input_bchw in inputs_bchw)
        padded_inputs: list[Float32[Tensor, "b c h_pad w_pad"]] = [F.pad(input_bchw, self._pad, mode="replicate") for input_bchw in inputs_bchw]
        return padded_inputs

    def unpad(self, x_bchw: Float32[Tensor, "b c h_pad w_pad"]) -> Float32[Tensor, "b c h w"]:
        """Remove this instance's padding from a float32 tensor.

        Args:
            x_bchw: Float32 padded tensor with shape ``(batch, channels, padded_height, padded_width)``.

        Returns:
            Float32 unpadded tensor with shape ``(batch, channels, height, width)``.
        """
        assert x_bchw.ndim == 4
        height: int = x_bchw.shape[-2]
        width: int = x_bchw.shape[-1]
        crop: list[int] = [self._pad[2], height - self._pad[3], self._pad[0], width - self._pad[1]]
        unpadded_bchw: Float32[Tensor, "b c h w"] = x_bchw[..., crop[0] : crop[1], crop[2] : crop[3]]
        return unpadded_bchw
