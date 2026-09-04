# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L103-L110  # noqa: E501


import torch
from jaxtyping import Float
from torch import Tensor, nn


class LayerScale(nn.Module):
    """Learnable per-channel residual scale."""

    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        """Initialize the residual scale parameter."""
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        """Scale the input by the learned channel weights."""
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
