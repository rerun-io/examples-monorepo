# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable

import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn


class SwiGLUFFN(nn.Module):
    """Dependency-free SwiGLU feed-forward network."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] | None = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        """Build the gated projections."""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Float[Tensor, "... features"]) -> Float[Tensor, "... output_features"]:
        """Apply the local SwiGLU projection."""
        x12: Float[Tensor, "... doubled_hidden"] = self.w12(x)
        x1: Float[Tensor, "... hidden"]
        x2: Float[Tensor, "... hidden"]
        x1, x2 = x12.chunk(2, dim=-1)
        hidden: Float[Tensor, "... hidden"] = F.silu(x1) * x2
        return self.w3(hidden)


class SwiGLUFFNFused(SwiGLUFFN):
    """Checkpoint-compatible name using the dependency-free inference kernel."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] | None = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        """Build the upstream-compatible width-adjusted SwiGLU module."""
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )
