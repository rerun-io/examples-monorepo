# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


# Implementation of 2D Rotary Position Embeddings (RoPE).

# This module provides a clean implementation of 2D Rotary Position Embeddings,
# which extends the original RoPE concept to handle 2D spatial positions.

# Inspired by:
#         https://github.com/meta-llama/codellama/blob/main/llama/model.py
#         https://github.com/naver-ai/rope-vit


from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionGetter:
    """Generates and caches 2D spatial positions for patches in a grid.

    This class efficiently manages the generation of spatial coordinates for patches
    in a 2D grid, caching results to avoid redundant computations.

    Attributes:
        position_cache: Dictionary storing precomputed position tensors for different
            grid dimensions.
    """

    def __init__(self):
        """Initializes the position generator with an empty cache."""
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(
        self, batch_size: int, height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        """Generates spatial positions for a batch of patches.

        Args:
            batch_size: Number of samples in the batch.
            height: Height of the grid in patches.
            width: Width of the grid in patches.
            device: Target device for the position tensor.

        Returns:
            Tensor of shape (batch_size, height*width, 2) containing y,x coordinates
            for each position in the grid, repeated for each batch item.
        """
        if (height, width) not in self.position_cache:
            # ONNX-safe equivalent of cartesian_prod: flattened (H*W, 2) grid
            # [(0,0),(0,1),...,(0,W-1),(1,0),...,(H-1,W-1)].
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            yy = y_coords.view(height, 1).expand(height, width)   # (H, W)
            xx = x_coords.view(1, width).expand(height, width)    # (H, W)
            positions = torch.stack([yy, xx], dim=-1).reshape(height * width, 2)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding implementation.

    This module applies rotary position embeddings to input tokens based on their
    2D spatial positions. It handles the position-dependent rotation of features
    separately for vertical and horizontal dimensions.

    Args:
        frequency: Base frequency for the position embeddings. Default: 100.0
        scaling_factor: Scaling factor for frequency computation. Default: 1.0

    Attributes:
        base_frequency: Base frequency for computing position embeddings.
        scaling_factor: Factor to scale the computed frequencies.
        frequency_cache: Cache for storing precomputed frequency components.
    """

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        """Initializes the 2D RoPE module."""
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes frequency components for rotary embeddings.

        Args:
            dim: Feature dimension (must be even).
            seq_len: Maximum sequence length.
            device: Target device for computations.
            dtype: Data type for the computed tensors.

        Returns:
            Tuple of (cosine, sine) tensors for frequency components.
        """
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            # Compute frequency bands
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)

            # Generate position-dependent frequencies
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)

            # Compute and cache frequency components
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)

        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        """Performs feature rotation by splitting and recombining feature dimensions.

        Args:
            x: Input tensor to rotate.

        Returns:
            Rotated feature tensor.
        """
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        cos_comp: torch.Tensor,
        sin_comp: torch.Tensor,
    ) -> torch.Tensor:
        """Applies 1D rotary position embeddings along one dimension (integer position path).

        Args:
            tokens: Input token features.
            positions: Integer position indices (long).
            cos_comp: Cosine components lookup table.
            sin_comp: Sine components lookup table.

        Returns:
            Tokens with applied rotary position embeddings.
        """
        # Embed positions with frequency components
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]
        # Apply rotation
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def _apply_1d_rope_continuous(
        self,
        tokens: torch.Tensor,
        positions_float: torch.Tensor,
    ) -> torch.Tensor:
        """1D RoPE for continuous (float) positions.

        Used by FishRoPE: positions are continuous angular coordinates rather than
        integer indices. cos/sin are computed on the fly instead of via lookup.

        Args:
            tokens: (B, n_heads, N, dim/2)
            positions_float: (B, N) float positions, pre-scaled to the target range.

        Returns:
            Rotated tokens.
        """
        feature_dim = tokens.size(-1)
        device, dtype = tokens.device, tokens.dtype
        # Frequency bands (same formula as the integer path)
        exponents = torch.arange(0, feature_dim, 2, device=device).float() / feature_dim
        inv_freq = (1.0 / (self.base_frequency ** exponents)).to(dtype)   # (dim/2,)
        angles = positions_float[..., None].to(dtype) * inv_freq[None, None, :]   # (B, N, dim/2)
        angles = torch.cat((angles, angles), dim=-1)                              # (B, N, dim)
        cos = angles.cos()[:, None, :, :]                                          # (B, 1, N, dim)
        sin = angles.sin()[:, None, :, :]
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Applies 2D rotary position embeddings to input tokens.

        Args:
            tokens:    (batch_size, n_heads, n_tokens, dim). dim must be a multiple of 4.
            positions: (batch_size, n_tokens, 2). Integer dtype uses the F.embedding
                       lookup path (pinhole pixel indices); float dtype uses the
                       on-the-fly cos/sin path (FishRoPE angular positions).

        Returns:
            Rotated tokens, same shape as input.
        """
        # Validate inputs
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert (
            positions.ndim == 3 and positions.shape[-1] == 2
        ), "Positions must have shape (batch_size, n_tokens, 2)"

        feature_dim = tokens.size(-1) // 2

        # Split features for vertical and horizontal processing
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)

        if positions.is_floating_point():
            # FishRoPE path: float angular positions, cos/sin computed on the fly
            vertical_features = self._apply_1d_rope_continuous(
                vertical_features, positions[..., 0]
            )
            horizontal_features = self._apply_1d_rope_continuous(
                horizontal_features, positions[..., 1]
            )
        else:
            # Integer path: pinhole pixel indices via precomputed lookup
            max_position = int(positions.max()) + 1
            cos_comp, sin_comp = self._compute_frequency_components(
                feature_dim, max_position, tokens.device, tokens.dtype
            )
            vertical_features = self._apply_1d_rope(
                vertical_features, positions[..., 0], cos_comp, sin_comp
            )
            horizontal_features = self._apply_1d_rope(
                horizontal_features, positions[..., 1], cos_comp, sin_comp
            )

        # Combine processed features
        return torch.cat((vertical_features, horizontal_features), dim=-1)
