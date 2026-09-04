"""Checkpoint-compatible identity used after removing stochastic-depth training."""

from jaxtyping import Float
from torch import Tensor, nn


class DropPath(nn.Module):
    """Preserve the upstream module name while applying the inference identity."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        """Initialize the compatibility module."""
        super().__init__()
        _ = drop_prob

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        """Return inference features unchanged."""
        return x
