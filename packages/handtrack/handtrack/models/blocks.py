"""MobileNetV2 blocks shared by the MEgATrack fast networks."""

from jaxtyping import Float32
from torch import Tensor, nn


class InvertedResidual(nn.Module):
    """Expand, depthwise filter, and linearly project; use a skip when sizes match.

    Args:
        in_channels: Input channels.
        out_channels: Projected channels.
        hidden_channels: Absolute hidden width, not an expansion multiplier.
            Equal input and hidden widths omit the expansion convolution.
        stride: Depthwise convolution stride.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, stride: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        if hidden_channels != in_channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU6(),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(hidden_channels, hidden_channels, 3, stride, 1, groups=hidden_channels, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(),
                nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.layers: nn.Sequential = nn.Sequential(*layers)
        self.residual: bool = stride == 1 and in_channels == out_channels

    def forward(self, features: Float32[Tensor, "b channels h w"]) -> Float32[Tensor, "b out_channels out_h out_w"]:
        """Filter Float32[Tensor, 'b channels h w'] features and optionally add a skip."""
        projected: Float32[Tensor, "b out_channels out_h out_w"] = self.layers(features)
        return features + projected if self.residual else projected


def inverted_residual_stack(in_channels: int, rows: list[tuple[int, int, int, int]]) -> nn.Sequential:
    """Build table rows, applying stride only to the first repeat of each row.

    Args:
        in_channels: Channels entering the first block.
        rows: (absolute hidden width, output channels, stride, repeats) per row.

    Returns:
        Sequential inverted-residual blocks in table order.
    """
    layers: list[nn.Module] = []
    for hidden_channels, out_channels, stride, repeats in rows:
        for repeat in range(repeats):
            layers.append(InvertedResidual(in_channels, out_channels, hidden_channels, stride if repeat == 0 else 1))
            in_channels = out_channels
    return nn.Sequential(*layers)
