"""Selective recurrent disparity updates for Fast-FoundationStereo."""

from typing import TypeAlias

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from monopriors.third_party.fast_foundationstereo.extractor import ConfigLike
from monopriors.third_party.fast_foundationstereo.submodule import EdgeNextConvEncoder

TensorPyramid: TypeAlias = list[Float[Tensor, "b _channels h w"]]
UpdateBlockOutput: TypeAlias = tuple[
    TensorPyramid,
    Float[Tensor, "b mask_channels h w"],
    Float[Tensor, "b 1 h w"],
]


class DispHead(nn.Module):
    """Predict a disparity update from the recurrent hidden state."""

    def __init__(self, input_dim: int = 128, output_dim: int = 1) -> None:
        """Initialize the disparity head.

        Args:
            input_dim: Recurrent hidden-state channel count.
            output_dim: Output disparity channel count.
        """
        super().__init__()
        self.conv: nn.Sequential = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            EdgeNextConvEncoder(input_dim, expan_ratio=4, kernel_size=7, norm=None),
            EdgeNextConvEncoder(input_dim, expan_ratio=4, kernel_size=7, norm=None),
            nn.Conv2d(input_dim, output_dim, 3, padding=1),
        )

    def forward(self, x_bchw: Float[Tensor, "b channels h w"]) -> Float[Tensor, "b output_channels h w"]:
        """Predict a floating-point disparity update.

        Args:
            x_bchw: Hidden state with shape ``(batch, channels, height, width)``.

        Returns:
            Disparity update with shape ``(batch, output_channels, height, width)``.
        """
        output_bchw: Float[Tensor, "b output_channels h w"] = self.conv(x_bchw)
        return output_bchw


class BasicMotionEncoder(nn.Module):
    """Encode geometric correlation and current disparity for the recurrent update."""

    def __init__(self, args: ConfigLike, ngroup: int = 8) -> None:
        """Initialize the motion encoder.

        Args:
            args: Fast-FoundationStereo inference configuration.
            ngroup: Number of groupwise-correlation channels.
        """
        super().__init__()
        self.args: ConfigLike = args
        correlation_planes: int = int(args["corr_levels"]) * (2 * int(args["corr_radius"]) + 1) * (ngroup + 1)
        self.convc1: nn.Conv2d = nn.Conv2d(correlation_planes, 256, kernel_size=1, padding=0)
        self.convc2: nn.Conv2d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.convd1: nn.Conv2d = nn.Conv2d(1, 64, kernel_size=7, padding=3)
        self.convd2: nn.Conv2d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv: nn.Conv2d = nn.Conv2d(320, int(args["hidden_dims"][0]) - 1, kernel_size=1, padding=0)

    def forward(
        self,
        disparity_b1hw: Float[Tensor, "b 1 h w"],
        correlation_bchw: Float[Tensor, "b correlation_channels h w"],
    ) -> Float[Tensor, "b hidden h w"]:
        """Encode floating-point disparity and correlation features.

        Args:
            disparity_b1hw: Disparity with shape ``(batch, 1, height, width)``.
            correlation_bchw: Correlation with shape ``(batch, correlation_channels, height, width)``.

        Returns:
            Motion features with shape ``(batch, hidden, height, width)``.
        """
        correlation1_bchw: Float[Tensor, "b correlation_hidden1 h w"] = F.relu(self.convc1(correlation_bchw))
        correlation2_bchw: Float[Tensor, "b correlation_hidden2 h w"] = F.relu(self.convc2(correlation1_bchw))
        disparity1_bchw: Float[Tensor, "b disparity_hidden1 h w"] = F.relu(self.convd1(disparity_b1hw))
        disparity2_bchw: Float[Tensor, "b disparity_hidden2 h w"] = F.relu(self.convd2(disparity1_bchw))
        combined_bchw: Float[Tensor, "b combined_channels h w"] = torch.cat([correlation2_bchw, disparity2_bchw], dim=1)
        encoded_bchw: Float[Tensor, "b hidden_minus_one h w"] = F.relu(self.conv(combined_bchw))
        motion_bchw: Float[Tensor, "b hidden h w"] = torch.cat([encoded_bchw, disparity_b1hw], dim=1)
        return motion_bchw


class RaftConvGRU(nn.Module):
    """RAFT-style convolutional gated recurrent unit."""

    def __init__(self, hidden_dim: int = 128, input_dim: int = 256, kernel_size: int = 3) -> None:
        """Initialize the GRU.

        Args:
            hidden_dim: Hidden-state channel count.
            input_dim: Input feature channel count.
            kernel_size: Gate convolution kernel size.
        """
        super().__init__()
        padding: int = kernel_size // 2
        self.convz: nn.Conv2d = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=padding)
        self.convr: nn.Conv2d = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=padding)
        self.convq: nn.Conv2d = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=padding)

    def forward(
        self,
        hidden_bchw: Float[Tensor, "b hidden h w"],
        input_bchw: Float[Tensor, "b input_channels h w"],
        hidden_input_bchw: Float[Tensor, "b combined_channels h w"],
    ) -> Float[Tensor, "b hidden h w"]:
        """Update a floating-point recurrent hidden state.

        Args:
            hidden_bchw: Hidden state with shape ``(batch, hidden, height, width)``.
            input_bchw: Input features with shape ``(batch, input_channels, height, width)``.
            hidden_input_bchw: Preprocessed hidden/input concatenation with shape ``(batch, combined_channels, height, width)``.

        Returns:
            Updated hidden state with shape ``(batch, hidden, height, width)``.
        """
        update_gate_bchw: Float[Tensor, "b hidden h w"] = torch.sigmoid(self.convz(hidden_input_bchw))
        reset_gate_bchw: Float[Tensor, "b hidden h w"] = torch.sigmoid(self.convr(hidden_input_bchw))
        candidate_input_bchw: Float[Tensor, "b combined_channels h w"] = torch.cat([reset_gate_bchw * hidden_bchw, input_bchw], dim=1)
        candidate_bchw: Float[Tensor, "b hidden h w"] = torch.tanh(self.convq(candidate_input_bchw))
        output_bchw: Float[Tensor, "b hidden h w"] = (1 - update_gate_bchw) * hidden_bchw + update_gate_bchw * candidate_bchw
        return output_bchw


class SelectiveConvGRU(nn.Module):
    """Blend small- and large-kernel GRU updates with learned spatial attention."""

    def __init__(self, hidden_dim: int = 128, input_dim: int = 256, small_kernel_size: int = 1, large_kernel_size: int = 3) -> None:
        """Initialize the selective GRU.

        Args:
            hidden_dim: Hidden-state channel count.
            input_dim: Input feature channel count.
            small_kernel_size: Small GRU kernel size.
            large_kernel_size: Large GRU kernel size.
        """
        super().__init__()
        self.conv0: nn.Sequential = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv1: nn.Sequential = nn.Sequential(
            nn.Conv2d(input_dim + hidden_dim, input_dim + hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.small_gru: RaftConvGRU = RaftConvGRU(hidden_dim, input_dim, small_kernel_size)
        self.large_gru: RaftConvGRU = RaftConvGRU(hidden_dim, input_dim, large_kernel_size)

    def forward(
        self,
        attention_b1hw: Float[Tensor, "b 1 h w"],
        hidden_bchw: Float[Tensor, "b hidden h w"],
        *inputs_bchw: Float[Tensor, "b _channels h w"],
    ) -> Float[Tensor, "b hidden h w"]:
        """Apply a selectively blended recurrent update.

        Args:
            attention_b1hw: Spatial weights with shape ``(batch, 1, height, width)``.
            hidden_bchw: Hidden state with shape ``(batch, hidden, height, width)``.
            *inputs_bchw: Input features with shape ``(batch, channels, height, width)``.

        Returns:
            Updated hidden state with shape ``(batch, hidden, height, width)``.
        """
        concatenated_bchw: Float[Tensor, "b input_channels h w"] = torch.cat(inputs_bchw, dim=1)
        encoded_input_bchw: Float[Tensor, "b input_channels h w"] = self.conv0(concatenated_bchw)
        hidden_input_bchw: Float[Tensor, "b combined_channels h w"] = self.conv1(torch.cat([encoded_input_bchw, hidden_bchw], dim=1))
        small_update_bchw: Float[Tensor, "b hidden h w"] = self.small_gru(hidden_bchw, encoded_input_bchw, hidden_input_bchw)
        large_update_bchw: Float[Tensor, "b hidden h w"] = self.large_gru(hidden_bchw, encoded_input_bchw, hidden_input_bchw)
        output_bchw: Float[Tensor, "b hidden h w"] = small_update_bchw * attention_b1hw + large_update_bchw * (1 - attention_b1hw)
        return output_bchw


class BasicSelectiveMultiUpdateBlock(nn.Module):
    """Run one recurrent disparity and upsampling-mask update."""

    def __init__(self, args: ConfigLike, hidden_dim: int = 128, volume_dim: int = 8) -> None:
        """Initialize the update block.

        Args:
            args: Fast-FoundationStereo inference configuration.
            hidden_dim: Recurrent hidden-state channel count.
            volume_dim: Groupwise/geometric correlation channel count.
        """
        super().__init__()
        self.args: ConfigLike = args
        self.encoder: BasicMotionEncoder = BasicMotionEncoder(args, volume_dim)
        self.gru04: SelectiveConvGRU = SelectiveConvGRU(hidden_dim, hidden_dim * 2)
        self.disp_head: DispHead = DispHead(hidden_dim)
        self.mask: nn.Sequential = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        net: TensorPyramid,
        inp: TensorPyramid,
        correlation_bchw: Float[Tensor, "b correlation_channels h w"],
        disparity_b1hw: Float[Tensor, "b 1 h w"],
        attention: TensorPyramid,
    ) -> UpdateBlockOutput:
        """Update the floating-point hidden state and disparity.

        Args:
            net: Recurrent states with shape ``(batch, channels, height, width)``.
            inp: Context inputs with shape ``(batch, channels, height, width)``.
            correlation_bchw: Correlation with shape ``(batch, correlation_channels, height, width)``.
            disparity_b1hw: Current disparity with shape ``(batch, 1, height, width)``.
            attention: Spatial weights with shape ``(batch, channels, height, width)``.

        Returns:
            Updated states, mask features, and disparity update.
        """
        motion_features_bchw: Float[Tensor, "b hidden h w"] = self.encoder(disparity_b1hw, correlation_bchw)
        recurrent_input_bchw: Float[Tensor, "b double_hidden h w"] = torch.cat([inp[0], motion_features_bchw], dim=1)
        net[0] = self.gru04(attention[0], net[0], recurrent_input_bchw)
        delta_disparity_b1hw: Float[Tensor, "b 1 h w"] = self.disp_head(net[0])
        mask_features_bchw: Float[Tensor, "b mask_channels h w"] = 0.25 * self.mask(net[0])
        return net, mask_features_bchw, delta_disparity_b1hw
