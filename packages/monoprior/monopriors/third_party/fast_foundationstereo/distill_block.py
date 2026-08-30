"""Sequential helper modules serialized by the released Fast-FoundationStereo checkpoint."""

from typing import Literal, TypeAlias, cast

import torch
from jaxtyping import Float
from torch import Tensor, nn

from monopriors.third_party.fast_foundationstereo.submodule import FeatureAtt

CostVolume: TypeAlias = Float[Tensor, "b channels disparities h w"]
ImageFeatures: TypeAlias = Float[Tensor, "b channels h w"]
PostLayer: TypeAlias = nn.Module | Literal["sum", "concat"]


class ForwardHelper(nn.Module):
    """Run a serialized sequence whose attention layers also consume image features."""

    def __init__(self, layers: list[nn.Module]) -> None:
        """Initialize the sequence.

        Args:
            layers: Cost-volume layers in execution order.
        """
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList(layers)

    def forward(self, x_bcdhw: CostVolume, left_feat_bchw: ImageFeatures | None = None) -> CostVolume:
        """Transform a cost volume.

        Args:
            x_bcdhw: Floating-point cost volume with shape ``(batch, channels, disparities, height, width)``.
            left_feat_bchw: Optional floating-point image features with shape ``(batch, channels, height, width)``.

        Returns:
            Floating-point cost volume with shape ``(batch, channels, disparities, height, width)``.
        """
        output_bcdhw: CostVolume = x_bcdhw
        for layer in self.layers:
            if isinstance(layer, FeatureAtt):
                if left_feat_bchw is None:
                    raise ValueError("FeatureAtt requires left-image features.")
                output_bcdhw = layer(output_bcdhw, left_feat_bchw)
            else:
                output_bcdhw = layer(output_bcdhw)
        return output_bcdhw


class PostForwardHelper(nn.Module):
    """Upsample and merge two serialized hourglass feature levels."""

    def __init__(self, layers: list[PostLayer]) -> None:
        """Initialize the upsample, merge, and output stages.

        Args:
            layers: Modules containing one ``"sum"`` or ``"concat"`` merge marker.

        Raises:
            ValueError: If no supported merge marker is present.
        """
        super().__init__()
        merge_position: int | None = None
        merge_operation: Literal["sum", "concat"] | None = None
        for position, layer in enumerate(layers):
            if layer in ("sum", "concat"):
                merge_position = position
                merge_operation = layer
                break
        if merge_position is None or merge_operation is None:
            raise ValueError("PostForwardHelper requires a 'sum' or 'concat' merge marker.")
        self.op: Literal["sum", "concat"] = merge_operation
        upsample_layers: list[nn.Module] = [cast(nn.Module, layer) for layer in layers[:merge_position]]
        output_layers: list[nn.Module] = [cast(nn.Module, layer) for layer in layers[merge_position + 1 :]]
        self.upsample: nn.Sequential = nn.Sequential(*upsample_layers)
        self.out: nn.ModuleList = nn.ModuleList(output_layers)

    def forward(
        self,
        conv2_bcdhw: CostVolume,
        conv3_bcdhw: CostVolume,
        left_feat_bchw: ImageFeatures | None = None,
    ) -> CostVolume:
        """Upsample ``conv3``, merge it with ``conv2``, and run the output layers.

        Args:
            conv2_bcdhw: Floating-point skip volume with shape ``(batch, channels, disparities, height, width)``.
            conv3_bcdhw: Floating-point lower-resolution volume with shape ``(batch, channels, disparities, height, width)``.
            left_feat_bchw: Optional floating-point image features with shape ``(batch, channels, height, width)``.

        Returns:
            Floating-point merged cost volume with shape ``(batch, channels, disparities, height, width)``.
        """
        conv3_up_bcdhw: CostVolume = self.upsample(conv3_bcdhw)
        if self.op == "sum":
            output_bcdhw: CostVolume = conv3_up_bcdhw + conv2_bcdhw
        else:
            output_bcdhw = torch.cat((conv3_up_bcdhw, conv2_bcdhw), dim=1)

        for layer in self.out:
            if isinstance(layer, FeatureAtt):
                if left_feat_bchw is None:
                    raise ValueError("FeatureAtt requires left-image features.")
                output_bcdhw = layer(output_bcdhw, left_feat_bchw)
            else:
                output_bcdhw = layer(output_bcdhw)
        return output_bcdhw
