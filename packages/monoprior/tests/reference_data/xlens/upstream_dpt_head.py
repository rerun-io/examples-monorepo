"""DPT depth-prediction head.

Features:
- Shared reassembly + depth fusion chain.
- Depth output (B, S, H, W) + confidence, optional non-ambiguous mask logits
  (shares the final 1x1 conv of the depth fusion chain).
"""

from typing import Dict as TyDict
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from .utils.head_utils import create_uv_grid, custom_interpolate, position_grid_to_embed


class DPTHead(nn.Module):
    """DPT head predicting depth (+ optional non-ambiguous mask).

    Args:
        dim_in: Input feature dimension.
        depth_output_dim: Depth output channels (2 = depth + confidence; +1 when predict_mask=True).
        features: Feature dimension of the fusion layers.
        out_channels: Output channels for each projection stage.
        activation: Depth activation function.
        conf_activation: Confidence activation function.
        predict_mask: Output non-ambiguous mask logits.
    """

    PATCH_SIZE = 14

    def __init__(
        self,
        dim_in: int,
        depth_output_dim: int = 2,
        activation: str = "exp",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = True,
        down_ratio: int = 1,
        predict_mask: bool = False,
    ) -> None:
        super().__init__()

        self.patch_size = self.PATCH_SIZE
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        # Depth channel layout: [0]=depth, [1]=depth_conf, [2]=mask_logits (optional)
        self.predict_mask = predict_mask
        self.depth_out_dim = depth_output_dim + (1 if predict_mask else 0)

        self.intermediate_layer_idx: Tuple[int, int, int, int] = (0, 1, 2, 3)

        # Shared layers: per-stage projection + scale alignment
        self.norm = nn.Identity()
        self.projects = nn.ModuleList(
            [nn.Conv2d(dim_in, oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
                nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )

        self.scratch = _make_scratch(list(out_channels), features, expand=False)

        # Depth fusion chain
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_features_1 = features
        head_features_2 = 32
        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
        )
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, self.depth_out_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
        chunk_size: int = 8,
        **kwargs,
    ) -> dict:
        B, S, N, C = feats[0][0].shape
        feats = [feat[0].reshape(B * S, N, C) for feat in feats]

        if chunk_size is None or chunk_size >= S:
            out_dict = self._forward_impl(feats, H, W, patch_start_idx)
            out_dict = {k: v.view(B, S, *v.shape[1:]) for k, v in out_dict.items()}
            return out_dict

        out_dicts: List[TyDict[str, torch.Tensor]] = []
        for s0 in range(0, B * S, chunk_size):
            s1 = min(s0 + chunk_size, B * S)
            out_dicts.append(
                self._forward_impl([f[s0:s1] for f in feats], H, W, patch_start_idx)
            )
        out_dict = {k: torch.cat([od[k] for od in out_dicts], dim=0) for k in out_dicts[0].keys()}
        out_dict = {k: v.view(B, S, *v.shape[1:]) for k, v in out_dict.items()}
        return out_dict

    def _forward_impl(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
    ) -> TyDict[str, torch.Tensor]:
        B, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size

        # 1) Per-stage feature projection and scale alignment
        resized_feats = []
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats[take_idx][:, patch_start_idx:]
            x = self.norm(x)
            x = x.permute(0, 2, 1).contiguous().reshape(B, C, ph, pw)
            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = self._add_pos_embed(x, W, H)
            x = self.resize_layers[stage_idx](x)
            resized_feats.append(x)

        # 2) scratch adaptation
        l1_rn = self.scratch.layer1_rn(resized_feats[0])
        l2_rn = self.scratch.layer2_rn(resized_feats[1])
        l3_rn = self.scratch.layer3_rn(resized_feats[2])
        l4_rn = self.scratch.layer4_rn(resized_feats[3])

        # 3) Depth branch: top-down fusion
        depth_out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        depth_out = self.scratch.refinenet3(depth_out, l3_rn, size=l2_rn.shape[2:])
        depth_out = self.scratch.refinenet2(depth_out, l2_rn, size=l1_rn.shape[2:])
        depth_out = self.scratch.refinenet1(depth_out, l1_rn)

        # 4) Upsample to target resolution
        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)

        depth_out = self.scratch.output_conv1(depth_out)
        depth_out = custom_interpolate(depth_out, (h_out, w_out), mode="bilinear", align_corners=True)
        if self.pos_embed:
            depth_out = self._add_pos_embed(depth_out, W, H)
        depth_logits = self.scratch.output_conv2(depth_out)

        # 5) Outputs
        outs: TyDict[str, torch.Tensor] = {}

        depth_fmap = depth_logits.permute(0, 2, 3, 1)  # (B, H, W, depth_out_dim)
        outs["depth"] = torch.nn.functional.softplus(depth_fmap[..., 0])  # (B, H, W)
        outs["depth_conf"] = self._apply_activation(depth_fmap[..., 1:2], self.conf_activation).squeeze(-1)

        if self.predict_mask:
            mask_logits = depth_fmap[..., 2]
            outs["mask_logits"] = mask_logits
            outs["mask"] = torch.sigmoid(mask_logits)

        return outs

    def _apply_activation(self, x: torch.Tensor, activation: str) -> torch.Tensor:
        act = activation.lower()
        if act == "exp":
            return torch.exp(x.clamp(max=10.0))
        if act == "expp1":
            return torch.exp(x.clamp(max=10.0)) + 1
        if act == "relu":
            return torch.relu(x)
        if act == "sigmoid":
            return torch.sigmoid(x)
        if act == "softplus":
            return torch.nn.functional.softplus(x)
        return x

    def _add_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        pw, ph = x.shape[-1], x.shape[-2]
        pe = create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pe = position_grid_to_embed(pe, x.shape[1]) * ratio
        pe = pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pe


# Builder helpers

def _make_fusion_block(
    features: int,
    size: Tuple[int, int] = None,
    has_residual: bool = True,
    groups: int = 1,
) -> nn.Module:
    return FeatureFusionBlock(
        features=features,
        activation=nn.ReLU(inplace=False),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


def _make_scratch(
    in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False
) -> nn.Module:
    scratch = nn.Module()
    c1 = out_shape
    c2 = out_shape * (2 if expand else 1)
    c3 = out_shape * (4 if expand else 1)
    c4 = out_shape * (8 if expand else 1)

    scratch.layer1_rn = nn.Conv2d(in_shape[0], c1, 3, 1, 1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], c2, 3, 1, 1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], c3, 3, 1, 1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], c4, 3, 1, 1, bias=False, groups=groups)
    return scratch


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int, activation: nn.Module, bn: bool, groups: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(
        self,
        features: int,
        activation: nn.Module,
        deconv: bool = False,
        bn: bool = False,
        expand: bool = False,
        align_corners: bool = True,
        size: Tuple[int, int] = None,
        has_residual: bool = True,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.align_corners = align_corners
        self.size = size
        self.has_residual = has_residual

        self.resConfUnit1 = (
            ResidualConvUnit(features, activation, bn, groups=groups) if has_residual else None
        )
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=groups)

        out_features = (features // 2) if expand else features
        self.out_conv = nn.Conv2d(features, out_features, 1, 1, 0, bias=True, groups=groups)

    def forward(self, *xs: torch.Tensor, size: Tuple[int, int] = None) -> torch.Tensor:
        y = xs[0]
        if self.has_residual and len(xs) > 1 and self.resConfUnit1 is not None:
            y = y + self.resConfUnit1(xs[1])

        y = self.resConfUnit2(y)

        if (size is None) and (self.size is None):
            up_kwargs = {"scale_factor": 2}
        elif size is None:
            up_kwargs = {"size": self.size}
        else:
            up_kwargs = {"size": size}

        y = custom_interpolate(y, **up_kwargs, mode="bilinear", align_corners=self.align_corners)
        y = self.out_conv(y)
        return y
