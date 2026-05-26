# Copyright (c) OpenMMLab. All rights reserved.
import os
from collections.abc import Callable
from functools import partial
from typing import Any, TypedDict, cast

import numpy as np
import torch
import torch.linalg as torch_linalg
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from torch import Tensor


class ViTManoParams(TypedDict):
    """Initial MANO parameters predicted by the ViT backbone."""

    global_orient: Float[Tensor, "batch 1 3 3"]
    hand_pose: Float[Tensor, "batch 15 3 3"]
    betas: Float[Tensor, "batch 10"]


class ViTManoFeats(TypedDict):
    """Intermediate MANO predictions used by RefineNet."""

    hand_pose: Float[Tensor, "batch n_pose=96"]
    betas: Float[Tensor, "batch 10"]
    # Vulture does not connect string-keyed TypedDict access to class-style fields.
    cam: Float[Tensor, "batch 3"]  # noqa


ViTOutput = tuple[
    ViTManoParams,
    Float[Tensor, "batch 3"],
    ViTManoFeats,
    Float[Tensor, "batch channels height width"],
]


def rot6d_to_rotmat(x: Float[Tensor, "*batch n_rot6"]) -> Float[Tensor, "n_rot 3 3"]:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Args:
        x (torch.Tensor): (B,6) Batch of 6-D rotation representations.
    Returns:
        torch.Tensor: Batch of corresponding rotation matrices with shape (B,3,3).
    """
    x = x.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch_linalg.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def vit(**kwargs: Any) -> "ViT":
    return ViT(
        img_size=(256, 192),
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.55,
        **kwargs,
    )


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Float[Tensor, "batch tokens channels"]) -> Float[Tensor, "batch tokens channels_out"]:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

        self.attn_drop = attn_drop
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Float[Tensor, "batch tokens channels"]) -> Float[Tensor, "batch tokens channels"]:
        B, N, _C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 使用 scaled_dot_product_attention
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_drop)

        x = attn.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        attn_head_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_head_dim=attn_head_dim,
        )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: Float[Tensor, "batch tokens channels"]) -> Float[Tensor, "batch tokens channels"]:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        ratio: int = 1,
    ) -> None:
        super().__init__()
        img_size = cast(tuple[int, int], tuple(int(value) for value in to_2tuple(img_size)))
        patch_size = cast(tuple[int, int], tuple(int(value) for value in to_2tuple(patch_size)))
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio**2)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=(patch_size[0] // ratio),
            padding=4 + 2 * (ratio // 2 - 1),
        )

    def forward(
        self,
        x: Float[Tensor, "batch channels height width"],
        **kwargs: Any,
    ) -> tuple[Float[Tensor, "batch n_patches embed_dim"], tuple[int, int]]:
        _B, _C, _H, _W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


class HybridEmbed(nn.Module):
    """CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(
        self,
        backbone: nn.Module,
        img_size: int | tuple[int, int] = 224,
        feature_size: tuple[int, int] | None = None,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = cast(tuple[int, int], tuple(int(value) for value in to_2tuple(img_size)))
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = cast(tuple[int, int], tuple(int(value) for value in to_2tuple(feature_size)))
            feature_info = getattr(self.backbone, "feature_info")  # noqa: B009
            feature_dim = feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x: Float[Tensor, "batch channels height width"]) -> Float[Tensor, "batch n_patches embed_dim"]:
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        num_classes: int = 80,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        hybrid_backbone: nn.Module | None = None,
        norm_layer: Callable[[int], nn.Module] | None = None,
        use_checkpoint: bool = False,
        frozen_stages: int = -1,
        ratio: int = 1,
        last_norm: bool = True,
        patch_padding: str = "pad",
        freeze_attn: bool = False,
        freeze_ffn: bool = False,
        **kwargs,
    ) -> None:
        # Protect mutable default arguments
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.depth = depth

        if hybrid_backbone is not None:
            self.patch_embed: HybridEmbed | PatchEmbed
            self.patch_embed = HybridEmbed(hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, ratio=ratio
            )
        num_patches = self.patch_embed.num_patches

        ##########################################
        self.joint_rep_type = "6d"
        self.joint_rep_dim = {"6d": 6, "aa": 3}[self.joint_rep_type]
        self.NUM_HAND_JOINTS = 15
        npose = self.joint_rep_dim * (self.NUM_HAND_JOINTS + 1)
        self.npose = npose
        mano_mean_path = kwargs.get("mano_mean_path")
        assert mano_mean_path and os.path.exists(mano_mean_path), f"{mano_mean_path} not exists!"
        mean_params = np.load(mano_mean_path)
        init_cam: Float[Tensor, "1 3"] = torch.from_numpy(mean_params["cam"].astype(np.float32)).unsqueeze(0)
        self.register_buffer("init_cam", init_cam)
        init_hand_pose: Float[Tensor, "1 n_pose=96"] = torch.from_numpy(mean_params["pose"].astype(np.float32)).unsqueeze(0)
        init_betas: Float[Tensor, "1 10"] = torch.from_numpy(mean_params["shape"].astype("float32")).unsqueeze(0)
        self.register_buffer("init_hand_pose", init_hand_pose)
        self.register_buffer("init_betas", init_betas)
        self.init_cam: Float[Tensor, "1 3"]
        self.init_hand_pose: Float[Tensor, "1 n_pose=96"]
        self.init_betas: Float[Tensor, "1 10"]

        self.pose_emb = nn.Linear(self.joint_rep_dim, embed_dim)
        self.shape_emb = nn.Linear(10, embed_dim)
        self.cam_emb = nn.Linear(3, embed_dim)

        self.decpose = nn.Linear(self.num_features, 6)
        self.decshape = nn.Linear(self.num_features, 10)
        self.deccam = nn.Linear(self.num_features, 3)

        # since the pretraining model has class token
        pos_embed: Float[Tensor, "1 n_patches_plus_cls embed_dim"] = torch.zeros(1, num_patches + 1, embed_dim)
        self.pos_embed = nn.Parameter(pos_embed)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

    def forward_features(self, x: Float[Tensor, "batch channels height width"]) -> ViTOutput:
        B, _C, _H, _W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)

        if self.pos_embed is not None:
            # fit for multiple GPU training
            # since the first element for pos embed (sin-cos manner) is zero, it will cause no difference
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        # X [B, 192, 1280]
        # x cat [ mean_pose, mean_shape, mean_cam] tokens
        pose_tokens = self.pose_emb(
            self.init_hand_pose.reshape(1, self.NUM_HAND_JOINTS + 1, self.joint_rep_dim)
        ).repeat(B, 1, 1)
        shape_tokens = self.shape_emb(self.init_betas).unsqueeze(1).repeat(B, 1, 1)
        cam_tokens = self.cam_emb(self.init_cam).unsqueeze(1).repeat(B, 1, 1)

        x = torch.cat([pose_tokens, shape_tokens, cam_tokens, x], 1)
        for blk in self.blocks:
            x = blk(x)

        x = self.last_norm(x)

        pose_feat = x[:, : (self.NUM_HAND_JOINTS + 1)]
        shape_feat = x[:, (self.NUM_HAND_JOINTS + 1) : 1 + (self.NUM_HAND_JOINTS + 1)]
        cam_feat = x[:, 1 + (self.NUM_HAND_JOINTS + 1) : 2 + (self.NUM_HAND_JOINTS + 1)]

        # print(pose_feat.shape, shape_feat.shape, cam_feat.shape)
        pred_hand_pose = self.decpose(pose_feat).reshape(B, -1) + self.init_hand_pose  # B , 96
        pred_betas = self.decshape(shape_feat).reshape(B, -1) + self.init_betas  # B , 10
        pred_cam = self.deccam(cam_feat).reshape(B, -1) + self.init_cam  # B , 3

        pred_mano_feats: ViTManoFeats = {
            "hand_pose": pred_hand_pose,
            "betas": pred_betas,
            "cam": pred_cam,
        }

        pred_hand_pose = rot6d_to_rotmat(pred_hand_pose).view(B, self.NUM_HAND_JOINTS + 1, 3, 3)
        pred_mano_params: ViTManoParams = {
            "global_orient": pred_hand_pose[:, [0]],
            "hand_pose": pred_hand_pose[:, 1:],
            "betas": pred_betas,
        }

        img_feat = x[:, 2 + (self.NUM_HAND_JOINTS + 1) :].reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2)
        return pred_mano_params, pred_cam, pred_mano_feats, img_feat

    def forward(self, x: Float[Tensor, "batch channels height width"]) -> ViTOutput:
        output = self.forward_features(x)
        return output
