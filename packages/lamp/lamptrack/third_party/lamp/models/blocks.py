# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Reusable neural network blocks for the LAMP lifter."""

from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from torch.nn import functional as F

from lamptrack.third_party.lamp.models.model_utils import rotation_6d_to_matrix

__all__ = [
    "MLP",
    "Attention",
    "Block",
    "MVRayFusion",
    "SMPLHeads",
]


class SMPLHeadOutput(TypedDict):
    """Raw SMPL parameters regressed by :class:`SMPLHeads`."""

    betas: Float[Tensor, "batch 10"]
    transl: Float[Tensor, "batch time 3"]
    body_pose_rotmat: Float[Tensor, "batch time 23 3 3"]
    global_orient_rotmat: Float[Tensor, "batch time 1 3 3"]


class MLP(nn.Module):
    """Two-linear feed-forward block (Linear -> act -> drop -> Linear -> drop)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Float[Tensor, "... in_features"]) -> Float[Tensor, "... out_features"]:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention with separate spatial / temporal modes."""

    mode: str

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        st_mode: str = "spatial",
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = st_mode

    def forward(self, x: Float[Tensor, "batch tokens channels"], seqlen: int = 1) -> Float[Tensor, "batch tokens channels"]:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # Index instead of tuple-unpack so TorchScript keeps the graph simple.
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.mode == "spatial":
            x = self.forward_spatial(q, k, v)
        elif self.mode == "temporal":
            x = self.forward_temporal(q, k, v, seqlen=seqlen)
        else:
            raise NotImplementedError(self.mode)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_spatial(
        self,
        q: Float[Tensor, "batch heads tokens head_dim"],
        k: Float[Tensor, "batch heads tokens head_dim"],
        v: Float[Tensor, "batch heads tokens head_dim"],
    ) -> Float[Tensor, "batch tokens channels"]:
        B, _, N, C = q.shape
        dropout_p = self.attn_drop.p if self.training else 0.0
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, scale=self.scale
        )
        x = x.transpose(1, 2).reshape(B, N, C * self.num_heads)
        return x

    def forward_temporal(
        self,
        q: Float[Tensor, "batch heads tokens head_dim"],
        k: Float[Tensor, "batch heads tokens head_dim"],
        v: Float[Tensor, "batch heads tokens head_dim"],
        seqlen: int = 8,
    ) -> Float[Tensor, "batch tokens channels"]:
        B, _, N, C = q.shape
        # Reinterpret `(B, H, N, C)` as `(B', T, H, N, C)` then rearrange to
        # `(B', H, N, T, C)` so SDPA attends across the T axis.
        qt = q.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)
        kt = k.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)
        vt = v.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)
        dropout_p = self.attn_drop.p if self.training else 0.0
        x = F.scaled_dot_product_attention(
            qt, kt, vt, dropout_p=dropout_p, scale=self.scale
        )
        x = x.permute(0, 3, 2, 1, 4).reshape(B, N, C * self.num_heads)
        return x


class Block(nn.Module):
    """Pre-norm block bundling a spatial half and a temporal half."""

    path_order: str

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        path_order: str = "spatial_first",
    ) -> None:
        super().__init__()
        self.path_order = path_order
        self.pre_attn_norm_spatial = norm_layer(dim)
        self.spatial_attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            st_mode="spatial",
        )
        self.pre_attn_norm_temporal = norm_layer(dim)
        self.temporal_attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            st_mode="temporal",
        )
        self.pre_mlp_norm_spatial = norm_layer(dim)
        self.pre_mlp_norm_temporal = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.spatial_mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.temporal_mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=drop,
        )

    def _spatial_step(self, x: Float[Tensor, "batch tokens channels"], seqlen: int) -> Float[Tensor, "batch tokens channels"]:
        x = x + self.spatial_attn(self.pre_attn_norm_spatial(x), seqlen)
        x = x + self.spatial_mlp(self.pre_mlp_norm_spatial(x))
        return x

    def _temporal_step(self, x: Float[Tensor, "batch tokens channels"], seqlen: int) -> Float[Tensor, "batch tokens channels"]:
        x = x + self.temporal_attn(self.pre_attn_norm_temporal(x), seqlen)
        x = x + self.temporal_mlp(self.pre_mlp_norm_temporal(x))
        return x

    def forward(self, x: Float[Tensor, "batch tokens channels"], seqlen: int = 1) -> Float[Tensor, "batch tokens channels"]:
        # path_order is a string config attribute — resolves at trace time.
        if self.path_order == "spatial_first":
            return self._temporal_step(self._spatial_step(x, seqlen), seqlen)
        if self.path_order == "temporal_first":
            return self._spatial_step(self._temporal_step(x, seqlen), seqlen)
        raise NotImplementedError(self.path_order)


class MVRayFusion(nn.Module):
    """Fuse `V` per-view embeddings into a single embedding per `(B*F*J)`."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 1.0,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.fusion_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.fusion_embedding, std=0.02)
        self.fusion_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(mlp_ratio * embed_dim),
                    dropout=drop_rate,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Float[Tensor, "samples views channels"]) -> Float[Tensor, "samples channels"]:
        # nn.TransformerEncoderLayer can't handle B > 65535 (PyTorch limit).
        max_batch_size = 65535
        fusion_embedding = self.fusion_embedding.expand(x.shape[0], -1, -1)
        x = torch.cat([fusion_embedding, x], dim=-2)
        B = x.shape[0]
        for blk in self.fusion_layers:
            outputs: list[Float[Tensor, "subbatch views channels"]] = []
            start_idx = 0
            while start_idx < B:
                end_idx = min(start_idx + max_batch_size, B)
                x_sub = x[start_idx:end_idx]  # (subB, N, C)
                out_sub = blk(x_sub)
                outputs.append(out_sub)
                start_idx = end_idx
            x = torch.cat(outputs, dim=0)
        # The fusion token is at index 0 — that's our readout.
        return x[:, 0, :]


class SMPLHeads(nn.Module):
    """SMPL prediction heads on top of the temporally-pooled readout."""

    num_smpl_betas: int
    rot_dim: int
    num_smpl_poses: int
    init_body_pose: Tensor
    init_betas: Tensor

    def __init__(
        self,
        dim_feat: int,
        init_body_pose: Float[Tensor, "1 1 pose_dim"] | None = None,
        init_betas: Float[Tensor, "1 10"] | None = None,
    ) -> None:
        super().__init__()
        self.num_smpl_betas = 10
        self.rot_dim = 6
        self.num_smpl_poses = 24

        self.dec_betas = nn.Linear(dim_feat, self.num_smpl_betas)
        self.dec_transl = nn.Linear(dim_feat, 3)
        self.dec_poses = nn.Linear(dim_feat, self.rot_dim * self.num_smpl_poses)

        # Tiny init so initial predictions stay close to the mean priors.
        nn.init.xavier_uniform_(self.dec_betas.weight, gain=0.01)
        nn.init.xavier_uniform_(self.dec_transl.weight, gain=0.01)
        nn.init.xavier_uniform_(self.dec_poses.weight, gain=0.01)

        if init_body_pose is None:
            init_body_pose = torch.zeros(1, 1, self.rot_dim * self.num_smpl_poses)
        if init_betas is None:
            init_betas = torch.zeros(1, self.num_smpl_betas)
        # Buffer names are part of the checkpoint format.
        self.register_buffer("init_body_pose", init_body_pose)
        self.register_buffer("init_betas", init_betas)

    def forward(self, readout_embedding: Float[Tensor, "batch time channels"]) -> SMPLHeadOutput:
        # Pool over the temporal axis for the per-clip shape head.
        x_mean_F = readout_embedding.mean(dim=1)

        betas = self.dec_betas(x_mean_F) + self.init_betas
        transl = self.dec_transl(readout_embedding)
        body_poses = self.dec_poses(readout_embedding)
        B, T, _ = body_poses.shape
        global_orient_6d = body_poses[..., : self.rot_dim].reshape(B, T, 1, 6)

        local_poses_6d = (
            body_poses[..., self.rot_dim :] + self.init_body_pose[..., self.rot_dim :]
        )
        local_poses_6d = local_poses_6d.reshape(B, T, -1, 6)
        local_poses_rotmat = rotation_6d_to_matrix(local_poses_6d)
        global_orient_rotmat = rotation_6d_to_matrix(global_orient_6d)
        return {
            "betas": betas,  # (B, 10)
            "transl": transl,  # (B, T, 3)
            "body_pose_rotmat": local_poses_rotmat,  # (B, T, 23, 3, 3)
            "global_orient_rotmat": global_orient_rotmat,  # (B, T, 1, 3, 3)
        }


# Dual Ray Temporal Transformer — `LampNet`
