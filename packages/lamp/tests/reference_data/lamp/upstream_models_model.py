# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Neural network modules for the LAMP SMPL lifter."""

# (PyTorch ships imprecise stubs that mark `torch.zeros`, `torch.cat`,
# `nn.ModuleList.__getitem__`, etc. as private re-exports or untyped
# — suppressing project-wide for this file. Matches the pattern in
# `lamp.models.lifter`, `lamp.detection.detector`, `lamp.tracking.tracker`, etc.)

from __future__ import annotations

import logging
from pathlib import Path

import smplx
import torch
import torch.nn as nn
from lamp.models.blocks import Block, MVRayFusion, SMPLHeads
from lamp.models.model_utils import (
    get_cam_ray,
    get_T_x_c,
    GRAVITY_DIRECTION_VIO,
    inverse_se3,
    R_CG_CGZ,
    se3_transform_points,
    smpl_forward_joints_lamp_outputs,
    transform_smpl_params,
)
from torch.nn import functional as F

logger: logging.Logger = logging.getLogger(__name__)


# smplx capture-safety patch

_SMPLX_PATCHED: bool = False


def _ensure_smplx_capture_safe(parents: torch.Tensor) -> None:
    """Patch `smplx` rigid transforms to avoid CUDA Graph host syncs."""
    global _SMPLX_PATCHED
    if _SMPLX_PATCHED:
        return
    import smplx.lbs  # local import keeps the cost off `from lamp.models.model import ...`

    parents_py: list[int] = parents.tolist()
    _orig_transform_mat = smplx.lbs.transform_mat

    def _patched_batch_rigid_transform(
        rot_mats: torch.Tensor,
        joints: torch.Tensor,
        parents: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joints = torch.unsqueeze(joints, dim=-1)
        rel_joints = joints.clone()
        rel_joints[:, 1:] -= joints[:, parents[1:]]
        transforms_mat = _orig_transform_mat(
            rot_mats.reshape(-1, 3, 3),
            rel_joints.reshape(-1, 3, 1),
        ).reshape(-1, joints.shape[1], 4, 4)
        transform_chain = [transforms_mat[:, 0]]
        for i in range(1, len(parents_py)):
            transform_chain.append(
                torch.matmul(transform_chain[parents_py[i]], transforms_mat[:, i])
            )
        transforms = torch.stack(transform_chain, dim=1)
        posed_joints = transforms[:, :, :3, 3]
        joints_homogen = F.pad(joints, [0, 0, 0, 1])
        rel_transforms = transforms - F.pad(
            torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0]
        )
        return posed_joints, rel_transforms

    smplx.lbs.batch_rigid_transform = _patched_batch_rigid_transform
    _SMPLX_PATCHED = True
    logger.debug(
        "Installed capture-safe smplx.lbs.batch_rigid_transform (parents=%s)",
        parents_py,
    )


__all__ = ["LampNet"]


class LampNet(nn.Module):
    """Dual Ray Temporal Transformer — the SMPL LAMP lifter model."""

    # Declared here so pyright sees them as typed attributes.
    dim_feat: int

    _gravity_w: torch.Tensor
    _r_cg_cgz: torch.Tensor

    def __init__(
        self,
        dim_in: int = 7,  # Plücker ray coords (6) + score (1)
        dim_feat: int = 256,
        depth: int = 3,
        num_heads: int = 8,
        mlp_ratio: float = 4,
        num_joints: int = 17,
        maxlen: int = 20,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        smpl_model_path: str | Path | None = None,
    ) -> None:
        super().__init__()

        if smpl_model_path is None:
            raise ValueError(
                "smpl_model_path is required. "
                "Download the SMPL neutral .pkl from https://smpl.is.tue.mpg.de "
                "and pass its path."
            )

        self.dim_feat = dim_feat

        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.floor_embed = nn.Linear(2, dim_feat)
        self.floor_joint_gate = nn.Parameter(torch.ones(num_joints))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.mv_fusion = MVRayFusion(embed_dim=dim_feat)

        self.blocks_spatial_first = nn.ModuleList(
            [
                Block(
                    dim=dim_feat,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                    path_order="spatial_first",
                )
                for _ in range(depth)
            ]
        )
        self.blocks_temporal_first = nn.ModuleList(
            [
                Block(
                    dim=dim_feat,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                    path_order="temporal_first",
                )
                for _ in range(depth)
            ]
        )

        self.temp_embed = nn.Parameter(torch.zeros(1, maxlen, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        nn.init.trunc_normal_(self.temp_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dec_proj = nn.Linear(dim_feat * num_joints, dim_feat)
        # Separate time embedding for the decoder so we can do temporal
        # upsampling independently of the encoder's per-frame embedding.
        self.dec_temp_embed = nn.Parameter(torch.zeros(1, maxlen, dim_feat))
        nn.init.trunc_normal_(self.dec_temp_embed, std=0.02)
        self.upsample_decoders = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=dim_feat,
                    nhead=num_heads,
                    dim_feedforward=int(mlp_ratio * dim_feat),
                    dropout=drop_rate,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        # (B, T, C) learnable readout query for the decoder cross-attention.
        self.readout_embedding = nn.Parameter(torch.zeros(1, 1, dim_feat))
        nn.init.trunc_normal_(self.readout_embedding, std=0.02)

        path_fusion_layers = [nn.Linear(dim_feat * 2, 2) for _ in range(depth)]
        for layer in path_fusion_layers:
            layer.weight.data.fill_(0)
            layer.bias.data.fill_(0.5)
        self.path_fusion = nn.ModuleList(path_fusion_layers)

        # Must run BEFORE we attach SMPL submodules, since the SMPL model
        # comes with its own pretrained weights we don't want to clobber.
        self.apply(self._init_weights)

        self.smpl_heads: SMPLHeads = SMPLHeads(dim_feat=self.dim_feat)
        self.smpl: smplx.SMPL = smplx.SMPL(
            model_path=str(smpl_model_path), gender="neutral"
        )

        _ensure_smplx_capture_safe(self.smpl.parents)

        # Register constants as buffers so dtype/device moves happen with the
        # rest of the model and never inside the captured forward path.
        self.register_buffer(
            "_r_cg_cgz",
            torch.tensor(list(R_CG_CGZ), dtype=torch.float32),  # (1, 3, 3)
        )
        self.register_buffer(
            "_gravity_w",
            torch.tensor(list(GRAVITY_DIRECTION_VIO), dtype=torch.float32),  # (3,)
        )

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Truncated-normal init for Linear, ones/zeros init for LayerNorm.

        The SMPL submodule is attached after this initializer runs so its
        pretrained weights stay intact.
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)

            if m.bias is not None:  # type: ignore[reportUnnecessaryComparison]
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_share(
        self,
        x_list: list[torch.Tensor],
        cam_params: list[torch.Tensor],
        Ts_wc: list[torch.Tensor],
        ground_planes: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """The shared transformer + decoder + SMPL-head pipeline."""
        anchor_cam = 0
        device = x_list[anchor_cam].device

        if ground_planes is None:
            ground_planes = torch.full(
                (x_list[anchor_cam].shape[0], 4, 3),
                float("nan"),
                device=device,
                dtype=Ts_wc[anchor_cam].dtype,
            )
        else:
            assert ground_planes.device == device, (
                f"ground_planes must be on {device}, got {ground_planes.device}"
            )
        _, T_w_x = get_T_x_c(
            Ts_wc[anchor_cam],
            gravity_w=self._gravity_w,
            R_cg_cgz=self._r_cg_cgz,
        )

        # Lift each view's 2D keypoints to Plücker rays in the local frame.
        rays: list[torch.Tensor] = []
        T_x_w = inverse_se3(T_w_x)
        for num_cam in range(len(x_list)):
            T_x_c = T_x_w @ Ts_wc[num_cam]
            rays.append(get_cam_ray(cam_params[num_cam], x_list[num_cam], T_x_c=T_x_c))

        x: torch.Tensor = torch.stack(rays, dim=1)
        # v: views, f: frames, j: joints, c: channels (lowercase to avoid
        # pyright `reportConstantRedefinition` — the joints_embed below
        # changes C from the input ray dim to dim_feat, so we rebind).
        b, v, f, j, c = x.shape
        x = x.reshape(-1, j, c)
        x = self.joints_embed(x)
        x = x + self.pos_embed
        _, j, c = x.shape
        x = x.reshape(-1, f, j, c) + self.temp_embed[:, :f, :, :]
        x = x.reshape(b, v, f, j, c)

        ground_planes_x = se3_transform_points(T_x_w.squeeze(1), ground_planes)
        ground_planes_x_z = ground_planes_x[:, 0, -1:]
        ground_planes_x_z_safe = torch.where(
            torch.isnan(ground_planes_x_z),
            torch.zeros_like(ground_planes_x_z),
            ground_planes_x_z,
        )
        floor_known = (
            (~torch.isnan(ground_planes)).any(dim=-1).any(dim=-1).float().unsqueeze(-1)
        )
        floor_input = torch.cat([ground_planes_x_z_safe, floor_known], dim=-1)
        floor_embedding = self.floor_embed(floor_input)
        x = x + (
            floor_embedding[:, None, None, None, :]
            * self.floor_joint_gate[None, None, None, :, None]
        )

        # Avoid an einops dependency by writing the permute+reshape inline.
        # The math: `(b, v, f, j, c) -> (b, f, j, v, c) -> (b*f*j, v, c)`.
        x = x.permute(0, 2, 3, 1, 4).reshape(b * f * j, v, c)
        x = self.mv_fusion(x)  # (b*f*j, c)
        # `(b*f*j, c) -> (b, f, j, c) -> (b*f, j, c)`.
        x = x.reshape(b, f, j, c).reshape(b * f, j, c)
        x = self.pos_drop(x)

        # Injecting time information during cross-attention is crucial
        # for decoding pose changes over time.
        readout_embedding = self.readout_embedding.expand(b, f, -1)
        # Slice the time embedding to the actual snippet length so shorter
        # clips work cleanly.
        readout_embedding = readout_embedding + self.dec_temp_embed[:, :f, :]

        # Fixed `depth` iterations — capture-safe.
        for idx, (blk_spatial_first, blk_temporal_first) in enumerate(
            zip(self.blocks_spatial_first, self.blocks_temporal_first, strict=True)
        ):
            x_spatial_first = blk_spatial_first(x, f)
            x_temporal_first = blk_temporal_first(x, f)
            fusion_layer = self.path_fusion[idx]
            alpha = torch.cat([x_spatial_first, x_temporal_first], dim=-1)
            alpha = fusion_layer(alpha)
            alpha = alpha.softmax(dim=-1)
            x = x_spatial_first * alpha[:, :, 0:1] + x_temporal_first * alpha[:, :, 1:2]

            # `(b*f, j, c) -> (b, f, j*c)`.
            memory = x.reshape(b, f, j * c)
            memory = self.dec_proj(memory)
            readout_embedding = self.upsample_decoders[idx](
                tgt=readout_embedding, memory=memory
            )

        smpl_outs = self.smpl_heads(readout_embedding)
        return smpl_outs, T_w_x

    def forward(
        self,
        x: list[torch.Tensor],
        cam_params: list[torch.Tensor],
        Ts_wc: list[torch.Tensor],
        ground_planes: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Lift per-view 2D keypoints into world-frame SMPL joints and parameters."""
        assert len(x) == len(cam_params) == len(Ts_wc)
        ret: dict[str, torch.Tensor] = {}
        smpl_outs, T_w_x = self.forward_share(x, cam_params, Ts_wc, ground_planes)

        ret["T_w_x"] = T_w_x
        ret.update(smpl_outs)

        B = ret["betas"].shape[0]
        device = ret["betas"].device
        dtype = ret["betas"].dtype

        smpl_out = self.smpl.forward(
            global_orient=torch.zeros((B, 3), device=device, dtype=dtype),  # type: ignore[arg-type]
            body_pose=torch.zeros(  # type: ignore[arg-type]
                B, self.smpl.NUM_BODY_JOINTS * 3, device=device, dtype=dtype
            ),
            betas=ret["betas"].detach(),  # type: ignore[arg-type]
            return_verts=False,
        )
        smpl_t_pose_pelvis = smpl_out.joints[:, 0]  # type: ignore[index]

        root_R_W, transl_W = transform_smpl_params(
            ret["global_orient_rotmat"].squeeze(-3),
            ret["transl"],
            T_w_x[..., :3, :3],
            T_w_x[..., :3, -1],
            smpl_t_pose_pelvis,
        )
        root_R_W = root_R_W.unsqueeze(-3)
        ret["global_orient_rotmat"] = root_R_W
        ret["transl"] = transl_W

        smpl_joints, _ = smpl_forward_joints_lamp_outputs(
            self.smpl,
            betas=ret["betas"],
            body_pose_rotmat=ret["body_pose_rotmat"],
            global_orient_rotmat=root_R_W,
            transl=transl_W,
            return_verts=False,
        )
        ret["skel_w"] = smpl_joints
        return ret
