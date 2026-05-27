# ## Inventory
# - [x] ManoOutput — type alias
# - [x] WiLorOutput — TypedDict
# - [x] WiLor.__init__ — Dims: none; int: focal_length, image_size
# - [x] WiLor.forward
import os
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import roma
import torch
from jaxtyping import Float
from simplecv.ops.mano.mano_torch import ManoSimpleLayer
from torch import Tensor, nn

from wilor_nano.models.refinement_net import RefineNet, RefineNetOutput
from wilor_nano.models.vit import ViTManoFeats, ViTManoParams, vit

ManoOutput = tuple[Float[Tensor, "b n_verts=778 3"], Float[Tensor, "b joints_and_tips=21 3"]]


class WiLorOutput(TypedDict):
    """Final WiLor model output after MANO decoding."""

    global_orient: Float[Tensor, "b 1 3"]
    hand_pose: Float[Tensor, "b 15 3"]
    betas: Float[Tensor, "b 10"]
    pred_cam: Float[Tensor, "b 3"]
    pred_keypoints_3d: Float[Tensor, "b joints_and_tips=21 3"]
    pred_vertices: Float[Tensor, "b n_verts=778 3"]


class WiLor(nn.Module):
    """
    WiLor for Onnx
    """

    def __init__(self, mano_root_dir: Path, **kwargs):
        super().__init__()
        # Create VIT backbone
        self.backbone = vit(**kwargs)
        # Create RefineNet head
        self.refine_net = RefineNet(feat_dim=1280, upscale=3)
        assert os.path.exists(mano_root_dir), f"MANO model {mano_root_dir} not exists!"
        # mano_cfg = {"model_path": mano_model_path, "create_body_pose": False}
        self.mano: ManoSimpleLayer = ManoSimpleLayer(mano_root=mano_root_dir, use_pca=False)
        self.FOCAL_LENGTH = kwargs.get("focal_length", 5000)
        self.IMAGE_SIZE = kwargs.get("image_size", 256)
        self.IMAGE_MEAN = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3))
        self.IMAGE_STD = torch.from_numpy(np.array([0.229, 0.224, 0.225])).reshape(1, 1, 1, 3)

    def forward(self, x: Float[Tensor, "b h=256 w=256 3"]) -> WiLorOutput:
        rgb: Float[Tensor, "b h=256 w=256 3"] = x.flip(dims=[-1]) / 255.0
        image_mean: Float[Tensor, "1 1 1 3"] = self.IMAGE_MEAN.to(rgb.device, dtype=rgb.dtype)
        image_std: Float[Tensor, "1 1 1 3"] = self.IMAGE_STD.to(rgb.device, dtype=rgb.dtype)
        normalized: Float[Tensor, "b h=256 w=256 3"] = (rgb - image_mean) / image_std
        nchw: Float[Tensor, "b 3 h=256 w=256"] = normalized.permute(0, 3, 1, 2)
        batch_size: int = nchw.shape[0]
        # Compute conditioning features using the backbone
        # if using ViT backbone, we need to use a different aspect ratio
        backbone_input: Float[Tensor, "b 3 h=256 crop_w=192"] = nchw[:, :, :, 32:-32]
        temp_mano_params: ViTManoParams
        pred_cam: Float[Tensor, "b 3"]
        pred_mano_feats: ViTManoFeats
        vit_out: Float[Tensor, "b channels height width"]
        temp_mano_params, pred_cam, pred_mano_feats, vit_out = self.backbone(backbone_input)  # B, 1280, 16, 12

        # Compute camera translation
        focal_length: Float[Tensor, "b 2"] = self.FOCAL_LENGTH * torch.ones(
            batch_size, 2, device=nchw.device, dtype=nchw.dtype
        )

        ## Temp MANO
        temp_mano_params["global_orient"] = temp_mano_params["global_orient"].reshape(batch_size, -1, 3, 3)
        temp_mano_params["hand_pose"] = temp_mano_params["hand_pose"].reshape(batch_size, -1, 3, 3)
        temp_mano_params["betas"] = temp_mano_params["betas"].reshape(batch_size, -1)

        # convert from rotation matricies to rotvecs
        temp_rotmat: Float[Tensor, "b n_joints=16 3 3"] = torch.cat(
            (
                temp_mano_params["global_orient"],
                temp_mano_params["hand_pose"],
            ),
            dim=1,
        )
        temp_pose_coeffs: Float[Tensor, "b n_poses=48"] = roma.rotmat_to_rotvec(temp_rotmat).reshape(batch_size, -1)
        temp_betas: Float[Tensor, "b n_betas=10"] = temp_mano_params["betas"]
        temp_trans: Float[Tensor, "b dim=3"] = torch.zeros(batch_size, 3, device=nchw.device, dtype=nchw.dtype)

        temp_mano_output: ManoOutput = self.mano.forward(
            th_pose_coeffs=temp_pose_coeffs, th_betas=temp_betas, th_trans=temp_trans
        )
        temp_vertices: Float[Tensor, "b n_verts=778 3"] = temp_mano_output[0].to(nchw.device, dtype=nchw.dtype) / 1000

        pred_mano_params: RefineNetOutput = self.refine_net(
            vit_out, temp_vertices, pred_cam, pred_mano_feats, focal_length
        )

        final_rotmat: Float[Tensor, "b n_joints=16 3 3"] = torch.cat(
            (
                pred_mano_params["global_orient"],
                pred_mano_params["hand_pose"],
            ),
            dim=1,
        )

        final_pose_coeffs: Float[Tensor, "b n_poses=48"] = roma.rotmat_to_rotvec(final_rotmat).reshape(batch_size, -1)
        final_betas: Float[Tensor, "b n_betas=10"] = pred_mano_params["betas"]
        # pred_cam is weak perspective transform so is in the local frame, the right thing to do is predict with zeros
        # https://chatgpt.com/share/68ae1c26-20e0-8008-ba27-fc4a8e4a4ad1
        final_trans: Float[Tensor, "b dim=3"] = torch.zeros(batch_size, 3, device=nchw.device, dtype=nchw.dtype)

        final_mano_output: ManoOutput = self.mano.forward(
            th_pose_coeffs=final_pose_coeffs, th_betas=final_betas, th_trans=final_trans
        )
        pred_keypoints_3d: Float[Tensor, "b joints_and_tips=21 3"] = final_mano_output[1] / 1000
        pred_vertices: Float[Tensor, "b n_verts=778 3"] = final_mano_output[0] / 1000

        pred_mano_output: WiLorOutput = cast(WiLorOutput, pred_mano_params)
        pred_mano_output["pred_keypoints_3d"] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        pred_mano_output["pred_vertices"] = pred_vertices.reshape(batch_size, -1, 3)
        pred_mano_output["global_orient"] = roma.rotmat_to_rotvec(pred_mano_params["global_orient"])
        pred_mano_output["hand_pose"] = roma.rotmat_to_rotvec(pred_mano_params["hand_pose"])
        return cast(WiLorOutput, pred_mano_output)
