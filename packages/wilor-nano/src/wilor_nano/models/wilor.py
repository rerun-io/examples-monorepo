import os
from pathlib import Path

import numpy as np
import roma
import torch
from jaxtyping import Float
from torch import Tensor, nn

from wilor_nano.mano_pytorch_simple import ManoSimpleLayer
from wilor_nano.models.refinement_net import RefineNet, RefineNetOutput
from wilor_nano.models.vit import vit


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

    def forward(self, x: Float[Tensor, "b h=256 w=256 3"]) -> RefineNetOutput:
        x = x.flip(dims=[-1]) / 255.0
        x = (x - self.IMAGE_MEAN.to(x.device, dtype=x.dtype)) / self.IMAGE_STD.to(x.device, dtype=x.dtype)
        x = x.permute(0, 3, 1, 2)
        batch_size = x.shape[0]
        # Compute conditioning features using the backbone
        # if using ViT backbone, we need to use a different aspect ratio
        temp_mano_params, pred_cam, pred_mano_feats, vit_out = self.backbone(x[:, :, :, 32:-32])  # B, 1280, 16, 12

        # Compute camera translation
        focal_length = self.FOCAL_LENGTH * torch.ones(batch_size, 2, device=x.device, dtype=x.dtype)

        ## Temp MANO
        temp_mano_params["global_orient"] = temp_mano_params["global_orient"].reshape(batch_size, -1, 3, 3)
        temp_mano_params["hand_pose"] = temp_mano_params["hand_pose"].reshape(batch_size, -1, 3, 3)
        temp_mano_params["betas"] = temp_mano_params["betas"].reshape(batch_size, -1)

        # convert from rotation matricies to rotvecs
        temp_rotmat: Float[Tensor, "b n_joints=16 3 3"] = torch.concat(
            [
                temp_mano_params["global_orient"],
                temp_mano_params["hand_pose"],
            ],
            dim=1,
        )
        temp_pose_coeffs: Float[Tensor, "b n_poses=48"] = roma.rotmat_to_rotvec(temp_rotmat).reshape(batch_size, -1)
        temp_betas: Float[Tensor, "b n_betas=10"] = temp_mano_params["betas"]
        temp_trans: Float[Tensor, "b dim=3"] = torch.zeros(batch_size, 3, device=x.device, dtype=x.dtype)

        temp_mano_output: tuple[Float[Tensor, "b n_verts=778 3"], Float[Tensor, "b joints_and_tips=21 3"]] = (
            self.mano.forward(th_pose_coeffs=temp_pose_coeffs, th_betas=temp_betas, th_trans=temp_trans)
        )
        temp_vertices = temp_mano_output[0].to(x.device, dtype=x.dtype) / 1000

        pred_mano_params: RefineNetOutput = self.refine_net(
            vit_out, temp_vertices, pred_cam, pred_mano_feats, focal_length
        )

        final_rotmat: Float[Tensor, "b n_joints=16 3 3"] = torch.concat(
            [
                pred_mano_params["global_orient"],
                pred_mano_params["hand_pose"],
            ],
            dim=1,
        )

        final_pose_coeffs: Float[Tensor, "b n_poses=48"] = roma.rotmat_to_rotvec(final_rotmat).reshape(batch_size, -1)
        final_betas: Float[Tensor, "b n_betas=10"] = pred_mano_params["betas"]
        # pred_cam is weak perspective transform so is in the local frame, the right thing to do is predict with zeros
        # https://chatgpt.com/share/68ae1c26-20e0-8008-ba27-fc4a8e4a4ad1
        final_trans: Float[Tensor, "b dim=3"] = torch.zeros(batch_size, 3, device=x.device, dtype=x.dtype)

        final_mano_output: tuple[Float[Tensor, "b n_verts=778 3"], Float[Tensor, "b joints_and_tips=21 3"]] = (
            self.mano.forward(th_pose_coeffs=final_pose_coeffs, th_betas=final_betas, th_trans=final_trans)
        )
        pred_keypoints_3d = final_mano_output[1] / 1000
        pred_vertices = final_mano_output[0] / 1000

        pred_mano_params["pred_keypoints_3d"] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        pred_mano_params["pred_vertices"] = pred_vertices.reshape(batch_size, -1, 3)
        pred_mano_params["global_orient"] = roma.rotmat_to_rotvec(pred_mano_params["global_orient"])
        pred_mano_params["hand_pose"] = roma.rotmat_to_rotvec(pred_mano_params["hand_pose"])
        return pred_mano_params
