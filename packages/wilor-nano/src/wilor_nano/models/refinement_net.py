# ## Inventory
# - [x] RefineNetOutput — TypedDict
# - [x] RefineInputManoFeats — TypedDict
# - [x] make_conv_layers — Dims: feat_dims elements, kernel, stride, padding; int: none
# - [x] make_deconv_layers — Dims: feat_dims elements; int: none
# - [x] sample_joint_features
# - [x] perspective_projection
# - [x] DeConvNet.__init__ — Dims: feat_dim; int: upscale
# - [x] DeConvNet.forward
# - [x] RefineNet.__init__ — Dims: feat_dim; int: upscale
# - [x] RefineNet.forward
import math
from collections.abc import Sequence
from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from .vit import rot6d_to_rotmat


class RefineNetOutput(TypedDict):
    """
    TypedDict for the output of RefineNet.

    This defines the structure and types of the dictionary returned by the RefineNet forward method.
    """

    global_orient: Float[Tensor, "batch 1 3 3"]
    hand_pose: Float[Tensor, "batch 15 3 3"]
    betas: Float[Tensor, "batch 10"]
    pred_cam: Float[Tensor, "batch 3"]


class RefineInputManoFeats(TypedDict):
    """Intermediate MANO feature predictions consumed by the refinement head."""

    hand_pose: Float[Tensor, "batch n_pose=96"]
    betas: Float[Tensor, "batch 10"]
    # Vulture does not connect string-keyed TypedDict access to class-style fields.
    cam: Float[Tensor, "batch 3"]  # noqa


def make_conv_layers(
    feat_dims: Sequence[int],
    kernel: int | tuple[int, int] = 3,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 1,
    bnrelu_final: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(feat_dims) - 1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i + 1],
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            )
        )
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def make_deconv_layers(feat_dims: Sequence[int], bnrelu_final: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(feat_dims) - 1):
        layers.append(
            nn.ConvTranspose2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i + 1],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False,
            )
        )

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def sample_joint_features(
    img_feat: Float[Tensor, "batch channels height width"],
    joint_xy: Float[Tensor, "batch joints 2"],
) -> Float[Tensor, "batch joints channels"]:
    height: int = img_feat.shape[2]
    width: int = img_feat.shape[3]
    x_norm: Float[Tensor, "batch joints"] = joint_xy[:, :, 0] / (width - 1) * 2 - 1
    y_norm: Float[Tensor, "batch joints"] = joint_xy[:, :, 1] / (height - 1) * 2 - 1
    grid: Float[Tensor, "batch joints sample_width=1 2"] = torch.stack((x_norm, y_norm), 2)[:, :, None, :]
    sampled: Float[Tensor, "batch channels joints sample_width=1"] = F.grid_sample(
        img_feat, grid, align_corners=True
    )
    sampled_squeezed: Float[Tensor, "batch channels joints"] = sampled[:, :, :, 0]
    joint_features: Float[Tensor, "batch joints channels"] = sampled_squeezed.permute(0, 2, 1).contiguous()
    return joint_features


def perspective_projection(
    points: Float[Tensor, "batch points 3"],
    translation: Float[Tensor, "batch 3"],
    focal_length: Float[Tensor, "batch 2"],
    camera_center: Float[Tensor, "batch 2"] | None = None,
    rotation: Float[Tensor, "batch 3 3"] | None = None,
) -> Float[Tensor, "batch points 2"]:
    """
    Computes the perspective projection of a set of 3D points.
    Args:
        points (Tensor): Tensor of shape (B, N, 3) containing the input 3D points.
        translation (Tensor): Tensor of shape (B, 3) containing the 3D camera translation.
        focal_length (Tensor): Tensor of shape (B, 2) containing the focal length in pixels.
        camera_center (Tensor): Tensor of shape (B, 2) containing the camera center in pixels.
        rotation (Tensor): Tensor of shape (B, 3, 3) containing the camera rotation.
    Returns:
        Tensor: Tensor of shape (B, N, 2) containing the projection of the input points.
    """
    batch_size: int = points.shape[0]
    if rotation is None:
        rotation = torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0).expand(batch_size, -1, -1)
    if camera_center is None:
        camera_center = torch.zeros(batch_size, 2, device=points.device, dtype=points.dtype)
    # Populate intrinsic camera matrix K.
    K: Float[Tensor, "batch 3 3"] = torch.zeros((batch_size, 3, 3), device=points.device, dtype=points.dtype)
    K[:, 0, 0] = focal_length[:, 0]
    K[:, 1, 1] = focal_length[:, 1]
    K[:, 2, 2] = 1.0
    K[:, :-1, -1] = camera_center
    # Transform points
    rotated_points: Float[Tensor, "batch points 3"] = torch.einsum("bij,bkj->bki", rotation, points)
    translated_points: Float[Tensor, "batch points 3"] = rotated_points + translation.unsqueeze(1)

    # Apply perspective distortion
    normalized_points: Float[Tensor, "batch points 3"] = translated_points / translated_points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points: Float[Tensor, "batch points 3"] = torch.einsum("bij,bkj->bki", K, normalized_points)

    return projected_points[:, :, :-1]


class DeConvNet(nn.Module):
    def __init__(self, feat_dim: int = 768, upscale: int = 4) -> None:
        super().__init__()
        self.first_conv = make_conv_layers([feat_dim, feat_dim // 2], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.deconv = nn.ModuleList([])
        for i in range(int(math.log2(upscale)) + 1):
            if i == 0:
                self.deconv.append(make_deconv_layers([feat_dim // 2, feat_dim // 4]))
            elif i == 1:
                self.deconv.append(make_deconv_layers([feat_dim // 2, feat_dim // 4, feat_dim // 8]))
            elif i == 2:
                self.deconv.append(make_deconv_layers([feat_dim // 2, feat_dim // 4, feat_dim // 8, feat_dim // 8]))

    def forward(self, img_feat: Float[Tensor, "batch channels height width"]) -> list[Float[Tensor, "batch channels height width"]]:
        face_img_feats: list[Float[Tensor, "batch channels height width"]] = []
        first_feat: Float[Tensor, "batch channels height width"] = self.first_conv(img_feat)
        face_img_feats.append(first_feat)
        for deconv in self.deconv:
            img_feat_i: Float[Tensor, "batch channels height width"] = deconv(first_feat)
            face_img_feat: Float[Tensor, "batch channels height width"] = img_feat_i
            face_img_feats.append(face_img_feat)
        return face_img_feats[::-1]  # high resolution -> low resolution


class RefineNet(nn.Module):
    def __init__(self, feat_dim: int = 1280, upscale: int = 3) -> None:
        super().__init__()

        self.deconv = DeConvNet(feat_dim=feat_dim, upscale=upscale)
        self.out_dim = feat_dim // 8 + feat_dim // 4 + feat_dim // 2
        self.dec_pose = nn.Linear(self.out_dim, 96)
        self.dec_cam = nn.Linear(self.out_dim, 3)
        self.dec_shape = nn.Linear(self.out_dim, 10)

        self.joint_rep_type = "6d"
        self.joint_rep_dim = {"6d": 6, "aa": 3}[self.joint_rep_type]

    def forward(
        self,
        img_feat: Float[Tensor, "batch channels height width"],
        verts_3d: Float[Tensor, "batch n_verts 3"],
        pred_cam: Float[Tensor, "batch 3"],
        pred_mano_feats: RefineInputManoFeats,
        focal_length: Float[Tensor, "batch 2"],
    ) -> RefineNetOutput:
        batch_size: int = img_feat.shape[0]

        img_feats: list[Float[Tensor, "batch channels height width"]] = self.deconv(img_feat)

        img_feat_sizes: list[int] = [img_feat_i.shape[2] for img_feat_i in img_feats]

        temp_cams: list[Float[Tensor, "batch 3"]] = [
            torch.stack(
                (pred_cam[:, 1], pred_cam[:, 2], 2 * focal_length[:, 0] / (img_feat_size * pred_cam[:, 0] + 1e-9)),
                dim=-1,
            )
            for img_feat_size in img_feat_sizes
        ]

        verts_2d: list[Float[Tensor, "batch n_verts 2"]] = [
            perspective_projection(verts_3d, translation=temp_cams[i], focal_length=focal_length / img_feat_sizes[i])
            for i in range(len(img_feat_sizes))
        ]

        vert_feats_by_scale: list[Float[Tensor, "batch channels"]] = [
            sample_joint_features(img_feats[i], verts_2d[i]).max(1).values for i in range(len(img_feat_sizes))
        ]

        vert_feats: Float[Tensor, "batch channels"] = torch.cat(vert_feats_by_scale, dim=-1)

        delta_pose: Float[Tensor, "batch n_pose=96"] = self.dec_pose(vert_feats)
        delta_betas: Float[Tensor, "batch 10"] = self.dec_shape(vert_feats)
        delta_cam: Float[Tensor, "batch 3"] = self.dec_cam(vert_feats)

        pred_hand_pose_6d: Float[Tensor, "batch n_pose=96"] = pred_mano_feats["hand_pose"] + delta_pose
        pred_betas: Float[Tensor, "batch 10"] = pred_mano_feats["betas"] + delta_betas
        refined_cam: Float[Tensor, "batch 3"] = pred_mano_feats["cam"] + delta_cam

        pred_hand_pose: Float[Tensor, "batch joints_and_root=16 3 3"] = rot6d_to_rotmat(pred_hand_pose_6d).view(
            batch_size, -1, 3, 3
        )

        pred_mano_params: RefineNetOutput = {
            "global_orient": pred_hand_pose[:, :1],
            "hand_pose": pred_hand_pose[:, 1:],
            "betas": pred_betas,
            "pred_cam": refined_cam,
        }

        return pred_mano_params
