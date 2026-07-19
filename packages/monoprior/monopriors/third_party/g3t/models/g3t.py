import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from monopriors.third_party.g3t.models.aggregator import Aggregator
from monopriors.third_party.g3t.heads.camera_head import CameraHead
from monopriors.third_party.g3t.heads.dpt_head import DPTHead


class G3T(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self, img_size=518, patch_size=14, embed_dim=1024,
        enable_point=True, enable_depth=True, enable_gravity_camera_heads=True,
    ):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        if enable_gravity_camera_heads:
            self.local_camera_head = CameraHead(dim_in=2 * embed_dim, pose_encoding_type="noT_quaR_FoV")
            self.global_camera_head = CameraHead(dim_in=2 * embed_dim, pose_encoding_type="absT_quaRy_noFoV")
        else:
            self.local_camera_head = None
            self.global_camera_head = None

        self.point_head = DPTHead(
            dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1"
        ) if enable_point else None

        self.depth_head = DPTHead(
            dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1"
        ) if enable_depth else None

    def forward(self, images: torch.Tensor):
        """
        Forward pass of the G3T model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            dict: A dictionary containing the following predictions:
                - local_pose_enc (torch.Tensor): Gravity-to-camera pose encoding from the last iteration
                - global_pose_enc (torch.Tensor): Relative gravity-frame pose encoding from the last iteration
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points/world_points_conf when the optional point head is enabled
        """
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        predictions = {}

        with torch.amp.autocast('cuda', enabled=False):

            if self.local_camera_head is not None and self.global_camera_head is not None:
                local_pose_enc_list = self.local_camera_head(aggregated_tokens_list)
                global_pose_enc_list = self.global_camera_head(aggregated_tokens_list)

                predictions["local_pose_enc"] = local_pose_enc_list[-1]
                predictions["global_pose_enc"] = global_pose_enc_list[-1]

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        return predictions
