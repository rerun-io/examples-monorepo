from collections.abc import Mapping

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from monopriors.third_party.g3t.models.aggregator import Aggregator
from monopriors.third_party.g3t.heads.camera_head import CameraHead
from monopriors.third_party.g3t.heads.dpt_head import DPTHead


class G3T(nn.Module, PyTorchModelHubMixin):
    """Inference-only G3T model used by Monopriors."""

    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.local_camera_head = CameraHead(dim_in=2 * embed_dim, pose_encoding_type="noT_quaR_FoV")
        self.global_camera_head = CameraHead(dim_in=2 * embed_dim, pose_encoding_type="absT_quaRy_noFoV")
        self.depth_head = DPTHead(
            dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1"
        )

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ) -> nn.modules.module._IncompatibleKeys:
        """Load every inference weight while explicitly ignoring the removed point head."""
        incompatible = super().load_state_dict(state_dict, strict=False, assign=assign)
        invalid_unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("point_head.")]
        if incompatible.missing_keys or invalid_unexpected:
            raise RuntimeError(
                "G3T checkpoint does not match the inference model: "
                f"missing={incompatible.missing_keys}, unexpected={invalid_unexpected}"
            )
        return incompatible

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
        """
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        predictions: dict[str, torch.Tensor] = {}

        with torch.amp.autocast('cuda', enabled=False):
            local_pose_enc_list = self.local_camera_head(aggregated_tokens_list)
            global_pose_enc_list = self.global_camera_head(aggregated_tokens_list)
            predictions["local_pose_enc"] = local_pose_enc_list[-1]
            predictions["global_pose_enc"] = global_pose_enc_list[-1]

        depth, depth_conf = self.depth_head(
            aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
        )
        predictions["depth"] = depth.float()
        predictions["depth_conf"] = depth_conf.float()

        return predictions
