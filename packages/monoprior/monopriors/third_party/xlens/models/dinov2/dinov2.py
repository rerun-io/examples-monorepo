"""
DINOv2 backbone wrapper.

Supports ViT-Base (vitb) and ViT-Large (vitl), alternating cross-view /
within-view attention, and loading pretrained weights from a local checkpoint.
"""

import logging
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange

from monopriors.third_party.xlens.models.dinov2.vision_transformer import DinoVisionTransformer

logger = logging.getLogger(__name__)


def vit_small(patch_size=14, **kwargs):
    """Build a ViT-Small model."""
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        num_register_tokens=0,
        block_chunks=0,
        **kwargs,
    )
    return model


def vit_base(patch_size=14, **kwargs):
    """Build a ViT-Base model."""
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_register_tokens=0,
        block_chunks=0,
        **kwargs,
    )
    return model


def vit_large(patch_size=14, **kwargs):
    """Build a ViT-Large model."""
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=0,
        block_chunks=0,
        **kwargs,
    )
    return model


class DinoV2(nn.Module):
    """
    DINOv2 backbone.

    Supports ViT-Base and ViT-Large. out_layers selects which transformer
    blocks to extract features from, for multi-scale fusion in a downstream
    DPT head.

    Args:
        name: model name, "vitb" or "vitl".
        out_layers: transformer block indices to extract features from.
        alt_start: layer index at which alternating (cross-view / within-view)
            attention begins; -1 disables it.
        qknorm_start: layer index at which QK-Norm begins; -1 disables it.
        rope_start: layer index at which RoPE begins; -1 disables it.
        cat_token: whether to concatenate local and global features.
    """

    def __init__(
        self,
        name: str,
        out_layers: List[int],
        alt_start: int = -1,
        qknorm_start: int = -1,
        rope_start: int = -1,
        cat_token: bool = True,
        use_dwc: bool = False,
        dwc_kernel_size: int = 3,
        n_cam_types: int = 0,
        # Stage 3: Calibration Tokens / Distortion Bias (off by default)
        use_calib_tokens: bool = False,
        calib_tokens_per_type: int = 4,
        calib_inject_types=(0,),
        use_distortion_bias: bool = False,
        distortion_bias_layers: str = "global_only",
        distortion_bias_hidden_dim: int = 64,
        distortion_bias_chunk_size: int = 1024,
        **kwargs,
    ):
        super().__init__()
        assert name in {"vits", "vitb", "vitl"}, f"expected one of vits, vitb, vitl, got {name}"
        self.name = name
        self.out_layers = out_layers
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.cat_token = cat_token
        self.use_dwc = use_dwc
        self.n_cam_types = n_cam_types
        self.use_calib_tokens = use_calib_tokens
        self.use_distortion_bias = use_distortion_bias

        encoder_map = {
            "vits": vit_small,
            "vitb": vit_base,
            "vitl": vit_large,
        }
        encoder_fn = encoder_map[self.name]
        self.pretrained = encoder_fn(
            img_size=518,
            patch_size=14,
            ffn_layer="mlp",
            alt_start=alt_start,
            qknorm_start=qknorm_start,
            rope_start=rope_start,
            cat_token=cat_token,
            use_dwc=use_dwc,
            dwc_kernel_size=dwc_kernel_size,
            n_cam_types=n_cam_types,
            use_calib_tokens=use_calib_tokens,
            calib_tokens_per_type=calib_tokens_per_type,
            calib_inject_types=calib_inject_types,
            use_distortion_bias=use_distortion_bias,
            distortion_bias_layers=distortion_bias_layers,
            distortion_bias_hidden_dim=distortion_bias_hidden_dim,
            distortion_bias_chunk_size=distortion_bias_chunk_size,
        )

    def forward(self, x, **kwargs):
        """
        Args:
            x: input images (B, S, 3, H, W), S views.

        Returns:
            feats: multi-scale feature list.
            aux_feats: auxiliary feature list.
            scale_tokens: (B, S, C) or (B, S, 2C), extracted from the backbone.
        """
        feats, aux_feats, scale_tokens = self.pretrained.get_intermediate_layers(
            x,
            self.out_layers,
            **kwargs,
        )
        return feats, aux_feats, scale_tokens

    def load_pretrained_weights(self, checkpoint_path: str):
        """Load DINOv2 pretrained weights from a local checkpoint."""
        logger.info(f"Loading DINOv2 pretrained weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Unwrap possible nesting.
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Partial load: pretrained weights lack scale_token and other added modules.
        missing_keys, unexpected_keys = self.pretrained.load_state_dict(
            state_dict, strict=False
        )

        if missing_keys:
            logger.info(f"Missing keys (expected): {missing_keys}")
        if unexpected_keys:
            logger.info(f"Unexpected keys (ignored): {unexpected_keys}")

        logger.info("DINOv2 pretrained weights loaded")
