# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
from torch import nn


# DINOv3 ViT-H+/16 as built by the facebookresearch/dinov3 hub factory ``dinov3_vith16plus``, which
# upstream loaded with torch.hub from unpinned GitHub main. The same architecture is vendored in
# posekit's dinov3-src package (DINOv3 5aa7e93, shared with EgoHand); build it from there instead.
_ARCH_KWARGS = {
    "dinov3_vith16plus": dict(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=6.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    ),
}


class Dinov3Backbone(nn.Module):
    def __init__(
        self, name="dinov3_vith16plus", pretrained_weight=None, cfg=None, *args, **kwargs
    ):
        super().__init__()
        from dinov3.models.vision_transformer import DinoVisionTransformer

        self.name = name
        self.cfg = cfg

        # The hub factory ignores the config's DROP_PATH_RATE (it hard-codes 0.0); inference never drops paths.
        self.encoder = DinoVisionTransformer(**_ARCH_KWARGS[name])
        self.encoder.init_weights()
        self.patch_size = self.encoder.patch_size
        self.embed_dim = self.embed_dims = self.encoder.embed_dim

    def forward(self, x, extra_embed=None):
        """
        Encode a RGB image using a ViT-backbone
        Args:
            - x: torch.Tensor of shape [bs,3,w,h]
        Return:
            - y: torch.Tensor of shape [bs,k,d] - image in patchified mode
        """
        assert extra_embed is None, "Not Implemented Yet"

        y = self.encoder.get_intermediate_layers(x, n=1, reshape=True, norm=True)[-1]

        return y

    def get_layer_depth(self, param_name: str, prefix: str = "encoder."):
        """Get the layer-wise depth of a parameter.
        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.
        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        Note:
            The first depth is the stem module (``layer_depth=0``), and the
            last depth is the subsequent module (``layer_depth=num_layers-1``)
        """
        num_layers = self.encoder.n_blocks + 2

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return num_layers - 1, num_layers

        param_name = param_name[len(prefix) :]

        if param_name in ("cls_token", "pos_embed", "storage_tokens") or param_name.startswith("patch_embed"):
            layer_depth = 0
        elif param_name.startswith("blocks"):
            layer_id = int(param_name.split(".")[1])
            layer_depth = layer_id + 1
        else:
            layer_depth = num_layers - 1

        return layer_depth, num_layers
