"""MammaNet: dense 512-landmark 2D network as a plain ``nn.Module``.

The original ``DenseLdmks2DViT`` is a LightningModule whose inference surface
is ``backbone(x) + mask_downscaling(masks) -> decoder(features, pos_embed)``;
this module mirrors exactly those attribute names so the converted
state dict (see ``tools/convert_ckpt.py``) loads strictly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float32
from numpy import ndarray
from torch import nn

from mamma.landmarks.config import DEFAULT_MAMMANET_CONFIG, MammaNetConfig
from mamma.landmarks.decoder import MammaNetDecoder
from mamma.landmarks.mask_embed import MaskEmbedding
from mamma.landmarks.vit import ViT

DEFAULT_MEAN: Float32[ndarray, "3"] = (255.0 * np.array([0.485, 0.456, 0.406])).astype(np.float32)
"""Per-channel RGB mean (0..255 scale) used to normalize crops."""
DEFAULT_STD: Float32[ndarray, "3"] = (255.0 * np.array([0.229, 0.224, 0.225])).astype(np.float32)
"""Per-channel RGB std (0..255 scale) used to normalize crops."""
MAMMANET_WEIGHTS_REPO: str = "pablovela5620/mamma-streaming-data"
MAMMANET_WEIGHTS_FILE: str = "weights/ma_2d/mamma_mask_full_cvpr.safetensors"


class MammaNet(nn.Module):
    """ViT backbone + mask embedding + DETR landmark decoder."""

    def __init__(self, config: MammaNetConfig = DEFAULT_MAMMANET_CONFIG) -> None:
        super().__init__()
        self.config = config
        self.backbone = ViT(
            img_size=(config.crop_width, config.crop_height),
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
        )
        self.decoder = MammaNetDecoder(
            d_model=config.embed_dim,
            n_heads=config.decoder_heads,
            n_layers=config.decoder_layers,
            n_landmarks=config.num_landmarks,
            transformer_dim_feedforward=config.decoder_ffn_dim,
            dropout=config.dropout,
            uncertainty=config.uncertainty,
            visibility=config.visibility,
            contact=config.contact,
            floor_contact=config.floor_contact,
        )
        self.mask_downscaling = MaskEmbedding(config.embed_dim, config.mask_patch_size)

    def forward(
        self,
        x: Float32[torch.Tensor, "b 3 ch cw"],
        masks: Float32[torch.Tensor, "b 1 ch cw"] | None,
    ) -> dict[str, torch.Tensor | None]:
        """Normalized crops (+0..1 mask crops) -> landmark/visibility/contact heads.

        Returns ``joints2d [b, 512, 3]`` (xy in [-1,1] crop coords + log-variance),
        and ``visibility``/``contact``/``floor_contact`` logits ``[b, 512, 1]``.
        """
        features = self.backbone(x)
        if self.config.use_mask and masks is not None:
            features = features + self.mask_downscaling(masks)
        return self.decoder(features, self.backbone.pos_embed)


def resolve_mammanet_weights_path(weights_path: Path | None = None) -> Path:
    """Resolve an explicit checkpoint or download the published MammaNet weights."""
    if weights_path is None:
        from huggingface_hub import hf_hub_download

        weights_path = Path(hf_hub_download(repo_id=MAMMANET_WEIGHTS_REPO, repo_type="dataset", filename=MAMMANET_WEIGHTS_FILE))
    resolved_path: Path = weights_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"MammaNet weights not found: {resolved_path}")
    return resolved_path


def load_mammanet(
    weights_path: Path | None = None,
    device: str = "cuda",
    config: MammaNetConfig = DEFAULT_MAMMANET_CONFIG,
) -> MammaNet:
    """Load MammaNet from the converted safetensors state dict.

    Args:
        weights_path: Local checkpoint path. ``None`` downloads the published
            MammaNet weights from Hugging Face.
        device: Device receiving the loaded model.
        config: MammaNet architecture configuration.

    Returns:
        The strictly loaded evaluation model.
    """
    resolved_path: Path = resolve_mammanet_weights_path(weights_path)
    from safetensors.torch import load_file

    model: MammaNet = MammaNet(config)
    state_dict: dict[str, torch.Tensor] = load_file(str(resolved_path))
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()
