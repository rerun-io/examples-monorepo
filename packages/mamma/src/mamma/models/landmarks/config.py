"""Resolved MammaNet inference configuration.

Values are the fully-resolved ``config_mammanet_mask_512.yaml`` +
``model/pose_mammanet.yaml`` from the original repo (hydra interpolations
evaluated by hand) — no omegaconf/hydra at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MammaNetConfig:
    """MammaNet (DenseLdmks2DViT) architecture + preprocessing constants."""

    crop_width: int = 384
    """Model input crop width in pixels (``image_size[0]``)."""
    crop_height: int = 512
    """Model input crop height in pixels (``image_size[1]``)."""
    num_landmarks: int = 512
    """Dense 2D landmarks predicted per person."""
    # ViT backbone (ViTPose-B variant).
    patch_size: int = 16
    """Backbone patch size."""
    embed_dim: int = 768
    """Backbone embedding dim."""
    depth: int = 12
    """Backbone transformer blocks."""
    num_heads: int = 12
    """Backbone attention heads."""
    mlp_ratio: float = 4.0
    """Backbone MLP expansion."""
    qkv_bias: bool = True
    """Backbone qkv bias."""
    # DETR-style decoder.
    decoder_heads: int = 8
    """Decoder attention heads."""
    decoder_layers: int = 6
    """Decoder layers."""
    decoder_ffn_dim: int = 2048
    """Decoder feedforward dim."""
    dropout: float = 0.1
    """Decoder dropout (inactive in eval)."""
    # Heads: ldmks_dim=2 + uncertainty => 3 outputs; visibility/contact/floor heads on.
    uncertainty: bool = True
    """Third joints2d channel is a log-variance head."""
    visibility: bool = True
    """Per-landmark visibility logit head."""
    contact: bool = True
    """Per-landmark self-contact logit head."""
    floor_contact: bool = True
    """Per-landmark floor-contact logit head."""
    use_mask: bool = True
    """Add SAM-style mask embedding to backbone features."""
    normalize_plus_min_one: bool = True
    """joints2d xy are in [-1, 1] crop coords; un-normalize with (v+1)/2."""
    bedlam_scale: float = 1.2
    """Box expansion factor before aspect-ratio fitting."""

    @property
    def mask_patch_size(self) -> int:
        """MaskEmbedding conv patch size (``backbone patch_size // 4``)."""
        return self.patch_size // 4


DEFAULT_MAMMANET_CONFIG: MammaNetConfig = MammaNetConfig()
"""Shared default config instance (frozen; safe as a module singleton)."""
