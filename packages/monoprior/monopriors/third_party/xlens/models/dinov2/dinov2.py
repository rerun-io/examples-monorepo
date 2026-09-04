"""
DINOv2 backbone wrapper.

Supports ViT-Base (vitb) and ViT-Large (vitl), alternating cross-view /
within-view attention, and loading pretrained weights from a local checkpoint.
"""

from collections.abc import Sequence
from typing import Literal, TypeAlias

from jaxtyping import Float, Int64
from torch import Tensor, nn

from monopriors.third_party.xlens.models.dinov2.vision_transformer import DinoVisionTransformer

BackboneName = Literal["vits", "vitb", "vitl"]
BackboneFeature: TypeAlias = tuple[Float[Tensor, "batch views patches features"], Float[Tensor, "batch views features"]]
BackboneOutput: TypeAlias = tuple[tuple[BackboneFeature, ...], Float[Tensor, "batch views features"] | None]


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
        name: BackboneName,
        out_layers: Sequence[int],
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
        calib_inject_types: tuple[int, ...] = (0,),
        use_distortion_bias: bool = False,
        distortion_bias_layers: Literal["global_only", "all"] = "global_only",
        distortion_bias_hidden_dim: int = 64,
        distortion_bias_chunk_size: int = 1024,
    ) -> None:
        """Build the selected DINOv2 backbone and X-Lens adapters."""
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

        embed_dim, depth, num_heads = {
            "vits": (384, 12, 6),
            "vitb": (768, 12, 12),
            "vitl": (1024, 24, 16),
        }[name]
        self.pretrained: DinoVisionTransformer = DinoVisionTransformer(
            img_size=518,
            patch_size=14,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            num_register_tokens=0,
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

    def forward(
        self,
        x: Float[Tensor, "batch views 3 height width"],
        *,
        ray_feat: Float[Tensor, "batch views features patch_height patch_width"] | None = None,
        d_cam: Float[Tensor, "batch views 3 height width"] | None = None,
        cam_types: Int64[Tensor, "batch views"] | None = None,
    ) -> BackboneOutput:
        """
        Args:
            x: input images (B, S, 3, H, W), S views.

        Returns:
            feats: multi-scale feature list.
            aux_feats: auxiliary feature list.
            scale_tokens: (B, S, C) or (B, S, 2C), extracted from the backbone.
        """
        feats, scale_tokens = self.pretrained.get_intermediate_layers(
            x,
            self.out_layers,
            ray_feat=ray_feat,
            d_cam=d_cam,
            cam_types=cam_types,
        )
        return feats, scale_tokens
