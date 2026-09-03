"""X-Lens model (depth-only variant).

Alternating (frame/global) cross-view attention architecture for multi-view
metric-depth estimation.
Pose prediction (camera head) and ray-map prediction (DPT aux branch) are
removed; geometry priors enter through a ray-map pathway instead.

Retains:
1. Variable view count S >= 2.
2. DPT head outputting depth + confidence (+ optional non-ambiguous mask).
3. ScaleHead predicting a metric scaling factor from backbone scale tokens.

Pipeline:
    images (B, S, 3, H, W)
      -> DINOv2 backbone (ViT-B/L) -> multi-scale features + scale tokens
      -> DPT head -> depth (B, S, H, W) + confidence (+ optional mask)
      -> scale head -> metric scaling factor (B,)
"""

import logging
from dataclasses import dataclass
from typing import Literal, TypedDict, cast

import torch
import torch.nn as nn
from jaxtyping import Float, Float32, Int64
from torch import Tensor

from monopriors.third_party.xlens.models.dinov2 import DinoV2
from monopriors.third_party.xlens.models.dpt_head import DPTHead
from monopriors.third_party.xlens.models.ray_map_encoder import RayMapEncoder

logger = logging.getLogger(__name__)

BackboneName = Literal["vits", "vitb", "vitl"]
ScaleHeadMode = Literal["cls", "attn_pool"]
DistortionBiasLayers = Literal["global_only", "all"]
BackboneFeature = tuple[Float[Tensor, "batch views patches features"], Float[Tensor, "batch views features"]]


class XLensNetOutput(TypedDict, total=False):
    """Tensor outputs used by calibrated rig-depth inference."""

    depth: Float32[Tensor, "batch views height width"]
    depth_metric: Float32[Tensor, "batch views height width"]
    depth_conf: Float32[Tensor, "batch views height width"]
    metric_scaling_factor: Float32[Tensor, "batch"]
    mask_logits: Float32[Tensor, "batch views height width"]
    mask: Float32[Tensor, "batch views height width"]


@dataclass(frozen=True, slots=True)
class VisionBackboneConfig:
    """Static dimensions and layer indices for one DINOv2 backbone size."""

    name: BackboneName
    embed_dim: int
    out_layers: tuple[int, int, int, int]
    alt_start: int
    qknorm_start: int
    rope_start: int
    checkpoint: str
    ray_intermediate_dims: tuple[int, ...]


class ScaleHead(nn.Module):
    """Scale head predicting a metric scaling factor.

    Two modes (via `mode`):

    1. "cls": average the last-layer scale_token across views, then an MLP.
    2. "attn_pool": K learnable queries cross-attention-pool the patch tokens
       of each output layer (own query set per layer); the K vectors from all
       layers are concatenated and passed through an MLP.

    metric_depth = normalized_depth * scale_factor
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        mode: ScaleHeadMode = "cls",
        num_queries: int = 4,
        num_layers: int = 4,
        num_heads: int = 8,
    ) -> None:
        """Build either the CLS-token or attention-pooling scale head."""
        super().__init__()
        assert mode in ("cls", "attn_pool"), f"unknown ScaleHead mode: {mode}"
        self.mode = mode
        self.embed_dim = embed_dim

        if mode == "cls":
            self.proj = nn.Linear(input_dim, embed_dim) if input_dim != embed_dim else nn.Identity()
            self.norm = nn.LayerNorm(embed_dim)
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, 1),
            )
        else:  # attn_pool
            self.num_queries = num_queries
            self.num_layers = num_layers

            # KV projection (input_dim is usually 2*embed_dim since cat_token=True).
            self.in_proj = nn.Linear(input_dim, embed_dim) if input_dim != embed_dim else nn.Identity()
            self.kv_norm = nn.LayerNorm(embed_dim)

            # Per-layer learnable queries: (num_layers, num_queries, embed_dim)
            self.queries = nn.Parameter(torch.zeros(num_layers, num_queries, embed_dim))
            nn.init.trunc_normal_(self.queries, std=0.02)

            # Per-layer cross-attention modules.
            self.attn_layers = nn.ModuleList([nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True) for _ in range(num_layers)])
            self.q_norm = nn.LayerNorm(embed_dim)

            # Concatenate all (num_layers x num_queries) vectors, regress a scalar.
            mlp_in = embed_dim * num_queries * num_layers
            self.mlp = nn.Sequential(
                nn.Linear(mlp_in, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, 1),
            )

            # Zero-init the last layer so scale ~ exp(0) = 1 early in training.
            last_layer = cast(nn.Linear, self.mlp[-1])
            nn.init.zeros_(last_layer.weight)
            nn.init.zeros_(last_layer.bias)

    def forward(
        self,
        scale_tokens: Float[Tensor, "batch views features"] | None = None,
        multilayer_feats: list[Float[Tensor, "batch views patches features"]] | None = None,
    ) -> Float32[Tensor, "batch"]:
        """
        Args:
            scale_tokens:     (B, S, C_in), required for "cls" mode
            multilayer_feats: list of (B, S, N_patch, C_in), required for "attn_pool" mode

        Returns:
            scale_factor: (B,) positive scalar
        """
        if self.mode == "cls":
            assert scale_tokens is not None, "cls mode requires scale_tokens"
            x: Float[Tensor, "batch features"] = scale_tokens.mean(dim=1)
            x = self.proj(x)
            x = self.norm(x)
            return torch.exp(self.mlp(x).squeeze(-1))  # (B,)

        # attn_pool
        assert multilayer_feats is not None, "attn_pool mode requires multilayer_feats"
        assert len(multilayer_feats) == self.num_layers, f"expected {self.num_layers} layers of patch tokens, got {len(multilayer_feats)}"

        B = multilayer_feats[0].shape[0]
        pooled_per_layer: list[Float[Tensor, "batch queries features"]] = []
        for li, feat in enumerate(multilayer_feats):
            # feat: (B, S, N_patch, C_in); merge S and N so queries pool across views.
            B_, S, N, _ = feat.shape
            kv: Float[Tensor, "batch tokens features"] = feat.reshape(B_, S * N, -1)
            kv = self.in_proj(kv)
            kv = self.kv_norm(kv)

            q: Float[Tensor, "batch queries features"] = self.queries[li].unsqueeze(0).expand(B_, -1, -1)
            pooled: Float[Tensor, "batch queries features"]
            pooled, _ = self.attn_layers[li](q, kv, kv, need_weights=False)
            pooled_per_layer.append(pooled)

        # (B, num_layers * num_queries, C)
        cat: Float[Tensor, "batch pooled_tokens features"] = torch.cat(pooled_per_layer, dim=1)
        cat = self.q_norm(cat)
        cat = cat.reshape(B, -1)  # (B, num_layers * num_queries * C)
        return torch.exp(self.mlp(cat).squeeze(-1))  # (B,)


VIT_CONFIGS: dict[BackboneName, VisionBackboneConfig] = {
    "vits": VisionBackboneConfig("vits", 384, (2, 5, 8, 11), 4, 4, 4, "dinov2_vits14_reg4_pretrain.pth", (256, 384)),
    "vitb": VisionBackboneConfig("vitb", 768, (2, 5, 8, 11), 4, 4, 4, "dinov2_vitb14_reg4_pretrain.pth", (512, 768)),
    "vitl": VisionBackboneConfig("vitl", 1024, (5, 11, 17, 23), 8, 8, 8, "dinov2_vitl14_reg4_pretrain.pth", (588, 768, 1024)),
}


class XLensNet(nn.Module):
    """X-Lens network (depth-only variant).

    Args:
        backbone_name: "vits" / "vitb" / "vitl"
        checkpoint_dir: DINOv2 pretrained-weights directory
        head_features: DPT head fusion-layer feature dim
        head_out_channels: DPT head per-level output channels
        freeze_backbone: freeze backbone weights
        predict_mask: enable the non-ambiguous mask branch
    """

    PATCH_SIZE = 14

    def __init__(
        self,
        backbone_name: BackboneName = "vitl",
        checkpoint_dir: str = "/path/to/checkpoints",
        head_features: int = 256,
        head_out_channels: tuple[int, int, int, int] = (256, 512, 1024, 1024),
        predict_mask: bool = False,
        scale_head_mode: ScaleHeadMode = "cls",
        scale_head_num_queries: int = 4,
        scale_head_num_heads: int = 8,
        use_dwc: bool = False,  # DWC bypass inside backbone attention
        dwc_kernel_size: int = 3,
        n_cam_types: int = 0,  # heterogeneous camera types (0=off, 2=fisheye+pinhole)
        # Stage 3: calibration tokens / distortion bias
        use_calib_tokens: bool = False,
        calib_tokens_per_type: int = 4,
        calib_inject_types: tuple[int, ...] = (0,),
        use_distortion_bias: bool = False,
        distortion_bias_layers: DistortionBiasLayers = "global_only",
        distortion_bias_hidden_dim: int = 64,
        distortion_bias_chunk_size: int = 1024,
    ) -> None:
        """Build the released X-Lens inference network."""
        super().__init__()

        assert backbone_name in VIT_CONFIGS, f"unsupported backbone: {backbone_name}"
        vit_cfg = VIT_CONFIGS[backbone_name]

        self.backbone_name = backbone_name
        self.embed_dim: int = vit_cfg.embed_dim
        self.n_cam_types = n_cam_types
        # cat_token=True doubles the feature dim (local + global concatenated).
        self.feat_dim = self.embed_dim * 2

        # 1. DINOv2 backbone
        self.backbone = DinoV2(
            name=vit_cfg.name,
            out_layers=list(vit_cfg.out_layers),
            alt_start=vit_cfg.alt_start,
            qknorm_start=vit_cfg.qknorm_start,
            rope_start=vit_cfg.rope_start,
            cat_token=True,
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

        # 2. Ray-map encoder (MapAnything dense_rep_encoder style).
        # Encodes the (B, S, 6, H, W) ray map to (B, S, embed_dim, H/p, W/p) and
        # adds it to patch tokens after patch_embed. The last layer is zero-init
        # so ray_feat=0 early in training and does not disturb the DINOv2 prior.
        self.ray_encoder = RayMapEncoder(
            embed_dim=self.embed_dim,
            patch_size=self.PATCH_SIZE,
            in_chans=6,
            intermediate_dims=vit_cfg.ray_intermediate_dims,
        )

        # 3. DPT head: depth (+ optional non-ambiguous mask).
        self.head = DPTHead(
            dim_in=self.feat_dim,
            depth_output_dim=2,  # depth + confidence (auto +1 when predict_mask=True)
            activation="exp",
            conf_activation="expp1",
            features=head_features,
            out_channels=head_out_channels,
            predict_mask=predict_mask,
        )
        self.predict_mask = predict_mask

        # 4. Scale head
        # "cls": average last-layer scale_token + MLP.
        # "attn_pool": K queries cross-attention-pool 4 layers of patch tokens.
        self.scale_head_mode = scale_head_mode
        self.scale_head = ScaleHead(
            input_dim=self.feat_dim,
            embed_dim=self.embed_dim,
            mode=scale_head_mode,
            num_queries=scale_head_num_queries,
            num_layers=len(vit_cfg.out_layers),
            num_heads=scale_head_num_heads,
        )

    def forward(
        self,
        images: Float[Tensor, "batch views 3 height width"],
        ray_map: Float32[Tensor, "batch views 6 height width"] | None = None,
        d_cam: Float32[Tensor, "batch views 3 height width"] | None = None,
        cam_types: Int64[Tensor, "batch views"] | None = None,
    ) -> XLensNetOutput:
        """
        Args:
            images:    (B, S, 3, H, W), ImageNet-normalized
            ray_map:   optional (B, S, 6, H, W) ray map = (d_world(3), t(3) broadcast),
                       view 0 is the world frame. Omit to skip geometry priors.
            d_cam:     optional (B, S, 3, H, W) per-pixel camera-frame unit rays.
                       Enables FishRoPE (correct fisheye angular positions); otherwise
                       pinhole pixel-index RoPE is used.
            cam_types: optional (B, S) int camera-type id (0=fisheye, 1=pinhole).
                       Enables cam_type_embed + per-cam attention bias when the
                       backbone is configured with n_cam_types>0. None disables it.

        Returns:
            dict:
                - depth: normalized-space depth (B, S, H, W)
                - depth_metric: metric depth = depth * scale (B, S, H, W)
                - depth_conf: depth confidence (B, S, H, W)
                - metric_scaling_factor: (B,)
                - mask_logits / mask: optional (B, S, H, W)
        """
        _, S, _, H, W = images.shape
        assert S >= 2, f"expected at least 2 views, got S={S}"

        ray_feat: Float32[Tensor, "batch views features patch_height patch_width"] | None = None
        if ray_map is not None:
            with torch.autocast(device_type=images.device.type, enabled=False):
                ray_feat = self.ray_encoder(ray_map.float())  # (B, S, embed_dim, H/p, W/p)

        # DINOv2 multi-scale features + scale tokens. d_cam feeds FishRoPE;
        # cam_types feeds cam_type_embed + per-cam attention bias.
        feats: tuple[BackboneFeature, ...]
        scale_tokens: Float[Tensor, "batch views features"] | None
        feats, scale_tokens = self.backbone(
            images,
            ray_feat=ray_feat,
            d_cam=d_cam,
            cam_types=cam_types,
        )

        # Scale head -> metric scaling factor.
        with torch.autocast(device_type=images.device.type, enabled=False):
            if self.scale_head_mode == "cls":
                scale_factor = self.scale_head(scale_tokens=scale_tokens.float())  # (B,)
            else:  # attn_pool
                # Backbone 4-layer patch tokens (B, S, N_patch, 2C).
                multilayer_patch: list[Float32[Tensor, "batch views patches features"]] = [feature[0].float() for feature in feats]
                scale_factor = self.scale_head(multilayer_feats=multilayer_patch)  # (B,)

        # DPT head -> normalized-space depth.
        with torch.autocast(device_type=images.device.type, enabled=False):
            output: XLensNetOutput = cast(XLensNetOutput, self.head(feats, H, W, patch_start_idx=0))

        # Metric-space depth.
        output["metric_scaling_factor"] = scale_factor
        sf: Float32[Tensor, "batch 1 1 1"] = scale_factor[:, None, None, None]
        output["depth_metric"] = output["depth"] * sf

        return output
