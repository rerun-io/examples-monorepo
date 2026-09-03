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
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .dinov2 import DinoV2
from .dpt_head import DPTHead
from .ray_map_encoder import RayMapEncoder

logger = logging.getLogger(__name__)


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
        mode: str = "cls",
        num_queries: int = 4,
        num_layers: int = 4,
        num_heads: int = 8,
    ):
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
            self.in_proj = (
                nn.Linear(input_dim, embed_dim) if input_dim != embed_dim else nn.Identity()
            )
            self.kv_norm = nn.LayerNorm(embed_dim)

            # Per-layer learnable queries: (num_layers, num_queries, embed_dim)
            self.queries = nn.Parameter(torch.zeros(num_layers, num_queries, embed_dim))
            nn.init.trunc_normal_(self.queries, std=0.02)

            # Per-layer cross-attention modules.
            self.attn_layers = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        embed_dim, num_heads=num_heads, batch_first=True
                    )
                    for _ in range(num_layers)
                ]
            )
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
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        scale_tokens: Optional[torch.Tensor] = None,
        multilayer_feats: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            scale_tokens:     (B, S, C_in), required for "cls" mode
            multilayer_feats: list of (B, S, N_patch, C_in), required for "attn_pool" mode

        Returns:
            scale_factor: (B,) positive scalar
        """
        if self.mode == "cls":
            assert scale_tokens is not None, "cls mode requires scale_tokens"
            x = scale_tokens.mean(dim=1)  # (B, C_in)
            x = self.proj(x)
            x = self.norm(x)
            return torch.exp(self.mlp(x).squeeze(-1))  # (B,)

        # attn_pool
        assert multilayer_feats is not None, "attn_pool mode requires multilayer_feats"
        assert len(multilayer_feats) == self.num_layers, (
            f"expected {self.num_layers} layers of patch tokens, got {len(multilayer_feats)}"
        )

        B = multilayer_feats[0].shape[0]
        pooled_per_layer = []
        for li, feat in enumerate(multilayer_feats):
            # feat: (B, S, N_patch, C_in); merge S and N so queries pool across views.
            B_, S, N, _ = feat.shape
            kv = feat.reshape(B_, S * N, -1)
            kv = self.in_proj(kv)
            kv = self.kv_norm(kv)

            q = self.queries[li].unsqueeze(0).expand(B_, -1, -1)  # (B, K, C)
            pooled, _ = self.attn_layers[li](q, kv, kv, need_weights=False)  # (B, K, C)
            pooled_per_layer.append(pooled)

        # (B, num_layers * num_queries, C)
        cat = torch.cat(pooled_per_layer, dim=1)
        cat = self.q_norm(cat)
        cat = cat.reshape(B, -1)  # (B, num_layers * num_queries * C)
        return torch.exp(self.mlp(cat).squeeze(-1))  # (B,)


VIT_CONFIGS = {
    "vits": {
        "name": "vits",
        "embed_dim": 384,
        "out_layers": [2, 5, 8, 11],
        "alt_start": 4,
        "qknorm_start": 4,
        "rope_start": 4,
        "depth": 12,
        "checkpoint": "dinov2_vits14_reg4_pretrain.pth",
        "ray_intermediate_dims": (256, 384),
    },
    "vitb": {
        "name": "vitb",
        "embed_dim": 768,
        "out_layers": [2, 5, 8, 11],
        "alt_start": 4,
        "qknorm_start": 4,
        "rope_start": 4,
        "depth": 12,
        "checkpoint": "dinov2_vitb14_reg4_pretrain.pth",
        "ray_intermediate_dims": (512, 768),
    },
    "vitl": {
        "name": "vitl",
        "embed_dim": 1024,
        "out_layers": [5, 11, 17, 23],
        "alt_start": 8,
        "qknorm_start": 8,
        "rope_start": 8,
        "depth": 24,
        "checkpoint": "dinov2_vitl14_reg4_pretrain.pth",
        "ray_intermediate_dims": (588, 768, 1024),
    },
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
        backbone_name: str = "vitl",
        checkpoint_dir: str = "/path/to/checkpoints",
        head_features: int = 256,
        head_out_channels: tuple = (256, 512, 1024, 1024),
        freeze_backbone: bool = False,
        predict_mask: bool = False,
        scale_head_mode: str = "cls",
        scale_head_num_queries: int = 4,
        scale_head_num_heads: int = 8,
        use_dwc: bool = False,        # DWC bypass inside backbone attention
        dwc_kernel_size: int = 3,
        n_cam_types: int = 0,         # heterogeneous camera types (0=off, 2=fisheye+pinhole)
        # Stage 3: calibration tokens / distortion bias
        use_calib_tokens: bool = False,
        calib_tokens_per_type: int = 4,
        calib_inject_types=(0,),
        use_distortion_bias: bool = False,
        distortion_bias_layers: str = "global_only",
        distortion_bias_hidden_dim: int = 64,
        distortion_bias_chunk_size: int = 1024,
        # Freeze the whole pretrained FMDE; only calibration tokens are trainable.
        # Overrides the partial unfreeze done by freeze_backbone.
        calib_tokens_only: bool = False,
        # Only the distortion_bias MLP is trainable (cross-camera geometric
        # alignment). Mutually exclusive with calib_tokens_only. Single-modality
        # forward passes are left unchanged.
        distortion_bias_only: bool = False,
    ):
        super().__init__()

        assert backbone_name in VIT_CONFIGS, f"unsupported backbone: {backbone_name}"
        vit_cfg = VIT_CONFIGS[backbone_name]

        self.backbone_name = backbone_name
        self.embed_dim = vit_cfg["embed_dim"]
        self.n_cam_types = n_cam_types
        # cat_token=True doubles the feature dim (local + global concatenated).
        self.feat_dim = self.embed_dim * 2

        # 1. DINOv2 backbone
        self.backbone = DinoV2(
            name=vit_cfg["name"],
            out_layers=vit_cfg["out_layers"],
            alt_start=vit_cfg["alt_start"],
            qknorm_start=vit_cfg["qknorm_start"],
            rope_start=vit_cfg["rope_start"],
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

        import os
        # DINOv2 weights dir; DINOV2_CKPT_DIR env overrides checkpoint_dir at eval time.
        ckpt_path = os.path.join(os.environ.get("DINOV2_CKPT_DIR", checkpoint_dir),
                                 vit_cfg["checkpoint"])
        if os.path.exists(ckpt_path):
            self.backbone.load_pretrained_weights(ckpt_path)
        else:
            logger.warning(f"DINOv2 pretrained weights not found: {ckpt_path}, using random init")

        if freeze_backbone:
            logger.info("freezing DINOv2 backbone weights")
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Keep three Stage 2/3 submodules trainable under a frozen backbone:
            # calib_tokens (fisheye->pinhole distribution alignment),
            # distortion_bias (cross-camera geometric alignment),
            # cam_type_embed (per-token camera-type hint).
            bb = self.backbone.pretrained
            for mod_name in ("calib_tokens", "distortion_bias"):
                mod = getattr(bb, mod_name, None)
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad = True
                    logger.info(f"  keeping {mod_name} trainable")
            if getattr(bb, "cam_type_embed", None) is not None:
                bb.cam_type_embed.weight.requires_grad = True
                logger.info("  keeping cam_type_embed trainable")

        # 2. Ray-map encoder (MapAnything dense_rep_encoder style).
        # Encodes the (B, S, 6, H, W) ray map to (B, S, embed_dim, H/p, W/p) and
        # adds it to patch tokens after patch_embed. The last layer is zero-init
        # so ray_feat=0 early in training and does not disturb the DINOv2 prior.
        self.ray_encoder = RayMapEncoder(
            embed_dim=self.embed_dim,
            patch_size=self.PATCH_SIZE,
            in_chans=6,
            intermediate_dims=vit_cfg["ray_intermediate_dims"],
        )

        # 3. DPT head: depth (+ optional non-ambiguous mask).
        self.head = DPTHead(
            dim_in=self.feat_dim,
            depth_output_dim=2,   # depth + confidence (auto +1 when predict_mask=True)
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
            num_layers=len(vit_cfg["out_layers"]),
            num_heads=scale_head_num_heads,
        )

        # Freeze the whole FMDE and train only calibration tokens. Must run after
        # all submodules are built, since freeze_backbone only touches
        # self.backbone and not these external heads.
        self.calib_tokens_only = calib_tokens_only
        self.distortion_bias_only = distortion_bias_only
        if calib_tokens_only and distortion_bias_only:
            raise ValueError("calib_tokens_only and distortion_bias_only are mutually exclusive")
        if calib_tokens_only:
            self._freeze_all_but_calib_tokens()
        if distortion_bias_only:
            self._freeze_all_but_distortion_bias()

    def _freeze_all_but_distortion_bias(self):
        """Freeze all parameters except the distortion_bias MLP.

        backbone / DPT / scale_head / ray_encoder / calib_tokens / cam_type_embed
        stay frozen, so single-modality (pure fisheye or pure pinhole) forward
        passes are bit-identical; only distortion_bias is trainable.
        """
        for p in self.parameters():
            p.requires_grad = False
        db = getattr(getattr(self.backbone, "pretrained", None), "distortion_bias", None)
        if db is None:
            raise ValueError(
                "distortion_bias_only=True but the model has no distortion_bias module; "
                "set use_distortion_bias=True (and n_cam_types > 0)."
            )
        n_db = 0
        for p in db.parameters():
            p.requires_grad = True
            n_db += p.numel()
        logger.info(
            f"[distortion_bias_only] whole model frozen, only distortion_bias trainable "
            f"({n_db / 1e3:.1f}K params)"
        )

    def _freeze_all_but_calib_tokens(self):
        """Freeze all parameters except the backbone's calibration tokens.

        cam_type_embed is zero-init, so frozen at 0 it is a no-op; distortion_bias
        / ray_encoder / scale_head / DPT head / DINOv2 trunk stay frozen. The
        fisheye->pinhole distribution alignment is carried entirely by
        calib_tokens, leaving the perspective decoder untouched.
        """
        for p in self.parameters():
            p.requires_grad = False
        calib = getattr(getattr(self.backbone, "pretrained", None), "calib_tokens", None)
        if calib is None:
            raise ValueError(
                "calib_tokens_only=True but the model has no calib_tokens module; "
                "set use_calib_tokens=True (and n_cam_types > 0)."
            )
        n_calib = 0
        for p in calib.parameters():
            p.requires_grad = True
            n_calib += p.numel()
        logger.info(
            f"[calib_tokens_only] whole FMDE frozen, only calib_tokens trainable "
            f"({n_calib / 1e3:.1f}K params)"
        )

    def forward(
        self,
        images: torch.Tensor,
        ray_map: Optional[torch.Tensor] = None,
        d_cam: Optional[torch.Tensor] = None,
        cam_types: Optional[torch.Tensor] = None,
        ray_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
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
                - backbone_patches_multilayer: list of (B, S, 1+N, D) for Gram distill
        """
        _, S, _, H, W = images.shape
        assert S >= 2, f"expected at least 2 views, got S={S}"

        # Ray-map encoding (autocast off; geometry runs in fp32). A precomputed
        # ray_feat (e.g. ONNX export, where fixed calibration makes ray_feat a
        # constant) is used directly, skipping the encoder.
        if ray_feat is None and ray_map is not None:
            with torch.autocast(device_type=images.device.type, enabled=False):
                ray_feat = self.ray_encoder(ray_map.float())  # (B, S, embed_dim, H/p, W/p)

        # DINOv2 multi-scale features + scale tokens. d_cam feeds FishRoPE;
        # cam_types feeds cam_type_embed + per-cam attention bias.
        feats, _, scale_tokens = self.backbone(
            images, ray_feat=ray_feat, d_cam=d_cam, cam_types=cam_types,
        )

        # Scale head -> metric scaling factor.
        with torch.autocast(device_type=images.device.type, enabled=False):
            if self.scale_head_mode == "cls":
                scale_factor = self.scale_head(scale_tokens=scale_tokens.float())  # (B,)
            else:  # attn_pool
                # Backbone 4-layer patch tokens (B, S, N_patch, 2C).
                multilayer_patch = [f[0].float() for f in feats]
                scale_factor = self.scale_head(multilayer_feats=multilayer_patch)  # (B,)

        # DPT head -> normalized-space depth.
        with torch.autocast(device_type=images.device.type, enabled=False):
            output = self.head(feats, H, W, patch_start_idx=0)

        # Expose all output-layer tokens for multi-layer Gram distillation.
        gram_layers = []
        for feat_patch, feat_cls in feats:
            cls_tok = feat_cls
            if cls_tok.dim() == 3:
                cls_tok = cls_tok.unsqueeze(2)
            gram_layers.append(torch.cat([cls_tok, feat_patch], dim=2))
        output["backbone_patches_multilayer"] = gram_layers

        # Metric-space depth.
        output["metric_scaling_factor"] = scale_factor
        sf = scale_factor[:, None, None, None]
        output["depth_metric"] = output["depth"] * sf

        return output

    def get_parameter_groups(self, lr: float, lr_backbone: float = None, lr_head: float = None,
                             lr_distortion_bias: float = None):
        """Build parameter groups with per-group learning rates."""
        if lr_backbone is None:
            lr_backbone = lr * 0.1
        if lr_head is None:
            lr_head = lr
        if lr_distortion_bias is None:
            lr_distortion_bias = lr

        pretrained = getattr(self.backbone, "pretrained", None)

        # calib_tokens get their own group at full lr (not lr_backbone): a
        # lightweight adapter, and the only trainable module in calib_tokens_only
        # mode, so the reduced backbone lr would train it too slowly.
        calib_mod = getattr(pretrained, "calib_tokens", None)
        calib_params = list(calib_mod.parameters()) if calib_mod is not None else []

        # distortion_bias (zero-init, added in Stage 3) likewise gets its own
        # group at lr_distortion_bias (default lr) since it trains from scratch.
        db_mod = getattr(pretrained, "distortion_bias", None)
        db_params = list(db_mod.parameters()) if db_mod is not None else []

        special_ids = {id(p) for p in calib_params} | {id(p) for p in db_params}
        backbone_params = [p for p in self.backbone.parameters() if id(p) not in special_ids]

        param_groups = [
            {"params": backbone_params, "lr": lr_backbone, "name": "backbone"},
            {"params": self.ray_encoder.parameters(), "lr": lr, "name": "ray_encoder"},
            {"params": self.scale_head.parameters(), "lr": lr, "name": "scale_head"},
            {"params": self.head.parameters(), "lr": lr_head, "name": "head"},
        ]
        if calib_params:
            param_groups.append({"params": calib_params, "lr": lr, "name": "calib_tokens"})
        if db_params:
            param_groups.append({"params": db_params, "lr": lr_distortion_bias, "name": "distortion_bias"})
        return param_groups
