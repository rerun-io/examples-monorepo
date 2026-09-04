"""DINOv2-based Vision Transformer.

Features:
- Alternating cross-view and within-view self-attention
- RoPE (rotary position embedding)
- QK-Norm
- Scale token injected at layer alt_start for ScaleHead

Geometry conditioning is provided through the ray-map path.
"""

import math
import logging
from typing import Callable, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange

from .layers import (
    Block,
    CalibrationTokens,
    DistortionBias,
    LayerScale,
    Mlp,
    PatchEmbed,
    PositionGetter,
    RotaryPositionEmbedding2D,
    SwiGLUFFNFused,
    build_calib_attention_mask,
    build_token_geometry,
)

logger = logging.getLogger(__name__)

# Reference-view selection is enabled only when the number of views is >= this
# threshold (a stereo pair has 2 views and never triggers it).
THRESH_FOR_REF_SELECTION = 3


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Build sin/cos position embeddings from a 1D grid.

    Args:
        embed_dim: Output dimension per position.
        pos: Positions to encode, shape (M,).

    Returns:
        emb: Position embeddings (M, D).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2)

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    """DINOv2 Vision Transformer.

    Architecture:
    1. Patch embedding.
    2. Transformer blocks with alternating attention:
       - First layers: within-view self-attention (each view independent).
       - Later layers: alternate cross-view and within-view attention.
    3. Scale token appended to each per-view sequence starting at layer alt_start.

    Args:
        img_size: Input image size.
        patch_size: Patch size.
        embed_dim: Embedding dimension.
        depth: Number of transformer layers.
        num_heads: Number of attention heads.
        alt_start: Layer index at which alternating attention begins.
        qknorm_start: Layer index at which QK-Norm begins.
        rope_start: Layer index at which RoPE begins.
        cat_token: Whether to concatenate local and global features.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=1.0,
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        alt_start=-1,
        qknorm_start=-1,
        rope_start=-1,
        rope_freq=100,
        cat_token=True,
        use_dwc=False,            # DWC bypass (local inductive bias)
        dwc_kernel_size=3,
        n_cam_types=0,            # Number of camera types (0=disabled; 2=fisheye+pinhole)
        # Stage 3 modules (disabled by default; compatible with Stage 1/2 checkpoints)
        use_calib_tokens=False,
        calib_tokens_per_type=4,
        calib_inject_types=(0,),  # Inject only for fisheye (cam_type=0) by default
        use_distortion_bias=False,
        distortion_bias_layers="global_only",  # "global_only" / "all" / "global_after_alt"
        distortion_bias_hidden_dim=64,
        distortion_bias_chunk_size=1024,       # Row chunk size for pairwise bias (memory)
    ):
        super().__init__()
        self.patch_start_idx = 1
        norm_layer = nn.LayerNorm
        self.num_features = self.embed_dim = embed_dim
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.cat_token = cat_token
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.use_dwc = use_dwc
        self.dwc_kernel_size = dwc_kernel_size
        self.n_cam_types = n_cam_types
        self.use_calib_tokens = use_calib_tokens
        self.calib_K = calib_tokens_per_type if use_calib_tokens else 0
        self.use_distortion_bias = use_distortion_bias
        self.distortion_bias_layers = distortion_bias_layers

        # Camera-type embedding: one learnable C-dim vector per camera type, added
        # to patch tokens. Zero-initialized so it starts as an identity w.r.t. the
        # base model (checkpoint compatible).
        if n_cam_types > 0:
            self.cam_type_embed = nn.Embedding(n_cam_types, embed_dim)
            nn.init.zeros_(self.cam_type_embed.weight)
        else:
            self.cam_type_embed = None

        # Stage 3: Calibration Tokens & Distortion Bias. Both disabled by default
        # (no extra parameters), enabled via config.
        if use_calib_tokens:
            assert n_cam_types > 0, "use_calib_tokens requires n_cam_types > 0"
            self.calib_tokens = CalibrationTokens(
                depth=depth,
                n_cam_types=n_cam_types,
                tokens_per_type=calib_tokens_per_type,
                embed_dim=embed_dim,
                inject_types=calib_inject_types,
            )
        else:
            self.calib_tokens = None

        if use_distortion_bias:
            self.distortion_bias = DistortionBias(
                num_heads=num_heads,
                hidden_dim=distortion_bias_hidden_dim,
                chunk_size=distortion_bias_chunk_size,
            )
        else:
            self.distortion_bias = None

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # CLS token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))

        # Scale token: injected at alt_start, participates in frame + global
        # attention, predicts metric scale.
        self.scale_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.scale_token, std=0.02)

        # Register tokens
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )

        # Drop path (stochastic depth)
        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # FFN layer type
        if ffn_layer == "mlp":
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            def f(*args, **kwargs):
                return nn.Identity()
            ffn_layer = f
        else:
            raise NotImplementedError

        # RoPE
        if self.rope_start != -1:
            self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
            self.position_getter = PositionGetter() if self.rope is not None else None
        else:
            self.rope = None

        # Transformer blocks
        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
                qk_norm=i >= qknorm_start if qknorm_start != -1 else False,
                rope=self.rope if i >= rope_start and rope_start != -1 else None,
                use_dwc=use_dwc,
                dwc_kernel_size=dwc_kernel_size,
                n_cam_types=n_cam_types,
            )
            for i in range(depth)
        ]
        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)

    def interpolate_pos_encoding(self, x, w, h):
        """Interpolate position embeddings for a different input resolution."""
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_cls_token(self, B, S):
        """Prepare CLS tokens."""
        cls_token = self.cls_token.expand(B, S, -1)
        cls_token = cls_token.reshape(B * S, -1, self.embed_dim)
        return cls_token

    def prepare_tokens_with_masks(self, x, masks=None, cls_token=None, ray_feat=None,
                                  cam_types=None, **kwargs):
        """Build the input token sequence.

        Composes patch embedding [+ ray_feat] [+ cam_type_embed] + CLS token +
        position embedding + register tokens.

        Args:
            x:         Input images (B, S, 3, H, W).
            ray_feat:  Optional (B, S, embed_dim, H/p, W/p) ray-map features from
                       RayMapEncoder, added element-wise to patch tokens.
            cam_types: Optional (B, S) int camera-type ids (0=fisheye, 1=pinhole),
                       looked up in cam_type_embed and broadcast over patch tokens.

        Returns:
            Token sequence (B, S, N, C).
        """
        B, S, nc, w, h = x.shape
        x = rearrange(x, "b s c h w -> (b s) c h w")
        x = self.patch_embed(x)                                 # (BS, N, C)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        # Add ray_feat to patch tokens (before CLS concat and pos_embed)
        if ray_feat is not None:
            assert ray_feat.shape[:2] == (B, S), (
                f"ray_feat batch/view mismatch: {ray_feat.shape[:2]} vs ({B}, {S})"
            )
            assert ray_feat.shape[2] == self.embed_dim, (
                f"ray_feat embed_dim {ray_feat.shape[2]} != backbone {self.embed_dim}"
            )
            # (B, S, C, Hp, Wp) -> (BS, N, C)
            rf = ray_feat.reshape(B * S, self.embed_dim, -1).transpose(1, 2)
            assert rf.shape[1] == x.shape[1], (
                f"ray_feat patch count {rf.shape[1]} != image patch count {x.shape[1]}"
            )
            x = x + rf.to(x.dtype)

        # Add camera-type embedding (per-view vector broadcast over all patch tokens)
        if self.cam_type_embed is not None and cam_types is not None:
            assert cam_types.shape == (B, S), (
                f"cam_types shape {cam_types.shape} != ({B}, {S})"
            )
            type_emb = self.cam_type_embed(cam_types)              # (B, S, C)
            type_emb = type_emb.reshape(B * S, 1, self.embed_dim)  # (BS, 1, C), broadcast over N
            x = x + type_emb.to(x.dtype)

        cls_token = self.prepare_cls_token(B, S)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        x = rearrange(x, "(b s) n c -> b s n c", b=B, s=S)
        return x

    def _prepare_rope(self, B, S, H, W, device, d_cam=None):
        """Build RoPE positions.

        - d_cam=None (pinhole): integer pixel-grid indices; RoPE uses F.embedding lookup.
        - d_cam given (fisheye/general): convert per-pixel unit rays to (azimuth,
          elevation) angles, scaled to the pixel-grid range (to keep frequencies
          matched), and feed as float positions.

        d_cam shape: (B, S, 3, H, W) or (B, S, H, W, 3), unit vectors, camera frame.
        OpenCV convention (+z forward, +y down, +x right):
          azimuth   = atan2(x, z)                horizontal angle
          elevation = atan2(y, sqrt(x^2 + z^2))  vertical angle
        """
        pos = None
        pos_nodiff = None
        if self.rope is None:
            return pos, pos_nodiff

        H_p = H // self.patch_size
        W_p = W // self.patch_size

        if d_cam is None:
            # Integer pixel grid
            pos = self.position_getter(B * S, H_p, W_p, device=device)
            pos = rearrange(pos, "(b s) n c -> b s n c", b=B)
            pos_nodiff = torch.zeros_like(pos).to(pos.dtype)
            if self.patch_start_idx > 0:
                pos = pos + 1
                pos_special = torch.zeros(B * S, self.patch_start_idx, 2, device=device, dtype=pos.dtype)
                pos_special = rearrange(pos_special, "(b s) n c -> b s n c", b=B)
                pos = torch.cat([pos_special, pos], dim=2)
                pos_nodiff = pos_nodiff + 1
                pos_nodiff = torch.cat([pos_special, pos_nodiff], dim=2)
            return pos, pos_nodiff

        # ---- FishRoPE: angular positions from d_cam ----
        # 1. Downsample d_cam to the patch grid (H_p, W_p): average pool + renormalize
        if d_cam.dim() == 5 and d_cam.shape[-1] == 3:
            # (B, S, H, W, 3) -> (B, S, 3, H, W)
            d_cam = d_cam.permute(0, 1, 4, 2, 3).contiguous()
        assert d_cam.shape[2] == 3, f"d_cam channels=3, got {d_cam.shape}"
        d_cam_bs = d_cam.reshape(B * S, 3, H, W)
        d_cam_p = torch.nn.functional.avg_pool2d(d_cam_bs, kernel_size=self.patch_size)  # (BS, 3, H_p, W_p)
        d_cam_p = torch.nn.functional.normalize(d_cam_p, dim=1, eps=1e-6)
        x_, y_, z_ = d_cam_p[:, 0], d_cam_p[:, 1], d_cam_p[:, 2]                          # (BS, H_p, W_p) each

        # 2. azimuth / elevation (rad), OpenCV convention
        azimuth = torch.atan2(x_, z_)                               # [-pi, pi]
        elevation = torch.atan2(y_, torch.sqrt(x_ * x_ + z_ * z_).clamp(min=1e-8))  # [-pi/2, pi/2]

        # 3. Scale to the pixel-grid range so RoPE frequency behavior is preserved:
        #    azimuth [-pi, pi] -> [0, W_p), elevation [-pi/2, pi/2] -> [0, H_p),
        #    linear so equal angular steps map to one RoPE cell.
        v_pos = (elevation / torch.pi + 0.5) * H_p          # (BS, H_p, W_p)
        u_pos = (azimuth / (2 * torch.pi) + 0.5) * W_p

        # 4. Stack into float (BS, N, 2)
        pos_flat = torch.stack([v_pos, u_pos], dim=-1).reshape(B * S, H_p * W_p, 2)
        pos_flat = pos_flat.to(torch.float32)
        pos = rearrange(pos_flat, "(b s) n c -> b s n c", b=B)

        # pos_nodiff: used by global attention; no cross-view geometric alignment,
        # so set to a constant (1.0)
        pos_nodiff = torch.ones_like(pos)

        # 5. Special tokens (CLS / register / scale): position 0 (no RoPE rotation)
        if self.patch_start_idx > 0:
            pos = pos + 1.0   # shift patch positions to distinguish from special tokens
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2, device=device, dtype=pos.dtype)
            pos_special = rearrange(pos_special, "(b s) n c -> b s n c", b=B)
            pos = torch.cat([pos_special, pos], dim=2)
            pos_nodiff = torch.cat([pos_special, pos_nodiff], dim=2)
        return pos, pos_nodiff

    def _get_intermediate_layers_not_chunked(self, x, n=1, export_feat_layers=[], **kwargs):
        """Extract intermediate features from selected layers.

        - Before alt_start: within-view (local) attention.
        - After alt_start, even layers: within-view (local) attention.
        - After alt_start, odd layers: cross-view (global) attention.
        - Scale token injected at alt_start.

        cam_types: optional (B, S) int in kwargs, used to (1) add cam_type_embed in
        prepare_tokens_with_masks and (2) forward per-cam bias into each block.
        """
        B, S, _, H, W = x.shape
        cam_types_BS = kwargs.get("cam_types", None)   # (B, S) int or None
        x = self.prepare_tokens_with_masks(
            x,
            ray_feat=kwargs.get("ray_feat", None),
            cam_types=cam_types_BS,
        )
        output, total_block_len, aux_output = [], len(self.blocks), []
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        pos, pos_nodiff = self._prepare_rope(
            B, S, H, W, x.device, d_cam=kwargs.get("d_cam", None),
        )
        scale_injected = False  # whether the scale token has been injected
        output_has_scale = []   # per-output flag: does it include the scale token

        # DWC bypass: precompute patch grid size, reused when building dwc_info
        H_p = H // self.patch_size
        W_p = W // self.patch_size
        dwc_prefix = 1 + self.num_register_tokens   # CLS + register tokens

        # Stage 3 runtime switches. Whether this batch actually runs
        # calib_tokens / distortion_bias:
        # - calib: use_calib_tokens=True and cam_types_BS given with an inject type present
        # - distortion bias: use_distortion_bias=True and d_cam given
        d_cam_input = kwargs.get("d_cam", None)
        calib_active = (
            self.calib_tokens is not None
            and cam_types_BS is not None
            and self.calib_tokens.needs_inject(cam_types_BS)
        )
        dist_bias_active = (
            self.distortion_bias is not None and d_cam_input is not None
        )

        for i, blk in enumerate(self.blocks):
            # Inject the scale token at alt_start
            if self.alt_start != -1 and i == self.alt_start:
                # Append the (shared) scale token to the end of each per-view sequence;
                # it participates in all subsequent local + global attention.
                scale_tok = self.scale_token.expand(B, S, -1, -1)  # (B, S, 1, C)
                x = torch.cat([x, scale_tok], dim=2)  # (B, S, N+1, C)
                scale_injected = True
                # Extend RoPE positions (scale token uses zero position)
                if pos is not None:
                    scale_pos = torch.zeros(B, S, 1, pos.shape[-1], device=pos.device, dtype=pos.dtype)
                    pos = torch.cat([pos, scale_pos], dim=2)
                    pos_nodiff = torch.cat([pos_nodiff, scale_pos], dim=2)

            # ---- Calib token injection (per-layer, multi-layer variant) ----
            # Append K tokens at the end of the sequence, dropped right after this
            # block. Must happen before g_pos/l_pos so RoPE positions stay aligned.
            K_layer = 0
            inject_mask = None
            calib_view_tokens = None
            if calib_active:
                K_layer = self.calib_tokens.K
                calib_view_tokens, inject_mask = self.calib_tokens.build_view_calib_tokens(
                    i, cam_types_BS
                )                                                              # (B, S, K, C), (B, S) bool
                x = torch.cat([x, calib_view_tokens.to(x.dtype)], dim=2)       # (B, S, N+K, C)
                if pos is not None:
                    calib_pos = torch.zeros(B, S, K_layer, pos.shape[-1],
                                            device=pos.device, dtype=pos.dtype)
                    pos = torch.cat([pos, calib_pos], dim=2)
                    pos_nodiff = torch.cat([pos_nodiff, calib_pos], dim=2)

            # Select RoPE positions (after scale + calib injection so dims match)
            if i < self.rope_start or self.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos = pos_nodiff
                l_pos = pos

            # Local vs global layer
            is_global_layer = (
                self.alt_start != -1 and i >= self.alt_start and i % 2 == 1
            )
            attn_type = "global" if is_global_layer else "local"

            # Build attention mask + distortion bias.
            # N_non_calib covers CLS + register + N_patches (+ scale if present).
            N_non_calib_per_view = x.shape[2] - K_layer  # x's N dim now includes calib
            attn_mask_combined = None  # final mask passed to the attention block

            # 1) calib mask
            if K_layer > 0:
                calib_mask = build_calib_attention_mask(
                    K_per_view=K_layer,
                    inject_mask=inject_mask,
                    N_non_calib=N_non_calib_per_view,
                    attn_type=attn_type,
                    device=x.device,
                    dtype=torch.float32,
                )
                if calib_mask is not None:
                    attn_mask_combined = calib_mask  # (B', L, L)

            # 2) distortion bias (global layers only, when configured)
            apply_dist_bias = False
            if dist_bias_active:
                if self.distortion_bias_layers == "all":
                    apply_dist_bias = True
                elif self.distortion_bias_layers == "global_only":
                    apply_dist_bias = is_global_layer
                elif self.distortion_bias_layers == "global_after_alt":
                    apply_dist_bias = is_global_layer  # same as global_only; kept for API
                else:
                    apply_dist_bias = False

            if apply_dist_bias:
                suffix_no_calib = 1 if scale_injected else 0
                d_tok, J_tok, valid_tok = build_token_geometry(
                    d_cam=d_cam_input,
                    H_p=H_p, W_p=W_p,
                    prefix_len=dwc_prefix,
                    suffix_len_no_calib=suffix_no_calib,
                    K_calib=K_layer,
                    n_views=S,
                    batch_size=B,
                    device=x.device,
                    dtype=torch.float32,
                    attn_type=attn_type,
                )
                if d_tok is not None:
                    db = self.distortion_bias(d_tok, J_tok, valid_mask=valid_tok)  # (B', H, L, L)
                    if attn_mask_combined is None:
                        attn_mask_combined = db
                    else:
                        # Broadcast the (B', L, L) calib mask over heads, then add bias
                        attn_mask_combined = db + attn_mask_combined.unsqueeze(1).to(db.dtype)

            # DWC bypass info; suffix must include the calib token slots
            dwc_suffix = (1 if scale_injected else 0) + K_layer
            dwc_info_local = (H_p, W_p, dwc_prefix, dwc_suffix, 1) if self.use_dwc else None
            dwc_info_global = (H_p, W_p, dwc_prefix, dwc_suffix, S) if self.use_dwc else None

            # Alternating attention
            if is_global_layer:
                # Cross-view attention
                external_mask = kwargs.get("attn_mask", None)
                final_mask = self._combine_masks(attn_mask_combined, external_mask)
                x = self.process_attention(
                    x, blk, "global", pos=g_pos,
                    attn_mask=final_mask,
                    dwc_info=dwc_info_global,
                    cam_types_BS=cam_types_BS,
                )
            else:
                # Within-view self-attention
                x = self.process_attention(
                    x, blk, "local", pos=l_pos, dwc_info=dwc_info_local,
                    cam_types_BS=cam_types_BS,
                    attn_mask=attn_mask_combined,
                )
                local_x = x  # keep local features for concatenation

            # Drop calib tokens (present only within this layer's attention)
            if K_layer > 0:
                x = x[:, :, :-K_layer, :]
                if pos is not None:
                    pos = pos[:, :, :-K_layer, :]
                    pos_nodiff = pos_nodiff[:, :, :-K_layer, :]
                # keep the calib-dropped version of local_x for alignment
                if not is_global_layer:
                    local_x = x

            # Collect outputs from selected layers
            if i in blocks_to_take:
                out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
                output.append((out_x[:, :, 0], out_x))
                output_has_scale.append(scale_injected)
            if i in export_feat_layers:
                aux_output.append(x)

        return output, aux_output, output_has_scale

    @staticmethod
    def _combine_masks(internal, external):
        """Combine the internal mask (calib + distortion bias) with an external mask.

        - Either None: return the other.
        - Both present: convert external to an additive float bias and add it.
        """
        if internal is None:
            return external
        if external is None:
            return internal
        # Convert external to an additive float bias, then combine
        if external.dtype == torch.bool:
            additive = torch.zeros_like(internal, dtype=internal.dtype)
            # external (B, N, N) -> broadcast over the head dim if internal is 4D
            if internal.dim() == 4 and external.dim() == 3:
                external = external.unsqueeze(1)
            additive.masked_fill_(~external.expand_as(internal), float("-inf"))
            return internal + additive
        else:
            if internal.dim() == 4 and external.dim() == 3:
                external = external.unsqueeze(1)
            return internal + external.to(internal.dtype).expand_as(internal)

    def process_attention(self, x, block, attn_type="global", pos=None, attn_mask=None,
                          dwc_info=None, cam_types_BS=None):
        """Run one attention layer.

        Args:
            x: Input tokens (B, S, N, C).
            block: Transformer block.
            attn_type: "local" (within-view) or "global" (cross-view).
            pos: Position embedding.
            attn_mask: Additional attention bias / mask. Shape may be:
                - (B', N, N) float additive bias / bool mask (broadcast over heads)
                - (B', H, N, N) per-head additive bias (DistortionBias output)
                B' = B*S (local) or B (global), depending on attn_type.
            dwc_info: DWC bypass info (H_p, W_p, prefix, suffix, S_per_batch); None disables DWC.
            cam_types_BS: (B, S) int camera-type id per view; None to skip per-cam bias.

        Local reshapes (B, S, N, C) to (B*S, N, C) so each view attends independently.
        Global reshapes to (B, S*N, C) so all views attend jointly.
        cam_types expand the same way; special tokens (CLS/register/scale) inherit
        the cam_type of their view.
        """
        b, s, n = x.shape[:3]
        # Build per-token cam_types matching the reshaped x
        cam_types_tokens = None
        if cam_types_BS is not None:
            # (B, S) -> broadcast to (B, S, N)
            cam_types_view = cam_types_BS.unsqueeze(-1).expand(b, s, n).contiguous()
            if attn_type == "local":
                # (B, S, N) -> (B*S, N)
                cam_types_tokens = cam_types_view.reshape(b * s, n)
            else:  # global
                # (B, S, N) -> (B, S*N)
                cam_types_tokens = cam_types_view.reshape(b, s * n)

        if attn_type == "local":
            x = rearrange(x, "b s n c -> (b s) n c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> (b s) n c")
        elif attn_type == "global":
            x = rearrange(x, "b s n c -> b (s n) c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> b (s n) c")
        else:
            raise ValueError(f"unsupported attention type: {attn_type}")

        x = block(x, pos=pos, attn_mask=attn_mask, dwc_info=dwc_info,
                  cam_types=cam_types_tokens)

        if attn_type == "local":
            x = rearrange(x, "(b s) n c -> b s n c", b=b, s=s)
        elif attn_type == "global":
            x = rearrange(x, "b (s n) c -> b s n c", b=b, s=s)
        return x

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
        export_feat_layers: List[int] = [],
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """Get intermediate-layer features.

        Args:
            x: Input images (B, S, 3, H, W).
            n: Layer indices to extract features from.

        Returns:
            (feats, aux_feats, scale_tokens). feats[i] = (patch_tokens, cls_tokens),
            where the second item is the token at the CLS position.
        """
        outputs, aux_outputs, output_has_scale = self._get_intermediate_layers_not_chunked(
            x, n, export_feat_layers=export_feat_layers, **kwargs
        )
        cls_tokens = [out[0] for out in outputs]

        # Extract the scale token (last position of the final output)
        if output_has_scale and output_has_scale[-1]:
            last_out = outputs[-1][1]  # (B, S, N, C) or (B, S, N, 2C)
            scale_tokens = last_out[:, :, -1, :]  # (B, S, C) or (B, S, 2C)
        else:
            scale_tokens = None

        # Normalize output features
        if outputs[0][1].shape[-1] == self.embed_dim:
            raw_outputs = [self.norm(out[1]) for out in outputs]
        elif outputs[0][1].shape[-1] == (self.embed_dim * 2):
            raw_outputs = [
                torch.cat(
                    [out[1][..., : self.embed_dim], self.norm(out[1][..., self.embed_dim :])],
                    dim=-1,
                )
                for out in outputs
            ]
        else:
            raise ValueError(f"unsupported output dimension: {outputs[0][1].shape}")

        aux_outputs = [self.norm(out) for out in aux_outputs]

        # Remove CLS/register tokens; also drop the trailing scale token if present
        prefix = 1 + self.num_register_tokens
        cleaned_outputs = []
        for idx, out in enumerate(raw_outputs):
            if output_has_scale[idx]:
                cleaned_outputs.append(out[..., prefix:-1, :])  # drop leading CLS/reg + trailing scale
            else:
                cleaned_outputs.append(out[..., prefix:, :])    # drop leading CLS/reg only
        # aux_outputs come from after alt_start, so a scale token is always present
        aux_outputs = [out[..., prefix:-1, :] if scale_tokens is not None else out[..., prefix:, :] for out in aux_outputs]

        return tuple(zip(cleaned_outputs, cls_tokens)), aux_outputs, scale_tokens
