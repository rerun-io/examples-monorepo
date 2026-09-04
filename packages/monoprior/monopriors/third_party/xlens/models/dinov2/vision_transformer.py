"""DINOv2-based Vision Transformer.

Features:
- Alternating cross-view and within-view self-attention
- RoPE (rotary position embedding)
- QK-Norm
- Scale token injected at layer alt_start for ScaleHead

Geometry conditioning is provided through the ray-map path.
"""

import logging
import math
from collections.abc import Sequence
from typing import Literal, TypeAlias, cast

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Bool, Float, Int64
from torch import Tensor

from monopriors.third_party.xlens.models.dinov2.layers import (
    Block,
    CalibrationTokens,
    DistortionBias,
    Mlp,
    PatchEmbed,
    PositionGetter,
    RotaryPositionEmbedding2D,
    build_calib_attention_mask,
    build_token_geometry,
)

logger = logging.getLogger(__name__)

AttentionType = Literal["local", "global"]
DwcInfo: TypeAlias = tuple[int, int, int, int, int]
BackboneFeature: TypeAlias = tuple[Float[Tensor, "batch views patches features"], Float[Tensor, "batch views features"]]
IntermediateOutput: TypeAlias = tuple[tuple[BackboneFeature, ...], Float[Tensor, "batch views features"] | None]


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
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 0,
        alt_start: int = -1,
        qknorm_start: int = -1,
        rope_start: int = -1,
        rope_freq: float = 100.0,
        cat_token: bool = True,
        use_dwc: bool = False,
        dwc_kernel_size: int = 3,
        n_cam_types: int = 0,
        # Stage 3 modules (disabled by default; compatible with Stage 1/2 checkpoints)
        use_calib_tokens: bool = False,
        calib_tokens_per_type: int = 4,
        calib_inject_types: tuple[int, ...] = (0,),
        use_distortion_bias: bool = False,
        distortion_bias_layers: Literal["global_only", "all"] = "global_only",
        distortion_bias_hidden_dim: int = 64,
        distortion_bias_chunk_size: int = 1024,
    ) -> None:
        """Build the inference-only multi-view vision transformer."""
        super().__init__()
        self.patch_start_idx = 1
        norm_layer = nn.LayerNorm
        self.num_features = self.embed_dim = embed_dim
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.cat_token = cat_token
        self.num_tokens = 1
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = False
        self.interpolate_offset = 0.1
        self.use_dwc = use_dwc
        self.dwc_kernel_size = dwc_kernel_size
        self.n_cam_types = n_cam_types
        self.use_calib_tokens = use_calib_tokens
        self.use_distortion_bias = use_distortion_bias
        self.distortion_bias_layers = distortion_bias_layers

        # Camera-type embedding: one learnable C-dim vector per camera type, added
        # to patch tokens. Zero-initialized so it starts as an identity w.r.t. the
        # base model (checkpoint compatible).
        self.cam_type_embed: nn.Embedding | None
        if n_cam_types > 0:
            self.cam_type_embed = nn.Embedding(n_cam_types, embed_dim)
            nn.init.zeros_(self.cam_type_embed.weight)
        else:
            self.cam_type_embed = None

        # Stage 3: Calibration Tokens & Distortion Bias. Both disabled by default
        # (no extra parameters), enabled via config.
        self.calib_tokens: CalibrationTokens | None
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

        self.distortion_bias: DistortionBias | None
        if use_distortion_bias:
            self.distortion_bias = DistortionBias(
                num_heads=num_heads,
                hidden_dim=distortion_bias_hidden_dim,
                chunk_size=distortion_bias_chunk_size,
            )
        else:
            self.distortion_bias = None

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
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
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None

        # RoPE
        self.rope: RotaryPositionEmbedding2D | None
        self.position_getter: PositionGetter | None
        if self.rope_start != -1:
            self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
            self.position_getter = PositionGetter() if self.rope is not None else None
        else:
            self.rope = None
            self.position_getter = None

        # Transformer blocks
        blocks_list = [
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=norm_layer,
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=1.0,
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

    def interpolate_pos_encoding(self, x: Float[Tensor, "batch tokens features"], w: int, h: int) -> Float[Tensor, "1 tokens features"]:
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
        if self.interpolate_offset:
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
                mode="bicubic",
                antialias=self.interpolate_antialias,
                scale_factor=(sx, sy),
            )
        else:
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
                mode="bicubic",
                antialias=self.interpolate_antialias,
                size=(w0, h0),
            )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_cls_token(self, B: int, S: int) -> Float[Tensor, "packed_views 1 features"]:
        """Prepare CLS tokens."""
        cls_token = self.cls_token.expand(B, S, -1)
        cls_token = cls_token.reshape(B * S, -1, self.embed_dim)
        return cls_token

    def prepare_tokens_with_masks(
        self,
        x: Float[Tensor, "batch views 3 height width"],
        ray_feat: Float[Tensor, "batch views features patch_height patch_width"] | None = None,
        cam_types: Int64[Tensor, "batch views"] | None = None,
    ) -> Float[Tensor, "batch views tokens features"]:
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
        B, S, _, w, h = x.shape
        x = rearrange(x, "b s c h w -> (b s) c h w")
        x = self.patch_embed(x)

        # Add ray_feat to patch tokens (before CLS concat and pos_embed)
        if ray_feat is not None:
            assert ray_feat.shape[:2] == (B, S), f"ray_feat batch/view mismatch: {ray_feat.shape[:2]} vs ({B}, {S})"
            assert ray_feat.shape[2] == self.embed_dim, f"ray_feat embed_dim {ray_feat.shape[2]} != backbone {self.embed_dim}"
            # (B, S, C, Hp, Wp) -> (BS, N, C)
            rf: Float[Tensor, "packed_views patches features"] = ray_feat.reshape(B * S, self.embed_dim, -1).transpose(1, 2)
            assert rf.shape[1] == x.shape[1], f"ray_feat patch count {rf.shape[1]} != image patch count {x.shape[1]}"
            x = x + rf.to(x.dtype)

        # Add camera-type embedding (per-view vector broadcast over all patch tokens)
        if self.cam_type_embed is not None and cam_types is not None:
            assert cam_types.shape == (B, S), f"cam_types shape {cam_types.shape} != ({B}, {S})"
            type_emb: Float[Tensor, "batch views features"] = self.cam_type_embed(cam_types)
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

    def _prepare_rope(
        self,
        B: int,
        S: int,
        H: int,
        W: int,
        device: torch.device,
        d_cam: Float[Tensor, "batch views _channels _height _width"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch views tokens 2"] | Int64[Tensor, "batch views tokens 2"] | None,
        Float[Tensor, "batch views tokens 2"] | Int64[Tensor, "batch views tokens 2"] | None,
    ]:
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
            if self.position_getter is None:
                raise RuntimeError("RoPE position getter was not initialized")
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
        x_, y_, z_ = d_cam_p[:, 0], d_cam_p[:, 1], d_cam_p[:, 2]  # (BS, H_p, W_p) each

        # 2. azimuth / elevation (rad), OpenCV convention
        azimuth = torch.atan2(x_, z_)  # [-pi, pi]
        elevation = torch.atan2(y_, torch.sqrt(x_ * x_ + z_ * z_).clamp(min=1e-8))  # [-pi/2, pi/2]

        # 3. Scale to the pixel-grid range so RoPE frequency behavior is preserved:
        #    azimuth [-pi, pi] -> [0, W_p), elevation [-pi/2, pi/2] -> [0, H_p),
        #    linear so equal angular steps map to one RoPE cell.
        v_pos = (elevation / torch.pi + 0.5) * H_p  # (BS, H_p, W_p)
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
            pos = pos + 1.0  # shift patch positions to distinguish from special tokens
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2, device=device, dtype=pos.dtype)
            pos_special = rearrange(pos_special, "(b s) n c -> b s n c", b=B)
            pos = torch.cat([pos_special, pos], dim=2)
            pos_nodiff = torch.cat([pos_special, pos_nodiff], dim=2)
        return pos, pos_nodiff

    def _get_intermediate_layers_not_chunked(
        self,
        x: Float[Tensor, "batch views 3 height width"],
        n: int | Sequence[int] = 1,
        *,
        ray_feat: Float[Tensor, "batch views features patch_height patch_width"] | None = None,
        d_cam: Float[Tensor, "batch views _channels _height _width"] | None = None,
        cam_types: Int64[Tensor, "batch views"] | None = None,
    ) -> tuple[list[tuple[Float[Tensor, "batch views features"], Float[Tensor, "batch views tokens features"]]], list[bool]]:
        """Extract intermediate features from selected layers.

        - Before alt_start: within-view (local) attention.
        - After alt_start, even layers: within-view (local) attention.
        - After alt_start, odd layers: cross-view (global) attention.
        - Scale token injected at alt_start.

        cam_types: optional (B, S) int in kwargs, used to (1) add cam_type_embed in
        prepare_tokens_with_masks and (2) forward per-cam bias into each block.
        """
        B, S, _, H, W = x.shape
        cam_types_BS = cam_types
        x = self.prepare_tokens_with_masks(
            x,
            ray_feat=ray_feat,
            cam_types=cam_types_BS,
        )
        output: list[tuple[Float[Tensor, "batch views features"], Float[Tensor, "batch views tokens features"]]] = []
        total_block_len = len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        pos, pos_nodiff = self._prepare_rope(
            B,
            S,
            H,
            W,
            x.device,
            d_cam=d_cam,
        )
        scale_injected = False  # whether the scale token has been injected
        output_has_scale = []  # per-output flag: does it include the scale token

        # DWC bypass: precompute patch grid size, reused when building dwc_info
        H_p = H // self.patch_size
        W_p = W // self.patch_size
        dwc_prefix = 1 + self.num_register_tokens  # CLS + register tokens

        # Stage 3 runtime switches. Whether this batch actually runs
        # calib_tokens / distortion_bias:
        # - calib: use_calib_tokens=True and cam_types_BS given with an inject type present
        # - distortion bias: use_distortion_bias=True and d_cam given
        d_cam_input = d_cam
        calib_active = self.calib_tokens is not None and cam_types_BS is not None and self.calib_tokens.needs_inject(cam_types_BS)
        dist_bias_active = self.distortion_bias is not None and d_cam_input is not None

        local_x: Float[Tensor, "batch views tokens features"] = x
        for i, block_module in enumerate(self.blocks):
            blk = cast(Block, block_module)
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
                    assert pos_nodiff is not None
                    pos_nodiff = torch.cat([pos_nodiff, scale_pos], dim=2)

            # ---- Calib token injection (per-layer, multi-layer variant) ----
            # Append K tokens at the end of the sequence, dropped right after this
            # block. Must happen before g_pos/l_pos so RoPE positions stay aligned.
            K_layer = 0
            inject_mask = None
            calib_view_tokens = None
            if calib_active:
                assert self.calib_tokens is not None
                assert cam_types_BS is not None
                K_layer = self.calib_tokens.K
                calib_view_tokens, inject_mask = self.calib_tokens.build_view_calib_tokens(i, cam_types_BS)  # (B, S, K, C), (B, S) bool
                x = torch.cat([x, calib_view_tokens.to(x.dtype)], dim=2)  # (B, S, N+K, C)
                if pos is not None:
                    calib_pos = torch.zeros(B, S, K_layer, pos.shape[-1], device=pos.device, dtype=pos.dtype)
                    pos = torch.cat([pos, calib_pos], dim=2)
                    assert pos_nodiff is not None
                    pos_nodiff = torch.cat([pos_nodiff, calib_pos], dim=2)

            # Select RoPE positions (after scale + calib injection so dims match)
            if i < self.rope_start or self.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos = pos_nodiff
                l_pos = pos

            # Local vs global layer
            is_global_layer = self.alt_start != -1 and i >= self.alt_start and i % 2 == 1
            attn_type = "global" if is_global_layer else "local"

            # Build attention mask + distortion bias.
            # N_non_calib covers CLS + register + N_patches (+ scale if present).
            N_non_calib_per_view = x.shape[2] - K_layer  # x's N dim now includes calib
            attn_mask_combined = None  # final mask passed to the attention block

            # 1) calib mask
            if K_layer > 0:
                assert inject_mask is not None
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
                    H_p=H_p,
                    W_p=W_p,
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
                    assert self.distortion_bias is not None
                    assert J_tok is not None
                    assert valid_tok is not None
                    db = self.distortion_bias(d_tok, J_tok, valid_mask=valid_tok)  # (B', H, L, L)
                    attn_mask_combined = db if attn_mask_combined is None else db + attn_mask_combined.unsqueeze(1).to(db.dtype)

            # DWC bypass info; suffix must include the calib token slots
            dwc_suffix = (1 if scale_injected else 0) + K_layer
            dwc_info_local = (H_p, W_p, dwc_prefix, dwc_suffix, 1) if self.use_dwc else None
            dwc_info_global = (H_p, W_p, dwc_prefix, dwc_suffix, S) if self.use_dwc else None

            # Alternating attention
            if is_global_layer:
                # Cross-view attention
                x = self.process_attention(
                    x,
                    blk,
                    "global",
                    pos=g_pos,
                    attn_mask=attn_mask_combined,
                    dwc_info=dwc_info_global,
                )
            else:
                # Within-view self-attention
                x = self.process_attention(
                    x,
                    blk,
                    "local",
                    pos=l_pos,
                    dwc_info=dwc_info_local,
                    attn_mask=attn_mask_combined,
                )
                local_x = x  # keep local features for concatenation

            # Drop calib tokens (present only within this layer's attention)
            if K_layer > 0:
                x = x[:, :, :-K_layer, :]
                if pos is not None:
                    pos = pos[:, :, :-K_layer, :]
                    assert pos_nodiff is not None
                    pos_nodiff = pos_nodiff[:, :, :-K_layer, :]
                # keep the calib-dropped version of local_x for alignment
                if not is_global_layer:
                    local_x = x

            # Collect outputs from selected layers
            if i in blocks_to_take:
                out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
                output.append((out_x[:, :, 0], out_x))
                output_has_scale.append(scale_injected)
        return output, output_has_scale

    def process_attention(
        self,
        x: Float[Tensor, "batch views tokens features"],
        block: Block,
        attn_type: AttentionType = "global",
        pos: Float[Tensor, "batch views tokens 2"] | Int64[Tensor, "batch views tokens 2"] | None = None,
        attn_mask: Float[Tensor, "packed_batch ... tokens tokens"] | Bool[Tensor, "packed_batch ... tokens tokens"] | None = None,
        dwc_info: DwcInfo | None = None,
    ) -> Float[Tensor, "batch views tokens features"]:
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

        Local reshapes (B, S, N, C) to (B*S, N, C) so each view attends independently.
        Global reshapes to (B, S*N, C) so all views attend jointly.
        """
        b, s, n = x.shape[:3]

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

        x = block(x, pos=pos, attn_mask=attn_mask, dwc_info=dwc_info)

        x = rearrange(x, "(b s) n c -> b s n c", b=b, s=s) if attn_type == "local" else rearrange(x, "b (s n) c -> b s n c", b=b, s=s)
        return x

    def get_intermediate_layers(
        self,
        x: Float[Tensor, "batch views 3 height width"],
        n: int | Sequence[int] = 1,
        *,
        ray_feat: Float[Tensor, "batch views features patch_height patch_width"] | None = None,
        d_cam: Float[Tensor, "batch views _channels _height _width"] | None = None,
        cam_types: Int64[Tensor, "batch views"] | None = None,
    ) -> IntermediateOutput:
        """Get intermediate-layer features.

        Args:
            x: Input images (B, S, 3, H, W).
            n: Layer indices to extract features from.

        Returns:
            ``(features, scale_tokens)``. Each feature is ``(patch_tokens, cls_tokens)``,
            where the second item is the token at the CLS position.
        """
        outputs, output_has_scale = self._get_intermediate_layers_not_chunked(
            x,
            n,
            ray_feat=ray_feat,
            d_cam=d_cam,
            cam_types=cam_types,
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

        # Remove CLS/register tokens; also drop the trailing scale token if present
        prefix = 1 + self.num_register_tokens
        cleaned_outputs = []
        for idx, out in enumerate(raw_outputs):
            if output_has_scale[idx]:
                cleaned_outputs.append(out[..., prefix:-1, :])  # drop leading CLS/reg + trailing scale
            else:
                cleaned_outputs.append(out[..., prefix:, :])  # drop leading CLS/reg only
        return tuple(zip(cleaned_outputs, cls_tokens, strict=True)), scale_tokens
