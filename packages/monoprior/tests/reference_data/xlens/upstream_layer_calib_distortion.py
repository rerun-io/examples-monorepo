# Calibration Tokens + Distortion Bias modules for heterogeneous multi-camera attention.
#
# Two independent modules, enabled via config in Stage 3:
#
# 1) CalibrationTokens
#    Multi-layer variant of "Extending Foundational Monocular Depth Estimators
#    to Fisheye Cameras with Calibration Tokens" (ICCV 2025). K learnable tokens
#    per camera type, independent per layer (L layers * T types * K tokens).
#    Injected into the sequence before each layer's attention and dropped after.
#    An attention mask blocks calib tokens from tokens they must not attend to.
#
# 2) DistortionBias
#    Per-patch Jacobian descriptor (finite differences over the d_cam ray map).
#    phi_ij = concat(d_i·d_j, log s_i - log s_j, d_i - d_j, rel_rot(J_i, J_j));
#    MLP(phi_ij) -> (B, H, N, N) bias added to attention logits. All global
#    attention layers share one MLP.

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# Calibration Tokens


class CalibrationTokens(nn.Module):
    """K learnable tokens per camera type, independent per layer.

    Parameters:
        depth: number of transformer layers L. Each layer has its own set.
        n_cam_types: number of camera types T (e.g. fisheye=0, pinhole=1).
        tokens_per_type: number of tokens per type K.
        embed_dim: token dimension C.
        inject_types: camera types actually injected. Default [0] (fisheye only);
            views of other types take the original path.

    Parameter shape: (L, T, K, C), zero-initialized so calib tokens start as
    constants and do not perturb existing representations, keeping Stage 1/2
    checkpoints compatible.
    """

    def __init__(
        self,
        depth: int,
        n_cam_types: int,
        tokens_per_type: int,
        embed_dim: int,
        inject_types: Sequence[int] = (0,),
    ) -> None:
        super().__init__()
        assert depth > 0 and n_cam_types > 0 and tokens_per_type > 0
        self.depth = depth
        self.n_cam_types = n_cam_types
        self.K = tokens_per_type
        self.embed_dim = embed_dim
        self.inject_types = tuple(int(t) for t in inject_types)
        for t in self.inject_types:
            assert 0 <= t < n_cam_types, f"inject_type {t} out of range [0, {n_cam_types})"

        # (L, T, K, C), zero-initialized.
        self.tokens = nn.Parameter(
            torch.zeros(depth, n_cam_types, tokens_per_type, embed_dim)
        )

    def needs_inject(self, cam_types_BS: torch.Tensor) -> bool:
        """Whether any view in the batch needs calib token injection."""
        if cam_types_BS is None or not self.inject_types:
            return False
        for t in self.inject_types:
            if (cam_types_BS == t).any():
                return True
        return False

    def get_layer_tokens(self, layer_idx: int) -> torch.Tensor:
        """Tokens for all camera types at a layer, shape (T, K, C)."""
        return self.tokens[layer_idx]

    def build_view_calib_tokens(
        self,
        layer_idx: int,
        cam_types_BS: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-view calib tokens and injection mask from (B, S) cam_types.

        Returns:
            view_tokens: (B, S, K, C) tokens for each view; views not in
                inject_types still get their (zero) tokens, gated by the mask.
            inject_mask: (B, S) bool — True where the view should inject K tokens.
        """
        B, S = cam_types_BS.shape
        T_K_C = self.tokens[layer_idx]                            # (T, K, C)
        view_tokens = T_K_C[cam_types_BS]                         # (B, S, K, C)

        inject_mask = torch.zeros_like(cam_types_BS, dtype=torch.bool)
        for t in self.inject_types:
            inject_mask = inject_mask | (cam_types_BS == t)
        return view_tokens, inject_mask


def build_calib_attention_mask(
    K_per_view: int,
    inject_mask: torch.Tensor,    # (B, S) bool, whether each view actually injected calib tokens
    N_non_calib: int,             # non-calib length per view (CLS + register + patches [+ scale])
    attn_type: str,               # "local" or "global"
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    """Build the additive attention mask for calib tokens (-inf blocks).

    Sequence layout (calib tokens at the end of each view, so CLS/register/scale
    keep fixed indices):
        [CLS, register..., patches..., (scale,) calib_1 ... calib_K]
        total (N_non_calib + K) tokens.

    Views that did not inject (placeholders) still reserve K slots, fully blocked
    (rows and columns) so they behave as if absent.

    Rules:
      - Injected calib tokens attend only to non-calib tokens of their own view.
      - Injected calib tokens attend only to calib tokens of their own view (no cross-view).
      - Placeholder calib tokens are fully isolated.
      - Non-calib tokens attend across all views: mask = 0.
      - Non-calib -> calib: only their own view's injected calib tokens.

    Args:
        attn_type: "local" => B' = B*S, length L = N_non_calib + K.
                   "global" => B' = B, length L = S * (N_non_calib + K).

    Returns:
        mask: (B', L, L) additive float mask. 0 = allow, NEG_INF = block.
              Returns None if K_per_view == 0 or no view injected.
    """
    if K_per_view <= 0:
        return None
    if not inject_mask.any():
        return None

    B, S = inject_mask.shape
    L_per_view = N_non_calib + K_per_view
    NEG_INF = torch.finfo(dtype).min

    if attn_type == "local":
        # One mask per view.
        Bp = B * S
        L = L_per_view
        mask = torch.zeros(Bp, L, L, device=device, dtype=dtype)

        # Placeholder views: calib at [N_non_calib, L), block whole rows/columns.
        not_inject = (~inject_mask).reshape(Bp)
        if not_inject.any():
            mask[not_inject, N_non_calib:, :] = NEG_INF
            mask[not_inject, :, N_non_calib:] = NEG_INF
        return mask

    elif attn_type == "global":
        # Full sequence L = S * L_per_view.
        L = S * L_per_view
        mask = torch.zeros(B, L, L, device=device, dtype=dtype)

        # view v occupies [v*L_per_view, (v+1)*L_per_view)
        # view v calib   = [v*L_per_view + N_non_calib, (v+1)*L_per_view)
        # view v patches = [v*L_per_view, v*L_per_view + N_non_calib)
        for b in range(B):
            for v in range(S):
                v_start = v * L_per_view
                calib_lo = v_start + N_non_calib
                calib_hi = (v + 1) * L_per_view

                if not inject_mask[b, v]:
                    # Placeholder view: calib fully isolated.
                    mask[b, calib_lo:calib_hi, :] = NEG_INF
                    mask[b, :, calib_lo:calib_hi] = NEG_INF
                else:
                    # Injected view: calib attends only within its own view.
                    # Block calib_v against every other view's tokens.
                    for v_other in range(S):
                        if v_other == v:
                            continue
                        o_start = v_other * L_per_view
                        o_end = (v_other + 1) * L_per_view
                        mask[b, calib_lo:calib_hi, o_start:o_end] = NEG_INF
                        mask[b, o_start:o_end, calib_lo:calib_hi] = NEG_INF
        return mask

    else:
        raise ValueError(f"unknown attn_type: {attn_type}")


# Distortion Bias (Jacobian-based geometric attention bias)


def jacobian_from_dcam(d_patch_BSN3: torch.Tensor, H_p: int, W_p: int) -> torch.Tensor:
    """Per-patch Jacobian J in R^(3x2) from the patch-grid ray directions d.

    Finite-difference approximation of dd/du (width) and dd/dv (height);
    boundary patches use one-sided differences.

    Args:
        d_patch_BSN3: (B, S, H_p * W_p, 3) — unit direction at each patch center.
        H_p, W_p: patch grid size.

    Returns:
        J: (B, S, H_p * W_p, 3, 2) — per-patch Jacobian, col 0 = dd/du, col 1 = dd/dv.
    """
    B, S, N, _ = d_patch_BSN3.shape
    assert N == H_p * W_p, f"N={N} != H_p*W_p={H_p*W_p}"
    d = d_patch_BSN3.view(B, S, H_p, W_p, 3)

    # dd/dv: difference along H (forward, last row uses backward).
    dv = torch.zeros_like(d)
    dv[:, :, :-1] = d[:, :, 1:] - d[:, :, :-1]
    dv[:, :, -1] = dv[:, :, -2]

    # dd/du: difference along W.
    du = torch.zeros_like(d)
    du[:, :, :, :-1] = d[:, :, :, 1:] - d[:, :, :, :-1]
    du[:, :, :, -1] = du[:, :, :, -2]

    # J shape: (B, S, H_p, W_p, 3, 2), col 0 = du, col 1 = dv
    J = torch.stack([du, dv], dim=-1)                # (B, S, H_p, W_p, 3, 2)
    return J.view(B, S, N, 3, 2)


class DistortionBias(nn.Module):
    """Pairwise attention bias from Jacobian descriptors.

    Input: per-token d_i in S^2 (unit direction) and J_i in R^(3x2) (Jacobian).
    Output: bias in (B, num_heads, L, L), added to attention logits.

    Features:
        phi_ij = concat(
            d_i · d_j,              (1,)
            log s_i - log s_j,      (1,)
            d_i - d_j,              (3,)
            vec(J_i^T J_j),         (4,)
        )                            => 9 dims

    MLP(phi_ij) -> num_heads bias. All global layers share one MLP instance.
    Last layer zero-initialized so the initial bias is 0.
    """

    PHI_DIM = 1 + 1 + 3 + 4  # = 9

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int = 64,
        chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        # Row-chunk size. Pairwise (L,L) peaks at O(L^2); with 6-view global,
        # L ~= 12.5k, which OOMs. Chunking rows plus gradient checkpointing caps
        # the peak at O(chunk*L). 0 disables chunking.
        self.chunk_size = int(chunk_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.PHI_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )
        # Zero-init last layer -> bias output is 0, identity to attention initially.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _bias_rows(
        self,
        d_i: torch.Tensor,      # (B, c, 3)    row-chunk d
        log_s_i: torch.Tensor,  # (B, c)       row-chunk log||J||
        Jt_i: torch.Tensor,     # (B, c, 2, 3) row-chunk J^T
        d: torch.Tensor,        # (B, L, 3)    all columns
        log_s: torch.Tensor,    # (B, L)
        J: torch.Tensor,        # (B, L, 3, 2)
    ) -> torch.Tensor:
        """Bias for one row chunk: (B, H, c, L). Same formula as the full
        computation, restricted to the chunk's rows to cut peak memory."""
        dot = torch.einsum("bid,bjd->bij", d_i, d).unsqueeze(-1)             # (B,c,L,1)
        d_diff = d_i.unsqueeze(2) - d.unsqueeze(1)                           # (B,c,L,3)
        log_s_diff = (log_s_i.unsqueeze(2) - log_s.unsqueeze(1)).unsqueeze(-1)  # (B,c,L,1)
        rel = torch.einsum("bikl,bjlm->bijkm", Jt_i, J)                      # (B,c,L,2,2)
        B, c, L = dot.shape[0], dot.shape[1], dot.shape[2]
        rel_flat = rel.reshape(B, c, L, 4)
        phi = torch.cat([dot, log_s_diff, d_diff, rel_flat], dim=-1)         # (B,c,L,9)
        b = self.mlp(phi)                                                    # (B,c,L,H)
        return b.permute(0, 3, 1, 2)                                         # (B,H,c,L)

    def forward(
        self,
        d: torch.Tensor,           # (B, L, 3)
        J: torch.Tensor,           # (B, L, 3, 2)
        valid_mask: Optional[torch.Tensor] = None,  # (B, L) bool — bias 0 where False
    ) -> torch.Tensor:
        """Return bias (B, num_heads, L, L), computed in row chunks to bound
        memory. During training each chunk uses gradient checkpointing."""
        # Non-geometric tokens (calib / CLS / register / scale) are marked by
        # valid_mask. At those positions set d/J to 0 and force bias to 0.
        if valid_mask is not None:
            m = valid_mask.unsqueeze(-1).to(d.dtype)
            d = d * m
            J = J * m.unsqueeze(-1)

        B, L, _ = d.shape
        # Precompute per-token quantities once, reused across chunks.
        s = torch.linalg.norm(J.reshape(B, L, 6), dim=-1).clamp(min=1e-6)
        log_s = torch.log(s)                                   # (B, L)
        Jt = J.transpose(-1, -2).contiguous()                  # (B, L, 2, 3)

        cs = self.chunk_size if self.chunk_size > 0 else L
        use_ckpt = self.training and torch.is_grad_enabled() and cs < L
        rows = []
        for i0 in range(0, L, cs):
            i1 = min(i0 + cs, L)
            args = (d[:, i0:i1], log_s[:, i0:i1], Jt[:, i0:i1], d, log_s, J)
            if use_ckpt:
                b = checkpoint(self._bias_rows, *args, use_reentrant=False)
            else:
                b = self._bias_rows(*args)
            rows.append(b)
        bias = torch.cat(rows, dim=2)                          # (B, H, L, L)

        if valid_mask is not None:
            # Zero bias where either end is invalid.
            row_valid = valid_mask.unsqueeze(1).unsqueeze(-1)   # (B, 1, L, 1)
            col_valid = valid_mask.unsqueeze(1).unsqueeze(-2)   # (B, 1, 1, L)
            bias = bias * (row_valid & col_valid).to(bias.dtype)
        return bias


def build_token_geometry(
    d_cam: Optional[torch.Tensor],      # (B, S, 3, H, W) or (B, S, H, W, 3) or None
    H_p: int,
    W_p: int,
    prefix_len: int,                    # CLS + register, before patches
    suffix_len_no_calib: int,           # tokens after patches but before calib (scale)
    K_calib: int,                       # trailing calib tokens per view; 0 if none injected
    n_views: int,                       # S
    batch_size: int,                    # B
    device: torch.device,
    dtype: torch.dtype,
    attn_type: str = "global",
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Build (d, J, valid_mask) aligned to the attention sequence from d_cam.

    Sequence layout (per view, L_per_view = prefix_len + N_patch + suffix_len_no_calib + K_calib):
        [CLS, register..., patches (H_p*W_p), (scale,) calib_1..K]

    Returns all None when d_cam is None (caller should skip distortion bias).

    Returns:
        d_tokens:    (B', L, 3) or None
        J_tokens:    (B', L, 3, 2) or None
        valid_mask:  (B', L) bool, True where the token is a patch (has geometry)
        B' = B*S (local) or B (global), L = L_per_view or S*L_per_view
    """
    N_patch = H_p * W_p
    L_per_view = prefix_len + N_patch + suffix_len_no_calib + K_calib

    if d_cam is None:
        return None, None, None

    # Normalize d_cam to (B, S, 3, H, W).
    if d_cam.dim() == 5 and d_cam.shape[-1] == 3 and d_cam.shape[2] != 3:
        d_cam = d_cam.permute(0, 1, 4, 2, 3).contiguous()
    assert d_cam.shape[:2] == (batch_size, n_views), (
        f"d_cam shape {d_cam.shape} does not match (B={batch_size}, S={n_views})"
    )
    B, S, _, H, W = d_cam.shape

    # 1) Downsample to the patch grid.
    d_bs = d_cam.reshape(B * S, 3, H, W)
    pool_kh = max(1, H // H_p)
    pool_kw = max(1, W // W_p)
    d_p = F.avg_pool2d(d_bs, kernel_size=(pool_kh, pool_kw))            # (BS, 3, H_p, W_p)
    d_p = F.normalize(d_p, dim=1, eps=1e-6)
    d_p_flat = d_p.permute(0, 2, 3, 1).reshape(B, S, N_patch, 3)

    # 2) Jacobian.
    J = jacobian_from_dcam(d_p_flat, H_p, W_p)                          # (B, S, N_patch, 3, 2)

    # 3) Assemble the full sequence; non-patch positions are 0.
    d_tokens = torch.zeros(B, S, L_per_view, 3, device=device, dtype=dtype)
    J_tokens = torch.zeros(B, S, L_per_view, 3, 2, device=device, dtype=dtype)
    valid_mask = torch.zeros(B, S, L_per_view, device=device, dtype=torch.bool)

    patch_lo = prefix_len
    patch_hi = patch_lo + N_patch
    d_tokens[:, :, patch_lo:patch_hi] = d_p_flat.to(dtype)
    J_tokens[:, :, patch_lo:patch_hi] = J.to(dtype)
    valid_mask[:, :, patch_lo:patch_hi] = True

    # 4) Reshape.
    if attn_type == "local":
        d_tokens = d_tokens.reshape(B * S, L_per_view, 3)
        J_tokens = J_tokens.reshape(B * S, L_per_view, 3, 2)
        valid_mask = valid_mask.reshape(B * S, L_per_view)
    elif attn_type == "global":
        d_tokens = d_tokens.reshape(B, S * L_per_view, 3)
        J_tokens = J_tokens.reshape(B, S * L_per_view, 3, 2)
        valid_mask = valid_mask.reshape(B, S * L_per_view)
    else:
        raise ValueError(f"unknown attn_type: {attn_type}")

    return d_tokens, J_tokens, valid_mask
