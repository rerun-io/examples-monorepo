"""MammaNet DETR-style landmark decoder — inference-only port.

Ports ``MammaNetDecoder`` + ``MLP`` from ``models_2d/mvhead.py`` and the
decoder half of ``models_2d/transformer_detr.py``. State-dict layout matches
the checkpoint (``query_embed.*``, ``decoder_detr.layers.N.*``,
``decoder_detr.norm.*``, ``landmarks.layers.N.*``, ``vis_prob/contact_prob/
floor_contact_prob``).
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    @staticmethod
    def _with_pos(tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        pos: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # normalize_before=False (forward_post) with gelu activation.
        q = k = self._with_pos(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.multihead_attn(
            query=self._with_pos(tgt, query_pos),
            key=self._with_pos(memory, pos),
            value=memory,
        )[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(F.gelu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: TransformerDecoderLayer, num_layers: int, norm: nn.LayerNorm) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        pos: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Inference only needs the final layer's normalized output (the
        # original returns all intermediates and indexes [-1] downstream).
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)
        return self.norm(output).unsqueeze(0)


class MLP(nn.Module):
    """Simple FFN head (relu between layers, linear last) — original ``MLP``."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        hidden: list[int] = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + hidden, hidden + [output_dim], strict=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MammaNetDecoder(nn.Module):
    """Landmark-query DETR decoder over ViT features."""

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 8,
        n_layers: int = 6,
        n_landmarks: int = 512,
        transformer_dim_feedforward: int = 2048,
        ldmks_dim: int = 2,
        dropout: float = 0.1,
        uncertainty: bool = True,
        visibility: bool = True,
        contact: bool = True,
        floor_contact: bool = True,
    ) -> None:
        super().__init__()
        self.output_dim: int = ldmks_dim + 1 if uncertainty else ldmks_dim
        self.visibility = visibility
        self.contact = contact
        self.floor_contact = floor_contact

        self.query_embed = nn.Embedding(n_landmarks, d_model)
        decoder_layer = TransformerDecoderLayer(d_model, n_heads, transformer_dim_feedforward, dropout)
        self.decoder_detr = TransformerDecoder(decoder_layer, n_layers, nn.LayerNorm(d_model))
        self.landmarks = MLP(d_model, d_model, self.output_dim, 3)
        if self.visibility:
            self.vis_prob = nn.Linear(d_model, 1)
        if self.contact:
            self.contact_prob = nn.Linear(d_model, 1)
        if self.floor_contact:
            self.floor_contact_prob = nn.Linear(d_model, 1)

    def forward(self, src: torch.Tensor, pos_embed: torch.Tensor) -> dict[str, torch.Tensor | None]:
        patch_pos_embed = pos_embed[:, 1:].permute(1, 0, 2)
        bs = src.shape[0]
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)
        src = rearrange(src, "B D H W -> (H W) B D")
        hs = self.decoder_detr(tgt, src, pos=patch_pos_embed, query_pos=query_embed)
        hs = hs.transpose(1, 2)
        return dict(
            joints2d=self.landmarks(hs)[-1],
            visibility=self.vis_prob(hs)[-1] if self.visibility else None,
            contact=self.contact_prob(hs)[-1] if self.contact else None,
            floor_contact=self.floor_contact_prob(hs)[-1] if self.floor_contact else None,
        )
