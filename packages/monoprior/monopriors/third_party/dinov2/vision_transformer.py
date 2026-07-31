# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0.

import math
from collections.abc import Callable, Sequence
from functools import partial
from typing import Literal, TypeAlias

import torch
from torch import Tensor, nn
from torch.nn.init import trunc_normal_

from monopriors.third_party.dinov2.layers import Block, PatchEmbed

IntermediateLayer: TypeAlias = tuple[Tensor, Tensor]
IntermediateLayers: TypeAlias = tuple[Tensor, ...] | tuple[IntermediateLayer, ...]


def named_apply(
    fn: Callable[..., None],
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class DinoVisionTransformer(nn.Module):
    """Inference-only DINOv2 vision transformer used by MoGe and Depth Anything V2."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        drop_path_rate: float = 0.0,
        init_values: float | None = None,
        ffn_layer: Literal["mlp"] = "mlp",
        block_chunks: Literal[0] = 0,
        num_register_tokens: Literal[0] = 0,
        interpolate_antialias: bool = False,
        interpolate_offset: float = 0.1,
        *,
        use_sdpa: bool,
        use_integer_grid_interpolation: bool,
    ) -> None:
        super().__init__()
        if drop_path_rate != 0.0:
            raise ValueError("The inference-only DINOv2 transformer requires drop_path_rate=0.0")
        if ffn_layer != "mlp":
            raise ValueError("The inference-only DINOv2 transformer supports only the MLP feed-forward layer")
        if block_chunks != 0:
            raise ValueError("The inference-only DINOv2 transformer does not support block chunks")
        if num_register_tokens != 0:
            raise ValueError("The inference-only DINOv2 transformer does not support register tokens")

        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6)
        self.num_features: int = embed_dim
        self.embed_dim: int = embed_dim
        self.num_tokens: int = 1
        self.n_blocks: int = depth
        self.num_heads: int = num_heads
        self.patch_size: int = patch_size
        self.interpolate_antialias: bool = interpolate_antialias
        self.interpolate_offset: float = interpolate_offset
        self.use_integer_grid_interpolation: bool = use_integer_grid_interpolation

        self.patch_embed: PatchEmbed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches: int = self.patch_embed.num_patches

        self.cls_token: nn.Parameter = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed: nn.Parameter = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.blocks: nn.ModuleList = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop_path=0.0,
                    norm_layer=norm_layer,
                    init_values=init_values,
                    use_sdpa=use_sdpa,
                )
                for _ in range(depth)
            ]
        )
        self.norm: nn.Module = norm_layer(embed_dim)
        self.mask_token: nn.Parameter = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def _interpolate_float_grid_pos_encoding(self, x: Tensor, w: int, h: int) -> Tensor:
        """Reproduce the Depth Anything V2 positional interpolation path."""
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
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset

        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            mode="bicubic",
            antialias=self.interpolate_antialias,
        )

        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def _interpolate_integer_grid_pos_encoding(self, x: Tensor, h: int, w: int) -> Tensor:
        """Reproduce the MoGe positional interpolation path."""
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0, :]
        patch_pos_embed = pos_embed[:, 1:, :]
        dim = x.shape[-1]
        h0, w0 = h // self.patch_size, w // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M
        kwargs: dict[str, tuple[float, float] | tuple[int, int]] = {}
        if self.interpolate_offset > 0:
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sy, sx)
        else:
            kwargs["size"] = (h0, w0)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )

        assert (h0, w0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
        return torch.cat((class_pos_embed[:, None, :].expand(patch_pos_embed.shape[0], -1, -1), patch_pos_embed), dim=1).to(previous_dtype)

    def interpolate_pos_encoding(self, x: Tensor, height: int, width: int) -> Tensor:
        if self.use_integer_grid_interpolation:
            return self._interpolate_integer_grid_pos_encoding(x, height, width)
        return self._interpolate_float_grid_pos_encoding(x, height, width)

    def prepare_tokens(self, x: Tensor) -> Tensor:
        height: int = x.shape[-2]
        width: int = x.shape[-1]
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, height, width)
        return x

    def get_intermediate_layers(
        self,
        x: Tensor,
        n: int | Sequence[int] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> IntermediateLayers:
        batch_size: int = x.shape[0]
        height: int = x.shape[-2]
        width: int = x.shape[-1]
        x = self.prepare_tokens(x)
        outputs: list[Tensor] = []
        total_block_len: int = len(self.blocks)
        blocks_to_take: range | Sequence[int] = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index in blocks_to_take:
                outputs.append(x)
        assert len(outputs) == len(blocks_to_take), f"only {len(outputs)} / {len(blocks_to_take)} blocks found"

        if norm:
            outputs = [self.norm(output) for output in outputs]
        class_tokens: list[Tensor] = [output[:, 0] for output in outputs]
        outputs = [output[:, 1:] for output in outputs]
        if reshape:
            outputs = [
                output.reshape(batch_size, height // self.patch_size, width // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for output in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens, strict=True))
        return tuple(outputs)


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _make_model(
    model_name: Literal["vits", "vitb", "vitl"],
    *,
    use_sdpa: bool,
    use_integer_grid_interpolation: bool,
) -> DinoVisionTransformer:
    architectures: dict[str, tuple[int, int, int]] = {
        "vits": (384, 12, 6),
        "vitb": (768, 12, 12),
        "vitl": (1024, 24, 16),
    }
    embed_dim: int
    depth: int
    num_heads: int
    embed_dim, depth, num_heads = architectures[model_name]
    return DinoVisionTransformer(
        img_size=518,
        patch_size=14,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        init_values=1.0,
        ffn_layer="mlp",
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        use_sdpa=use_sdpa,
        use_integer_grid_interpolation=use_integer_grid_interpolation,
    )


def DINOv2(model_name: Literal["vits", "vitb", "vitl"]) -> DinoVisionTransformer:
    """Build the manual-attention variant used by Depth Anything V2."""
    return _make_model(model_name, use_sdpa=False, use_integer_grid_interpolation=False)


def _make_moge_model(model_name: Literal["vits", "vitb", "vitl"], *, pretrained: bool) -> DinoVisionTransformer:
    if pretrained:
        raise ValueError("Pretrained DINOv2 downloads were removed; load a consumer checkpoint instead")
    return _make_model(model_name, use_sdpa=True, use_integer_grid_interpolation=True)


def dinov2_vits14(*, pretrained: bool = False) -> DinoVisionTransformer:
    return _make_moge_model("vits", pretrained=pretrained)


def dinov2_vitb14(*, pretrained: bool = False) -> DinoVisionTransformer:
    return _make_moge_model("vitb", pretrained=pretrained)


def dinov2_vitl14(*, pretrained: bool = False) -> DinoVisionTransformer:
    return _make_moge_model("vitl", pretrained=pretrained)
