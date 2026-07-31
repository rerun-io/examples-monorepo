# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0.

import math
from collections.abc import Callable, Sequence
from functools import partial
from typing import Literal, TypeAlias

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn.init import trunc_normal_

from monopriors.third_party.dinov2.layers import Block, PatchEmbed

FeatureTensor: TypeAlias = Float[Tensor, "*shape"]
IntermediateLayer: TypeAlias = tuple[Float[Tensor, "b n c"], Float[Tensor, "b c"]]
IntermediateLayers: TypeAlias = tuple[FeatureTensor, ...] | tuple[IntermediateLayer, ...]
ModelSize: TypeAlias = Literal["vits", "vitb", "vitl"]


def named_apply(
    fn: Callable[..., None],
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    """Apply a named callback recursively to a module tree.

    Args:
        fn: Callback accepting ``module`` and ``name`` keyword arguments.
        module: Root Torch module.
        name: Dotted name assigned to the root module.
        depth_first: Whether children are visited before their parent.
        include_root: Whether to apply the callback to this root.

    Returns:
        The unchanged root module.
    """
    if not depth_first and include_root:
        fn(module=module, name=name)
    for raw_child_name, child_module in module.named_children():
        child_name: str = ".".join((name, raw_child_name)) if name else raw_child_name
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

    def init_weights(self) -> None:
        """Initialize positional, class-token, and linear-layer weights."""
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def _interpolate_float_grid_pos_encoding(
        self,
        x_bnc: Float[Tensor, "b n c"],
        first_spatial_size: int,
        second_spatial_size: int,
    ) -> Float[Tensor, "1 n c"]:
        """Reproduce the Depth Anything V2 positional interpolation path."""
        previous_dtype: torch.dtype = x_bnc.dtype
        patch_count: int = x_bnc.shape[1] - 1
        pretrained_patch_count: int = self.pos_embed.shape[1] - 1
        if patch_count == pretrained_patch_count and first_spatial_size == second_spatial_size:
            return self.pos_embed
        pos_embed_1nc: Float[Tensor, "1 n c"] = self.pos_embed.float()
        class_pos_embed_1c: Float[Tensor, "1 c"] = pos_embed_1nc[:, 0]
        patch_pos_embed_1nc: Float[Tensor, "1 n c"] = pos_embed_1nc[:, 1:]
        channels: int = x_bnc.shape[-1]
        first_grid_size: int = first_spatial_size // self.patch_size
        second_grid_size: int = second_spatial_size // self.patch_size
        offset_first_grid_size: float = first_grid_size + self.interpolate_offset
        offset_second_grid_size: float = second_grid_size + self.interpolate_offset

        pretrained_grid_size: float = math.sqrt(pretrained_patch_count)
        first_scale: float = float(offset_first_grid_size) / pretrained_grid_size
        second_scale: float = float(offset_second_grid_size) / pretrained_grid_size
        patch_pos_embed_1chw: Float[Tensor, "1 c grid_h grid_w"] = nn.functional.interpolate(
            patch_pos_embed_1nc.reshape(
                1,
                int(pretrained_grid_size),
                int(pretrained_grid_size),
                channels,
            ).permute(0, 3, 1, 2),
            scale_factor=(first_scale, second_scale),
            mode="bicubic",
            antialias=self.interpolate_antialias,
        )

        assert int(offset_first_grid_size) == patch_pos_embed_1chw.shape[-2]
        assert int(offset_second_grid_size) == patch_pos_embed_1chw.shape[-1]
        patch_pos_embed_1nc = patch_pos_embed_1chw.permute(0, 2, 3, 1).view(1, -1, channels)
        output_1nc: Float[Tensor, "1 n c"] = torch.cat((class_pos_embed_1c.unsqueeze(0), patch_pos_embed_1nc), dim=1).to(previous_dtype)
        return output_1nc

    def _interpolate_integer_grid_pos_encoding(
        self,
        x_bnc: Float[Tensor, "b n c"],
        height: int,
        width: int,
    ) -> Float[Tensor, "1 n c"]:
        """Reproduce the MoGe positional interpolation path."""
        previous_dtype: torch.dtype = x_bnc.dtype
        patch_count: int = x_bnc.shape[1] - 1
        pretrained_patch_count: int = self.pos_embed.shape[1] - 1
        if patch_count == pretrained_patch_count and width == height:
            return self.pos_embed
        pos_embed_1nc: Float[Tensor, "1 n c"] = self.pos_embed.float()
        class_pos_embed_1c: Float[Tensor, "1 c"] = pos_embed_1nc[:, 0, :]
        patch_pos_embed_1nc: Float[Tensor, "1 n c"] = pos_embed_1nc[:, 1:, :]
        channels: int = x_bnc.shape[-1]
        grid_height: int = height // self.patch_size
        grid_width: int = width // self.patch_size
        pretrained_grid_size: int = int(math.sqrt(pretrained_patch_count))
        assert pretrained_patch_count == pretrained_grid_size * pretrained_grid_size
        interpolation_args: dict[str, tuple[float, float] | tuple[int, int]] = {}
        if self.interpolate_offset > 0:
            scale_x: float = float(grid_width + self.interpolate_offset) / pretrained_grid_size
            scale_y: float = float(grid_height + self.interpolate_offset) / pretrained_grid_size
            interpolation_args["scale_factor"] = (scale_y, scale_x)
        else:
            interpolation_args["size"] = (grid_height, grid_width)

        patch_pos_embed_1chw: Float[Tensor, "1 c grid_h grid_w"] = nn.functional.interpolate(
            patch_pos_embed_1nc.reshape(
                1,
                pretrained_grid_size,
                pretrained_grid_size,
                channels,
            ).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **interpolation_args,
        )

        assert (grid_height, grid_width) == patch_pos_embed_1chw.shape[-2:]
        patch_pos_embed_1nc = patch_pos_embed_1chw.permute(0, 2, 3, 1).flatten(1, 2)
        output_1nc: Float[Tensor, "1 n c"] = torch.cat(
            (class_pos_embed_1c[:, None, :].expand(patch_pos_embed_1nc.shape[0], -1, -1), patch_pos_embed_1nc),
            dim=1,
        ).to(previous_dtype)
        return output_1nc

    def interpolate_pos_encoding(
        self,
        x_bnc: Float[Tensor, "b n c"],
        height: int,
        width: int,
    ) -> Float[Tensor, "1 n c"]:
        """Interpolate learned positional encodings for an input token grid.

        Args:
            x_bnc: Float token tensor shaped ``b n c``.
            height: Source image height in pixels.
            width: Source image width in pixels.

        Returns:
            Float positional encoding tensor shaped ``1 n c``.
        """
        if self.use_integer_grid_interpolation:
            return self._interpolate_integer_grid_pos_encoding(x_bnc, height, width)
        return self._interpolate_float_grid_pos_encoding(x_bnc, height, width)

    def prepare_tokens(self, image_bchw: Float[Tensor, "b c h w"]) -> Float[Tensor, "b n embed"]:
        """Embed patches and add class and positional tokens.

        Args:
            image_bchw: Float image tensor shaped ``b c h w``.

        Returns:
            Float token tensor shaped ``b n embed``.
        """
        height: int = image_bchw.shape[-2]
        width: int = image_bchw.shape[-1]
        tokens_bne: Float[Tensor, "b n embed"] = self.patch_embed(image_bchw)
        tokens_bne = torch.cat((self.cls_token.expand(tokens_bne.shape[0], -1, -1), tokens_bne), dim=1)
        tokens_bne = tokens_bne + self.interpolate_pos_encoding(tokens_bne, height, width)
        return tokens_bne

    def get_intermediate_layers(
        self,
        image_bchw: Float[Tensor, "b c h w"],
        n: int | Sequence[int] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> IntermediateLayers:
        """Return selected transformer block outputs.

        Args:
            image_bchw: Float image tensor shaped ``b c h w``.
            n: Number of trailing blocks or explicit block indices to return.
            reshape: Whether patch tokens become float feature maps shaped ``b c patch_h patch_w``.
            return_class_token: Whether each output is paired with a float class token shaped ``b c``.
            norm: Whether to apply the final transformer normalization.

        Returns:
            Tuple of float token tensors shaped ``b n c`` or feature maps shaped ``b c patch_h patch_w``; optionally paired with class tokens.
        """
        batch_size: int = image_bchw.shape[0]
        height: int = image_bchw.shape[-2]
        width: int = image_bchw.shape[-1]
        tokens_bnc: Float[Tensor, "b n c"] = self.prepare_tokens(image_bchw)
        outputs: list[FeatureTensor] = []
        total_block_len: int = len(self.blocks)
        blocks_to_take: range | Sequence[int] = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for index, block in enumerate(self.blocks):
            tokens_bnc = block(tokens_bnc)
            if index in blocks_to_take:
                outputs.append(tokens_bnc)
        assert len(outputs) == len(blocks_to_take), f"only {len(outputs)} / {len(blocks_to_take)} blocks found"

        if norm:
            outputs = [self.norm(output) for output in outputs]
        class_tokens: list[Float[Tensor, "b c"]] = [output[:, 0] for output in outputs]
        outputs = [output[:, 1:] for output in outputs]
        if reshape:
            outputs = [
                output.reshape(batch_size, height // self.patch_size, width // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for output in outputs
            ]
        if return_class_token:
            paired_outputs: tuple[IntermediateLayer, ...] = tuple(zip(outputs, class_tokens, strict=True))
            return paired_outputs
        return tuple(outputs)


def init_weights_vit_timm(module: nn.Module, name: str = "") -> None:
    """Initialize a linear module with the reproducible timm ViT scheme.

    Args:
        module: Torch module visited by ``named_apply``.
        name: Dotted module name supplied by ``named_apply``.
    """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _make_model(
    model_name: ModelSize,
    *,
    use_sdpa: bool,
    use_integer_grid_interpolation: bool,
) -> DinoVisionTransformer:
    architectures: dict[str, tuple[int, int, int]] = {
        "vits": (384, 12, 6),
        "vitb": (768, 12, 12),
        "vitl": (1024, 24, 16),
    }
    architecture: tuple[int, int, int] = architectures[model_name]
    embed_dim: int = architecture[0]
    depth: int = architecture[1]
    num_heads: int = architecture[2]
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


def DINOv2(model_name: ModelSize) -> DinoVisionTransformer:
    """Build the manual-attention variant used by Depth Anything V2."""
    return _make_model(model_name, use_sdpa=False, use_integer_grid_interpolation=False)


def _make_moge_model(model_name: ModelSize, *, pretrained: bool) -> DinoVisionTransformer:
    if pretrained:
        raise ValueError("Pretrained DINOv2 downloads were removed; load a consumer checkpoint instead")
    return _make_model(model_name, use_sdpa=True, use_integer_grid_interpolation=True)


def dinov2_vits14(*, pretrained: bool = False) -> DinoVisionTransformer:
    """Build the DINOv2 ViT-S/14 variant used by MoGe."""
    return _make_moge_model("vits", pretrained=pretrained)


def dinov2_vitb14(*, pretrained: bool = False) -> DinoVisionTransformer:
    """Build the DINOv2 ViT-B/14 variant used by MoGe."""
    return _make_moge_model("vitb", pretrained=pretrained)


def dinov2_vitl14(*, pretrained: bool = False) -> DinoVisionTransformer:
    """Build the DINOv2 ViT-L/14 variant used by MoGe."""
    return _make_moge_model("vitl", pretrained=pretrained)
