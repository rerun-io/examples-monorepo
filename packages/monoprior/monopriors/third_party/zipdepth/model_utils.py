"""Checkpoint cleanup and inference-fusion utilities for ZipDepth models."""

from typing import TypeAlias

import torch
from jaxtyping import Float, Num
from torch import nn

StateValue: TypeAlias = Num[torch.Tensor, "..."]
StateDict: TypeAlias = dict[str, StateValue]


def strip_state_dict_prefixes(state_dict: StateDict) -> StateDict:
    """Remove DDP and ``torch.compile`` key prefixes.

    Single-GPU training wraps the model in ``torch.compile``, whose ``state_dict()``
    prefixes every key with ``_orig_mod.``; DDP adds ``module.``. Both (in any order
    or nesting) are stripped so checkpoints load into a plain model.

    Args:
        state_dict: Mapping from parameter names to numeric tensors of arbitrary shape.

    Returns:
        A new dictionary with the same tensors and all leading ``module.`` and ``_orig_mod.`` prefixes removed.
    """
    prefixes: tuple[str, str] = ("module.", "_orig_mod.")

    def _clean(key: str) -> str:
        changed: bool = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        return key

    stripped_state_dict: StateDict = {_clean(key): value for key, value in state_dict.items()}
    return stripped_state_dict


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Return one convolution equivalent to a convolution and batch normalization.

    Args:
        conv: Convolution whose float weights have shape
            ``(output_channels, input_channels_per_group, kernel_height, kernel_width)``.
        bn: Batch normalization with one float statistic per output channel.

    Returns:
        A convolution with fused float weights of shape
        ``(output_channels, input_channels_per_group, kernel_height, kernel_width)`` and float bias
        of shape ``(output_channels,)``.
    """
    device: torch.device = conv.weight.device
    fused: nn.Conv2d = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        (conv.kernel_size[0], conv.kernel_size[1]),
        stride=(conv.stride[0], conv.stride[1]),
        padding=conv.padding if isinstance(conv.padding, str) else (conv.padding[0], conv.padding[1]),
        dilation=(conv.dilation[0], conv.dilation[1]),
        groups=conv.groups,
        bias=True,
    ).to(device)
    if bn.running_mean is None or bn.running_var is None:
        raise ValueError("BatchNorm without running statistics cannot be fused")
    if bn.weight is None or bn.bias is None:
        raise ValueError("BatchNorm without affine parameters cannot be fused")
    running_var: Float[torch.Tensor, "c_out"] = bn.running_var
    running_mean: Float[torch.Tensor, "c_out"] = bn.running_mean
    bn_weight: Float[torch.Tensor, "c_out"] = bn.weight
    bn_bias: Float[torch.Tensor, "c_out"] = bn.bias
    std: Float[torch.Tensor, "c_out"] = (running_var + bn.eps).sqrt()
    scale: Float[torch.Tensor, "c_out 1 1 1"] = (bn_weight / std).reshape(-1, 1, 1, 1)
    fused.weight.data = conv.weight.data.clone() * scale
    conv_bias: Float[torch.Tensor, "c_out"] = conv.bias.data if conv.bias is not None else torch.zeros(conv.out_channels, device=device)
    if fused.bias is None:
        raise RuntimeError("fused convolution unexpectedly has no bias")
    fused.bias.data = bn_bias + (conv_bias - running_mean) * bn_weight / std
    return fused


def fuse_remaining_conv_bn(model: nn.Module) -> int:
    """Fuse consecutive convolution and batch-normalization pairs.

    Args:
        model: Model whose sequential containers are modified in place.

    Returns:
        Number of fused pairs.
    """
    count: int = 0
    for module in model.modules():
        if not isinstance(module, nn.Sequential):
            continue
        children: list[tuple[str, nn.Module]] = list(module.named_children())
        for i in range(len(children) - 1):
            first_name, first_module = children[i]
            second_name, second_module = children[i + 1]
            if isinstance(first_module, nn.Conv2d) and isinstance(second_module, nn.BatchNorm2d):
                setattr(module, first_name, _fuse_conv_bn(first_module, second_module))
                setattr(module, second_name, nn.Identity())
                count += 1
    for module in model.modules():
        if isinstance(module, nn.Sequential):
            kept_children: list[tuple[str, nn.Module]] = [
                (name, child) for name, child in module.named_children() if not isinstance(child, nn.Identity)
            ]
            if len(kept_children) < len(list(module.children())):
                current_children: list[tuple[str, nn.Module]] = list(module.named_children())
                for current_name, _current_module in current_children:
                    delattr(module, current_name)
                for kept_name, kept_module in kept_children:
                    module.add_module(kept_name, kept_module)
    return count
