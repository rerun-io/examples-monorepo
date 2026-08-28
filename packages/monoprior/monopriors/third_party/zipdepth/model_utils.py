from typing import TypeAlias, cast

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
            prefix: str
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        return key

    stripped_state_dict: StateDict = {}
    key: str
    value: StateValue
    for key, value in state_dict.items():
        stripped_state_dict[_clean(key)] = value
    return stripped_state_dict


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Return one convolution equivalent to a consecutive convolution and batch normalization."""
    device: torch.device = conv.weight.device
    kernel_size: tuple[int, int] = cast(tuple[int, int], conv.kernel_size)
    stride: tuple[int, int] = cast(tuple[int, int], conv.stride)
    padding: str | tuple[int, int] = cast(str | tuple[int, int], conv.padding)
    dilation: tuple[int, int] = cast(tuple[int, int], conv.dilation)
    fused: nn.Conv2d = cast(
        nn.Conv2d,
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=conv.groups,
            bias=True,
        ).to(device),
    )
    running_var: Float[torch.Tensor, "c_out"] = cast(Float[torch.Tensor, "c_out"], bn.running_var)
    running_mean: Float[torch.Tensor, "c_out"] = cast(Float[torch.Tensor, "c_out"], bn.running_mean)
    bn_weight: Float[torch.Tensor, "c_out"] = cast(Float[torch.Tensor, "c_out"], bn.weight)
    bn_bias: Float[torch.Tensor, "c_out"] = cast(Float[torch.Tensor, "c_out"], bn.bias)
    std: Float[torch.Tensor, "c_out"] = (running_var + bn.eps).sqrt()
    scale: Float[torch.Tensor, "c_out 1 1 1"] = (bn_weight / std).reshape(-1, 1, 1, 1)
    fused.weight.data = conv.weight.data.clone() * scale
    conv_bias: Float[torch.Tensor, "c_out"] = conv.bias.data if conv.bias is not None else torch.zeros(conv.out_channels, device=device)
    fused_bias: Float[torch.Tensor, "c_out"] = cast(Float[torch.Tensor, "c_out"], fused.bias)
    fused_bias.data = bn_bias + (conv_bias - running_mean) * bn_weight / std
    return fused


def fuse_remaining_conv_bn(model: nn.Module) -> int:
    """Fuse consecutive convolution and batch-normalization pairs.

    Args:
        model: Model whose sequential containers are modified in place.

    Returns:
        Number of fused pairs.
    """
    count: int = 0
    module: nn.Module
    for module in model.modules():
        if not isinstance(module, nn.Sequential):
            continue
        children: list[tuple[str, nn.Module]] = list(module.named_children())
        i: int
        for i in range(len(children) - 1):
            first_child: tuple[str, nn.Module] = children[i]
            first_name: str = first_child[0]
            first_module: nn.Module = first_child[1]
            second_child: tuple[str, nn.Module] = children[i + 1]
            second_name: str = second_child[0]
            second_module: nn.Module = second_child[1]
            if isinstance(first_module, nn.Conv2d) and isinstance(second_module, nn.BatchNorm2d):
                setattr(module, first_name, _fuse_conv_bn(first_module, second_module))
                setattr(module, second_name, nn.Identity())
                count += 1
    for module in model.modules():
        if isinstance(module, nn.Sequential):
            kept_children: list[tuple[str, nn.Module]] = []
            name: str
            child: nn.Module
            for name, child in module.named_children():
                if not isinstance(child, nn.Identity):
                    kept_children.append((name, child))
            if len(kept_children) < len(list(module.children())):
                current_children: list[tuple[str, nn.Module]] = list(module.named_children())
                current_name: str
                _current_module: nn.Module
                for current_name, _current_module in current_children:
                    delattr(module, current_name)
                kept_name: str
                kept_module: nn.Module
                for kept_name, kept_module in kept_children:
                    module.add_module(kept_name, kept_module)
    return count
