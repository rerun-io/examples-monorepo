"""The SAM-initialized DINOv3 encoder layout and its activation observer, shared by EgoHand and SAM 3D Body."""

from collections.abc import Callable

import torch
from dinov3.layers.attention import SelfAttention
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

DINO_BLOCKS: int = 32
"""Transformer blocks in the SAM-initialized DINOv3 encoder (EgoHand, SAM 3D Body); each has one native Attention op."""
LINEARS_PER_BLOCK: int = 5
"""Linears per block: attention qkv and proj, and the gated MLP's w1, w2 and w3."""
CALIBRATED_LINEARS: int = DINO_BLOCKS * LINEARS_PER_BLOCK
"""Linear inputs with a calibrated range."""
CALIBRATED_ATTENTION_INPUTS: int = DINO_BLOCKS * 3
"""Query, key and value inputs with a calibrated range."""


class DinoActivationObserver:
    """Measure linear inputs and post-RoPE Q/K/V without retaining frame tensors."""

    def __init__(self, backbone: nn.Module) -> None:
        self.backbone: nn.Module = backbone
        self.inputs: dict[str, Tensor] = {}
        self.attention: dict[str, Tensor] = {}
        self.handles: list[RemovableHandle] = []
        self.originals: list[tuple[SelfAttention, Callable[..., Tensor]]] = []

    def _linear_hook(self, name: str) -> Callable[[nn.Module, tuple[Tensor, ...]], None]:
        def record(module: nn.Module, args: tuple[Tensor, ...]) -> None:
            value: Tensor = args[0].detach().abs().amax()
            self.inputs[name] = torch.maximum(self.inputs[name], value) if name in self.inputs else value
        return record

    def _attention_hook(self, name: str, module: SelfAttention, original: Callable[..., Tensor]) -> Callable[..., Tensor]:
        def record(qkv: Tensor, attn_bias: Tensor | None = None, rope: tuple[Tensor, Tensor] | None = None) -> Tensor:
            batch, tokens = qkv.shape[:2]
            channels: int = module.qkv.in_features
            q, k, v = (item.transpose(1, 2) for item in qkv.reshape(batch, tokens, 3, module.num_heads, channels // module.num_heads).unbind(2))
            if rope is not None:
                q, k = module.apply_rope(q, k, rope)
            for tag, tensor in zip("qkv", (q, k, v), strict=True):
                key: str = f"{name}.{tag}"
                value: Tensor = tensor.detach().abs().amax()
                self.attention[key] = torch.maximum(self.attention[key], value) if key in self.attention else value
            return original(qkv, attn_bias, rope)
        return record

    def __enter__(self) -> "DinoActivationObserver":
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_pre_hook(self._linear_hook(name)))
            if isinstance(module, SelfAttention):
                self.originals.append((module, module.compute_attention))
                module.compute_attention = self._attention_hook(name, module, module.compute_attention)
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self.handles:
            handle.remove()
        for module, original in self.originals:
            module.compute_attention = original

    def ranges(self) -> tuple[dict[str, float], dict[str, float]]:
        """Copy only scalar maxima to the host after the complete calibration run."""
        return ({name: float(value) for name, value in self.inputs.items()}, {name: float(value) for name, value in self.attention.items()})
