"""Adam with bfloat16 moments, stochastically rounded, in one fused Triton kernel per parameter (HiGS 4.2 + 4.3, PennyGS 3.7).

The arithmetic is torch.optim.Adam's (non-fused, ``maximize=False``, no weight decay, no amsgrad) in float32; only the
two moment stores are narrowed to bfloat16. Rounding to nearest would drop every moment change below half a unit in
the last place (2^-9 relative); stochastic rounding keeps the stored value equal to the exact one in expectation
(HiGS 4.3). Parameters stay float32. The state keeps torch's keys (``step``, ``exp_avg``, ``exp_avg_sq``), so TRiGS's
row-wise state helpers (relocation zeroes rows, pruning slices them) work unchanged.
"""

from collections.abc import Iterable

import torch
import triton
import triton.language as tl
from torch import Tensor

BLOCK: int = 1024


@triton.jit
def _round_bf16_stochastic(x, rand):
    """float32 -> bfloat16, rounding up with probability equal to the dropped low 16 bits' fraction."""
    bits = x.to(tl.uint32, bitcast=True)
    bits = (bits + (rand & 0xFFFF)) & 0xFFFF0000
    return bits.to(tl.float32, bitcast=True).to(tl.bfloat16)


@triton.jit
def _adam_bf16_kernel(p_ptr, g_ptr, m_ptr, v_ptr, n, lr, beta1, beta2, eps, step_size, bc2_sqrt, seed, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    p = tl.load(p_ptr + offs, mask=mask)
    g = tl.load(g_ptr + offs, mask=mask)
    m = tl.load(m_ptr + offs, mask=mask).to(tl.float32)
    v = tl.load(v_ptr + offs, mask=mask).to(tl.float32)
    m = m + (g - m) * (1.0 - beta1)
    v = v * beta2 + g * g * (1.0 - beta2)
    p = p - step_size * m / (tl.sqrt(v) / bc2_sqrt + eps)
    tl.store(p_ptr + offs, p, mask=mask)
    r = tl.randint4x(seed, offs)
    tl.store(m_ptr + offs, _round_bf16_stochastic(m, r[0].to(tl.uint32, bitcast=True)), mask=mask)
    tl.store(v_ptr + offs, _round_bf16_stochastic(v, r[1].to(tl.uint32, bitcast=True)), mask=mask)


class AdamBf16(torch.optim.Optimizer):
    """``torch.optim.Adam(params, lr, betas, eps)`` with bfloat16 moments stored by stochastic rounding (CUDA float32 params)."""

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        seed: int = 0,
    ) -> None:
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps})
        self._calls: int = seed * 1_000_003

    @torch.no_grad()
    def step(self, closure: None = None) -> None:  # pyrefly: ignore[bad-override]
        del closure
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dtype != torch.float32 or not p.is_contiguous() or not p.grad.is_contiguous():
                    raise ValueError("AdamBf16 needs contiguous float32 parameters and gradients")
                state = self.state[p]
                if not state:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.bfloat16)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.bfloat16)
                state["step"] += 1
                t: float = float(state["step"])
                n: int = p.numel()
                self._calls += 1
                _adam_bf16_kernel[(triton.cdiv(n, BLOCK),)](
                    p,
                    p.grad,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    n,
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    group["lr"] / (1.0 - beta1**t),
                    (1.0 - beta2**t) ** 0.5,
                    self._calls & 0x7FFFFFFF,
                    BLOCK=BLOCK,  # pyrefly: ignore[bad-argument-type]  # Triton constexpr
                )
