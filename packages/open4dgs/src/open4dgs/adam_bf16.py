"""Adam with bfloat16 moments, stochastically rounded, in one fused Triton kernel per parameter (HiGS 4.2 + 4.3, PennyGS 3.7).

The arithmetic is torch.optim.Adam's (non-fused, ``maximize=False``, no weight decay, no amsgrad) in float32; only the
two moment stores are narrowed to bfloat16. Rounding to nearest would drop every moment change below half a unit in
the last place (2^-9 relative); stochastic rounding keeps the stored value equal to the exact one in expectation
(HiGS 4.3). Parameters stay float32. The state keeps torch's keys (``step``, ``exp_avg``, ``exp_avg_sq``), so TRiGS's
row-wise state helpers (relocation zeroes rows, pruning slices them) work unchanged.
"""

from collections.abc import Iterable, Sequence

import numpy as np
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
def _adam_bf16_update(p_ptr, g_ptr, m_ptr, v_ptr, offs, mask, beta1, beta2, eps, step_size, bc2_sqrt, seed):
    """The Adam update of the elements ``offs`` of one parameter (both kernels' body, so their results stay bit-identical)."""
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


@triton.jit
def _adam_bf16_kernel(p_ptr, g_ptr, m_ptr, v_ptr, n, beta1, beta2, eps, step_size, bc2_sqrt, seed, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    _adam_bf16_update(p_ptr, g_ptr, m_ptr, v_ptr, offs, offs < n, beta1, beta2, eps, step_size, bc2_sqrt, seed)


@triton.jit
def _adam_bf16_multi_kernel(table_ptr, block_tensor_ptr, block_first_ptr, BLOCK: tl.constexpr):
    """``_adam_bf16_kernel`` for many parameters in one launch: program ``i`` updates block ``i - block_first[t]`` of tensor
    ``t = block_tensor[i]``; row ``t`` of the int64 table holds its pointers, size and seed, then its five float32 scalars
    (beta1, beta2, eps, step size, sqrt bias correction 2) packed two per int64 slot."""
    pid = tl.program_id(0)
    t = tl.load(block_tensor_ptr + pid).to(tl.int64)
    row = table_ptr + t * TABLE_COLS
    p_ptr = tl.load(row + 0).to(tl.pointer_type(tl.float32))
    g_ptr = tl.load(row + 1).to(tl.pointer_type(tl.float32))
    m_ptr = tl.load(row + 2).to(tl.pointer_type(tl.bfloat16))
    v_ptr = tl.load(row + 3).to(tl.pointer_type(tl.bfloat16))
    n = tl.load(row + 4)
    seed = tl.load(row + 5).to(tl.int32)
    scalars = (row + 6).to(tl.pointer_type(tl.float32), bitcast=True)
    beta1 = tl.load(scalars + 0)
    beta2 = tl.load(scalars + 1)
    eps = tl.load(scalars + 2)
    step_size = tl.load(scalars + 3)
    bc2_sqrt = tl.load(scalars + 4)
    offs = (pid - tl.load(block_first_ptr + t)) * BLOCK + tl.arange(0, BLOCK)
    _adam_bf16_update(p_ptr, g_ptr, m_ptr, v_ptr, offs, offs < n, beta1, beta2, eps, step_size, bc2_sqrt, seed)


TABLE_COLS: tl.constexpr = tl.constexpr(9)
"""int64 columns of ``fused_step``'s table: p, grad, exp_avg, exp_avg_sq pointers, numel, seed, then 3 slots = 6 float32."""


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
                state, step_size, bc2_sqrt, seed = self._advance(p, group)
                _adam_bf16_kernel[(triton.cdiv(p.numel(), BLOCK),)](
                    p,
                    p.grad,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    p.numel(),
                    beta1,
                    beta2,
                    group["eps"],
                    step_size,
                    bc2_sqrt,
                    seed,
                    BLOCK=BLOCK,  # pyrefly: ignore[bad-argument-type]  # Triton constexpr
                )

    def _advance(self, p: Tensor, group: dict) -> tuple[dict, float, float, int]:
        """One step of ``p``'s state (created on its first step): (state, step size, sqrt bias correction 2, launch seed)."""
        state = self.state[p]
        if not state:
            state["step"] = torch.tensor(0.0)
            state["exp_avg"] = torch.zeros_like(p, dtype=torch.bfloat16)
            state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.bfloat16)
        state["step"] += 1
        t: float = float(state["step"])
        beta1, beta2 = group["betas"]
        self._calls += 1
        return state, group["lr"] / (1.0 - beta1**t), (1.0 - beta2**t) ** 0.5, self._calls & 0x7FFFFFFF


_BLOCK_MAPS: dict[tuple[torch.device, tuple[int, ...]], tuple[Tensor, Tensor]] = {}


def _block_map(device: torch.device, sizes: tuple[int, ...]) -> tuple[Tensor, Tensor]:
    """(tensor index per block, first block per tensor) for ``fused_step``'s launch, built once per set of sizes."""
    key = (device, sizes)
    out = _BLOCK_MAPS.get(key)
    if out is None:
        if len(_BLOCK_MAPS) > 64:
            _BLOCK_MAPS.clear()
        blocks: Tensor = torch.tensor([triton.cdiv(n, BLOCK) for n in sizes], dtype=torch.int64)
        first: Tensor = torch.cumsum(blocks, 0) - blocks
        out = _BLOCK_MAPS[key] = (torch.repeat_interleave(torch.arange(len(sizes), dtype=torch.int32), blocks).to(device), first.to(device))
    return out


@torch.no_grad()
def fused_step(optimizers: Sequence[AdamBf16]) -> None:
    """``opt.step(); opt.zero_grad(set_to_none=True)`` for every optimizer, as ONE kernel launch over all their parameters
    with a gradient (the per-parameter launches and the optimizers' Python step wrappers cost host time every step).
    Bit-identical to the per-optimizer steps: same arithmetic, same per-parameter seeds and offsets. Parameters shared
    between optimizers or spread over devices step one optimizer after the other (the first one clears a shared
    parameter's gradient, so the next skips it); every parameter is checked before any state advances."""
    todo: list[tuple[AdamBf16, dict, Tensor]] = [
        (opt, group, p) for opt in optimizers for group in opt.param_groups for p in group["params"] if p.grad is not None
    ]
    for _, _, p in todo:
        if p.dtype != torch.float32 or not p.is_contiguous() or not p.grad.is_contiguous():  # pyrefly: ignore[missing-attribute]
            raise ValueError("AdamBf16 needs contiguous float32 parameters and gradients")
    if len({id(p) for _, _, p in todo}) != len(todo) or len({p.device for _, _, p in todo}) > 1:
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)
        return
    rows: list[tuple[int, ...]] = []
    scalars: list[tuple[float, ...]] = []
    for opt, group, p in todo:
        beta1, beta2 = group["betas"]
        state, step_size, bc2_sqrt, seed = opt._advance(p, group)
        rows.append((p.data_ptr(), p.grad.data_ptr(), state["exp_avg"].data_ptr(), state["exp_avg_sq"].data_ptr(), p.numel(), seed))  # pyrefly: ignore[missing-attribute]
        scalars.append((beta1, beta2, group["eps"], step_size, bc2_sqrt, 0.0))
    if todo:
        device: torch.device = todo[0][2].device
        sizes: list[int] = [p.numel() for _, _, p in todo]
        table: np.ndarray = np.zeros((len(rows), int(TABLE_COLS)), dtype=np.int64)
        table[:, :6] = rows
        table[:, 6:].view(np.float32)[:] = scalars
        # Pinned + non_blocking: no host wait; the caching host allocator keeps the buffer until the copy has run.
        table_dev: Tensor = torch.from_numpy(table).pin_memory().to(device, non_blocking=True)
        block_tensor, block_first = _block_map(device, tuple(sizes))
        _adam_bf16_multi_kernel[(int(block_tensor.numel()),)](table_dev, block_tensor, block_first, BLOCK=BLOCK)  # pyrefly: ignore[bad-argument-type]
    for opt in optimizers:
        for group in opt.param_groups:
            for p in group["params"]:
                p.grad = None
