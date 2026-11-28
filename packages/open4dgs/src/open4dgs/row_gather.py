"""TRiGS's per-step row gather (the active rows of every row-wise parameter) in one kernel launch, its backward in one more.

``Open4dgsTrainer._slice_rowwise_params`` gathers ~20 float32 parameters with ``index_select`` every step: ~20 forward
launches and, in the backward, a zero fill plus an ``index_add_`` per parameter that needs a gradient. Here the forward
is one Triton launch over a pointer table of all the tensors, and the backward writes every full-size gradient in one
launch (the active row's gradient where the row was gathered, zero elsewhere) from an inverse row map. ``idx`` has no
repeats (the active rows), so each full-size gradient equals ``index_add_`` into zeros; an output that gets no gradient
leaves its tensor's gradient None, as ``index_select`` does.
"""

from collections.abc import Sequence

import numpy as np
import torch
import triton
import triton.language as tl
from jaxtyping import Int64
from torch import Tensor

BLOCK: int = 1024

TABLE_COLS: tl.constexpr = tl.constexpr(4)
"""int64 columns of a launch table: source pointer, destination pointer, row width, destination numel."""


@triton.jit
def _gather_kernel(table_ptr, idx_ptr, block_tensor_ptr, block_first_ptr, BLOCK: tl.constexpr):
    """dst_t[i, :] = src_t[idx[i], :] for tensor ``t = block_tensor[program]``."""
    pid = tl.program_id(0)
    t = tl.load(block_tensor_ptr + pid).to(tl.int64)
    row = table_ptr + t * TABLE_COLS
    src = tl.load(row + 0).to(tl.pointer_type(tl.float32))
    dst = tl.load(row + 1).to(tl.pointer_type(tl.float32))
    width = tl.load(row + 2)
    numel = tl.load(row + 3)
    offs = (pid - tl.load(block_first_ptr + t)) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < numel
    i = offs // width
    r = tl.load(idx_ptr + i, mask=mask, other=0)
    tl.store(dst + offs, tl.load(src + r * width + (offs - i * width), mask=mask), mask=mask)


@triton.jit
def _scatter_kernel(table_ptr, pos_ptr, block_tensor_ptr, block_first_ptr, BLOCK: tl.constexpr):
    """dst_t[r, :] = src_t[pos[r], :] where pos[r] >= 0, else 0 (the full-size gradient of the gather)."""
    pid = tl.program_id(0)
    t = tl.load(block_tensor_ptr + pid).to(tl.int64)
    row = table_ptr + t * TABLE_COLS
    src = tl.load(row + 0).to(tl.pointer_type(tl.float32))
    dst = tl.load(row + 1).to(tl.pointer_type(tl.float32))
    width = tl.load(row + 2)
    numel = tl.load(row + 3)
    offs = (pid - tl.load(block_first_ptr + t)) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < numel
    r = offs // width
    p = tl.load(pos_ptr + r, mask=mask, other=-1)
    hit = mask & (p >= 0)
    tl.store(dst + offs, tl.load(src + p * width + (offs - r * width), mask=hit, other=0.0), mask=mask)


def _launch(kernel: object, pairs: Sequence[tuple[Tensor, Tensor]], index: Tensor) -> None:
    """One ``kernel`` launch over (source, destination) pairs; the destination's numel sets each tensor's blocks."""
    table: np.ndarray = np.array([(s.data_ptr(), d.data_ptr(), d[0].numel() if d.shape[0] else 1, d.numel()) for s, d in pairs], dtype=np.int64)
    blocks: np.ndarray = -(-table[:, 3] // BLOCK)
    first: np.ndarray = np.cumsum(blocks) - blocks
    meta: np.ndarray = np.concatenate([table.reshape(-1), first, np.repeat(np.arange(len(pairs)), blocks)])
    # One pinned, non-blocking upload (no host wait); the caching host allocator keeps the buffer until the copy has run.
    dev: Tensor = torch.from_numpy(meta).pin_memory().to(index.device, non_blocking=True)
    n_table: int = table.size
    total: int = int(blocks.sum())
    if total == 0:
        return
    kernel[(total,)](dev[:n_table], index, dev[n_table + len(pairs) :], dev[n_table : n_table + len(pairs)], BLOCK=BLOCK)  # pyrefly: ignore[index-error]


class _RowGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, idx: Tensor, *tensors: Tensor) -> tuple[Tensor, ...]:
        outs: tuple[Tensor, ...] = tuple(torch.empty((idx.numel(), *t.shape[1:]), dtype=t.dtype, device=t.device) for t in tensors)
        _launch(_gather_kernel, list(zip(tensors, outs, strict=True)), idx)
        ctx.save_for_backward(idx)
        # An unused output's gradient stays None (not zeros), so its parameter's gradient stays None as with
        # index_select: Adam then skips it (TRiGS's SH warm-up leaves shN unused at first).
        ctx.set_materialize_grads(False)
        ctx.shapes = [tuple(t.shape) for t in tensors]  # pyrefly: ignore[missing-attribute]
        return outs

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, *grads: Tensor | None) -> tuple[Tensor | None, ...]:  # pyrefly: ignore[bad-override]
        (idx,) = ctx.saved_tensors  # pyrefly: ignore[missing-attribute]
        shapes: list[tuple[int, ...]] = ctx.shapes  # pyrefly: ignore[missing-attribute]
        n: int = shapes[0][0]
        pos: Tensor = torch.full((n,), -1, dtype=torch.int64, device=idx.device)
        pos[idx] = torch.arange(idx.numel(), device=idx.device)
        full: list[Tensor | None] = [None] * len(grads)
        pairs: list[tuple[Tensor, Tensor]] = []
        for i, g in enumerate(grads):
            if g is None or not ctx.needs_input_grad[i + 1]:  # pyrefly: ignore[missing-attribute]
                continue
            out: Tensor = torch.empty(shapes[i], dtype=g.dtype, device=g.device)
            full[i] = out
            pairs.append((g.contiguous(), out))
        if pairs:
            _launch(_scatter_kernel, pairs, pos)
        return (None, *full)


def gather_rows(tensors: Sequence[Tensor], idx: Int64[Tensor, " a"]) -> tuple[Tensor, ...]:
    """``tuple(t.index_select(0, idx) for t in tensors)`` for contiguous float32 CUDA tensors of one row count and an
    ``idx`` without repeats, in one launch forward and one backward (plus the inverse row map)."""
    return _RowGather.apply(idx.contiguous(), *tensors)  # pyrefly: ignore[bad-return]
