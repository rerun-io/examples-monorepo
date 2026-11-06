"""Drop-in for lietorch's compiled `lietorch_backends` module, backed by Mojo.

Every function has the name, argument order, output shapes and gradient
conventions of the pybind11 function of the same name in lietorch
`src/lietorch.cpp` (VSLAM-LAB/lietorch @ 3e2695dd). `group_index` uses
lietorch's ids: SO3=1, RxSO3=2, SE3=3, Sim3=4. Groups without a Mojo
implementation raise `NotImplementedError`.

Deviations from lietorch, all on inputs lietorch mishandles:
- A dtype other than float32/float64 raises `TypeError` (lietorch silently
  returns zeros).
- Inputs on different devices, with different dtypes, on a CUDA device other
  than cuda:0, or of the wrong shape (not (batch, width) with the op's row
  width and one batch size) raise `ValueError`/`RuntimeError` (lietorch reads
  or writes out of bounds or uses the wrong device).

All kernel launches go through `_run`, the only place that knows how the
Mojo extension is called. The extension owns the row widths (`row_widths`),
so the shape check here and the kernels cannot disagree. GPU kernels run on
PyTorch's current CUDA stream (asynchronously, like a native PyTorch op),
unlike lietorch, whose kernels always run on the legacy default stream.
"""

from collections.abc import Callable
from typing import Final

import torch
from torch import Tensor

from lietorch_mojo import _lietorch_mojo_backends as _ext

# Op codes; keep in sync with lietorch_mojo/mojo/group_ops.mojo.
_OP_EXP: Final[int] = 0
_OP_EXP_BACKWARD: Final[int] = 1
_OP_LOG: Final[int] = 2
_OP_LOG_BACKWARD: Final[int] = 3
_OP_INV: Final[int] = 4
_OP_INV_BACKWARD: Final[int] = 5
_OP_MUL: Final[int] = 6
_OP_MUL_BACKWARD: Final[int] = 7
_OP_ADJ: Final[int] = 8
_OP_ADJ_BACKWARD: Final[int] = 9
_OP_ADJT: Final[int] = 10
_OP_ADJT_BACKWARD: Final[int] = 11
_OP_ACT: Final[int] = 12
_OP_ACT_BACKWARD: Final[int] = 13
_OP_ACT4: Final[int] = 14
_OP_ACT4_BACKWARD: Final[int] = 15
_OP_AS_MATRIX: Final[int] = 16
_OP_PROJECTOR: Final[int] = 17
_OP_JINV: Final[int] = 18
_NUM_OPS: Final[int] = 19

# lietorch group id -> (embedded dim N, manifold dim K).
_GROUP_DIMS: Final[dict[int, tuple[int, int]]] = {1: (4, 3), 2: (5, 4), 3: (7, 6), 4: (8, 7)}
_IMPLEMENTED_DIMS: Final[dict[int, tuple[int, int]]] = {2: _GROUP_DIMS[2], 3: _GROUP_DIMS[3], 4: _GROUP_DIMS[4]}
_GROUP_NAMES: Final[dict[int, str]] = {1: "SO3", 2: "RxSO3", 3: "SE3", 4: "Sim3"}
# (group id, op) -> width of each tensor the op takes, inputs then outputs (from the extension).
_ROW_WIDTHS: Final[dict[tuple[int, int], tuple[int, ...]]] = {
    (g, op): tuple(_ext.row_widths(op | (g << 8))) for g in _IMPLEMENTED_DIMS for op in range(_NUM_OPS)
}

_FLOAT32: Final[torch.dtype] = torch.float32
_FLOAT64: Final[torch.dtype] = torch.float64
# Raw cudaStream_t of PyTorch's current stream on a device index (0 = legacy default stream).
_current_raw_stream: Final[Callable[[int], int]] = torch._C._cuda_getCurrentRawStream
_launch_cuda: Final[Callable[[object, int, int, int, tuple[int, ...]], None]] = _ext.launch_cuda
_launch_cpu: Final[Callable[[object, int, int, tuple[int, ...]], None]] = _ext.launch_cpu

# The Mojo runtimes (`GPURuntime`, `CPURuntime`), created on first use: CPU-only use never touches the GPU.
_gpu_runtime: object | None = None
_cpu_runtime: object | None = None


def _runtime(on_gpu: bool) -> object:
    global _gpu_runtime, _cpu_runtime
    if on_gpu:
        if _gpu_runtime is None:
            _gpu_runtime = _ext.create_gpu_runtime()
        return _gpu_runtime
    if _cpu_runtime is None:
        _cpu_runtime = _ext.create_cpu_runtime()
    return _cpu_runtime


def _dims(group_index: int) -> tuple[int, int]:
    dims: tuple[int, int] | None = _IMPLEMENTED_DIMS.get(group_index)
    if dims is not None:
        return dims
    if group_index not in _GROUP_DIMS:
        raise ValueError(f"unknown lietorch group id {group_index}")
    raise NotImplementedError(f"lietorch_mojo: {_GROUP_NAMES[group_index]} is not implemented yet")


def _bad_inputs(tensors: tuple[Tensor, ...], names: tuple[str, ...], widths: tuple[int, ...], n: int) -> Exception:
    """The error for input tensors that failed `_run`'s combined check (the slow path, off the hot loop)."""
    ref: Tensor = tensors[0]
    for t, name, width in zip(tensors, names, widths[: len(tensors)], strict=True):
        if not t.is_contiguous():
            return RuntimeError(f"{name} must be contiguous")  # lietorch's CHECK_CONTIGUOUS message
        if t.dtype != ref.dtype:
            return ValueError(f"lietorch_mojo: mixed dtypes {ref.dtype} and {t.dtype}")
        if t.device != ref.device:
            return ValueError(f"lietorch_mojo: mixed devices {ref.device} and {t.device}")
        if tuple(t.shape) != (n, width):
            return ValueError(f"lietorch_mojo: {name} must have shape ({n}, {width}), got {tuple(t.shape)}")
    return AssertionError("unreachable: the tensors passed every check")


def _run(op: int, group_index: int, inputs: tuple[Tensor, ...], names: tuple[str, ...]) -> list[Tensor]:
    """Run Mojo op `op` on `inputs` (roles per op in group_ops.mojo) and return its fresh (batch, width) outputs.

    The one launch site. Checks every input (contiguous, one dtype, one
    device, shape (batch, the op's row width) with one batch size; `names`
    label them in errors), allocates the outputs like the first input, then
    enqueues the kernel on PyTorch's current CUDA stream, or runs the CPU
    loop. GPU launches are asynchronous and ordered with PyTorch's own work on
    that stream, like any PyTorch op. This runs on every op call, so it does
    one combined check per input and leaves the error messages to
    `_bad_inputs`.
    """
    widths: tuple[int, ...] | None = _ROW_WIDTHS.get((group_index, op))
    if widths is None:
        _dims(group_index)  # raises for unknown or unimplemented groups
        raise ValueError(f"lietorch_mojo: unknown op code {op}")
    ref: Tensor = inputs[0]
    n: int = ref.shape[0] if ref.dim() == 2 else -1
    dtype: torch.dtype = ref.dtype
    device: int = ref.get_device()  # -1 on CPU
    ptrs: list[int] = []
    for t, width in zip(inputs, widths, strict=False):
        if not t.is_contiguous() or t.dtype is not dtype or t.get_device() != device or t.shape != (n, width):
            raise _bad_inputs(inputs, names, widths, ref.shape[0] if ref.dim() > 0 else 0)
        ptrs.append(t.data_ptr())
    if dtype is _FLOAT32:
        spec: int = op | (group_index << 8)
    elif dtype is _FLOAT64:
        spec = op | (group_index << 8) | (1 << 16)
    else:
        raise TypeError(f"lietorch_mojo supports float32 and float64 tensors, got {dtype}")
    outputs: list[Tensor] = [ref.new_empty(n, width) for width in widths[len(inputs) :]]
    if n == 0:
        return outputs
    ptrs.extend(t.data_ptr() for t in outputs)
    if device == 0 and ref.is_cuda:
        _launch_cuda(_gpu_runtime or _runtime(True), _current_raw_stream(0), spec, n, tuple(ptrs))
    elif device < 0 and ref.is_cpu:
        _launch_cpu(_cpu_runtime or _runtime(False), spec, n, tuple(ptrs))
    elif ref.is_cuda:
        raise RuntimeError(f"lietorch_mojo runs on cuda:0 only, got {ref.device}")
    else:
        raise RuntimeError(f"lietorch_mojo supports CPU and CUDA tensors, got {ref.device}")
    return outputs


# ── unary operations ─────────────────────────────────────────────────────────


def expm(group_index: int, a: Tensor) -> Tensor:
    """Exponential map forward: a (B, K) -> X (B, N)."""
    return _run(_OP_EXP, group_index, (a,), ("a",))[0]


def expm_backward(group_index: int, grad: Tensor, a: Tensor) -> list[Tensor]:
    """Exponential map backward: grad (B, N) -> [da (B, K)]."""
    return _run(_OP_EXP_BACKWARD, group_index, (grad, a), ("grad", "a"))


def logm(group_index: int, X: Tensor) -> Tensor:
    """Logarithm map forward: X (B, N) -> a (B, K)."""
    return _run(_OP_LOG, group_index, (X,), ("X",))[0]


def logm_backward(group_index: int, grad: Tensor, X: Tensor) -> list[Tensor]:
    """Logarithm map backward: grad (B, K) -> [dX (B, N)]."""
    return _run(_OP_LOG_BACKWARD, group_index, (grad, X), ("grad", "X"))


def inv(group_index: int, X: Tensor) -> Tensor:
    """Group inverse forward: X (B, N) -> Y (B, N)."""
    return _run(_OP_INV, group_index, (X,), ("X",))[0]


def inv_backward(group_index: int, grad: Tensor, X: Tensor) -> list[Tensor]:
    """Group inverse backward: grad (B, N) -> [dX (B, N)]."""
    return _run(_OP_INV_BACKWARD, group_index, (grad, X), ("grad", "X"))


# ── binary operations ────────────────────────────────────────────────────────


def mul(group_index: int, X: Tensor, Y: Tensor) -> Tensor:
    """Group multiplication forward: X, Y (B, N) -> Z (B, N)."""
    return _run(_OP_MUL, group_index, (X, Y), ("X", "Y"))[0]


def mul_backward(group_index: int, grad: Tensor, X: Tensor, Y: Tensor) -> list[Tensor]:
    """Group multiplication backward: grad (B, N) -> [dX (B, N), dY (B, N)]."""
    return _run(_OP_MUL_BACKWARD, group_index, (grad, X, Y), ("grad", "X", "Y"))


def adj(group_index: int, X: Tensor, a: Tensor) -> Tensor:
    """Adjoint forward: X (B, N), a (B, K) -> b = Adj(X) a (B, K)."""
    return _run(_OP_ADJ, group_index, (X, a), ("X", "a"))[0]


def adj_backward(group_index: int, grad: Tensor, X: Tensor, a: Tensor) -> list[Tensor]:
    """Adjoint backward: grad (B, K) -> [dX (B, N), da (B, K)]."""
    return _run(_OP_ADJ_BACKWARD, group_index, (grad, X, a), ("grad", "X", "a"))


def adjT(group_index: int, X: Tensor, a: Tensor) -> Tensor:
    """Transposed adjoint forward: X (B, N), a (B, K) -> b = Adj(X)^T a (B, K)."""
    return _run(_OP_ADJT, group_index, (X, a), ("X", "a"))[0]


def adjT_backward(group_index: int, grad: Tensor, X: Tensor, a: Tensor) -> list[Tensor]:
    """Transposed adjoint backward: grad (B, K) -> [dX (B, N), da (B, K)]."""
    return _run(_OP_ADJT_BACKWARD, group_index, (grad, X, a), ("grad", "X", "a"))


def act(group_index: int, X: Tensor, p: Tensor) -> Tensor:
    """Action on points forward: X (B, N), p (B, 3) -> q (B, 3)."""
    return _run(_OP_ACT, group_index, (X, p), ("X", "p"))[0]


def act_backward(group_index: int, grad: Tensor, X: Tensor, p: Tensor) -> list[Tensor]:
    """Action on points backward: grad (B, 3) -> [dX (B, N), dp (B, 3)]."""
    return _run(_OP_ACT_BACKWARD, group_index, (grad, X, p), ("grad", "X", "p"))


def act4(group_index: int, X: Tensor, p: Tensor) -> Tensor:
    """Action on homogeneous points forward: X (B, N), p (B, 4) -> q (B, 4)."""
    return _run(_OP_ACT4, group_index, (X, p), ("X", "p"))[0]


def act4_backward(group_index: int, grad: Tensor, X: Tensor, p: Tensor) -> list[Tensor]:
    """Action on homogeneous points backward: grad (B, 4) -> [dX (B, N), dp (B, 4)]."""
    return _run(_OP_ACT4_BACKWARD, group_index, (grad, X, p), ("grad", "X", "p"))


# ── functions with no gradient ───────────────────────────────────────────────


def as_matrix(group_index: int, X: Tensor) -> Tensor:
    """4x4 matrix of each element: X (B, N) -> T (B, 4, 4)."""
    return _run(_OP_AS_MATRIX, group_index, (X,), ("X",))[0].unflatten(-1, (4, 4))


def projector(group_index: int, X: Tensor) -> Tensor:
    """Orthogonal projector onto the tangent space: X (B, N) -> P (B, N, N)."""
    n_dim: int = _dims(group_index)[0]
    return _run(_OP_PROJECTOR, group_index, (X,), ("X",))[0].unflatten(-1, (n_dim, n_dim))


def Jinv(group_index: int, X: Tensor, a: Tensor) -> Tensor:
    """Left Jacobian inverse at Log(X) applied to a: X (B, N), a (B, K) -> b (B, K)."""
    return _run(_OP_JINV, group_index, (X, a), ("X", "a"))[0]
