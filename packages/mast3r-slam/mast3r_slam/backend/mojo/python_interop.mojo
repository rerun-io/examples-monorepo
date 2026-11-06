"""Helpers for the shared-library Mojo ↔ PyTorch interop boundary.

Why this file exists
--------------------
Mojo GPU kernels need two things that PyTorch doesn't natively provide:

  1. **A DeviceContext** — Mojo's GPU runtime handle for enqueueing kernels.
     Unlike CUDA C++ extensions (which just call `<<<...>>>` launch syntax),
     Mojo requires an explicit context object. Since Mojo shared libraries
     can't hold module-level globals, we allocate a `DeviceContext` on the
     heap at import time and store its address as a Python integer attribute
     on the extension module (`_ctx_addr`). Every kernel launch recovers
     the pointer via `get_cached_context_ptr()`.

  2. **Typed tensors** over PyTorch tensor storage. PyTorch's
     `tensor.data_ptr()` returns an integer address; `torch_tensor()` and
     `torch_imm_tensor()` wrap it in a row-major `TileTensor` that kernels
     index by coordinate.

Safety contract
---------------
`torch_tensor()` / `torch_imm_tensor()` assume:
  1. The tensor has already been cast to the requested dtype (.float(), etc.).
  2. The tensor is contiguous in memory (.contiguous()).
  3. The Python tensor object stays alive until the launched kernel finishes
     (i.e. the caller holds a reference and calls synchronize before dropping).

These conditions are enforced at the call sites in `gn.mojo` and `matching.mojo`.
"""

from max.gpu.host import DeviceContext
from layout import ImmTileTensor, MutTileTensor, TensorLayout, TileTensor
from std.memory import Layout, alloc
from std.python import Python, PythonObject


# ── Module imports (cached by Python's import system) ────────────────────────

def get_backend_module() raises -> PythonObject:
    """Import the Mojo extension module itself (to access _ctx_addr)."""
    return Python.import_module("mast3r_slam_mojo_backends")


def get_torch_module() raises -> PythonObject:
    """Import PyTorch (used for tensor allocation and CUDA sync)."""
    return Python.import_module("torch")


def get_cuda_backend_module() raises -> PythonObject:
    """Import the original CUDA extension (used for point/calib fallbacks)."""
    return Python.import_module("mast3r_slam._backends")


def install_cached_context(module: PythonObject) raises:
    """Attach a process-lifetime DeviceContext pointer to the Python module.

    Mojo shared libraries cannot store a module-global `DeviceContext`, so the
    Python module owns one heap allocation for the lifetime of the extension.
    The allocation is leaked on purpose; its address is stored as an integer
    and recovered later by `get_cached_context_ptr()`.
    """
    var ctx_storage = alloc(Layout[DeviceContext](count=1)).unsafe_leak()
    ctx_storage.unsafe_write(DeviceContext())
    Python.add_object(module, "_ctx_addr", PythonObject(Int(ctx_storage)))


def get_cached_context_ptr() raises -> MutPointer[DeviceContext, MutUntrackedOrigin]:
    """Return the shared DeviceContext for kernels launched from this module."""
    var module = get_backend_module()
    return MutPointer[DeviceContext, MutUntrackedOrigin](
        unsafe_from_address=Int(py=module._ctx_addr)
    )


# ── Tensor → TileTensor views ────────────────────────────────────────────────
# Each helper reinterprets tensor.data_ptr() (a Python int) as typed device
# memory. The caller MUST ensure the tensor dtype matches `dtype`.

def torch_tensor[
    dtype: DType, LT: TensorLayout
](tensor: PythonObject, layout: LT) raises -> MutTileTensor[dtype, LT, MutUntrackedOrigin]:
    """View a contiguous torch tensor that a kernel writes."""
    var ptr = MutPointer[Scalar[dtype], MutUntrackedOrigin](
        unsafe_from_address=Int(py=tensor.data_ptr())
    )
    return TileTensor(ptr, layout)


def torch_imm_tensor[
    dtype: DType, LT: TensorLayout
](tensor: PythonObject, layout: LT) raises -> ImmTileTensor[dtype, LT, ImmUntrackedOrigin]:
    """View a contiguous torch tensor that a kernel only reads."""
    var ptr = ImmPointer[Scalar[dtype], ImmUntrackedOrigin](
        unsafe_from_address=Int(py=tensor.data_ptr())
    )
    return TileTensor(ptr, layout)
