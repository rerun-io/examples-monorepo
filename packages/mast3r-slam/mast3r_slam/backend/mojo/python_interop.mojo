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

  2. **Raw typed pointers** into PyTorch tensor storage. PyTorch's
     `tensor.data_ptr()` returns an integer address; the `torch_*_ptr()`
     helpers here reinterpret-cast it to the appropriate Mojo pointer type.

Safety contract
---------------
Every `torch_*_ptr()` helper assumes:
  1. The tensor has already been cast to the expected dtype (.float(), etc.).
  2. The tensor is contiguous in memory (.contiguous()).
  3. The Python tensor object stays alive until the launched kernel finishes
     (i.e. the caller holds a reference and calls synchronize before dropping).

These conditions are enforced at the call sites in `gn.mojo` and `matching.mojo`.
"""

from std.gpu.host import DeviceContext
from std.memory import alloc
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
    The pointer is stored as an integer and recovered later by
    `get_cached_context_ptr()`.
    """
    var ctx_storage = alloc[DeviceContext](1)
    var cached_ctx = DeviceContext()
    ctx_storage.init_pointee_move(cached_ctx^)
    Python.add_object(module, "_ctx_addr", PythonObject(Int(ctx_storage)))


def get_cached_context_ptr() raises -> UnsafePointer[DeviceContext, MutAnyOrigin]:
    """Return the shared DeviceContext for kernels launched from this module."""
    var module = get_backend_module()
    var ctx_addr = Int(py=module._ctx_addr)
    return UnsafePointer[DeviceContext, MutAnyOrigin](
        unsafe_from_address=ctx_addr
    )


# ── Tensor → UnsafePointer casts ─────────────────────────────────────────────
# Each helper reinterpret-casts tensor.data_ptr() (a Python int) to a typed
# Mojo UnsafePointer. The caller MUST ensure the tensor dtype matches.

@always_inline
def torch_float32_ptr(
    tensor: PythonObject,
) raises -> UnsafePointer[Float32, MutAnyOrigin]:
    """Cast a float32 tensor's data_ptr to a Mojo Float32 pointer."""
    return UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=Int(py=tensor.data_ptr())
    )


@always_inline
def torch_float16_ptr(
    tensor: PythonObject,
) raises -> UnsafePointer[Float16, MutAnyOrigin]:
    """Cast a float16 tensor's data_ptr to a Mojo Float16 pointer."""
    return UnsafePointer[Float16, MutAnyOrigin](
        unsafe_from_address=Int(py=tensor.data_ptr())
    )


@always_inline
def torch_int64_ptr(
    tensor: PythonObject,
) raises -> UnsafePointer[Int64, MutAnyOrigin]:
    """Cast an int64 tensor's data_ptr to a Mojo Int64 pointer."""
    return UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=Int(py=tensor.data_ptr())
    )


@always_inline
def torch_uint8_ptr(
    tensor: PythonObject,
) raises -> UnsafePointer[UInt8, MutAnyOrigin]:
    """Cast a uint8/bool tensor's data_ptr to a Mojo UInt8 pointer."""
    return UnsafePointer[UInt8, MutAnyOrigin](
        unsafe_from_address=Int(py=tensor.data_ptr())
    )
