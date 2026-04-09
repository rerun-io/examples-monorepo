"""Helpers for the shared-library Mojo <-> PyTorch interop boundary.

This module centralizes the intentionally unsafe parts of the shared-lib
backend: retrieving the process-lifetime DeviceContext pointer cached on the
Python module object and reinterpreting `tensor.data_ptr()` addresses as typed
Mojo pointers.

The safety contract for every `torch_*_ptr()` helper is:
1. the caller has already converted the tensor to the expected dtype,
2. the caller has already made the tensor contiguous, and
3. the Python tensor object remains alive until the launched kernel finishes.

Those conditions already hold in the existing call sites; keeping the raw
pointer logic here makes that boundary easier to audit.

`LayoutTensor` is intentionally not used here. It is a better fit once we are
already inside Mojo kernel code with a typed pointer or shared allocation to
wrap. At the Python extension boundary, we still need one explicit
`tensor.data_ptr()` -> typed pointer conversion, so this module keeps that
unsafe step isolated in one place instead of scattering it across wrappers.
"""

from std.gpu.host import DeviceContext
from std.memory import alloc
from std.python import Python, PythonObject


def get_backend_module() raises -> PythonObject:
    return Python.import_module("mast3r_slam_mojo_backends")


def get_torch_module() raises -> PythonObject:
    return Python.import_module("torch")


def get_cuda_backend_module() raises -> PythonObject:
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


@always_inline
def torch_float32_ptr(
    tensor: PythonObject,
) raises -> UnsafePointer[Float32, MutAnyOrigin]:
    return UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=Int(py=tensor.data_ptr())
    )


@always_inline
def torch_float16_ptr(
    tensor: PythonObject,
) raises -> UnsafePointer[Float16, MutAnyOrigin]:
    return UnsafePointer[Float16, MutAnyOrigin](
        unsafe_from_address=Int(py=tensor.data_ptr())
    )


@always_inline
def torch_int64_ptr(
    tensor: PythonObject,
) raises -> UnsafePointer[Int64, MutAnyOrigin]:
    return UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=Int(py=tensor.data_ptr())
    )


@always_inline
def torch_uint8_ptr(
    tensor: PythonObject,
) raises -> UnsafePointer[UInt8, MutAnyOrigin]:
    return UnsafePointer[UInt8, MutAnyOrigin](
        unsafe_from_address=Int(py=tensor.data_ptr())
    )
