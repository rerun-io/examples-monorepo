"""Bridge between PyTorch tensors and the Mojo kernels of this extension module.

The module owns one process-lifetime `DeviceContext`. A Mojo shared library cannot hold a module-level global, so its
address lives on the Python module as `_ctx_addr`.

`torch_ptr` turns `tensor.data_ptr()` into a typed pointer. Callers must keep contiguous tensors of the matching dtype
alive until the kernel has finished. Matching and ray-step wrappers synchronise before returning; the Python
Gauss-Newton loop owns the final synchronisation after retraction.
"""

from max.gpu.host import DeviceContext
from std.memory import Layout, alloc
from std.python import Python, PythonObject


def install_cached_context(module: PythonObject) raises:
    """Create the shared DeviceContext and store its address on `module._ctx_addr`; it is never freed."""
    var ctx = alloc(Layout[DeviceContext](count=1)).unsafe_leak()
    ctx.unsafe_write(DeviceContext())
    Python.add_object(module, "_ctx_addr", PythonObject(Int(ctx)))


def cached_context() raises -> MutPointer[DeviceContext, MutAnyOrigin]:
    """The DeviceContext created by `install_cached_context`."""
    var module = Python.import_module("mast3r_slam_mojo_backends")
    return MutPointer[DeviceContext, MutAnyOrigin](unsafe_from_address=Int(py=module._ctx_addr))


@inline(.always)
def torch_ptr[dtype: DType](tensor: PythonObject) raises -> MutPointer[Scalar[dtype], MutAnyOrigin]:
    """`tensor.data_ptr()` as a typed pointer; the tensor must be contiguous and hold `dtype`."""
    return MutPointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=Int(py=tensor.data_ptr()))
