"""Metal GPU addresses of PyTorch MPS tensors, for the Mojo kernels on an Apple GPU.

An MPS tensor's storage is an `MTLBuffer` and its `data_ptr()` is that object's
address, not device memory. The Mojo Metal kernels take device pointers, so
`gpu_address` reads the buffer's `gpuAddress` (Metal 3) through the Objective-C
runtime and adds the tensor's byte offset. PyTorch's MPS queue and the Mojo
context's queue are not ordered with each other: callers run
`torch.mps.synchronize()` before a launch, and the Mojo side waits for its own
kernel before it returns (`GPURuntime.finish`).
"""

import ctypes
import functools
from collections.abc import Callable

from torch import Tensor


@functools.cache
def _buffer_gpu_address() -> Callable[[int], int]:
    """`[buffer gpuAddress]` for an `id<MTLBuffer>`, bound once (libobjc is loaded on the first MPS launch only)."""
    objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    selector: int = objc.sel_registerName(b"gpuAddress")
    send = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
    return lambda buffer: send(buffer, selector)


def gpu_address(t: Tensor) -> int:
    """Device address of the first element of the MPS tensor `t`."""
    return _buffer_gpu_address()(t.untyped_storage().data_ptr()) + t.storage_offset() * t.element_size()
