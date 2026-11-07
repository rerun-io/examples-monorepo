"""Metal GPU addresses of PyTorch MPS tensors, for the Mojo kernels on an Apple GPU.

An MPS tensor's storage is an `MTLBuffer` and its `data_ptr()` is that object's
address, not device memory. The Mojo Metal kernels take device pointers, so
`gpu_addresses` reads each buffer's `gpuAddress` (Metal 3) through the Objective-C
runtime and adds the tensor's byte offset. PyTorch's MPS queue and the Mojo
context's queue are not ordered with each other: callers run
`torch.mps.synchronize()` before a launch, and the Mojo side waits for its own
kernel before it returns (`GPURuntime.finish`).

A kernel that reaches a buffer only through its address must also have the
buffer resident: Metal makes a command buffer's resources resident through
`useResource:` or a residency set, and the Mojo runtime does that for its own
allocations only. A PyTorch buffer that is not resident reads as zeros (MetricTrail's
SharedK calibration on pool.mp4 got an all-zero `SE3.inv` from a valid pose row
after a few hundred launches). So `gpu_addresses` first puts the launch's buffers
into one `MTLResidencySet` and requests its residency (macOS 15).
"""

import ctypes
import functools
import threading
from collections.abc import Callable, Sequence

from torch import Tensor

_c_id = ctypes.c_void_p


class _Residency:
    """One `MTLResidencySet` on the system GPU, holding the buffers of the current launch."""

    def __init__(self) -> None:
        objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
        objc.sel_registerName.restype = _c_id
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.objc_getClass.restype = _c_id
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        metal = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Metal.framework/Metal")
        metal.MTLCreateSystemDefaultDevice.restype = _c_id

        def send(restype: type | None, *argtypes: type) -> Callable[..., int | None]:
            return ctypes.CFUNCTYPE(restype, _c_id, _c_id, *argtypes)(("objc_msgSend", objc))

        def sel(name: bytes) -> int:
            return objc.sel_registerName(name)

        send_id = send(_c_id)
        self._send_void = send(None)
        self._send_void_id = send(None, _c_id)
        device: int | None = metal.MTLCreateSystemDefaultDevice()
        descriptor = send_id(send_id(objc.objc_getClass(b"MTLResidencySetDescriptor"), sel(b"alloc")), sel(b"init"))
        error = _c_id(0)
        new_set = send(_c_id, _c_id, ctypes.POINTER(_c_id))
        self._set: int | None = new_set(device, sel(b"newResidencySetWithDescriptor:error:"), descriptor, ctypes.byref(error))
        if not self._set:
            raise RuntimeError("lietorch_mojo: Metal refused a residency set (needs macOS 15)")
        self._remove_all, self._add, self._commit, self._request = (
            sel(b"removeAllAllocations"),
            sel(b"addAllocation:"),
            sel(b"commit"),
            sel(b"requestResidency"),
        )
        self._lock = threading.Lock()

    def make_resident(self, buffers: Sequence[int]) -> None:
        """Replace the set's allocations by `buffers` (`id<MTLBuffer>` values; 0 = none) and request their residency."""
        with self._lock:
            self._send_void(self._set, self._remove_all)
            for buffer in dict.fromkeys(buffers):
                if buffer:  # an empty tensor has no MTLBuffer (data_ptr 0); addAllocation:nil crashes
                    self._send_void_id(self._set, self._add, buffer)
            self._send_void(self._set, self._commit)
            self._send_void(self._set, self._request)


@functools.cache
def _residency() -> _Residency:
    return _Residency()


@functools.cache
def _buffer_gpu_address() -> Callable[[int], int]:
    """`[buffer gpuAddress]` for an `id<MTLBuffer>`, bound once (libobjc is loaded on the first MPS launch only)."""
    objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
    objc.sel_registerName.restype = _c_id
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    selector: int = objc.sel_registerName(b"gpuAddress")
    send = ctypes.CFUNCTYPE(ctypes.c_uint64, _c_id, _c_id)(("objc_msgSend", objc))
    return lambda buffer: send(buffer, selector)


def gpu_addresses(tensors: Sequence[Tensor]) -> tuple[int, ...]:
    """Device addresses of the first elements of the MPS `tensors` (0 for an empty one), whose buffers are made resident first."""
    buffers: list[int] = [t.untyped_storage().data_ptr() for t in tensors]
    _residency().make_resident(buffers)
    address: Callable[[int], int] = _buffer_gpu_address()
    return tuple(address(b) + int(t.storage_offset()) * t.element_size() if b else 0 for b, t in zip(buffers, tensors, strict=True))
