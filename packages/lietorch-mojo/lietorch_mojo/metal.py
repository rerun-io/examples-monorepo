"""Metal GPU addresses of PyTorch MPS tensors, for the Mojo kernels on an Apple GPU.

An MPS tensor's storage is an `MTLBuffer` and its `data_ptr()` is that object's
address, not device memory. The Mojo Metal kernels take device pointers, so
`gpu_addresses` reads each buffer's `gpuAddress` (Metal 3) through the Objective-C
runtime and adds the tensor's byte offset. PyTorch's MPS queue and the Mojo
context's queue are not ordered with each other: callers run
`torch.mps.synchronize()` before a launch, and the Mojo side waits for its own
kernel before it returns (`GPURuntime.finish`).

A kernel that reaches a buffer only through its address also needs the buffer
resident. At each launch Mojo calls `useResource:` for its own buffers only
(`MetalDeviceContext.cpp`, Mojo 1.2): a raw address carries no buffer handle, and
Mojo 1.2 has neither a Metal queue accessor nor a host-memory wrap. A PyTorch buffer
that is not resident reads as zeros (MetricTrail's SharedK calibration on pool.mp4
got an all-zero `SE3.inv` from a valid pose row). Workaround, until Mojo can track
external buffers (draft: /tmp/fleet-artifacts/spark-mac-ports/modular-upstream.md):
`install_queue_residency` interposes `-[MTLDevice newCommandQueue]` so that every
queue created afterwards, Mojo's included, carries one `MTLResidencySet`, and
`gpu_addresses` makes the set hold exactly the `MTLHeap`s of the launch's buffers
(PyTorch sub-allocates its shared-storage buffers from heaps). The set retains what
it holds, so it must not keep earlier launches' heaps: once PyTorch releases a heap,
only the set would keep it alive (a SharedK calibration on the M4 grew to 13 GiB of
Metal memory that way). The Mojo side finishes each kernel before it returns, so the
previous launch's heaps no longer need residency. The set is committed only when the
heaps change, so back-to-back launches on the same heaps cost a lookup.
"""

import ctypes
import functools
import threading
from collections.abc import Callable, Sequence

from torch import Tensor

_c_id = ctypes.c_void_p


class _ObjC:
    """The few Objective-C runtime and Metal calls this module makes, bound once."""

    def __init__(self) -> None:
        self.lib = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
        lib = self.lib
        lib.sel_registerName.restype = _c_id
        lib.sel_registerName.argtypes = [ctypes.c_char_p]
        lib.objc_getClass.restype = _c_id
        lib.objc_getClass.argtypes = [ctypes.c_char_p]
        lib.object_getClass.restype = _c_id
        lib.object_getClass.argtypes = [_c_id]
        lib.class_getInstanceMethod.restype = _c_id
        lib.class_getInstanceMethod.argtypes = [_c_id, _c_id]
        lib.method_getImplementation.restype = _c_id
        lib.method_getImplementation.argtypes = [_c_id]
        lib.method_setImplementation.restype = _c_id
        lib.method_setImplementation.argtypes = [_c_id, _c_id]
        metal = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Metal.framework/Metal")
        metal.MTLCreateSystemDefaultDevice.restype = _c_id
        self.device: int | None = metal.MTLCreateSystemDefaultDevice()
        self.send_id = self.send(_c_id)
        self.send_u64 = self.send(ctypes.c_uint64)
        self.send_void = self.send(None)
        self.send_void_id = self.send(None, _c_id)

    def send(self, restype: type | None, *argtypes: type) -> Callable[..., int | None]:
        """`objc_msgSend` typed as `restype (id, SEL, *argtypes)`."""
        return ctypes.CFUNCTYPE(restype, _c_id, _c_id, *argtypes)(("objc_msgSend", self.lib))

    def sel(self, name: bytes) -> int:
        return self.lib.sel_registerName(name)


@functools.cache
def _objc() -> _ObjC:
    return _ObjC()


class _QueueResidency:
    """One `MTLResidencySet` attached to every command queue created after `install`, holding PyTorch's heaps."""

    def __init__(self, objc: _ObjC) -> None:
        self._objc = objc
        descriptor = objc.send_id(objc.send_id(objc.lib.objc_getClass(b"MTLResidencySetDescriptor"), objc.sel(b"alloc")), objc.sel(b"init"))
        error = _c_id(0)
        new_set = objc.send(_c_id, _c_id, ctypes.POINTER(_c_id))
        self.set: int | None = new_set(objc.device, objc.sel(b"newResidencySetWithDescriptor:error:"), descriptor, ctypes.byref(error))
        if not self.set:
            raise RuntimeError("lietorch_mojo: Metal refused a residency set (needs macOS 15)")
        self._members: set[int] = set()
        self._lock = threading.Lock()
        self._heap, self._add, self._commit = objc.sel(b"heap"), objc.sel(b"addAllocation:"), objc.sel(b"commit")
        self._remove = objc.sel(b"removeAllocation:")
        self._add_to_queue = objc.sel(b"addResidencySet:")
        self._imp: object | None = None  # the ctypes callback must outlive the swizzle

    def install(self) -> None:
        """Interpose `newCommandQueue` on the device's class: each new queue gets the set (call before Mojo's context)."""
        objc = self._objc
        method = objc.lib.class_getInstanceMethod(objc.lib.object_getClass(objc.device), objc.sel(b"newCommandQueue"))
        prototype = ctypes.CFUNCTYPE(_c_id, _c_id, _c_id)
        original = prototype(objc.lib.method_getImplementation(method))

        def new_command_queue(device: int, selector: int) -> int | None:
            queue: int | None = original(device, selector)
            if queue:
                objc.send_void_id(queue, self._add_to_queue, self.set)
            return queue

        self._imp = prototype(new_command_queue)
        objc.lib.method_setImplementation(method, ctypes.cast(self._imp, _c_id))

    def cover(self, buffers: Sequence[int]) -> None:
        """Make the set hold the heap of each `id<MTLBuffer>` (or the buffer, if it has none) and nothing else."""
        objc = self._objc
        # An empty tensor has no MTLBuffer (data_ptr 0).
        allocations: set[int] = {objc.send_id(buffer, self._heap) or buffer for buffer in buffers if buffer}
        with self._lock:
            if allocations == self._members:
                return
            for allocation in self._members - allocations:
                objc.send_void_id(self.set, self._remove, allocation)
            for allocation in allocations - self._members:
                objc.send_void_id(self.set, self._add, allocation)
            self._members = allocations
            objc.send_void(self.set, self._commit)


@functools.cache
def _queue_residency() -> _QueueResidency:
    return _QueueResidency(_objc())


def install_queue_residency() -> None:
    """Make PyTorch's MPS buffers resident for the Mojo context's queue; call once before the GPU runtime exists."""
    _queue_residency().install()


def gpu_addresses(tensors: Sequence[Tensor]) -> tuple[int, ...]:
    """Device addresses of the first elements of the MPS `tensors` (0 for an empty one), their heaps made resident."""
    buffers: list[int] = [t.untyped_storage().data_ptr() for t in tensors]
    _queue_residency().cover(buffers)
    objc: _ObjC = _objc()
    gpu_address = objc.sel(b"gpuAddress")
    return tuple(
        (objc.send_u64(b, gpu_address) or 0) + int(t.storage_offset()) * t.element_size() if b else 0 for b, t in zip(buffers, tensors, strict=True)
    )
