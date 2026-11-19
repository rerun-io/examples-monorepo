"""NVIDIA's hardware optical-flow engine (NVOF) on torch CUDA tensors, through ctypes on the driver's library.

The driver ships ``libnvidia-opticalflow.so.1`` beside NVDEC's ``libnvcuvid`` and NVENC's ``libnvidia-encode``, so
nothing is packaged: the struct layouts below are the NV_OF API 2.0 interface of NVIDIA's BSD-3 headers
(github.com/NVIDIA/NVIDIAOpticalFlowSDK, ``nvOpticalFlowCommon.h`` / ``nvOpticalFlowCuda.h``, the ones OpenCV's
``NvidiaOpticalFlow_2_0`` builds on). A newer driver serves the older API version.

Convention: ``flow(input, reference)`` gives, per ``grid`` x ``grid`` block of ``input``, the displacement ``f`` with
``input(x) ~ reference(x + f)``, in pixels.
"""

from __future__ import annotations

import ctypes
import functools
from typing import ClassVar, Literal, TypeAlias

import torch
from jaxtyping import Float32, UInt8
from torch import Tensor

_API_VERSION: int = (2 << 4) | 0
"""NV_OF_API_VERSION 2.0."""
_SUCCESS: int = 0
_MODE_OPTICALFLOW: int = 1
_USAGE_INPUT, _USAGE_OUTPUT = 1, 2
_FORMAT_GRAYSCALE8, _FORMAT_SHORT2 = 1, 5
_CUDA_DEVICEPTR: int = 2
PerfLevel: TypeAlias = Literal["slow", "medium", "fast"]
_PERF_LEVELS: dict[PerfLevel, int] = {"slow": 5, "medium": 10, "fast": 20}
"""``NV_OF_PERF_LEVEL``: slow is the best quality."""


class _FunctionList(ctypes.Structure):
    _fields_: ClassVar = [
        (name, ctypes.c_void_p)
        for name in (
            "create",
            "init",
            "create_buffer",
            "buffer_cuarray",
            "buffer_deviceptr",
            "buffer_stride",
            "set_streams",
            "execute",
            "destroy_buffer",
            "destroy",
            "last_error",
            "caps",
        )
    ]


class _InitParams(ctypes.Structure):
    _fields_: ClassVar = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("out_grid", ctypes.c_int),
        ("hint_grid", ctypes.c_int),
        ("mode", ctypes.c_int),
        ("perf_level", ctypes.c_int),
        ("external_hints", ctypes.c_int),
        ("output_cost", ctypes.c_int),
        ("private", ctypes.c_void_p),
        ("disparity_range", ctypes.c_int),
        ("roi", ctypes.c_int),
    ]


class _BufferDescriptor(ctypes.Structure):
    _fields_: ClassVar = [("width", ctypes.c_uint32), ("height", ctypes.c_uint32), ("usage", ctypes.c_int), ("format", ctypes.c_int)]


class _PlaneStride(ctypes.Structure):
    _fields_: ClassVar = [("row_bytes", ctypes.c_uint32), ("rows", ctypes.c_uint32)]


class _StrideInfo(ctypes.Structure):
    _fields_: ClassVar = [("planes", _PlaneStride * 3), ("count", ctypes.c_uint32)]


class _ExecuteInput(ctypes.Structure):
    _fields_: ClassVar = [
        ("input", ctypes.c_void_p),
        ("reference", ctypes.c_void_p),
        ("hints", ctypes.c_void_p),
        ("disable_temporal_hints", ctypes.c_int),
        ("padding", ctypes.c_uint32),
        ("private", ctypes.c_void_p),
        ("padding2", ctypes.c_uint32),
        ("rois", ctypes.c_uint32),
        ("roi_data", ctypes.c_void_p),
    ]


class _ExecuteOutput(ctypes.Structure):
    _fields_: ClassVar = [("output", ctypes.c_void_p), ("cost", ctypes.c_void_p), ("private", ctypes.c_void_p)]


_Status = ctypes.c_int
_Handle = ctypes.c_void_p
_CREATE = ctypes.CFUNCTYPE(_Status, ctypes.c_void_p, ctypes.POINTER(_Handle))
_INIT = ctypes.CFUNCTYPE(_Status, _Handle, ctypes.POINTER(_InitParams))
_SET_STREAMS = ctypes.CFUNCTYPE(_Status, _Handle, ctypes.c_void_p, ctypes.c_void_p)
_CREATE_BUFFER = ctypes.CFUNCTYPE(_Status, _Handle, ctypes.POINTER(_BufferDescriptor), ctypes.c_int, ctypes.POINTER(_Handle))
_BUFFER_STRIDE = ctypes.CFUNCTYPE(_Status, _Handle, ctypes.POINTER(_StrideInfo))
_BUFFER_DEVICEPTR = ctypes.CFUNCTYPE(ctypes.c_uint64, _Handle)
_EXECUTE = ctypes.CFUNCTYPE(_Status, _Handle, ctypes.POINTER(_ExecuteInput), ctypes.POINTER(_ExecuteOutput))
_DESTROY = ctypes.CFUNCTYPE(_Status, _Handle)
"""Prototypes of the entries of ``NV_OF_CUDA_API_FUNCTION_LIST`` this module calls."""


class _DeviceArray:
    """A device pointer as ``__cuda_array_interface__``, so ``torch.as_tensor`` views it without a copy."""

    def __init__(self, pointer: int, shape: tuple[int, int, int], strides: tuple[int, int, int], typestr: str) -> None:
        self.__cuda_array_interface__: dict[str, object] = {"shape": shape, "strides": strides, "typestr": typestr, "data": (pointer, False), "version": 3, "stream": None}


@functools.cache
def _load(name: str) -> ctypes.CDLL:
    """A driver library, loaded once per process (ctypes never unloads it).

    Raises:
        RuntimeError: If the library is missing (no NVIDIA driver, or a container without its video capability).
    """
    try:
        return ctypes.CDLL(name)
    except OSError as error:
        raise RuntimeError(f"NVOF needs the NVIDIA driver's {name}: {error}") from error


def _check(status: int, call: str) -> None:
    if status != _SUCCESS:
        raise RuntimeError(f"NVOF {call} failed with NV_OF_STATUS {status}")


class OpticalFlow:
    """One NVOF session for ``width`` x ``height`` grayscale pairs, bound to the CUDA device and stream current when it is
    made: NVOF reuses one set of buffers, so every call must come from that stream.

    With temporal hints on (the default), each ``flow`` call seeds its search from the previous call's flow: feed one
    sequence's pairs in order.

    Raises:
        RuntimeError: If the driver has no NVOF, no CUDA context is current, or a call fails.
    """

    def __init__(self, width: int, height: int, *, grid: int = 4, perf: PerfLevel = "slow", temporal_hints: bool = True) -> None:
        """``grid``: one output vector per ``grid`` x ``grid`` block (1, 2 or 4)."""
        self._temporal_hints: bool = temporal_hints
        self._api: _FunctionList = _FunctionList()
        _check(_load("libnvidia-opticalflow.so.1").NvOFAPICreateInstanceCuda(_API_VERSION, ctypes.byref(self._api)), "NvOFAPICreateInstanceCuda")
        torch.cuda.synchronize()  # creates and binds torch's primary context on this thread
        context = ctypes.c_void_p()
        _load("libcuda.so.1").cuCtxGetCurrent(ctypes.byref(context))
        if not context:
            raise RuntimeError("NVOF needs a current CUDA context")
        self._stream: torch.cuda.Stream = torch.cuda.current_stream()
        self._handle: _Handle = _Handle()
        self._buffers: list[_Handle] = []
        self._views: list[Tensor] = []
        _check(_CREATE(self._api.create)(context, ctypes.byref(self._handle)), "create")
        try:
            params = _InitParams(width, height, grid, 0, _MODE_OPTICALFLOW, _PERF_LEVELS[perf], 0, 0, None, 0, 0)
            _check(_INIT(self._api.init)(self._handle, ctypes.byref(params)), "init")
            stream: int = self._stream.cuda_stream
            _check(_SET_STREAMS(self._api.set_streams)(self._handle, stream, stream), "set streams")
            self._input = self._buffer(width, height, _USAGE_INPUT, _FORMAT_GRAYSCALE8, "|u1", 1)
            self._reference = self._buffer(width, height, _USAGE_INPUT, _FORMAT_GRAYSCALE8, "|u1", 1)
            self._output = self._buffer(-(-width // grid), -(-height // grid), _USAGE_OUTPUT, _FORMAT_SHORT2, "<i2", 2)
        except BaseException:
            self.close()
            raise
        self._execute = _EXECUTE(self._api.execute)

    def flow(self, input: UInt8[Tensor, "h w"], reference: UInt8[Tensor, "h w"]) -> Float32[Tensor, "gh gw 2"]:
        """``f`` per grid block with ``input(x) ~ reference(x + f)``, pixels (NVOF's S10.5 fixed point divided out).

        Raises:
            RuntimeError: After ``close``, from another stream than the session's, or if NVOF fails.
        """
        if not self._handle:
            raise RuntimeError("the NVOF session is closed")
        if torch.cuda.current_stream() != self._stream:
            raise RuntimeError("an NVOF session runs on the CUDA stream it was made on")
        input_view, reference_view, output_view = self._views
        input_view.copy_(input[..., None])
        reference_view.copy_(reference[..., None])
        args_in = _ExecuteInput(self._input, self._reference, None, int(not self._temporal_hints), 0, None, 0, 0, None)
        args_out = _ExecuteOutput(self._output, None, None)
        _check(self._execute(self._handle, ctypes.byref(args_in), ctypes.byref(args_out)), "execute")
        return output_view.float().div_(32.0)

    def close(self) -> None:
        """Free the buffers and the session (after its stream's pending work); ``flow`` refuses afterwards."""
        self._stream.synchronize()
        self._views.clear()
        for buffer in self._buffers:
            _DESTROY(self._api.destroy_buffer)(buffer)
        self._buffers.clear()
        if self._handle:
            _DESTROY(self._api.destroy)(self._handle)
            self._handle = _Handle()

    def _buffer(self, width: int, height: int, usage: int, format: int, typestr: str, channels: int) -> _Handle:
        """A device buffer of NVOF's own pitch; its torch view (``height x width x channels``) joins ``_views``."""
        handle: _Handle = _Handle()
        descriptor = _BufferDescriptor(width, height, usage, format)
        _check(_CREATE_BUFFER(self._api.create_buffer)(self._handle, ctypes.byref(descriptor), _CUDA_DEVICEPTR, ctypes.byref(handle)), "create buffer")
        self._buffers.append(handle)
        stride = _StrideInfo()
        _check(_BUFFER_STRIDE(self._api.buffer_stride)(handle, ctypes.byref(stride)), "buffer stride")
        pointer: int = _BUFFER_DEVICEPTR(self._api.buffer_deviceptr)(handle)
        itemsize: int = int(typestr[-1])
        view = _DeviceArray(pointer, (height, width, channels), (stride.planes[0].row_bytes, channels * itemsize, itemsize), typestr)
        self._views.append(torch.as_tensor(view, device="cuda"))
        return handle
