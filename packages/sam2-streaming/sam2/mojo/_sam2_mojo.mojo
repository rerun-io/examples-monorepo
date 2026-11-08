"""CPython entry point of SAM 2's connected-components kernel in Mojo (`sam2._sam2_mojo`).

`connected_components(runtime, scalars, tensors)`, as vipe_mojo's ops:
- `runtime`: lietorch-mojo's `GPURuntime`, from `lietorch_mojo.backends._runtime(True)`. This module is built
  against the same Mojo structs (`-I <lietorch-mojo>/lietorch_mojo/mojo`) and must not register them again.
- `scalars`: `(stream,)`, the raw CUDA stream of PyTorch's current stream.
- `tensors`: `(address, *shape)` of contiguous CUDA tensors: mask u8 (N, 1, H, W), labels, counts_init, counts i32
  (N, 1, H, W), the last three zeroed. `sam2.mojo_cc` owns the dtype and shape checks.
"""

from std.math import ceildiv
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from connected_components import (
    BLOCK,
    I32Ptr,
    U8Ptr,
    compression_kernel,
    final_counting_kernel,
    final_labeling_kernel,
    init_counting_kernel,
    init_labeling_kernel,
    merge_kernel,
)
from launch import GPURuntime


@always_inline
def _ptr[dtype: DType](spec: PythonObject) raises -> MutPointer[Scalar[dtype], MutUntrackedOrigin]:
    return MutPointer[Scalar[dtype], MutUntrackedOrigin](unsafe_from_address=Int(py=spec[0]))


def _connected_components_py(runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    var batch = Int(py=t[0][1])
    var height = Int(py=t[0][3])
    var width = Int(py=t[0][4])
    var img = _ptr[DType.uint8](t[0])
    var labels = _ptr[DType.int32](t[1])
    var counts_init = _ptr[DType.int32](t[2])
    var counts = _ptr[DType.int32](t[3])
    var h = Int32(height)
    var w = Int32(width)
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    var stream = rt[].torch_stream(Int(py=s[0]))
    # One thread per 2x2 block for the labelling, one per pixel for the counting; grid z = batch item.
    var blocks = (ceildiv((width + 1) // 2, BLOCK), ceildiv((height + 1) // 2, BLOCK), batch)
    var pixels = (ceildiv(width, BLOCK), ceildiv(height, BLOCK), batch)
    stream.enqueue_function(rt[].ctx.compile_function[init_labeling_kernel](), labels, h, w, grid_dim=blocks, block_dim=(BLOCK, BLOCK))
    stream.enqueue_function(rt[].ctx.compile_function[merge_kernel](), img, labels, h, w, grid_dim=blocks, block_dim=(BLOCK, BLOCK))
    stream.enqueue_function(rt[].ctx.compile_function[compression_kernel](), labels, h, w, grid_dim=blocks, block_dim=(BLOCK, BLOCK))
    stream.enqueue_function(rt[].ctx.compile_function[final_labeling_kernel](), img, labels, h, w, grid_dim=blocks, block_dim=(BLOCK, BLOCK))
    stream.enqueue_function(rt[].ctx.compile_function[init_counting_kernel](), labels, counts_init, h, w, grid_dim=pixels, block_dim=(BLOCK, BLOCK))
    stream.enqueue_function(
        rt[].ctx.compile_function[final_counting_kernel](), labels, counts_init, counts, h, w, grid_dim=pixels, block_dim=(BLOCK, BLOCK)
    )
    return Python.none()


@export
def PyInit__sam2_mojo() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_sam2_mojo")
        m.def_function[_connected_components_py]("connected_components")
        return m.finalize()
    except e:
        abort(t"failed to create _sam2_mojo: {e}")
