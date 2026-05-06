from compiler import register
from tensor import InputTensor, OutputTensor
from std.gpu import global_idx
from std.math import ceildiv
from std.runtime.asyncrt import DeviceContextPtr
from std.utils.index import IndexList


# MAX CustomOpLibrary smoke op for the standalone fastba package.
#
# The real DPVO fastba kernels live in `native/dpvo_fastba_mojo_backends.mojo`.
# This file intentionally stays small: it validates that the package can load a
# MAX/PyTorch custom op without duplicating bundle-adjustment math.


@always_inline
def _load_f32_1d(tensor: InputTensor[dtype=DType.float32, rank=1, ...], i0: Int) -> Float32:
    return tensor.load[1](IndexList[1](i0))[0]


@register("fastba_smoke_scale")
struct FastBASmokeScale:
    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor[dtype=DType.float32, rank=1, ...],
        x: InputTensor[dtype=DType.float32, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        comptime if target != "gpu":
            raise Error("fastba_smoke_scale currently requires a GPU tensor")
        var gpu_ctx = ctx.get_device_context()
        var element_count = output.dim_size[0]()
        comptime BLOCK_SIZE = 256

        @parameter
        def smoke_scale_kernel(element_count: Int):
            var tid = global_idx.x
            if tid < element_count:
                var idx = tid
                var value = _load_f32_1d(x, idx) * 2.0
                output.store[1](IndexList[1](idx), SIMD[DType.float32, 1](value))

        gpu_ctx.enqueue_function_experimental[smoke_scale_kernel](
            element_count,
            grid_dim=ceildiv(element_count, BLOCK_SIZE),
            block_dim=BLOCK_SIZE,
        )
