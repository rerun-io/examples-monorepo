"""Test ops: verify CustomOpLibrary auto-compile, GPU launch, and Vec3 package import."""

from compiler import register
from tensor import InputTensor, OutputTensor, foreach
from std.runtime.asyncrt import DeviceContextPtr
from std.utils.index import IndexList


@register("test_add_one")
struct TestAddOne:
    """Minimal GPU op that adds 1 to every element."""

    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor,
        input: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        def add_one[width: Int](idx: IndexList[input.rank]) -> SIMD[input.dtype, width]:
            return input.load[width](idx) + 1

        foreach[add_one, target=target, simd_width=1](output, ctx)


@register("test_vec3_norm")
struct TestVec3Norm:
    """Test op: compute squared norm of [B, N, 3] points. Verifies Vec3 from __init__.mojo."""

    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor[dtype=DType.float32, rank=2, ...],
        input: InputTensor[dtype=DType.float32, rank=3, ...],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        def compute_norm[width: Int](idx: IndexList[output.rank]) -> SIMD[DType.float32, width]:
            var b = idx[0]
            var n = idx[1]
            # Vec3 is available from __init__.mojo package scope
            var v = Vec3(
                input.load[1](IndexList[3](b, n, 0))[0],
                input.load[1](IndexList[3](b, n, 1))[0],
                input.load[1](IndexList[3](b, n, 2))[0],
            )
            return SIMD[DType.float32, width](v.squared_norm())

        foreach[compute_norm, target=target, simd_width=1](output, ctx)
