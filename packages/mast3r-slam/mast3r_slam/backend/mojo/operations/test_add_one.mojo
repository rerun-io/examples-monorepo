"""Minimal test op: adds 1.0 to every element. Used to verify CustomOpLibrary auto-compile works."""

from compiler import register
from tensor import InputTensor, OutputTensor, foreach
from std.runtime.asyncrt import DeviceContextPtr
from std.utils.index import IndexList


@register("test_add_one")
struct TestAddOne:
    """Minimal GPU op that adds 1 to every element. Verifies the full pipeline:
    CustomOpLibrary(directory) → auto-compile → DLPack marshal → GPU launch → result.
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        def add_one[width: Int](idx: IndexList[input.rank]) -> SIMD[input.dtype, width]:
            return input.load[width](idx) + 1

        foreach[add_one, target=target, simd_width=1](output, ctx)
