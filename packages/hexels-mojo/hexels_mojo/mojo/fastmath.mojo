"""nvcc `--use_fast_math` float arithmetic, one PTX instruction per call.

hexel-kernel is built with `--use_fast_math`: float ops flush denormals (`.ftz`), and division, reciprocal, `__expf`,
`logf`, `sincosf` and `rsqrtf` are `.approx` instructions. A kernel that must match it bit for bit spells every
operation with these functions in the order of nvcc's PTX: each lowers to that one instruction and LLVM never fuses
it with its neighbours. Off the GPU (the host half of the shared library lowers them too) they fall back to the
plain operation.
"""

from std import math
from std.sys import llvm_intrinsic
from std.sys._assembly import inlined_assembly
from std.sys.info import is_nvidia_gpu

comptime LOG2E: Float32 = 1.4426950408889634  # 0f3FB8AA3B, nvcc's __expf constant
comptime LN2: Float32 = 0.6931471805599453  # 0f3F317218, nvcc's fast logf constant


@inline(.always)
def _ftz[op: StaticString](a: Float32, b: Float32) -> Float32:
    """`<op>.ftz.f32` as inline PTX: this LLVM has no `llvm.nvvm.{add,sub,mul}.rn.ftz.f` intrinsics."""
    return inlined_assembly[op + ".ftz.f32 $0, $1, $2;", Float32, constraints="=f,f,f", has_side_effect=False](a, b)


@inline(.always)
def add(a: Float32, b: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return _ftz["add"](a, b)
    else:
        return a + b


@inline(.always)
def sub(a: Float32, b: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return _ftz["sub"](a, b)
    else:
        return a - b


@inline(.always)
def mul(a: Float32, b: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return _ftz["mul"](a, b)
    else:
        return a * b


@inline(.always)
def fma(a: Float32, b: Float32, c: Float32) -> Float32:
    """a * b + c, rounded once."""
    comptime if is_nvidia_gpu():
        return llvm_intrinsic["llvm.nvvm.fma.rn.ftz.f", Float32](a, b, c)
    else:
        return math.fma(a, b, c)


@inline(.always)
def div(a: Float32, b: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return llvm_intrinsic["llvm.nvvm.div.approx.ftz.f", Float32](a, b)
    else:
        return a / b


@inline(.always)
def rcp(a: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return llvm_intrinsic["llvm.nvvm.rcp.approx.ftz.f", Float32](a)
    else:
        return 1.0 / a


@inline(.always)
def rsqrt(a: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return llvm_intrinsic["llvm.nvvm.rsqrt.approx.ftz.f", Float32](a)
    else:
        return math.rsqrt(a)


@inline(.always)
def sqrt(a: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return llvm_intrinsic["llvm.nvvm.sqrt.approx.ftz.f", Float32](a)
    else:
        return math.sqrt(a)


@inline(.always)
def sin(a: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return llvm_intrinsic["llvm.nvvm.sin.approx.ftz.f", Float32](a)
    else:
        return math.sin(a)


@inline(.always)
def cos(a: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return llvm_intrinsic["llvm.nvvm.cos.approx.ftz.f", Float32](a)
    else:
        return math.cos(a)


@inline(.always)
def ex2(a: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return llvm_intrinsic["llvm.nvvm.ex2.approx.ftz.f", Float32](a)
    else:
        return math.exp2(a)


@inline(.always)
def lg2(a: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return llvm_intrinsic["llvm.nvvm.lg2.approx.ftz.f", Float32](a)
    else:
        return math.log2(a)


@inline(.always)
def expf(a: Float32) -> Float32:
    """`__expf`: 2^(a log2 e)."""
    return ex2(mul(a, LOG2E))
