"""nvcc `--use_fast_math` float arithmetic, one PTX instruction per call.

hexel-kernel is built with `--use_fast_math`: every float op flushes denormals (`.ftz`), division, reciprocal,
`__expf`, `logf`, `sincosf` and `rsqrtf` are the `.approx` instructions, and nvcc chooses which multiply-adds to
fuse. LLVM would fuse differently, so a kernel that must match it bit for bit spells every operation with these
functions, in the order of nvcc's PTX. Each is one inline PTX instruction, which LLVM never fuses or reorders
(this LLVM has no `llvm.nvvm.{add,mul}.rn.ftz` intrinsics). Off the GPU (the host half of the shared library
also lowers these functions) they fall back to the plain operation.
"""

from std import math
from std.sys._assembly import inlined_assembly
from std.sys.info import is_nvidia_gpu

comptime LOG2E: Float32 = 1.4426950408889634  # 0f3FB8AA3B, nvcc's __expf constant
comptime LN2: Float32 = 0.6931471805599453  # 0f3F317218, nvcc's fast logf constant


@always_inline
def add(a: Float32, b: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return inlined_assembly["add.ftz.f32 $0, $1, $2;", Float32, constraints="=f,f,f", has_side_effect=False](a, b)
    else:
        return a + b


@always_inline
def sub(a: Float32, b: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return inlined_assembly["sub.ftz.f32 $0, $1, $2;", Float32, constraints="=f,f,f", has_side_effect=False](a, b)
    else:
        return a - b


@always_inline
def mul(a: Float32, b: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return inlined_assembly["mul.ftz.f32 $0, $1, $2;", Float32, constraints="=f,f,f", has_side_effect=False](a, b)
    else:
        return a * b


@always_inline
def fma(a: Float32, b: Float32, c: Float32) -> Float32:
    """a * b + c, rounded once."""
    comptime if is_nvidia_gpu():
        return inlined_assembly["fma.rn.ftz.f32 $0, $1, $2, $3;", Float32, constraints="=f,f,f,f", has_side_effect=False](a, b, c)
    else:
        return math.fma(a, b, c)


@always_inline
def div(a: Float32, b: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return inlined_assembly["div.approx.ftz.f32 $0, $1, $2;", Float32, constraints="=f,f,f", has_side_effect=False](a, b)
    else:
        return a / b


@always_inline
def rcp(a: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return inlined_assembly["rcp.approx.ftz.f32 $0, $1;", Float32, constraints="=f,f", has_side_effect=False](a)
    else:
        return 1.0 / a


@always_inline
def rsqrt(a: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return inlined_assembly["rsqrt.approx.ftz.f32 $0, $1;", Float32, constraints="=f,f", has_side_effect=False](a)
    else:
        return math.rsqrt(a)


@always_inline
def sin(a: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return inlined_assembly["sin.approx.ftz.f32 $0, $1;", Float32, constraints="=f,f", has_side_effect=False](a)
    else:
        return math.sin(a)


@always_inline
def cos(a: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return inlined_assembly["cos.approx.ftz.f32 $0, $1;", Float32, constraints="=f,f", has_side_effect=False](a)
    else:
        return math.cos(a)


@always_inline
def ex2(a: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return inlined_assembly["ex2.approx.ftz.f32 $0, $1;", Float32, constraints="=f,f", has_side_effect=False](a)
    else:
        return math.exp2(a)


@always_inline
def lg2(a: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return inlined_assembly["lg2.approx.ftz.f32 $0, $1;", Float32, constraints="=f,f", has_side_effect=False](a)
    else:
        return math.log2(a)


@always_inline
def expf(a: Float32) -> Float32:
    """`__expf`: 2^(a log2 e)."""
    return ex2(mul(a, LOG2E))
