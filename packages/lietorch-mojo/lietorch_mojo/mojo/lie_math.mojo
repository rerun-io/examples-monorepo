"""Scalar math that runs the same on CPU and on NVIDIA GPUs.

Why this file exists
--------------------
The Mojo standard library lowers `sin`, `cos` and `sqrt` for float32 on NVIDIA
GPUs to the PTX `*.approx.ftz.f32` instructions, and it does not implement
float64 `sin`, `cos`, `atan` or `sqrt` there at all (checked against
Mojo 1.0.0b3). lietorch's CUDA build uses the IEEE-accurate CUDA math library,
so the approximate instructions would break parity near the small-angle
branches. This file gives one implementation for both targets:

- `sqrt_`: the `llvm.sqrt` intrinsic, which lowers to `sqrt.rn.f32/f64` on
  NVIDIA and to the IEEE square root on CPU (correctly rounded on both).
- `sincos_`, `atan_` in float64: ports of FreeBSD msun (fdlibm)
  `__kernel_sin`, `__kernel_cos`, the medium path of `__ieee754_rem_pio2`
  (three-round Cody-Waite reduction, accurate for |x| < 2^20 * pi/2) and
  `atan`.
- `sincos_` in float32: the Cephes `sinf`/`cosf` algorithm (octant reduction
  with the three-part DP1/DP2/DP3 constants, exact for |x| < 8192).
- `atan_` in float32: FreeBSD msun `atanf`.

The errors of these routines are at or below one ulp in their ranges; the CUDA
library guarantees two ulp for `sinf`/`cosf`/`atanf` and one or two for the
double versions. Float32 arguments with |x| > 8192 go through the float64
routine. Float64 arguments beyond 2^20 * pi/2 lose accuracy gradually (the
Cody-Waite products stop being exact); lietorch only feeds rotation angles
to these functions.
"""

# TODO(no ticket filed yet): drop this file once std.math lowers float32 sin/cos/sqrt to the
# IEEE (non-.approx) PTX instructions and implements float64 sin/cos/atan/sqrt on NVIDIA GPUs.

from std.memory import bitcast
from std.math import floor
from std.sys import llvm_intrinsic


# ── Square root ──────────────────────────────────────────────────────────────


@always_inline
def sqrt_[dtype: DType, //](x: Scalar[dtype]) -> Scalar[dtype]:
    """Correctly rounded square root on CPU and GPU."""
    comptime if dtype == DType.float32:
        return llvm_intrinsic["llvm.sqrt.f32", Scalar[dtype], has_side_effect=False](x)
    else:
        comptime assert dtype == DType.float64, "lietorch_mojo supports float32 and float64"
        return llvm_intrinsic["llvm.sqrt.f64", Scalar[dtype], has_side_effect=False](x)


# ── float64 sin/cos (fdlibm) ─────────────────────────────────────────────────

comptime _S1: Float64 = -1.66666666666666324348e-01
comptime _S2: Float64 = 8.33333333332248946124e-03
comptime _S3: Float64 = -1.98412698298579493134e-04
comptime _S4: Float64 = 2.75573137070700676789e-06
comptime _S5: Float64 = -2.50507602534068634195e-08
comptime _S6: Float64 = 1.58969099521155010221e-10

comptime _C1: Float64 = 4.16666666666666019037e-02
comptime _C2: Float64 = -1.38888888888741095749e-03
comptime _C3: Float64 = 2.48015872894767294178e-05
comptime _C4: Float64 = -2.75573143513906633035e-07
comptime _C5: Float64 = 2.08757232129817482790e-09
comptime _C6: Float64 = -1.13596475577881948265e-11

comptime _INVPIO2: Float64 = 6.36619772367581382433e-01
comptime _PIO2_1: Float64 = 1.57079632673412561417e00
comptime _PIO2_1T: Float64 = 6.07710050650619224932e-11
comptime _PIO2_2: Float64 = 6.07710050630396597660e-11
comptime _PIO2_2T: Float64 = 2.02226624879595063154e-21
comptime _PIO2_3: Float64 = 2.02226624871116645580e-21
comptime _PIO2_3T: Float64 = 8.47842766036889956997e-32
comptime _PIO4_F64: Float64 = 7.85398163397448278999e-01


@always_inline
def _kernel_sin(x: Float64, y: Float64) -> Float64:
    # fdlibm __kernel_sin with iy = 1 (y is the tail of the reduced argument).
    var z = x * x
    var w = z * z
    var r = _S2 + z * (_S3 + z * _S4) + z * w * (_S5 + z * _S6)
    var v = z * x
    return x - ((z * (0.5 * y - v * r) - y) - v * _S1)


@always_inline
def _kernel_cos(x: Float64, y: Float64) -> Float64:
    var z = x * x
    var w = z * z
    var r = z * (_C1 + z * (_C2 + z * _C3)) + w * w * (_C4 + z * (_C5 + z * _C6))
    var hz = 0.5 * z
    w = 1.0 - hz
    return w + (((1.0 - w) - hz) + (z * r - x * y))


@always_inline
def _biased_exponent(x: Float64) -> Int:
    return Int((bitcast[DType.uint64](x) >> 52) & 0x7FF)


@always_inline
def _sincos_f64(x: Float64) -> Tuple[Float64, Float64]:
    # __ieee754_rem_pio2, medium path, also for |x| <= pi/4 (n = 0, y = (x, 0)):
    # one code path instead of fdlibm's two keeps GPU kernels smaller.
    var n_f = floor(x * _INVPIO2 + 0.5)
    var n = Int(n_f)
    var r = x - n_f * _PIO2_1
    var w = n_f * _PIO2_1T
    var j = _biased_exponent(x)
    var y0 = r - w
    if j - _biased_exponent(y0) > 16:
        var t = r
        w = n_f * _PIO2_2
        r = t - w
        w = n_f * _PIO2_2T - ((t - r) - w)
        y0 = r - w
        if j - _biased_exponent(y0) > 49:
            t = r
            w = n_f * _PIO2_3
            r = t - w
            w = n_f * _PIO2_3T - ((t - r) - w)
            y0 = r - w
    var y1 = (r - y0) - w
    var ks = _kernel_sin(y0, y1)
    var kc = _kernel_cos(y0, y1)
    var q = n & 3
    if q == 0:
        return (ks, kc)
    if q == 1:
        return (kc, -ks)
    if q == 2:
        return (-ks, -kc)
    return (-kc, ks)


# ── float32 sin/cos (Cephes) ─────────────────────────────────────────────────

comptime _FOPI: Float32 = 1.27323954473516
comptime _DP1: Float32 = 0.78515625
comptime _DP2: Float32 = 2.4187564849853515625e-4
comptime _DP3: Float32 = 3.77489497744594108e-8
comptime _PIO4F: Float32 = 0.7853981633974483096
comptime _LOSSTH: Float32 = 8192.0


# TODO(no ticket filed yet): `@no_inline` stops the NVPTX backend from if-converting this rare
# FP64 path into every float32 call; replace it with a branch-weight hint once Mojo exposes one.
@no_inline
def _sincos_f32_via_f64(xx: Float32) -> Tuple[Float32, Float32]:
    var r = _sincos_f64(Float64(xx))
    return (Float32(r[0]), Float32(r[1]))


@always_inline
def _sincos_f32(xx: Float32) -> Tuple[Float32, Float32]:
    var x = abs(xx)
    if x > _LOSSTH:
        # Outside Cephes' exact-reduction range: use the float64 routine and round.
        return _sincos_f32_via_f64(xx)
    var j = Int(_FOPI * x)
    var y = Float32(j)
    if (j & 1) == 1:
        j += 1
        y += 1.0
    j &= 7
    var sin_sign: Float32 = -1.0 if xx < 0.0 else 1.0
    var cos_sign: Float32 = 1.0
    if j > 3:
        sin_sign = -sin_sign
        cos_sign = -cos_sign
        j -= 4
    if j > 1:
        cos_sign = -cos_sign
    x = ((x - y * _DP1) - y * _DP2) - y * _DP3
    var z = x * x
    var cos_poly = ((2.443315711809948e-005 * z - 1.388731625493765e-003) * z + 4.166664568298827e-002) * z * z
    cos_poly -= 0.5 * z
    cos_poly += 1.0
    var sin_poly = ((-1.9515295891e-4 * z + 8.3321608736e-3) * z - 1.6666654611e-1) * z * x
    sin_poly += x
    if j == 2:
        return (sin_sign * cos_poly, cos_sign * sin_poly)
    return (sin_sign * sin_poly, cos_sign * cos_poly)


@always_inline
def sincos_[dtype: DType, //](x: Scalar[dtype]) -> Tuple[Scalar[dtype], Scalar[dtype]]:
    """Return (sin(x), cos(x)) with at most about one ulp of error."""
    comptime if dtype == DType.float32:
        var r = _sincos_f32(x.cast[DType.float32]())
        return (r[0].cast[dtype](), r[1].cast[dtype]())
    else:
        comptime assert dtype == DType.float64, "lietorch_mojo supports float32 and float64"
        var r = _sincos_f64(x.cast[DType.float64]())
        return (r[0].cast[dtype](), r[1].cast[dtype]())


# ── atan (fdlibm s_atan.c / s_atanf.c) ───────────────────────────────────────


@always_inline
def _atan_reduce[dtype: DType, //](x_in: Scalar[dtype], ax: Scalar[dtype]) -> Tuple[Scalar[dtype], Int]:
    """fdlibm's atan argument reduction: (reduced x, interval id; -1 means no reduction)."""
    var x = x_in
    var id = -1
    if ax >= 0.4375:
        x = ax
        if ax < 1.1875:
            if ax < 0.6875:
                id = 0
                x = (2.0 * x - 1.0) / (2.0 + x)
            else:
                id = 1
                x = (x - 1.0) / (x + 1.0)
        elif ax < 2.4375:
            id = 2
            x = (x - 1.5) / (1.0 + 1.5 * x)
        else:
            id = 3
            x = -1.0 / x
    return (x, id)


@always_inline
def _atan_f64(x_in: Float64) -> Float64:
    var neg = x_in < 0.0
    var ax = abs(x_in)
    var x, id = _atan_reduce(x_in, ax)
    var z = x * x
    var w = z * z
    var s1 = z * (
        3.33333333333329318027e-01
        + w * (1.42857142725034663711e-01 + w * (9.09088713343650656196e-02 + w * (6.66107313738753120669e-02 + w * (4.97687799461593236017e-02 + w * 1.62858201153657823623e-02))))
    )
    var s2 = w * (
        -1.99999999998764832476e-01
        + w * (-1.11111104054623557880e-01 + w * (-7.69187620504482999495e-02 + w * (-5.83357013379057348645e-02 + w * -3.65315727442169155270e-02)))
    )
    if id < 0:
        return x - x * (s1 + s2)
    var hi: Float64
    var lo: Float64
    if id == 0:
        hi = 4.63647609000806093515e-01
        lo = 2.26987774529616870924e-17
    elif id == 1:
        hi = 7.85398163397448278999e-01
        lo = 3.06161699786838301793e-17
    elif id == 2:
        hi = 9.82793723247329054082e-01
        lo = 1.39033110312309984516e-17
    else:
        hi = 1.57079632679489655800e00
        lo = 6.12323399573676603587e-17
    var r = hi - ((x * (s1 + s2) - lo) - x)
    return -r if neg else r


@always_inline
def _atan_f32(x_in: Float32) -> Float32:
    var neg = x_in < 0.0
    var ax = abs(x_in)
    var x, id = _atan_reduce(x_in, ax)
    var z = x * x
    var w = z * z
    var s1 = z * (3.3333328366e-01 + w * (1.4253635705e-01 + w * 6.1687607318e-02))
    var s2 = w * (-1.9999158382e-01 + w * -1.0648017377e-01)
    if id < 0:
        return x - x * (s1 + s2)
    var hi: Float32
    var lo: Float32
    if id == 0:
        hi = 4.6364760399e-01
        lo = 5.0121582440e-09
    elif id == 1:
        hi = 7.8539812565e-01
        lo = 3.7748947079e-08
    elif id == 2:
        hi = 9.8279368877e-01
        lo = 3.4473217170e-08
    else:
        hi = 1.5707962513e00
        lo = 7.5497894159e-08
    var r = hi - ((x * (s1 + s2) - lo) - x)
    return -r if neg else r


@always_inline
def atan_[dtype: DType, //](x: Scalar[dtype]) -> Scalar[dtype]:
    """Arctangent with at most about one ulp of error."""
    comptime if dtype == DType.float32:
        return _atan_f32(x.cast[DType.float32]()).cast[dtype]()
    else:
        comptime assert dtype == DType.float64, "lietorch_mojo supports float32 and float64"
        return _atan_f64(x.cast[DType.float64]()).cast[dtype]()
