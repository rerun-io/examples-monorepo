"""`sqrt_` and `sincos_`: the scalar math the Mojo 1.2.0 standard library does not give us on NVIDIA GPUs.

Measured against mpmath on sm_120 and on the CPU (study in the lietorch-mojo
Mojo 1.2.0 port report): `std.math.sin`/`cos` lower to `sin.approx.ftz.f32`/
`cos.approx.ftz.f32` for float32 and do not compile for float64, and
`std.math.sqrt` lowers to `sqrt.approx.ftz.f32` for float32. `std.math.atan`
is accurate on both devices, so the group code calls it directly.

- `sqrt_`: the `llvm.sqrt` intrinsic, `sqrt.rn` on NVIDIA and the IEEE square
  root on CPU (correctly rounded on both).
- `sincos_` in float64: FreeBSD msun (fdlibm) `__kernel_sin`/`__kernel_cos`
  after the medium path of `__ieee754_rem_pio2` (three-round Cody-Waite) for
  |x| < 2^20 pi/2, and a Payne-Hanek reduction against 1216 bits of 2/pi above.
- `sincos_` in float32: Cody-Waite reduction into a float-float argument for
  |x| <= 1024 (pi/2 in five pieces, error-free sums), then the fdlibm kernels
  with float32 Taylor coefficients; larger or non-finite arguments take the
  float64 routine.

Both are within one ulp of the exact result everywhere, including arguments up
to the float range.
"""

from std.bit import count_leading_zeros
from std.math import floor
from std.memory import bitcast
from std.sys import llvm_intrinsic


# ── Square root ──────────────────────────────────────────────────────────────


@inline(.always)
def sqrt_[dtype: DType, //](x: Scalar[dtype]) -> Scalar[dtype]:
    """Correctly rounded square root on CPU and GPU."""
    comptime match dtype:
        case .float32:
            return llvm_intrinsic["llvm.sqrt.f32", Scalar[dtype], has_side_effect=False](x)
        case .float64:
            return llvm_intrinsic["llvm.sqrt.f64", Scalar[dtype], has_side_effect=False](x)
        case _:
            comptime assert False, "lietorch_mojo supports float32 and float64"


# ── float64 kernels on [-pi/4, pi/4] (fdlibm) ────────────────────────────────

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


@inline(.always)
def _kernel_sin(x: Float64, y: Float64) -> Float64:
    """sin(x + y) for |x + y| <= pi/4; y is the tail of the reduced argument."""
    var z = x * x
    var w = z * z
    var r = _S2 + z * (_S3 + z * _S4) + z * w * (_S5 + z * _S6)
    var v = z * x
    return x - ((z * (0.5 * y - v * r) - y) - v * _S1)


@inline(.always)
def _kernel_cos(x: Float64, y: Float64) -> Float64:
    var z = x * x
    var w = z * z
    var r = z * (_C1 + z * (_C2 + z * _C3)) + w * w * (_C4 + z * (_C5 + z * _C6))
    var hz = 0.5 * z
    w = 1.0 - hz
    return w + (((1.0 - w) - hz) + (z * r - x * y))


@inline(.always)
def _quadrant[dtype: DType, //](n: Int, s: Scalar[dtype], c: Scalar[dtype]) -> Tuple[Scalar[dtype], Scalar[dtype]]:
    """(sin, cos) of n pi/2 + r from (sin r, cos r)."""
    match n & 3:
        case 0:
            return (s, c)
        case 1:
            return (c, -s)
        case 2:
            return (-s, -c)
        case _:
            return (-c, s)


# ── float64 argument reduction ───────────────────────────────────────────────

comptime _INVPIO2: Float64 = 6.36619772367581382433e-01
comptime _PIO2_1: Float64 = 1.57079632673412561417e00
comptime _PIO2_1T: Float64 = 6.07710050650619224932e-11
comptime _PIO2_2: Float64 = 6.07710050630396597660e-11
comptime _PIO2_2T: Float64 = 2.02226624879595063154e-21
comptime _PIO2_3: Float64 = 2.02226624871116645580e-21
comptime _PIO2_3T: Float64 = 8.47842766036889956997e-32
# pi/2 as a double plus its remainder.
comptime _PIO2_HI: Float64 = 1.5707963267948965580e00
comptime _PIO2_LO: Float64 = 6.1232339957367658e-17
comptime _MEDIUM_LIMIT: Float64 = 1647099.3291652855  # 2^20 * pi/2

# 2/pi in 64-bit words, most significant bit first, after one zero word, so the
# window for a small exponent can start before the binary point.
comptime _TWO_OVER_PI: Array[UInt64, 22] = [
    0x0000000000000000,
    0xA2F9836E4E441529,
    0xFC2757D1F534DDC0,
    0xDB6295993C439041,
    0xFE5163ABDEBBC561,
    0xB7246E3A424DD2E0,
    0x06492EEA09D1921C,
    0xFE1DEB1CB129A73E,
    0xE88235F52EBB4484,
    0xE99C7026B45F7E41,
    0x3991D639835339F4,
    0x9C845F8BBDF9283B,
    0x1FF897FFDE05980F,
    0xEF2F118B5A0A6D1F,
    0x6D367ECF27CB09B7,
    0x4F463F669E5FEA2D,
    0x7527BAC7EBE5F17B,
    0x3D0739F78A5292EA,
    0x6BFB5FB11F8D5D08,
    0x56033046FC7B6BAB,
    0xF0CFBC209AF4361D,
    0xA9E391615EE61B08,
]


@inline(.always)
def _biased_exponent(x: Float64) -> Int:
    return Int((bitcast[.uint64](x) >> 52) & 0x7FF)


@inline(.always)
def _rem_pio2_medium(x: Float64) -> Tuple[Float64, Float64, Int]:
    """fdlibm `__ieee754_rem_pio2`, medium path: x = n pi/2 + y0 + y1 for |x| < 2^20 pi/2."""
    var n_f = floor(x * _INVPIO2 + 0.5)
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
    return (y0, (r - y0) - w, Int(n_f))


@inline(.always)
def _mul_wide(a: UInt64, b: UInt64) -> Tuple[UInt64, UInt64]:
    """(low, high) 64-bit halves of a * b."""
    var p = UInt128(a) * UInt128(b)
    return (UInt64(p), UInt64(p >> 64))


@inline(.always)
def _window(words: Array[UInt64, 22], idx: Int, shift: Int) -> UInt64:
    """64 bits of `words` starting `shift` bits into word `idx`."""
    if shift == 0:
        return words[idx]
    return (words[idx] << UInt64(shift)) | (words[idx + 1] >> UInt64(64 - shift))


@inline(.never)
def _rem_pio2_huge(x: Float64) -> Tuple[Float64, Float64, Int]:
    """Payne-Hanek: x = n pi/2 + y0 + y1 for finite |x| >= 2^20 pi/2.

    |x| = m 2^e with a 53-bit integer m. Bits of 2/pi above 2^(2-e) only add
    multiples of 8 to x 2/pi, so a 192-bit window from there gives x 2/pi mod 8
    with 189 fraction bits; the rest of 2/pi adds less than 2^-136.
    """
    var bits = bitcast[.uint64](x)
    var e = Int((bits >> 52) & 0x7FF) - 1075
    var m = (bits & 0xFFFFFFFFFFFFF) | 0x10000000000000
    var pos = e + 61  # bit index of 2/pi's fraction bit e - 2, counting the zero word
    var idx, shift = divmod(pos, 64)
    var words = materialize[_TWO_OVER_PI]()
    var lo0, hi0 = _mul_wide(m, _window(words, idx + 2, shift))
    var lo1, hi1 = _mul_wide(m, _window(words, idx + 1, shift))
    var limb1 = hi0 + lo1
    var limb2 = hi1 + m * _window(words, idx, shift) + UInt64(Int(limb1 < hi0))
    var limb0 = lo0
    # Integer part mod 8 in the top 3 bits; round to the nearest integer.
    var q = Int(limb2 >> 61)
    limb2 &= 0x1FFFFFFFFFFFFFFF
    var negative = (limb2 >> 60) != 0
    if negative:
        q += 1
        # Fraction - 1, as a magnitude: 2^189 - fraction (two's complement over 189 bits).
        limb0 = ~limb0 + 1
        limb1 = ~limb1 + UInt64(Int(limb0 == 0))
        limb2 = (~limb2 + UInt64(Int(limb0 == 0 and limb1 == 0))) & 0x1FFFFFFFFFFFFFFF
    # Normalise the 189-bit magnitude so its top bit is bit 63 of `limb2` (it is never 0: 2/pi is irrational).
    var lz = 0
    for _ in range(2):
        if limb2 == 0:
            limb2 = limb1
            limb1 = limb0
            limb0 = 0
            lz += 64
    var s = Int(count_leading_zeros(limb2))
    if s > 0:
        limb2 = (limb2 << UInt64(s)) | (limb1 >> UInt64(64 - s))
        limb1 = (limb1 << UInt64(s)) | (limb0 >> UInt64(64 - s))
    lz += s
    # fraction = (limb2 + limb1 2^-64) 2^(-61 - lz), split into two doubles.
    var scale = bitcast[.float64](UInt64(1023 - 61 - lz) << 52)
    var f_hi = Float64(limb2 >> 11) * (2048.0 * scale)
    var f_lo = (Float64(limb2 & 0x7FF) + Float64(limb1) * 5.421010862427522e-20) * scale  # limb1 2^-64
    # y = fraction * pi/2 in double-double. Written with explicit FMAs: Mojo marks every multiply and add
    # `contract`, and a product rounded on its own would be fused into the next add on some targets.
    var tail = f_hi * _PIO2_LO + f_lo * _PIO2_HI
    var y0 = f_hi.fma(_PIO2_HI, tail)
    var y1 = f_hi.fma(_PIO2_HI, -y0) + tail
    if negative:
        y0 = -y0
        y1 = -y1
    if x < 0:
        return (-y0, -y1, -q)
    return (y0, y1, q)


@inline(.always)
def _sincos_f64(x: Float64) -> Tuple[Float64, Float64]:
    var y0: Float64
    var y1: Float64
    var n: Int
    if abs(x) < _MEDIUM_LIMIT:
        y0, y1, n = _rem_pio2_medium(x)
    elif _biased_exponent(x) == 0x7FF:
        var nan = x - x
        return (nan, nan)
    else:
        y0, y1, n = _rem_pio2_huge(x)
    return _quadrant(n, _kernel_sin(y0, y1), _kernel_cos(y0, y1))


# ── float32 ──────────────────────────────────────────────────────────────────

comptime _F32_LIMIT: Float32 = 1024.0
comptime _INVPIO2_F32: Float32 = 0.636619772367581343
# pi/2 = A + B + C + D + E to 81 bits. A..D have at most 14 bits, so j A .. j D are exact for
# |j| < 2^10 and stay exact whether or not the compiler fuses them into the next add.
comptime _PIO2_A: Float32 = 1.57080078125
comptime _PIO2_B: Float32 = -4.454515874385834e-06
comptime _PIO2_C: Float32 = 6.077272018956137e-11
comptime _PIO2_D: Float32 = -1.7150994166548195e-15
comptime _PIO2_E: Float32 = -2.5082788353727258e-20
# Taylor coefficients: the truncation error on [-pi/4, pi/4] is below 0.05 ulp.
comptime _SF1: Float32 = -1.0 / 6.0
comptime _SF2: Float32 = 1.0 / 120.0
comptime _SF3: Float32 = -1.0 / 5040.0
comptime _SF4: Float32 = 1.0 / 362880.0
comptime _CF1: Float32 = 1.0 / 24.0
comptime _CF2: Float32 = -1.0 / 720.0
comptime _CF3: Float32 = 1.0 / 40320.0
comptime _CF4: Float32 = -1.0 / 3628800.0


@inline(.always)
def _two_sum(a: Float32, b: Float32) -> Tuple[Float32, Float32]:
    """(a + b, its exact rounding error) (Knuth)."""
    var s = a + b
    var bb = s - a
    return (s, (a - (s - bb)) + (b - bb))


@inline(.never)
def _sincos_f32_wide(x: Float32) -> Tuple[Float32, Float32]:
    """|x| > 1024 or non-finite: the float64 routine, rounded. Out of line so float32 kernels carry no FP64 code."""
    var s, c = _sincos_f64(Float64(x))
    return (Float32(s), Float32(c))


@inline(.always)
def _sincos_f32(x: Float32) -> Tuple[Float32, Float32]:
    if not abs(x) <= _F32_LIMIT:
        return _sincos_f32_wide(x)
    var j = floor(x * _INVPIO2_F32 + 0.5)
    # r = x - j pi/2 as hi + lo: x - j A is exact (Sterbenz), each later step keeps its rounding error.
    var hi = x - j * _PIO2_A
    var lo: Float32 = 0
    comptime for piece in [_PIO2_B, _PIO2_C, _PIO2_D]:
        var err: Float32
        hi, err = _two_sum(hi, -(j * piece))
        lo += err
    lo -= j * _PIO2_E
    var y0 = hi + lo
    var y1 = lo - (y0 - hi)
    var z = y0 * y0
    var s_poly = _SF2 + z * (_SF3 + z * _SF4)
    var v = z * y0
    var s = y0 - ((z * (0.5 * y1 - v * s_poly) - y1) - v * _SF1)
    var c_poly = z * (_CF1 + z * (_CF2 + z * (_CF3 + z * _CF4)))
    var hz = 0.5 * z
    var w = 1.0 - hz
    var c = w + (((1.0 - w) - hz) + (z * c_poly - y0 * y1))
    return _quadrant(Int(j), s, c)


@inline(.always)
def sincos_[dtype: DType, //](x: Scalar[dtype]) -> Tuple[Scalar[dtype], Scalar[dtype]]:
    """Return (sin(x), cos(x)) within about one ulp, for every finite x."""
    comptime match dtype:
        case .float32:
            var s, c = _sincos_f32(x.cast[.float32]())
            return (s.cast[dtype](), c.cast[dtype]())
        case .float64:
            var s, c = _sincos_f64(x.cast[.float64]())
            return (s.cast[dtype](), c.cast[dtype]())
        case _:
            comptime assert False, "lietorch_mojo supports float32 and float64"
