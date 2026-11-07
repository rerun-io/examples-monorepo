"""`exp_` and `log_` for RxSO3/Sim3's scale: accurate on CPU and NVIDIA GPU.

Measured on Mojo 1.2.0 (RTX 5090 and CPU) through RxSO3 `expm`/`logm`
against mpmath: `std.math.exp` is off by up to ~3.7e4 ulp in float64 (CPU and
GPU) and ~7.6 ulp in float32 on the GPU; `std.math.log` is within 0.53 ulp in
float64 but loses almost all relative accuracy near 1 in float32 on the GPU
(lg2.approx). So:

- `exp_` is fdlibm's `__ieee754_exp` (Cody-Waite reduction by ln 2 and its
  Remez polynomial; < 1 ulp), in float64. float32 rounds the float64 result.
- `log_` is `std.math.log` in float64; float32 rounds the float64 result.

float32 therefore spends one FP64 exp or log per call (two in Sim3 `Exp`). An
Apple GPU has no float64: there float32 takes `std.math.exp`/`log` in float32.
"""

from std.math import exp, floor, log
from std.sys.info import is_apple_gpu
from std.memory import bitcast

comptime _LN2_HI: Float64 = 6.93147180369123816490e-01
comptime _LN2_LO: Float64 = 1.90821492927058770002e-10
comptime _INV_LN2: Float64 = 1.44269504088896338700e00
comptime _P1: Float64 = 1.66666666666666019037e-01
comptime _P2: Float64 = -2.77777777770155933842e-03
comptime _P3: Float64 = 6.61375632143793436117e-05
comptime _P4: Float64 = -1.65339022054652515390e-06
comptime _P5: Float64 = 4.13813679705723846039e-08
comptime _O_THRESHOLD: Float64 = 7.09782712893383973096e02
comptime _U_THRESHOLD: Float64 = -7.45133219101941108420e02


@inline(.always)
def _pow2(m: Int) -> Float64:
    """2^m for m in [-1022, 1023], exactly."""
    return bitcast[.float64](UInt64(m + 1023) << 52)


@inline(.always)
def _exp_f64(x: Float64) -> Float64:
    if x != x:
        return x
    if x > _O_THRESHOLD:
        return Float64.MAX * Float64.MAX  # inf
    if x < _U_THRESHOLD:
        return 0.0
    if abs(x) < 3.725290298461914e-09:  # 2^-28: exp(x) rounds to 1 + x
        return 1.0 + x
    var k_f = floor(x * _INV_LN2 + 0.5)
    var hi = x - k_f * _LN2_HI  # exact: _LN2_HI has 11 trailing zero bits
    var lo = k_f * _LN2_LO
    var r = hi - lo
    var t = r * r
    var c = r - t * (_P1 + t * (_P2 + t * (_P3 + t * (_P4 + t * _P5))))
    var k = Int(k_f)
    if k == 0:
        return 1.0 - ((r * c) / (c - 2.0) - r)
    var y = 1.0 - ((lo - (r * c) / (2.0 - c)) - hi)
    # Two exact power-of-two factors keep 2^k representable down to the subnormal range.
    var k1 = k // 2
    return y * _pow2(k1) * _pow2(k - k1)


@inline(.always)
def exp_[dtype: DType, //](x: Scalar[dtype]) -> Scalar[dtype]:
    """exp(x) within one ulp on CPU and GPU (float32 and float64)."""
    comptime match dtype:
        case .float32 | .float64:
            comptime if is_apple_gpu():
                return Scalar[dtype](exp(Float32(x)))
            else:
                return Scalar[dtype](_exp_f64(Float64(x)))
        case _:
            comptime assert False, "lietorch_mojo supports float32 and float64"


@inline(.always)
def log_[dtype: DType, //](x: Scalar[dtype]) -> Scalar[dtype]:
    """log(x) within one ulp on CPU and GPU (float32 and float64)."""
    comptime match dtype:
        case .float32 | .float64:
            comptime if is_apple_gpu():
                return Scalar[dtype](log(Float32(x)))
            else:
                return Scalar[dtype](log(Float64(x)))
        case _:
            comptime assert False, "lietorch_mojo supports float32 and float64"
