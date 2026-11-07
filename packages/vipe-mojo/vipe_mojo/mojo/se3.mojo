"""The SE3 helpers of ViPE's SLAM kernels (`csrc/slam_ext/geom_kernels.cu`, the same functions as DPVO's `ba_cuda.cu`): poses are (tx ty tz qx qy qz qw) rows.

Ported line for line from the CUDA device functions so that the BA and reprojection kernels keep
the CUDA arithmetic order. Vectors travel as SIMD values: a 3-vector or a homogeneous point in
SIMD[float32, 4] (lane 3 unused for 3-vectors), a quaternion (x y z w) in SIMD[float32, 4], a
tangent or Jacobian row (tau, phi) in the first 6 lanes of SIMD[float32, 8].

The CUDA source writes some constants as doubles (`2.0 * (...)`, `0.5 - (1.0/48.0) * theta_sq`).
Metal has no float64, so they are float32 here: the factor 2 is exact either way, and the series
terms differ by at most a float32 rounding.

sin, cos and sqrt come from lietorch-mojo's `lie_math` (`sincos_`, `sqrt_`), within one ulp on CPU
and NVIDIA. Mojo 1.2's `std.math.cos` lowers to `cos.approx.ftz.f32` on NVIDIA, and
`(1 - cos(theta)) / theta^2` in `exp_se3` cancels: at theta ~ 1e-3 that moved poses by 1e-3.
"""

from lie_math import sincos_, sqrt_

comptime Vec4 = SIMD[DType.float32, 4]
comptime Vec8 = SIMD[DType.float32, 8]


@always_inline
def act_so3(q: Vec4, x: Vec4) -> Vec4:
    """Rotate the 3-vector `x` by the unit quaternion `q` (`actSO3`)."""
    var uv0 = 2.0 * (q[1] * x[2] - q[2] * x[1])
    var uv1 = 2.0 * (q[2] * x[0] - q[0] * x[2])
    var uv2 = 2.0 * (q[0] * x[1] - q[1] * x[0])
    return Vec4(
        x[0] + q[3] * uv0 + (q[1] * uv2 - q[2] * uv1),
        x[1] + q[3] * uv1 + (q[2] * uv0 - q[0] * uv2),
        x[2] + q[3] * uv2 + (q[0] * uv1 - q[1] * uv0),
        0.0,
    )


@always_inline
def act_se3(t: Vec4, q: Vec4, x: Vec4) -> Vec4:
    """Transform the homogeneous point `x` = (X, Y, Z, W) by (t, q) (`actSE3`)."""
    var y = act_so3(q, x)
    return Vec4(y[0] + x[3] * t[0], y[1] + x[3] * t[1], y[2] + x[3] * t[2], x[3])


@always_inline
def adj_se3(t: Vec4, q: Vec4, x: Vec8) -> Vec8:
    """Map a Jacobian row from the target frame to the source frame (`adjSE3`)."""
    var qinv = Vec4(-q[0], -q[1], -q[2], q[3])
    var y0 = act_so3(qinv, Vec4(x[0], x[1], x[2], 0.0))
    var y1 = act_so3(qinv, Vec4(x[3], x[4], x[5], 0.0))
    var u = Vec4(t[2] * x[1] - t[1] * x[2], t[0] * x[2] - t[2] * x[0], t[1] * x[0] - t[0] * x[1], 0.0)
    var v = act_so3(qinv, u)
    return Vec8(y0[0], y0[1], y0[2], y1[0] + v[0], y1[1] + v[1], y1[2] + v[2], 0.0, 0.0)


@always_inline
def rel_se3(ti: Vec4, qi: Vec4, tj: Vec4, qj: Vec4) -> Tuple[Vec4, Vec4]:
    """(tij, qij) of Gij = Tj * Ti^-1 (`relSE3`)."""
    var qij = Vec4(
        -qj[3] * qi[0] + qj[0] * qi[3] - qj[1] * qi[2] + qj[2] * qi[1],
        -qj[3] * qi[1] + qj[1] * qi[3] - qj[2] * qi[0] + qj[0] * qi[2],
        -qj[3] * qi[2] + qj[2] * qi[3] - qj[0] * qi[1] + qj[1] * qi[0],
        qj[3] * qi[3] + qj[0] * qi[0] + qj[1] * qi[1] + qj[2] * qi[2],
    )
    var r = act_so3(qij, ti)
    return (Vec4(tj[0] - r[0], tj[1] - r[1], tj[2] - r[2], 0.0), qij)


@always_inline
def exp_so3(phi: Vec4) -> Vec4:
    """Unit quaternion of the axis-angle `phi` (`expSO3`)."""
    var theta_sq = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2]
    var theta_p4 = theta_sq * theta_sq
    var theta = sqrt_(theta_sq)
    var imag: Float32
    var real: Float32
    if theta_sq < 1e-8:
        imag = 0.5 - (1.0 / 48.0) * theta_sq + (1.0 / 3840.0) * theta_p4
        real = 1.0 - (1.0 / 8.0) * theta_sq + (1.0 / 384.0) * theta_p4
    else:
        var s, c = sincos_(0.5 * theta)
        imag = s / theta
        real = c
    return Vec4(imag * phi[0], imag * phi[1], imag * phi[2], real)


@always_inline
def _cross(a: Vec4, b: Vec4) -> Vec4:
    return Vec4(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0], 0.0)


@always_inline
def exp_se3(xi: Vec8) -> Tuple[Vec4, Vec4]:
    """(t, q) of Exp(xi) for xi = (tau, phi) (`expSE3`; the V matrix applied through two cross products)."""
    var phi = Vec4(xi[3], xi[4], xi[5], 0.0)
    var q = exp_so3(phi)
    var tau = Vec4(xi[0], xi[1], xi[2], 0.0)
    var theta_sq = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2]
    var theta = sqrt_(theta_sq)
    var t = tau
    if theta > 1e-4:
        var s, c = sincos_(theta)
        var a = (1 - c) / theta_sq
        tau = _cross(phi, tau)
        t = Vec4(t[0] + a * tau[0], t[1] + a * tau[1], t[2] + a * tau[2], 0.0)
        var b = (theta - s) / (theta * theta_sq)
        tau = _cross(phi, tau)
        t = Vec4(t[0] + b * tau[0], t[1] + b * tau[1], t[2] + b * tau[2], 0.0)
    return (t, q)


@always_inline
def retr_se3(xi: Vec8, t: Vec4, q: Vec4) -> Tuple[Vec4, Vec4]:
    """(t1, q1) of Exp(xi) * (t, q) (`retrSE3`)."""
    var dt, dq = exp_se3(xi)
    var q1 = Vec4(
        dq[3] * q[0] + dq[0] * q[3] + dq[1] * q[2] - dq[2] * q[1],
        dq[3] * q[1] + dq[1] * q[3] + dq[2] * q[0] - dq[0] * q[2],
        dq[3] * q[2] + dq[2] * q[3] + dq[0] * q[1] - dq[1] * q[0],
        dq[3] * q[3] - dq[0] * q[0] - dq[1] * q[1] - dq[2] * q[2],
    )
    var r = act_so3(dq, t)
    return (Vec4(r[0] + dt[0], r[1] + dt[1], r[2] + dt[2], 0.0), q1)
