"""Camera models the factor kernels are generic over.

A model projects a camera-frame point with the calibration row (COLMAP `camera.params` order, zero-padded to
`MAX_PARAMS`) and gives the two Jacobians the factor kernels chain: d pixel / d camera point (2x3) and
d pixel / d focal-and-extra (2x`NUM_CALIB`, CASPAR's FocalAndExtra group; the principal point stays constant).
Every Jacobian is row-major. Adding a model is one struct; the kernels hold no model assumptions.
"""

from std.math import atan2, sqrt

comptime MAX_PARAMS = 12
comptime Params = Array[Float32, MAX_PARAMS]
# The generated code's projection epsilon: (X, Y) / (Z + copysign(eps, Z)).
comptime EPS: Float32 = 1e-6


@fieldwise_init
struct Projection[K: Int](Movable):
    var u: Float32
    var v: Float32
    var d_point: Array[Float32, 6]
    var d_calib: Array[Float32, 2 * Self.K]


trait CameraModel(TrivialRegisterPassable):
    comptime NUM_CALIB: Int
    """Focal-and-extra parameters, the width of the refined intrinsics block."""

    @staticmethod
    def calib_column(i: Int) -> Int:
        """`camera.params` column of focal-and-extra parameter `i`."""
        ...

    @staticmethod
    def project(x: Float32, y: Float32, z: Float32, params: Params) -> Projection[Self.NUM_CALIB]:
        ...


@always_inline
def _safe(z: Float32) -> Float32:
    """z + copysign(eps, z), Symforce's sign_no_zero (0 counts as positive)."""
    return z + (EPS if z >= 0.0 else -EPS)


@always_inline
def _through_perspective(
    a00: Float32, a01: Float32, a10: Float32, a11: Float32, u: Float32, v: Float32, inv_z: Float32
) -> Array[Float32, 6]:
    """A d(u, v)/d point for A = d pixel / d(u, v), with (u, v) = (X, Y) / safe(Z)."""
    return [a00 * inv_z, a01 * inv_z, -(a00 * u + a01 * v) * inv_z, a10 * inv_z, a11 * inv_z, -(a10 * u + a11 * v) * inv_z]


@fieldwise_init
struct Pinhole(CameraModel):
    """`[fx, fy, cx, cy]`; focal-and-extra `[fx, fy]`."""

    comptime NUM_CALIB = 2

    @staticmethod
    def calib_column(i: Int) -> Int:
        return i

    @staticmethod
    @always_inline
    def project(x: Float32, y: Float32, z: Float32, params: Params) -> Projection[Self.NUM_CALIB]:
        var fx = params[0]
        var fy = params[1]
        var inv_z = 1.0 / _safe(z)
        var u = x * inv_z
        var v = y * inv_z
        var d_point: Array[Float32, 6] = [fx * inv_z, 0.0, -fx * u * inv_z, 0.0, fy * inv_z, -fy * v * inv_z]
        var d_calib: Array[Float32, 4] = [u, 0.0, 0.0, v]
        return Projection[Self.NUM_CALIB](fx * u + params[2], fy * v + params[3], d_point^, d_calib^)


@fieldwise_init
struct SimpleRadial(CameraModel):
    """`[f, cx, cy, k]`; focal-and-extra `[f, k]`."""

    comptime NUM_CALIB = 2

    @staticmethod
    def calib_column(i: Int) -> Int:
        return 0 if i == 0 else 3

    @staticmethod
    @always_inline
    def project(x: Float32, y: Float32, z: Float32, params: Params) -> Projection[Self.NUM_CALIB]:
        var f = params[0]
        var k = params[3]
        var inv_z = 1.0 / _safe(z)
        var u = x * inv_z
        var v = y * inv_z
        var q = u * u + v * v
        var radial = 1.0 + k * q
        # d pixel / d(u, v) = f (radial I + 2 k (u, v)(u, v)^T)
        var two_fk = 2.0 * f * k
        var a00 = f * radial + two_fk * u * u
        var a01 = two_fk * u * v
        var a11 = f * radial + two_fk * v * v
        var d_calib: Array[Float32, 4] = [radial * u, f * q * u, radial * v, f * q * v]
        return Projection[Self.NUM_CALIB](
            f * radial * u + params[1], f * radial * v + params[2], _through_perspective(a00, a01, a01, a11, u, v, inv_z), d_calib^
        )


@fieldwise_init
struct OpenCVFisheye(CameraModel):
    """`[fx, fy, cx, cy, k1, k2, k3, k4]`; focal-and-extra `[fx, fy, k1, k2, k3, k4]`.

    Kannala-Brandt on the unit ray, with the generator's two norm epsilons and atan2's epsilon."""

    comptime NUM_CALIB = 6

    @staticmethod
    def calib_column(i: Int) -> Int:
        return i if i < 2 else i + 2

    @staticmethod
    @always_inline
    def project(x: Float32, y: Float32, z: Float32, params: Params) -> Projection[Self.NUM_CALIB]:
        var fx = params[0]
        var fy = params[1]
        var k1 = params[4]
        var k2 = params[5]
        var k3 = params[6]
        var k4 = params[7]
        var inv_norm = 1.0 / sqrt(x * x + y * y + z * z + EPS)
        var rx = x * inv_norm
        var ry = y * inv_norm
        var rz = z * inv_norm
        var rho = sqrt(rx * rx + ry * ry + EPS)
        var w = _safe(rz)
        var theta = atan2(rho, w)
        var t2 = theta * theta
        var radial = t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))
        var theta_d = theta * (1.0 + radial)
        var d_theta_d = 1.0 + t2 * (3.0 * k1 + t2 * (5.0 * k2 + t2 * (7.0 * k3 + t2 * 9.0 * k4)))
        var inv_rho = 1.0 / rho
        var scale = theta_d * inv_rho
        var inv_den = 1.0 / (rho * rho + w * w)
        # d scale / d ray = (d_theta_d / rho) d theta - (theta_d / rho^2) d rho, with d rho = (rx, ry, 0) / rho and
        # d theta = (w d rho - rho e_z) / (rho^2 + w^2).
        var c_rho = (d_theta_d * inv_rho * w * inv_den - theta_d * inv_rho * inv_rho) * inv_rho
        var sx = c_rho * rx
        var sy = c_rho * ry
        var sz = -d_theta_d * inv_den
        # Rows of d pixel / d ray, then through d ray / d point = (I - r r^T) / |P|.
        var g0x = fx * (rx * sx + scale)
        var g0y = fx * rx * sy
        var g0z = fx * rx * sz
        var g1x = fy * ry * sx
        var g1y = fy * (ry * sy + scale)
        var g1z = fy * ry * sz
        var gr0 = g0x * rx + g0y * ry + g0z * rz
        var gr1 = g1x * rx + g1y * ry + g1z * rz
        var d_point: Array[Float32, 6] = [
            (g0x - gr0 * rx) * inv_norm,
            (g0y - gr0 * ry) * inv_norm,
            (g0z - gr0 * rz) * inv_norm,
            (g1x - gr1 * rx) * inv_norm,
            (g1y - gr1 * ry) * inv_norm,
            (g1z - gr1 * rz) * inv_norm,
        ]
        var d_calib = Array[Float32, 12](fill=0.0)
        d_calib[0] = scale * rx
        d_calib[7] = scale * ry
        var power = theta * inv_rho
        comptime for c in range(2, 6):
            power *= t2
            d_calib[c] = fx * rx * power
            d_calib[6 + c] = fy * ry * power
        return Projection[Self.NUM_CALIB](fx * scale * rx + params[2], fy * scale * ry + params[3], d_point^, d_calib^)


@fieldwise_init
struct FullOpenCV(CameraModel):
    """`[fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6]`; focal-and-extra `[fx, fy, k1, k2, p1, p2, k3, k4, k5, k6]`."""

    comptime NUM_CALIB = 10

    @staticmethod
    def calib_column(i: Int) -> Int:
        return i if i < 2 else i + 2

    @staticmethod
    @always_inline
    def project(x: Float32, y: Float32, z: Float32, params: Params) -> Projection[Self.NUM_CALIB]:
        var fx = params[0]
        var fy = params[1]
        var k1 = params[4]
        var k2 = params[5]
        var p1 = params[6]
        var p2 = params[7]
        var k3 = params[8]
        var k4 = params[9]
        var k5 = params[10]
        var k6 = params[11]
        var inv_z = 1.0 / _safe(z)
        var u = x * inv_z
        var v = y * inv_z
        var u2 = u * u
        var uv = u * v
        var v2 = v * v
        var q = u2 + v2
        var numerator = 1.0 + q * (k1 + q * (k2 + q * k3))
        var inv_den = 1.0 / (1.0 + q * (k4 + q * (k5 + q * k6)))
        var a = numerator * inv_den
        # d a / d q, the quotient rule over the two cubics in q.
        var da = ((k1 + q * (2.0 * k2 + q * 3.0 * k3)) - a * (k4 + q * (2.0 * k5 + q * 3.0 * k6))) * inv_den
        var xd = u * a + 2.0 * p1 * uv + p2 * (q + 2.0 * u2)
        var yd = v * a + 2.0 * p2 * uv + p1 * (q + 2.0 * v2)
        var cross_term = 2.0 * (uv * da + p1 * u + p2 * v)
        var a00 = fx * (a + 2.0 * u2 * da + 2.0 * p1 * v + 6.0 * p2 * u)
        var a01 = fx * cross_term
        var a10 = fy * cross_term
        var a11 = fy * (a + 2.0 * v2 * da + 2.0 * p2 * u + 6.0 * p1 * v)
        var d_calib = Array[Float32, 20](fill=0.0)
        d_calib[0] = xd
        d_calib[11] = yd
        # k1, k2, k3 move the numerator, k4, k5, k6 the denominator: d a / d k = (+-) q^n / den (times a).
        var fu = fx * u * inv_den
        var fv = fy * v * inv_den
        var q3 = q * q * q
        d_calib[2] = fu * q
        d_calib[3] = fu * q * q
        d_calib[6] = fu * q3
        d_calib[7] = -fu * a * q
        d_calib[8] = -fu * a * q * q
        d_calib[9] = -fu * a * q3
        d_calib[12] = fv * q
        d_calib[13] = fv * q * q
        d_calib[16] = fv * q3
        d_calib[17] = -fv * a * q
        d_calib[18] = -fv * a * q * q
        d_calib[19] = -fv * a * q3
        d_calib[4] = fx * 2.0 * uv
        d_calib[14] = fy * (q + 2.0 * v2)
        d_calib[5] = fx * (q + 2.0 * u2)
        d_calib[15] = fy * 2.0 * uv
        return Projection[Self.NUM_CALIB](
            fx * xd + params[2], fy * yd + params[3], _through_perspective(a00, a01, a10, a11, u, v, inv_z), d_calib^
        )
