"""Pose3 as CASPAR stores it: quaternion `[qx, qy, qz, qw]` + translation, `x_out = R(q) x_in + t`.

The rotation matrix uses the unit-quaternion formula, as Symforce's Rot3 does; quaternions drift off unit
norm between write-backs and are normalised only then (COLMAP's adapter).
"""

from std.math import cos, sin, sqrt


@fieldwise_init
struct Pose(TrivialRegisterPassable):
    var qx: Float32
    var qy: Float32
    var qz: Float32
    var qw: Float32
    var tx: Float32
    var ty: Float32
    var tz: Float32


@fieldwise_init
struct Mat3(TrivialRegisterPassable):
    var m: SIMD[DType.float32, 16]
    """Row-major 3x3 in lanes 0..8."""

    @always_inline
    def __getitem__(self, i: Int, j: Int) -> Float32:
        return self.m[3 * i + j]


@always_inline
def rotation(p: Pose) -> Mat3:
    var x = p.qx
    var y = p.qy
    var z = p.qz
    var w = p.qw
    var m = SIMD[DType.float32, 16](0.0)
    m[0] = 1.0 - 2.0 * (y * y + z * z)
    m[1] = 2.0 * (x * y - z * w)
    m[2] = 2.0 * (x * z + y * w)
    m[3] = 2.0 * (x * y + z * w)
    m[4] = 1.0 - 2.0 * (x * x + z * z)
    m[5] = 2.0 * (y * z - x * w)
    m[6] = 2.0 * (x * z - y * w)
    m[7] = 2.0 * (y * z + x * w)
    m[8] = 1.0 - 2.0 * (x * x + y * y)
    return Mat3(m)


@fieldwise_init
struct Vec3(TrivialRegisterPassable):
    var x: Float32
    var y: Float32
    var z: Float32


@always_inline
def apply(r: Mat3, t: Vec3, v: Vec3) -> Vec3:
    return Vec3(
        r[0, 0] * v.x + r[0, 1] * v.y + r[0, 2] * v.z + t.x,
        r[1, 0] * v.x + r[1, 1] * v.y + r[1, 2] * v.z + t.y,
        r[2, 0] * v.x + r[2, 1] * v.y + r[2, 2] * v.z + t.z,
    )


@always_inline
def translation(p: Pose) -> Vec3:
    return Vec3(p.tx, p.ty, p.tz)


@always_inline
def cross(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)


@always_inline
def retract(p: Pose, wx: Float32, wy: Float32, wz: Float32, dx: Float32, dy: Float32, dz: Float32) -> Pose:
    """CASPAR's Pose3 retraction: q' = q (x) exp(w) (right rotation), t' = t + dt."""
    var a = sqrt(wx * wx + wy * wy + wz * wz + 1e-12)
    var s = sin(a * 0.5) / a
    var ex = wx * s
    var ey = wy * s
    var ez = wz * s
    var ew = cos(a * 0.5)
    return Pose(
        p.qw * ex + p.qx * ew + p.qy * ez - p.qz * ey,
        p.qw * ey - p.qx * ez + p.qy * ew + p.qz * ex,
        p.qw * ez + p.qx * ey - p.qy * ex + p.qz * ew,
        p.qw * ew - p.qx * ex - p.qy * ey - p.qz * ez,
        p.tx + dx,
        p.ty + dy,
        p.tz + dz,
    )
