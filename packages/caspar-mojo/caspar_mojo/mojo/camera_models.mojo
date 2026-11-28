"""Camera models the factor kernels are generic over.

A model projects a camera-frame point with the calibration row (COLMAP `camera.params` order, zero-padded to
`MAX_PARAMS`) and gives the two Jacobians the factor kernels chain: d pixel / d camera point (2x3) and
d pixel / d focal-and-extra (2x`NUM_CALIB`, CASPAR's FocalAndExtra group; the principal point stays constant).
Every Jacobian is row-major. Adding a model is one struct; the kernels hold no model assumptions.
"""

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
def _safe_z(z: Float32) -> Float32:
    return z + (EPS if z >= 0.0 else -EPS)


@fieldwise_init
struct Pinhole(CameraModel):
    """`[fx, fy, cx, cy]`; focal-and-extra `[fx, fy]`."""

    comptime NUM_CALIB = 2

    @staticmethod
    def calib_column(i: Int) -> Int:
        return i

    @staticmethod
    def project(x: Float32, y: Float32, z: Float32, params: Params) -> Projection[Self.NUM_CALIB]:
        var fx = params[0]
        var fy = params[1]
        var inv_z = 1.0 / _safe_z(z)
        var u = x * inv_z
        var v = y * inv_z
        var d_point: Array[Float32, 6] = [fx * inv_z, 0.0, -fx * u * inv_z, 0.0, fy * inv_z, -fy * v * inv_z]
        var d_calib: Array[Float32, 4] = [u, 0.0, 0.0, v]
        return Projection[Self.NUM_CALIB](fx * u + params[2], fy * v + params[3], d_point^, d_calib^)
