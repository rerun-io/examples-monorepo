"""SO3: unit quaternions, a line-by-line port of lietorch `include/so3.h`.

Data row: qx qy qz qw (Eigen coefficient order, scalar last). Every
constructor normalises the quaternion, as `SO3(Quaternion)` and
`SO3(const Scalar*)` do upstream; copies do not.

Precision notes (lietorch's semantics, checked against the PTX of the
ai-demos lietorch 1.0 build 2):
- `EPS` is a double (1e-6) and every `x < EPS` test compares in double, as
  in so3.h. The branch structure and thresholds are lietorch's.
- `sin`/`cos`/`atan` of a float32 value are float32 functions, as in the CUDA
  build. so3.h also promotes two expressions to double through a double
  literal (`sin(.5 * theta) / theta`, `cos(.5 * theta)` in `Exp`, and
  `(1.0 - cos(theta)) / theta2` in `left_jacobian`); this port keeps them in
  the data dtype, which changes float32 results by at most ~1 ulp and keeps
  float32 kernels free of FP64 (1/64 rate on GeForce GPUs).
- So for float32 data this port reproduces lietorch's own float32 rounding,
  including its known loss at small angles: for theta below ~2.4e-4,
  `cos(theta)` rounds to 1 and `left_jacobian`'s first-order coefficient
  becomes 0. The one exception is `SE3.calcQ` (se3.mojo).
- One deliberate float64 deviation: `left_jacobian` evaluates
  (1 - cos(theta)) / theta^2 as sin(theta)^2 / ((1 + cos(theta)) theta^2) for
  theta < 1. so3.h's form loses ~1e-16 / theta^2 to cancellation just above EPS;
  `Exp`'s translation then carries ~1e-11 of error at theta = 1e-6, which
  turns float64 `torch.autograd.gradcheck` (step 1e-6) into a failure at the
  identity for lietorch and this port alike. The rewrite is exact algebra and
  moves results by at most lietorch's own error there.
"""

from std.math import atan
from std.sys.info import is_apple_gpu

from lie_group import LieGroup
from lie_linalg import Mat
from lie_math import sincos_, sqrt_

comptime EPS: Float64 = 1e-6
comptime PI: Float64 = 3.14159265358979323846


@inline(.always)
def from_double[dtype: DType](v: Float64) -> Scalar[dtype]:
    """`Scalar[dtype](v)` for a double constant such as `1.0 / 24.0`: round to double first, then to dtype, as C++ does."""
    return Scalar[dtype](v)


@inline(.always)
def less_than[dtype: DType, //, bound: Float64](x: Scalar[dtype]) -> Bool:
    """`x < bound` as lietorch evaluates it, with `x` widened to double.

    An Apple GPU has no double: there `x` is compared with `bound` rounded to `dtype` at compile time. The two
    tests disagree only when `x` lies between `bound` and its rounding, i.e. for at most one float32 value.
    """
    comptime if is_apple_gpu():
        comptime rounded = Scalar[dtype](bound)
        return x < rounded
    else:
        return Float64(x) < bound


@inline(.always)
def hat[dtype: DType, //](phi: Mat[dtype, 3, 1]) -> Mat[dtype, 3, 3]:
    var m = Mat[dtype, 3, 3]()
    m[0, 1] = -phi[2]
    m[0, 2] = phi[1]
    m[1, 0] = phi[2]
    m[1, 2] = -phi[0]
    m[2, 0] = -phi[1]
    m[2, 1] = phi[0]
    return m


@inline(.always)
def vec3[dtype: DType, //](x: Scalar[dtype], y: Scalar[dtype], z: Scalar[dtype]) -> Mat[dtype, 3, 1]:
    var v = Mat[dtype, 3, 1]()
    v[0] = x
    v[1] = y
    v[2] = z
    return v


@inline(.always)
def cross[dtype: DType, //](a: Mat[dtype, 3, 1], b: Mat[dtype, 3, 1]) -> Mat[dtype, 3, 1]:
    # Explicit FMAs (see `SO3.act`): a1 b2 - a2 b1 = fma(a1, b2, -(a2 b1)), the form nvcc picks for se3.h.
    return vec3(a[1].fma(b[2], -(a[2] * b[1])), a[2].fma(b[0], -(a[0] * b[2])), a[0].fma(b[1], -(a[1] * b[0])))


struct SO3[dt: DType](LieGroup):
    comptime dtype = Self.dt
    comptime N = 4
    comptime K = 3

    var x: Scalar[Self.dt]
    var y: Scalar[Self.dt]
    var z: Scalar[Self.dt]
    var w: Scalar[Self.dt]

    @inline(.always)
    def __init__(out self, x: Scalar[Self.dt], y: Scalar[Self.dt], z: Scalar[Self.dt], w: Scalar[Self.dt]):
        """Normalising constructor (Eigen `Quaternion::normalize`: divide by the norm if it is > 0)."""
        # Explicit FMA chain (Eigen's sequential squaredNorm as nvcc contracts it): left to the compiler,
        # the sum was contracted differently per kernel, so one data row normalised to different bits.
        var sq = w.fma(w, z.fma(z, y.fma(y, x * x)))
        if sq > 0:
            var n = sqrt_(sq)
            self.x = x / n
            self.y = y / n
            self.z = z / n
            self.w = w / n
        else:
            self.x = x
            self.y = y
            self.z = z
            self.w = w

    @inline(.always)
    def __init__(out self):
        """Identity."""
        self.x = 0
        self.y = 0
        self.z = 0
        self.w = 1

    @inline(.always)
    def vec(self) -> Mat[Self.dt, 3, 1]:
        return vec3(self.x, self.y, self.z)

    @staticmethod
    @inline(.always)
    def from_data(x: Mat[Self.dtype, Self.N, 1]) -> Self:
        return Self(x[0], x[1], x[2], x[3])

    @inline(.always)
    def data(self) -> Mat[Self.dtype, Self.N, 1]:
        var r = Mat[Self.dtype, Self.N, 1]()
        r[0] = self.x
        r[1] = self.y
        r[2] = self.z
        r[3] = self.w
        return r

    @inline(.always)
    def inv(self) -> Self:
        return Self(-self.x, -self.y, -self.z, self.w)

    @inline(.always)
    def mul(self, o: Self) -> Self:
        # Eigen quat_product, then the normalising SO3 constructor.
        var w = self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z
        var x = self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y
        var y = self.w * o.y + self.y * o.w + self.z * o.x - self.x * o.z
        var z = self.w * o.z + self.z * o.w + self.x * o.y - self.y * o.x
        return Self(x, y, z, w)

    @inline(.always)
    def act(self, p: Mat[Self.dtype, 3, 1]) -> Mat[Self.dtype, 3, 1]:
        # Explicit FMAs: FP contraction otherwise pairs these differently per kernel, so X.inv() * X
        # stops cancelling exactly (README, "FMA").
        var v = self.vec()
        var uv = cross(v, p)
        uv = uv + uv
        var c = cross(v, uv)
        return vec3(self.w.fma(uv[0], p[0]) + c[0], self.w.fma(uv[1], p[1]) + c[1], self.w.fma(uv[2], p[2]) + c[2])

    @inline(.always)
    def act4(self, p: Mat[Self.dtype, 4, 1]) -> Mat[Self.dtype, 4, 1]:
        var r = Mat[Self.dt, 4, 1]()
        r.set_block[0, 0](self.act(p.block[0, 0, 3, 1]()))
        r[3] = p[3]
        return r

    @inline(.always)
    def Matrix(self) -> Mat[Self.dt, 3, 3]:
        # Eigen QuaternionBase::toRotationMatrix.
        var tx = 2 * self.x
        var ty = 2 * self.y
        var tz = 2 * self.z
        var twx = tx * self.w
        var twy = ty * self.w
        var twz = tz * self.w
        var txx = tx * self.x
        var txy = ty * self.x
        var txz = tz * self.x
        var tyy = ty * self.y
        var tyz = tz * self.y
        var tzz = tz * self.z
        var m = Mat[Self.dt, 3, 3]()
        m[0, 0] = 1 - (tyy + tzz)
        m[0, 1] = txy - twz
        m[0, 2] = txz + twy
        m[1, 0] = txy + twz
        m[1, 1] = 1 - (txx + tzz)
        m[1, 2] = tyz - twx
        m[2, 0] = txz - twy
        m[2, 1] = tyz + twx
        m[2, 2] = 1 - (txx + tyy)
        return m

    @inline(.always)
    def Adj(self) -> Mat[Self.dtype, Self.K, Self.K]:
        return self.Matrix()

    @inline(.always)
    def Matrix4x4(self) -> Mat[Self.dtype, 4, 4]:
        var T = Mat[Self.dt, 4, 4].identity()
        T.set_block[0, 0](self.Matrix())
        return T

    @inline(.always)
    def orthogonal_projector(self) -> Mat[Self.dtype, Self.N, Self.N]:
        var J = Mat[Self.dt, 4, 4]()
        var neg_v = -self.vec()
        J.set_block[0, 0](0.5 * (self.w * Mat[Self.dt, 3, 3].identity() + hat(neg_v)))
        J.set_block[3, 0]((0.5 * neg_v).T())
        return J

    @staticmethod
    @inline(.always)
    def adj(a: Mat[Self.dtype, Self.K, 1]) -> Mat[Self.dtype, Self.K, Self.K]:
        return hat(a)

    @inline(.always)
    def Log(self) -> Mat[Self.dtype, Self.K, 1]:
        comptime assert Self.dt.is_floating_point(), "SO3 needs a floating-point dtype"
        var squared_n = self.x * self.x + self.y * self.y + self.z * self.z
        var w = self.w
        var two_atan_nbyw_by_n: Scalar[Self.dt]
        if less_than[EPS * EPS](squared_n):
            var squared_w = w * w
            two_atan_nbyw_by_n = 2 / w - from_double[Self.dt](2.0 / 3.0) * squared_n / (w * squared_w)
        else:
            var n = sqrt_(squared_n)
            if less_than[EPS](abs(w)):
                if w > 0:
                    two_atan_nbyw_by_n = from_double[Self.dt](PI) / n
                else:
                    two_atan_nbyw_by_n = -from_double[Self.dt](PI) / n
            else:
                two_atan_nbyw_by_n = 2 * atan(n / w) / n
        return two_atan_nbyw_by_n * self.vec()

    @staticmethod
    @inline(.always)
    def Exp(phi: Mat[Self.dtype, Self.K, 1]) -> Self:
        var theta2 = phi.squared_norm()
        var theta = sqrt_(theta2)
        var imag_factor: Scalar[Self.dt]
        var real_factor: Scalar[Self.dt]
        if less_than[EPS](theta):
            var theta4 = theta2 * theta2
            imag_factor = 0.5 - from_double[Self.dt](1.0 / 48.0) * theta2 + from_double[Self.dt](1.0 / 3840.0) * theta4
            real_factor = 1 - from_double[Self.dt](1.0 / 8.0) * theta2 + from_double[Self.dt](1.0 / 384.0) * theta4
        else:
            var s, c = sincos_(0.5 * theta)
            imag_factor = s / theta
            real_factor = c
        return Self(imag_factor * phi[0], imag_factor * phi[1], imag_factor * phi[2], real_factor)

    @staticmethod
    @inline(.always)
    def left_jacobian(phi: Mat[Self.dtype, Self.K, 1]) -> Mat[Self.dtype, Self.K, Self.K]:
        var I = Mat[Self.dt, 3, 3].identity()
        var Phi = hat(phi)
        var Phi2 = Phi @ Phi
        var theta2 = phi.squared_norm()
        var theta = sqrt_(theta2)
        var coef1: Scalar[Self.dt]
        var coef2: Scalar[Self.dt]
        if less_than[EPS](theta):
            coef1 = 0.5 - from_double[Self.dt](1.0 / 24.0) * theta2
            coef2 = from_double[Self.dt](1.0 / 6.0) - from_double[Self.dt](1.0 / 120.0) * theta2
        else:
            var s, c = sincos_(theta)
            comptime if Self.dt == .float64:
                # 1 - cos = sin^2 / (1 + cos) for theta < 1: no cancellation (module docstring). Select, then
                # divide once: an FP64 division per branch cost 4% of expm's kernel time.
                var small = theta < 1.0
                var num = s * s if small else 1 - c
                var den = (1 + c) * theta2 if small else theta2
                coef1 = num / den
            else:
                coef1 = (1 - c) / theta2
            coef2 = (theta - s) / (theta2 * theta)
        return I + coef1 * Phi + coef2 * Phi2

    @staticmethod
    @inline(.always)
    def left_jacobian_inverse(phi: Mat[Self.dtype, Self.K, 1]) -> Mat[Self.dtype, Self.K, Self.K]:
        var I = Mat[Self.dt, 3, 3].identity()
        var Phi = hat(phi)
        var Phi2 = Phi @ Phi
        var theta2 = phi.squared_norm()
        var theta = sqrt_(theta2)
        var coef2: Scalar[Self.dt]
        if less_than[EPS](theta):
            coef2 = from_double[Self.dt](1.0 / 12.0)
        else:
            var s, c = sincos_(0.5 * theta)
            coef2 = (1 - theta * c / (2 * s)) / (theta * theta)
        return I + (-0.5) * Phi + coef2 * Phi2

    @staticmethod
    @inline(.always)
    def act_jacobian(p: Mat[Self.dtype, 3, 1]) -> Mat[Self.dtype, 3, Self.K]:
        return hat(-p)

    @staticmethod
    @inline(.always)
    def act4_jacobian(p: Mat[Self.dtype, 4, 1]) -> Mat[Self.dtype, 4, Self.K]:
        var J = Mat[Self.dt, 4, 3]()
        J.set_block[0, 0](hat(-p.block[0, 0, 3, 1]()))
        return J
