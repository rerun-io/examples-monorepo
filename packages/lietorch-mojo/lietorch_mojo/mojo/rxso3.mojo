"""RxSO3: rotation times positive scale, a port of lietorch `include/rxso3.h`.

Data row: qx qy qz qw s (unit quaternion, scalar last, then the scale).
Tangent: phi (3, rotation) then sigma (log-scale), as in lietorch.

The rotation part is exactly `SO3` (rxso3.h repeats so3.h's quaternion code
line for line: same normalising constructor, `Exp`, `Log`, product and
action), so this struct holds an `SO3` and a scale. `calcW` (the
translation coupling of Sim3) keeps rxso3.h's closed forms and branches. Like
rxso3.h, `left_jacobian` and its inverse act on the rotation block only (the
scale row and column are the identity).

`exp`/`log` of the scale are `exp_`/`log_` (lie_explog.mojo): Mojo 1.2.0's
`std.math.exp` and float32 GPU `log` are not accurate enough.
"""

from lie_explog import exp_, log_
from lie_group import LieGroup
from lie_linalg import Mat
from lie_math import sincos_, sqrt_
from so3 import EPS, SO3, from_double, hat, less_than


struct RxSO3[dt: DType](LieGroup):
    comptime dtype = Self.dt
    comptime N = 5
    comptime K = 4

    var so3: SO3[Self.dt]
    var scale: Scalar[Self.dt]

    @inline(.always)
    def __init__(out self, so3: SO3[Self.dt], scale: Scalar[Self.dt]):
        self.so3 = so3
        self.scale = scale

    @staticmethod
    @inline(.always)
    def from_data(x: Mat[Self.dtype, Self.N, 1]) -> Self:
        return Self(SO3[Self.dt].from_data(x.block[0, 0, 4, 1]()), x[4])

    @inline(.always)
    def data(self) -> Mat[Self.dtype, Self.N, 1]:
        var r = Mat[Self.dtype, Self.N, 1]()
        r.set_block[0, 0](self.so3.data())
        r[4] = self.scale
        return r

    @inline(.always)
    def inv(self) -> Self:
        # rxso3.h: 1.0 / scale is a double division; rounding it to float32 equals the float32 quotient.
        return Self(self.so3.inv(), 1 / self.scale)

    @inline(.always)
    def mul(self, o: Self) -> Self:
        return Self(self.so3.mul(o.so3), self.scale * o.scale)

    @inline(.always)
    def act(self, p: Mat[Self.dtype, 3, 1]) -> Mat[Self.dtype, 3, 1]:
        return self.scale * self.so3.act(p)

    @inline(.always)
    def act4(self, p: Mat[Self.dtype, 4, 1]) -> Mat[Self.dtype, 4, 1]:
        var r = Mat[Self.dt, 4, 1]()
        r.set_block[0, 0](self.act(p.block[0, 0, 3, 1]()))
        r[3] = p[3]
        return r

    @inline(.always)
    def Matrix(self) -> Mat[Self.dt, 3, 3]:
        return self.scale * self.so3.Matrix()

    @inline(.always)
    def Adj(self) -> Mat[Self.dtype, Self.K, Self.K]:
        var Ad = Mat[Self.dt, 4, 4].identity()
        Ad.set_block[0, 0](self.so3.Matrix())
        return Ad

    @inline(.always)
    def Matrix4x4(self) -> Mat[Self.dtype, 4, 4]:
        var T = Mat[Self.dt, 4, 4].identity()
        T.set_block[0, 0](self.Matrix())
        return T

    @inline(.always)
    def orthogonal_projector(self) -> Mat[Self.dtype, Self.N, Self.N]:
        var J = Mat[Self.dt, 5, 5]()
        J.set_block[0, 0](self.so3.orthogonal_projector())
        J[4, 3] = self.scale
        return J

    @staticmethod
    @inline(.always)
    def adj(phi_sigma: Mat[Self.dtype, Self.K, 1]) -> Mat[Self.dtype, Self.K, Self.K]:
        var ad = Mat[Self.dt, 4, 4]()
        ad.set_block[0, 0](hat(phi_sigma.block[0, 0, 3, 1]()))
        return ad

    @inline(.always)
    def Log(self) -> Mat[Self.dtype, Self.K, 1]:
        var phi_sigma = Mat[Self.dt, 4, 1]()
        phi_sigma.set_block[0, 0](self.so3.Log())
        phi_sigma[3] = log_(self.scale)
        return phi_sigma

    @staticmethod
    @inline(.always)
    def Exp(phi_sigma: Mat[Self.dtype, Self.K, 1]) -> Self:
        return Self(SO3[Self.dt].Exp(phi_sigma.block[0, 0, 3, 1]()), exp_(phi_sigma[3]))

    @staticmethod
    @inline(.always)
    def calcW(phi_sigma: Mat[Self.dt, 4, 1]) -> Mat[Self.dt, 3, 3]:
        """V matrix of Sim3's Exp: t = W tau (rxso3.h `calcW`)."""
        var phi = phi_sigma.block[0, 0, 3, 1]()
        var sigma = phi_sigma[3]
        var theta = sqrt_(phi.squared_norm())
        var Phi = hat(phi)
        var Phi2 = Phi @ Phi
        var scale = exp_(sigma)
        var A: Scalar[Self.dt]
        var B: Scalar[Self.dt]
        var C: Scalar[Self.dt]
        if less_than[EPS](abs(sigma)):
            C = 1
            if less_than[EPS](abs(theta)):
                A = 0.5
                B = from_double[Self.dt](1.0 / 6.0)
            else:
                var theta_sq = theta * theta
                var s, c = sincos_(theta)
                A = (1 - c) / theta_sq
                B = (theta - s) / (theta_sq * theta)
        else:
            C = (scale - 1) / sigma
            if less_than[EPS](abs(theta)):
                var sigma_sq = sigma * sigma
                A = ((sigma - 1) * scale + 1) / sigma_sq
                B = (scale * 0.5 * sigma_sq + scale - 1 - sigma * scale) / (sigma_sq * sigma)
            else:
                var theta_sq = theta * theta
                var s, c = sincos_(theta)
                var a = scale * s
                var b = scale * c
                var cc = theta_sq + sigma * sigma
                A = (a * sigma + (1 - b) * theta) / (theta * cc)
                B = (C - ((b - 1) * sigma + a * theta) / cc) * 1 / theta_sq
        return A * Phi + B * Phi2 + C * Mat[Self.dt, 3, 3].identity()

    @staticmethod
    @inline(.always)
    def left_jacobian(phi_sigma: Mat[Self.dtype, Self.K, 1]) -> Mat[Self.dtype, Self.K, Self.K]:
        var J = Mat[Self.dt, 4, 4].identity()
        J.set_block[0, 0](SO3[Self.dt].left_jacobian(phi_sigma.block[0, 0, 3, 1]()))
        return J

    @staticmethod
    @inline(.always)
    def left_jacobian_inverse(phi_sigma: Mat[Self.dtype, Self.K, 1]) -> Mat[Self.dtype, Self.K, Self.K]:
        var J = Mat[Self.dt, 4, 4].identity()
        J.set_block[0, 0](SO3[Self.dt].left_jacobian_inverse(phi_sigma.block[0, 0, 3, 1]()))
        return J

    @staticmethod
    @inline(.always)
    def act_jacobian(p: Mat[Self.dtype, 3, 1]) -> Mat[Self.dtype, 3, Self.K]:
        var J = Mat[Self.dt, 3, 4]()
        J.set_block[0, 0](hat(-p))
        J.set_block[0, 3](p)
        return J

    @staticmethod
    @inline(.always)
    def act4_jacobian(p: Mat[Self.dtype, 4, 1]) -> Mat[Self.dtype, 4, Self.K]:
        var J = Mat[Self.dt, 4, 4]()
        var p3 = p.block[0, 0, 3, 1]()
        J.set_block[0, 0](hat(-p3))
        J.set_block[0, 3](p3)
        return J
