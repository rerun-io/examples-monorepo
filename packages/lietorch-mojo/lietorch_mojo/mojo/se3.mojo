"""SE3: rigid transforms, a line-by-line port of lietorch `include/se3.h`.

Data row: tx ty tz qx qy qz qw. Tangent: tau (3, translation part) then
phi (3, rotation part), as in lietorch.

One deliberate deviation, float32 only: `calcQ` evaluates its three
coefficients from their Taylor series for theta < 2 (8 terms, truncation
< 3e-10 relative), because se3.h's closed forms cancel catastrophically in
float32 at small theta (lietorch's own CPU and CUDA builds disagree there by
up to ~1 normwise). float64 keeps se3.h's closed forms unchanged.
"""

from lie_group import LieGroup
from lie_linalg import Mat
from lie_math import sincos_, sqrt_
from so3 import EPS, SO3, from_double, hat, less_than


@inline(.always)
def _series8[
    dtype: DType, //
](x: Scalar[dtype], c0: Float64, c1: Float64, c2: Float64, c3: Float64, c4: Float64, c5: Float64, c6: Float64, c7: Float64) -> Scalar[dtype]:
    """c0 + c1 x + ... + c7 x^7 by Horner's rule in `dtype`."""
    var r = from_double[dtype](c7)
    r = r * x + from_double[dtype](c6)
    r = r * x + from_double[dtype](c5)
    r = r * x + from_double[dtype](c4)
    r = r * x + from_double[dtype](c3)
    r = r * x + from_double[dtype](c2)
    r = r * x + from_double[dtype](c1)
    r = r * x + from_double[dtype](c0)
    return r


struct SE3[dt: DType](LieGroup):
    comptime dtype = Self.dt
    comptime N = 7
    comptime K = 6

    var so3: SO3[Self.dt]
    var t: Mat[Self.dt, 3, 1]

    @inline(.always)
    def __init__(out self, so3: SO3[Self.dt], t: Mat[Self.dt, 3, 1]):
        self.so3 = so3
        self.t = t

    @staticmethod
    @inline(.always)
    def from_data(x: Mat[Self.dtype, Self.N, 1]) -> Self:
        return Self(SO3[Self.dt].from_data(x.block[3, 0, 4, 1]()), x.block[0, 0, 3, 1]())

    @inline(.always)
    def data(self) -> Mat[Self.dtype, Self.N, 1]:
        var r = Mat[Self.dtype, Self.N, 1]()
        r.set_block[0, 0](self.t)
        r.set_block[3, 0](self.so3.data())
        return r

    @inline(.always)
    def inv(self) -> Self:
        var r = self.so3.inv()
        return Self(r, -(r.act(self.t)))

    @inline(.always)
    def mul(self, o: Self) -> Self:
        return Self(self.so3.mul(o.so3), self.t + self.so3.act(o.t))

    @inline(.always)
    def act(self, p: Mat[Self.dtype, 3, 1]) -> Mat[Self.dtype, 3, 1]:
        return self.so3.act(p) + self.t

    @inline(.always)
    def act4(self, p: Mat[Self.dtype, 4, 1]) -> Mat[Self.dtype, 4, 1]:
        var r = Mat[Self.dt, 4, 1]()
        r.set_block[0, 0](self.so3.act(p.block[0, 0, 3, 1]()) + self.t * p[3])
        r[3] = p[3]
        return r

    @inline(.always)
    def Adj(self) -> Mat[Self.dtype, Self.K, Self.K]:
        var R = self.so3.Matrix()
        var Ad = Mat[Self.dt, 6, 6]()
        Ad.set_block[0, 0](R)
        Ad.set_block[0, 3](hat(self.t) @ R)
        Ad.set_block[3, 3](R)
        return Ad

    @inline(.always)
    def Matrix4x4(self) -> Mat[Self.dtype, 4, 4]:
        var T = Mat[Self.dt, 4, 4].identity()
        T.set_block[0, 0](self.so3.Matrix())
        T.set_block[0, 3](self.t)
        return T

    @inline(.always)
    def orthogonal_projector(self) -> Mat[Self.dtype, Self.N, Self.N]:
        var J = Mat[Self.dt, 7, 7]()
        J.set_block[0, 0](Mat[Self.dt, 3, 3].identity())
        J.set_block[0, 3](hat(-self.t))
        J.set_block[3, 3](self.so3.orthogonal_projector())
        return J

    @staticmethod
    @inline(.always)
    def adj(tau_phi: Mat[Self.dtype, Self.K, 1]) -> Mat[Self.dtype, Self.K, Self.K]:
        var Tau = hat(tau_phi.block[0, 0, 3, 1]())
        var Phi = hat(tau_phi.block[3, 0, 3, 1]())
        var ad = Mat[Self.dt, 6, 6]()
        ad.set_block[0, 0](Phi)
        ad.set_block[0, 3](Tau)
        ad.set_block[3, 3](Phi)
        return ad

    @inline(.always)
    def Log(self) -> Mat[Self.dtype, Self.K, 1]:
        var phi = self.so3.Log()
        var tau_phi = Mat[Self.dt, 6, 1]()
        tau_phi.set_block[0, 0](SO3[Self.dt].left_jacobian_inverse(phi) @ self.t)
        tau_phi.set_block[3, 0](phi)
        return tau_phi

    @staticmethod
    @inline(.always)
    def Exp(tau_phi: Mat[Self.dtype, Self.K, 1]) -> Self:
        var tau = tau_phi.block[0, 0, 3, 1]()
        var phi = tau_phi.block[3, 0, 3, 1]()
        return Self(SO3[Self.dt].Exp(phi), SO3[Self.dt].left_jacobian(phi) @ tau)

    @staticmethod
    @inline(.always)
    def calcQ(tau_phi: Mat[Self.dt, 6, 1]) -> Mat[Self.dt, 3, 3]:
        var Tau = hat(tau_phi.block[0, 0, 3, 1]())
        var phi = tau_phi.block[3, 0, 3, 1]()
        var Phi = hat(phi)
        var theta = sqrt_(phi.squared_norm())
        var theta_pow2 = theta * theta
        var theta_pow4 = theta_pow2 * theta_pow2
        var use_series: Bool
        comptime if Self.dt == .float32:
            use_series = theta < 2.0  # see the module docstring
        else:
            use_series = False
        var coef1: Scalar[Self.dt]
        var coef2: Scalar[Self.dt]
        var coef3: Scalar[Self.dt]
        if less_than[EPS](theta):
            coef1 = from_double[Self.dt](1.0 / 6.0) - from_double[Self.dt](1.0 / 120.0) * theta_pow2
            coef2 = from_double[Self.dt](1.0 / 24.0) - from_double[Self.dt](1.0 / 720.0) * theta_pow2
            coef3 = from_double[Self.dt](1.0 / 120.0) - from_double[Self.dt](1.0 / 2520.0) * theta_pow2
        elif use_series:
            var t2 = theta_pow2
            coef1 = _series8(
                t2,
                1.0 / 6.0,
                -1.0 / 120.0,
                1.0 / 5040.0,
                -1.0 / 362880.0,
                1.0 / 39916800.0,
                -1.0 / 6227020800.0,
                1.0 / 1307674368000.0,
                -1.0 / 355687428096000.0,
            )
            coef2 = _series8(
                t2,
                1.0 / 24.0,
                -1.0 / 720.0,
                1.0 / 40320.0,
                -1.0 / 3628800.0,
                1.0 / 479001600.0,
                -1.0 / 87178291200.0,
                1.0 / 20922789888000.0,
                -1.0 / 6402373705728000.0,
            )
            coef3 = _series8(
                t2,
                1.0 / 120.0,
                -1.0 / 2520.0,
                1.0 / 120960.0,
                -1.0 / 9979200.0,
                1.0 / 1245404160.0,
                -1.0 / 217945728000.0,
                1.0 / 50812489728000.0,
                -1.0 / 15205637551104000.0,
            )
        else:
            var s, c = sincos_(theta)
            coef1 = (theta - s) / (theta_pow2 * theta)
            coef2 = (theta_pow2 + 2 * c - 2) / (2 * theta_pow4)
            coef3 = (2 * theta - 3 * s + theta * c) / (2 * theta_pow4 * theta)
        var PhiTau = Phi @ Tau
        var PhiTauPhi = PhiTau @ Phi
        var PhiPhi = Phi @ Phi
        var Q = 0.5 * Tau
        Q = Q + coef1 * (PhiTau + Tau @ Phi + PhiTauPhi)
        Q = Q + coef2 * (PhiPhi @ Tau + (Tau @ Phi) @ Phi - ((3 * Phi) @ Tau) @ Phi)
        Q = Q + coef3 * (PhiTauPhi @ Phi + (PhiPhi @ Tau) @ Phi)
        return Q

    @staticmethod
    @inline(.always)
    def left_jacobian(tau_phi: Mat[Self.dtype, Self.K, 1]) -> Mat[Self.dtype, Self.K, Self.K]:
        var J = SO3[Self.dt].left_jacobian(tau_phi.block[3, 0, 3, 1]())
        var J6 = Mat[Self.dt, 6, 6]()
        J6.set_block[0, 0](J)
        J6.set_block[0, 3](Self.calcQ(tau_phi))
        J6.set_block[3, 3](J)
        return J6

    @staticmethod
    @inline(.always)
    def left_jacobian_inverse(tau_phi: Mat[Self.dtype, Self.K, 1]) -> Mat[Self.dtype, Self.K, Self.K]:
        var Jinv = SO3[Self.dt].left_jacobian_inverse(tau_phi.block[3, 0, 3, 1]())
        var J6 = Mat[Self.dt, 6, 6]()
        J6.set_block[0, 0](Jinv)
        J6.set_block[0, 3](((-Jinv) @ Self.calcQ(tau_phi)) @ Jinv)
        J6.set_block[3, 3](Jinv)
        return J6

    @staticmethod
    @inline(.always)
    def act_jacobian(p: Mat[Self.dtype, 3, 1]) -> Mat[Self.dtype, 3, Self.K]:
        var J = Mat[Self.dt, 3, 6]()
        J.set_block[0, 0](Mat[Self.dt, 3, 3].identity())
        J.set_block[0, 3](hat(-p))
        return J

    @staticmethod
    @inline(.always)
    def act4_jacobian(p: Mat[Self.dtype, 4, 1]) -> Mat[Self.dtype, 4, Self.K]:
        var J = Mat[Self.dt, 4, 6]()
        J.set_block[0, 0](p[3] * Mat[Self.dt, 3, 3].identity())
        J.set_block[0, 3](hat(-p.block[0, 0, 3, 1]()))
        return J
