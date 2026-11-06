"""Sim3: similarity transforms, a port of lietorch `include/sim3.h`.

Data row: tx ty tz qx qy qz qw s. Tangent: tau (3, translation), phi (3,
rotation), sigma (log-scale), as in lietorch.

Two sim3.h behaviours are kept on purpose, because lietorch's gradients
depend on them:
- `left_jacobian` is the series I + Xi/2 + Xi^2/6 + Xi^3/24 + Xi^4/120; sim3.h
  ends that statement with a stray `;` before its Xi^5/720 term, so the term is
  never added.
- `left_jacobian_inverse` is the truncated series I - Xi/2 + Xi^2/12 - Xi^4/720.
Both are approximations whose error grows with the tangent norm (~|Xi|^5/720
and ~|Xi|^6/30240), so Sim3 `expm_backward`, `logm_backward` and `Jinv` are as
approximate as lietorch's (upstream run_tests.py checks Sim3 gradients at
tol 1e-3).

`Log` inverts `calcW` with the cofactor formula of Eigen's 3x3 `inverse()`.
"""

from lie_group import LieGroup
from lie_linalg import Mat
from rxso3 import RxSO3
from so3 import from_double, hat


@inline(.always)
def _cofactor[dtype: DType, //, i: Int, j: Int](m: Mat[dtype, 3, 3]) -> Scalar[dtype]:
    comptime i1 = (i + 1) % 3
    comptime i2 = (i + 2) % 3
    comptime j1 = (j + 1) % 3
    comptime j2 = (j + 2) % 3
    return m[i1, j1] * m[i2, j2] - m[i1, j2] * m[i2, j1]


@inline(.always)
def inverse3[dtype: DType, //](m: Mat[dtype, 3, 3]) -> Mat[dtype, 3, 3]:
    """Eigen's `compute_inverse<3,3>`: the adjugate times 1 / det, det expanded along column 0."""
    var c00 = _cofactor[0, 0](m)
    var c10 = _cofactor[1, 0](m)
    var c20 = _cofactor[2, 0](m)
    var det = c00 * m[0, 0] + c10 * m[1, 0] + c20 * m[2, 0]
    var invdet = 1 / det
    var r = Mat[dtype, 3, 3]()
    r[0, 0] = c00 * invdet
    r[0, 1] = c10 * invdet
    r[0, 2] = c20 * invdet
    r[1, 0] = _cofactor[0, 1](m) * invdet
    r[1, 1] = _cofactor[1, 1](m) * invdet
    r[1, 2] = _cofactor[2, 1](m) * invdet
    r[2, 0] = _cofactor[0, 2](m) * invdet
    r[2, 1] = _cofactor[1, 2](m) * invdet
    r[2, 2] = _cofactor[2, 2](m) * invdet
    return r


struct Sim3[dt: DType](LieGroup):
    comptime dtype = Self.dt
    comptime N = 8
    comptime K = 7

    var rxso3: RxSO3[Self.dt]
    var t: Mat[Self.dt, 3, 1]

    @inline(.always)
    def __init__(out self, rxso3: RxSO3[Self.dt], t: Mat[Self.dt, 3, 1]):
        self.rxso3 = rxso3
        self.t = t

    @staticmethod
    @inline(.always)
    def from_data(x: Mat[Self.dtype, Self.N, 1]) -> Self:
        return Self(RxSO3[Self.dt].from_data(x.block[3, 0, 5, 1]()), x.block[0, 0, 3, 1]())

    @inline(.always)
    def data(self) -> Mat[Self.dtype, Self.N, 1]:
        var r = Mat[Self.dtype, Self.N, 1]()
        r.set_block[0, 0](self.t)
        r.set_block[3, 0](self.rxso3.data())
        return r

    @inline(.always)
    def inv(self) -> Self:
        var r = self.rxso3.inv()
        return Self(r, -(r.act(self.t)))

    @inline(.always)
    def mul(self, o: Self) -> Self:
        return Self(self.rxso3.mul(o.rxso3), self.t + self.rxso3.act(o.t))

    @inline(.always)
    def act(self, p: Mat[Self.dtype, 3, 1]) -> Mat[Self.dtype, 3, 1]:
        return self.rxso3.act(p) + self.t

    @inline(.always)
    def act4(self, p: Mat[Self.dtype, 4, 1]) -> Mat[Self.dtype, 4, 1]:
        var r = Mat[Self.dt, 4, 1]()
        r.set_block[0, 0](self.rxso3.act(p.block[0, 0, 3, 1]()) + p[3] * self.t)
        r[3] = p[3]
        return r

    @inline(.always)
    def Adj(self) -> Mat[Self.dtype, Self.K, Self.K]:
        var R = self.rxso3.so3.Matrix()
        var Ad = Mat[Self.dt, 7, 7].identity()
        Ad.set_block[0, 0](self.rxso3.scale * R)
        Ad.set_block[0, 3](hat(self.t) @ R)
        Ad.set_block[0, 6](-self.t)
        Ad.set_block[3, 3](R)
        return Ad

    @inline(.always)
    def Matrix4x4(self) -> Mat[Self.dtype, 4, 4]:
        var T = Mat[Self.dt, 4, 4].identity()
        T.set_block[0, 0](self.rxso3.Matrix())
        T.set_block[0, 3](self.t)
        return T

    @inline(.always)
    def orthogonal_projector(self) -> Mat[Self.dtype, Self.N, Self.N]:
        var J = Mat[Self.dt, 8, 8]()
        J.set_block[0, 0](Mat[Self.dt, 3, 3].identity())
        J.set_block[0, 3](hat(-self.t))
        J.set_block[0, 6](self.t)
        J.set_block[3, 3](self.rxso3.orthogonal_projector())
        return J

    @staticmethod
    @inline(.always)
    def adj(tau_phi_sigma: Mat[Self.dtype, Self.K, 1]) -> Mat[Self.dtype, Self.K, Self.K]:
        var tau = tau_phi_sigma.block[0, 0, 3, 1]()
        var Phi = hat(tau_phi_sigma.block[3, 0, 3, 1]())
        var sigma = tau_phi_sigma[6]
        var ad = Mat[Self.dt, 7, 7]()
        ad.set_block[0, 0](Phi + sigma * Mat[Self.dt, 3, 3].identity())
        ad.set_block[0, 3](hat(tau))
        ad.set_block[0, 6](-tau)
        ad.set_block[3, 3](Phi)
        return ad

    @inline(.always)
    def Log(self) -> Mat[Self.dtype, Self.K, 1]:
        var phi_sigma = self.rxso3.Log()
        var W = RxSO3[Self.dt].calcW(phi_sigma)
        var r = Mat[Self.dt, 7, 1]()
        r.set_block[0, 0](inverse3(W) @ self.t)
        r.set_block[3, 0](phi_sigma)
        return r

    @staticmethod
    @inline(.always)
    def Exp(tau_phi_sigma: Mat[Self.dtype, Self.K, 1]) -> Self:
        var phi_sigma = tau_phi_sigma.block[3, 0, 4, 1]()
        var W = RxSO3[Self.dt].calcW(phi_sigma)
        return Self(RxSO3[Self.dt].Exp(phi_sigma), W @ tau_phi_sigma.block[0, 0, 3, 1]())

    @staticmethod
    @inline(.always)
    def left_jacobian(tau_phi_sigma: Mat[Self.dtype, Self.K, 1]) -> Mat[Self.dtype, Self.K, Self.K]:
        var Xi = Self.adj(tau_phi_sigma)
        var Xi2 = Xi @ Xi
        var Xi4 = Xi2 @ Xi2
        # sim3.h drops its Xi^5 / 720 term (module docstring).
        return (
            Mat[Self.dt, 7, 7].identity()
            + from_double[Self.dt](1.0 / 2.0) * Xi
            + from_double[Self.dt](1.0 / 6.0) * Xi2
            + from_double[Self.dt](1.0 / 24.0) * (Xi @ Xi2)
            + from_double[Self.dt](1.0 / 120.0) * Xi4
        )

    @staticmethod
    @inline(.always)
    def left_jacobian_inverse(tau_phi_sigma: Mat[Self.dtype, Self.K, 1]) -> Mat[Self.dtype, Self.K, Self.K]:
        var Xi = Self.adj(tau_phi_sigma)
        var Xi2 = Xi @ Xi
        var Xi4 = Xi2 @ Xi2
        return (
            Mat[Self.dt, 7, 7].identity()
            - from_double[Self.dt](1.0 / 2.0) * Xi
            + from_double[Self.dt](1.0 / 12.0) * Xi2
            - from_double[Self.dt](1.0 / 720.0) * Xi4
        )

    @staticmethod
    @inline(.always)
    def act_jacobian(p: Mat[Self.dtype, 3, 1]) -> Mat[Self.dtype, 3, Self.K]:
        var J = Mat[Self.dt, 3, 7]()
        J.set_block[0, 0](Mat[Self.dt, 3, 3].identity())
        J.set_block[0, 3](hat(-p))
        J.set_block[0, 6](p)
        return J

    @staticmethod
    @inline(.always)
    def act4_jacobian(p: Mat[Self.dtype, 4, 1]) -> Mat[Self.dtype, 4, Self.K]:
        var J = Mat[Self.dt, 4, 7]()
        var p3 = p.block[0, 0, 3, 1]()
        J.set_block[0, 0](p[3] * Mat[Self.dt, 3, 3].identity())
        J.set_block[0, 3](hat(-p3))
        J.set_block[0, 6](p3)
        return J
