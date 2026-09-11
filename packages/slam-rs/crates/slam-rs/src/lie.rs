//! SO(3), SE(3), group Jacobians and decoupled pose increments.
//! SO(3) operations delegate to kornia-algebra through scalar adapters (S33).
//! This module supplies reported-angle conventions, composition normalization,
//! inverse Jacobians and SE(3) operations.
//!
//! Tangents put translation first. Estimator updates add translation directly
//! and left-multiply rotation by `exp(inc[3..6])`; every state block and prior
//! uses this convention. Coupled SE(3) operations instead use a translation factor.
//! Their small-angle Taylor branches stop one term earlier than the standalone
//! SO(3) Jacobians, so each call retains its own formula.

use kornia_algebra::{SO3F32, SO3F64, Vec3AF32, Vec3F64};
use nalgebra::{
    Matrix3, Matrix3x4, Matrix4, Matrix6, Quaternion, RealField, UnitQuaternion, Vector3, Vector6,
};

/// A scalar the Lie module can run in: `f32` and `f64`.
///
/// The extra methods carry `Sophus::Constants<Scalar>`,
/// whose epsilon is `1e-10` in double and `1e-5` in float. `nalgebra`'s own
/// `RealField::default_epsilon` is the machine epsilon and would move every
/// small-angle branch, so it is not used.
pub trait LieScalar: RealField + Copy {
    /// `Sophus::Constants<Scalar>::epsilon()`.
    ///
    /// A `const` rather than a method so a context with no value to call it on
    /// can still name it: `gpu::kernels` aliases it the way it aliases
    /// `PATCH_BORDER`, instead of re-declaring `1e-5`.
    const SOPHUS_EPSILON: Self;

    /// `Sophus::Constants<Scalar>::epsilonSqrt()`.
    fn sophus_epsilon_sqrt() -> Self {
        Self::SOPHUS_EPSILON.sqrt()
    }

    /// Near-antiparallel tolerance: 1e-12 for f64 and 1e-5 for f32.
    /// This is coarser than machine epsilon to stabilize gravity initialization.
    fn eigen_dummy_precision() -> Self;

    /// Smallest positive normal scalar, used to give tiny LDLT pivots zero weight.
    /// This is not the most negative finite value returned by `RealField::min_value`.
    fn min_positive() -> Self;

    /// Largest finite scalar, used to initialize eviction score minima.
    fn largest() -> Self;

    /// SO(3) exponential through kornia-algebra, in `[qx, qy, qz, qw]` order.
    /// The scalar adapters bridge concrete f32/f64 types without allocation.
    /// Reported theta retains this module's small-angle convention even where the
    /// underlying exponential uses a different Taylor threshold.
    fn so3_exp(omega: &[Self; 3]) -> [Self; 4];

    /// SO(3)'s logarithm, `SO3F32::log` or `SO3F64::log`, from `[qx, qy, qz, qw]`.
    ///
    /// Upstream folds `w < 0` by negating both parts before one `atan2`, where
    /// Sophus folds it into `atan2(-n, -w)`; the two are the same expression.
    /// See [`Self::so3_exp`] for the branch thresholds.
    fn so3_log(quaternion_xyzw: &[Self; 4]) -> [Self; 3];

    /// The 3x3 rotation matrix, `SO3F32::matrix` or `SO3F64::matrix`, **column
    /// major**.
    ///
    fn so3_matrix(quaternion_xyzw: &[Self; 4]) -> [Self; 9];

    /// The group inverse, `SO3F32::inverse` or `SO3F64::inverse`: the conjugate
    /// of a unit quaternion, which is what Sophus takes too.
    fn so3_inverse(quaternion_xyzw: &[Self; 4]) -> [Self; 4];

    /// The action on a point, `SO3F32 * Vec3AF32` or `SO3F64 * Vec3F64`.
    fn so3_act(quaternion_xyzw: &[Self; 4], point: &[Self; 3]) -> [Self; 3];

    /// Convert a literal to the selected scalar as accurately as possible.
    fn from_literal(value: f64) -> Self;

    /// Convert to `f64` for diagnostics and scalar conversion.
    fn to_f64(self) -> f64;
}

impl LieScalar for f64 {
    const SOPHUS_EPSILON: Self = 1e-10;

    fn eigen_dummy_precision() -> Self {
        1e-12
    }

    fn min_positive() -> Self {
        Self::MIN_POSITIVE
    }

    fn largest() -> Self {
        Self::MAX
    }

    fn so3_exp(omega: &[Self; 3]) -> [Self; 4] {
        SO3F64::exp(Vec3F64::from_array(*omega)).to_array()
    }

    fn so3_log(quaternion_xyzw: &[Self; 4]) -> [Self; 3] {
        SO3F64::from_array(*quaternion_xyzw).log().to_array()
    }

    fn so3_matrix(quaternion_xyzw: &[Self; 4]) -> [Self; 9] {
        SO3F64::from_array(*quaternion_xyzw)
            .matrix()
            .to_cols_array()
    }

    fn so3_inverse(quaternion_xyzw: &[Self; 4]) -> [Self; 4] {
        SO3F64::from_array(*quaternion_xyzw).inverse().to_array()
    }

    fn so3_act(quaternion_xyzw: &[Self; 4], point: &[Self; 3]) -> [Self; 3] {
        (SO3F64::from_array(*quaternion_xyzw) * Vec3F64::from_array(*point)).to_array()
    }

    fn from_literal(value: f64) -> Self {
        value
    }

    fn to_f64(self) -> f64 {
        self
    }
}

impl LieScalar for f32 {
    const SOPHUS_EPSILON: Self = 1e-5;

    fn eigen_dummy_precision() -> Self {
        1e-5
    }

    fn min_positive() -> Self {
        Self::MIN_POSITIVE
    }

    fn largest() -> Self {
        Self::MAX
    }

    fn so3_exp(omega: &[Self; 3]) -> [Self; 4] {
        SO3F32::exp(Vec3AF32::from_array(*omega)).to_array()
    }

    fn so3_log(quaternion_xyzw: &[Self; 4]) -> [Self; 3] {
        SO3F32::from_array(*quaternion_xyzw).log().to_array()
    }

    fn so3_matrix(quaternion_xyzw: &[Self; 4]) -> [Self; 9] {
        SO3F32::from_array(*quaternion_xyzw)
            .matrix()
            .to_cols_array()
    }

    fn so3_inverse(quaternion_xyzw: &[Self; 4]) -> [Self; 4] {
        SO3F32::from_array(*quaternion_xyzw).inverse().to_array()
    }

    fn so3_act(quaternion_xyzw: &[Self; 4], point: &[Self; 3]) -> [Self; 3] {
        (SO3F32::from_array(*quaternion_xyzw) * Vec3AF32::from_array(*point)).to_array()
    }

    fn from_literal(value: f64) -> Self {
        value as Self
    }

    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

/// A literal in the caller's scalar type, through `S::from_literal`.
#[inline]
pub(crate) fn c<S: LieScalar>(value: f64) -> S {
    S::from_literal(value)
}

/// Return the larger value, preserving a NaN on the left.
/// The damped solver must retry on invalid input rather than suppress its NaN.
pub(crate) fn eigen_maxi<S: LieScalar>(a: S, b: S) -> S {
    if a < b { b } else { a }
}

/// A rotation, stored as a unit quaternion exactly as `Sophus::SO3` does.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct So3<S: LieScalar> {
    quaternion: UnitQuaternion<S>,
}

impl<S: LieScalar> Default for So3<S> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<S: LieScalar> So3<S> {
    /// The identity rotation.
    pub fn identity() -> Self {
        Self {
            quaternion: UnitQuaternion::identity(),
        }
    }

    /// Wrap a quaternion that is already of unit length.
    pub fn from_unit_quaternion(quaternion: UnitQuaternion<S>) -> Self {
        Self { quaternion }
    }

    /// Normalize `(x, y, z, w)` into a rotation, correcting small calibration drift.
    /// Return `None` for a zero-norm quaternion rather than producing NaNs.
    pub fn from_quaternion_xyzw(x: S, y: S, z: S, w: S) -> Option<Self> {
        let quaternion: Quaternion<S> = Quaternion::new(w, x, y, z);
        if !quaternion.norm().is_finite() || quaternion.norm() <= S::zero() {
            return None;
        }
        Some(Self {
            quaternion: UnitQuaternion::new_normalize(quaternion),
        })
    }

    /// `Sophus::SO3::cast` : the same rotation in
    /// another scalar.
    ///
    /// Sophus casts the four coefficients and hands them to the quaternion
    /// constructor, which normalizes; the narrowing is what makes that
    /// normalization not a no-op. A rotation that cannot be normalized in the
    /// target scalar — only reachable from a non-finite input — comes back as
    /// the identity, which is what `Calibration::cast` needs to stay total.
    pub fn cast<T: LieScalar>(&self) -> So3<T> {
        let [x, y, z, w]: [S; 4] = self.quaternion_xyzw();
        So3::from_quaternion_xyzw(
            T::from_literal(x.to_f64()),
            T::from_literal(y.to_f64()),
            T::from_literal(z.to_f64()),
            T::from_literal(w.to_f64()),
        )
        .unwrap_or_else(So3::identity)
    }

    /// The underlying unit quaternion.
    pub fn quaternion(&self) -> &UnitQuaternion<S> {
        &self.quaternion
    }

    /// Quaternion coefficients in JSON order `(qx, qy, qz, qw)`.
    pub fn quaternion_xyzw(&self) -> [S; 4] {
        let q = self.quaternion.as_ref();
        [q.i, q.j, q.k, q.w]
    }

    /// The 3x3 skew-symmetric matrix of a vector, `Sophus::SO3::hat`.
    pub fn hat(v: &Vector3<S>) -> Matrix3<S> {
        let zero: S = S::zero();
        Matrix3::new(zero, -v.z, v.y, v.z, zero, -v.x, -v.y, v.x, zero)
    }

    /// The exponential map, `kornia_algebra::lie::SO3F32::exp` / `SO3F64::exp`.
    pub fn exp(omega: &Vector3<S>) -> Self {
        Self::from_kornia_quaternion(S::so3_exp(&[omega.x, omega.y, omega.z]))
    }

    /// [`So3::exp`] plus `theta = |omega|`, which is zero on the Taylor branch.
    ///
    /// `Sophus::SO3::expAndTheta` reports the same
    /// `theta` it branched on, and [`Se3::exp`] feeds it straight into the left
    /// Jacobian, so the two travel together. The **rotation** is upstream's; the
    /// `theta` is still Sophus's convention, including the zero it reports on its
    /// own small-angle branch and the threshold it uses. Upstream branches
    /// slightly earlier in `f32` ([`LieScalar::so3_exp`]), which moves nothing
    /// but the last bits of the quaternion there, and `Se3::exp`'s left Jacobian
    /// keeps reading exactly the `theta` it read before.
    pub fn exp_and_theta(omega: &Vector3<S>) -> (Self, S) {
        let theta_sq: S = omega.norm_squared();
        let epsilon: S = S::SOPHUS_EPSILON;
        let theta: S = if theta_sq < epsilon * epsilon {
            S::zero()
        } else {
            theta_sq.sqrt()
        };
        (Self::exp(omega), theta)
    }

    /// A `[qx, qy, qz, qw]` from `kornia-algebra` as an `So3`.
    ///
    /// `new_unchecked` because every upstream operation that produces one starts
    /// from a unit quaternion and stays on the sphere to within rounding — the
    /// same assumption Sophus makes when it asserts rather than normalises
    /// The one place the assumption is not free is
    /// composition, which is why [`So3::mul`] normalises.
    #[inline]
    fn from_kornia_quaternion(xyzw: [S; 4]) -> Self {
        Self {
            quaternion: UnitQuaternion::new_unchecked(Quaternion::new(
                xyzw[3], xyzw[0], xyzw[1], xyzw[2],
            )),
        }
    }

    /// The logarithm, `kornia_algebra::lie::SO3F32::log` / `SO3F64::log`.
    pub fn log(&self) -> Vector3<S> {
        Vector3::from(S::so3_log(&self.quaternion_xyzw()))
    }

    /// [`So3::log`] plus Sophus's `theta`, which is **not** `|log|` on the
    /// Taylor branch.
    ///
    /// `Sophus::SO3::logAndTheta` returns `2 n^2 / w` for a near-identity
    /// rotation, i.e. second order in the vector
    /// part, while the tangent itself is first order. [`Se3::log`] passes that
    /// `theta` to the inverse left Jacobian, and substituting `|log|` there
    /// moves it onto the ill-conditioned branch, so the pair travels together
    /// and the port keeps Sophus's branch and its threshold.
    ///
    /// Off that branch, `theta` is `2 atan2(n, w)` with the `w < 0` wrap folded
    /// in, which is the *signed magnitude* of the tangent: Sophus computes it as
    /// `two_atan_nbyw_by_n * n` and the tangent as `two_atan_nbyw_by_n * vec`,
    /// so the two differ only by `|vec|`, and the sign is negative exactly when
    /// `w < 0`. Taking it from the vector upstream returned is the same
    /// quantity and leaves one implementation of the `atan2`, not two.
    pub fn log_and_theta(&self) -> (Vector3<S>, S) {
        let q = self.quaternion.as_ref();
        let squared_n: S = q.vector().norm_squared();
        let w: S = q.w;
        let epsilon: S = S::SOPHUS_EPSILON;

        let tangent: Vector3<S> = self.log();
        let theta: S = if squared_n < epsilon * epsilon {
            // A unit quaternion with a vanishing vector part has |w| ~ 1, so the
            // division is safe; Sophus asserts the same thing.
            c::<S>(2.0) * squared_n / w
        } else if w < S::zero() {
            // w < 0 means the rotation is past pi; Sophus wraps it to a negative
            // angle rather than reporting the reflex one.
            -tangent.norm()
        } else {
            tangent.norm()
        };
        (tangent, theta)
    }

    /// The inverse rotation, `kornia_algebra::lie::SO3F32::inverse` /
    /// `SO3F64::inverse` — the conjugate of a unit quaternion, which is what
    /// Sophus takes too.
    ///
    /// Sophus's constructor then renormalizes, so this one does
    /// too. Conjugating only flips signs, so the renormalization is a no-op
    /// here — it is kept for the same reason Sophus keeps it: every path that
    /// produces an `So3` leaves it unit length.
    pub fn inverse(&self) -> Self {
        let [x, y, z, w]: [S; 4] = S::so3_inverse(&self.quaternion_xyzw());
        Self {
            quaternion: normalized(Quaternion::new(w, x, y, z)),
        }
    }

    /// The rotation as a 3x3 matrix, `kornia_algebra::lie::SO3F32::matrix` /
    /// `SO3F64::matrix`.
    ///
    pub fn matrix(&self) -> Matrix3<S> {
        // Both `glam` and nalgebra store column major, so the array transfers
        // without a transpose.
        Matrix3::from_column_slice(&S::so3_matrix(&self.quaternion_xyzw()))
    }
}

impl<S: LieScalar> std::ops::Mul for So3<S> {
    type Output = Self;

    /// Compose two rotations, renormalizing the product.
    ///
    /// Sophus's `operator*` hands the raw quaternion product to the `SO3`
    /// quaternion constructor, which calls
    /// `normalize()`. `nalgebra`'s
    /// `UnitQuaternion * UnitQuaternion` does not: it trusts the invariant and
    /// lets rounding accumulate. Over a long chain that matters — composing one
    /// small rotation 100,000 times in `f32` drifts the norm to 1.00105 without
    /// the renormalization and stays at 1 with it.
    fn mul(self, rhs: Self) -> Self {
        Self {
            quaternion: normalized(self.quaternion.into_inner() * rhs.quaternion.into_inner()),
        }
    }
}

/// `Sophus::SO3::normalize()`.
///
/// Sophus refuses a quaternion shorter than its epsilon; here the inputs are
/// always products or conjugates of unit quaternions, so the norm is within a
/// few ulps of 1 and `new_normalize` cannot divide by zero.
#[inline]
fn normalized<S: LieScalar>(quaternion: Quaternion<S>) -> UnitQuaternion<S> {
    UnitQuaternion::new_normalize(quaternion)
}

impl<S: LieScalar> std::ops::Mul<Vector3<S>> for So3<S> {
    type Output = Vector3<S>;

    /// Rotate a point, `SO3F32 * Vec3AF32` / `SO3F64 * Vec3F64`.
    ///
    /// Rotate a point with the scalar-specific kornia quaternion action.
    fn mul(self, rhs: Vector3<S>) -> Vector3<S> {
        Vector3::from(S::so3_act(&self.quaternion_xyzw(), &[rhs.x, rhs.y, rhs.z]))
    }
}

/// A rigid transform, `Sophus::SE3` as a rotation plus a translation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Se3<S: LieScalar> {
    /// The rotation part.
    pub rotation: So3<S>,
    /// The translation part.
    pub translation: Vector3<S>,
}

impl<S: LieScalar> Default for Se3<S> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<S: LieScalar> Se3<S> {
    /// A transform from its two parts.
    pub fn new(rotation: So3<S>, translation: Vector3<S>) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// The identity transform.
    pub fn identity() -> Self {
        Self {
            rotation: So3::identity(),
            translation: Vector3::zeros(),
        }
    }

    /// Coupled exponential: translation is `V(omega) upsilon`.
    /// State updates use [`Se3::exp_decoupled`] and [`Se3::apply_inc`] instead.
    pub fn exp(tangent: &Vector6<S>) -> Self {
        let upsilon: Vector3<S> = tangent.fixed_rows::<3>(0).into_owned();
        let omega: Vector3<S> = tangent.fixed_rows::<3>(3).into_owned();
        let (rotation, theta) = So3::exp_and_theta(&omega);
        let v: Matrix3<S> = sophus_left_jacobian_so3(&omega, theta);
        Self {
            rotation,
            translation: v * upsilon,
        }
    }

    /// Coupled logarithm.
    pub fn log(&self) -> Vector6<S> {
        let (omega, theta) = self.rotation.log_and_theta();
        let v_inv: Matrix3<S> = sophus_left_jacobian_inv_so3(&omega, theta);
        let upsilon: Vector3<S> = v_inv * self.translation;
        Vector6::new(upsilon.x, upsilon.y, upsilon.z, omega.x, omega.y, omega.z)
    }

    /// Decoupled exponential: tangent head becomes translation and the tail uses SO(3) exp.
    pub fn exp_decoupled(tangent: &Vector6<S>) -> Self {
        Self {
            rotation: So3::exp(&tangent.fixed_rows::<3>(3).into_owned()),
            translation: tangent.fixed_rows::<3>(0).into_owned(),
        }
    }

    /// Decoupled logarithm, inverse to [`Self::exp_decoupled`].
    /// Used by round-trip and finite-difference tests; production updates do not read tangents back.
    pub fn log_decoupled(&self) -> Vector6<S> {
        let omega: Vector3<S> = self.rotation.log();
        Vector6::new(
            self.translation.x,
            self.translation.y,
            self.translation.z,
            omega.x,
            omega.y,
            omega.z,
        )
    }

    /// `Sophus::SE3::cast`: the same transform in another scalar.
    pub fn cast<T: LieScalar>(&self) -> Se3<T> {
        Se3 {
            rotation: self.rotation.cast(),
            translation: Vector3::new(
                T::from_literal(self.translation.x.to_f64()),
                T::from_literal(self.translation.y.to_f64()),
                T::from_literal(self.translation.z.to_f64()),
            ),
        }
    }

    /// The homogeneous 4x4 matrix, `Sophus::SE3::matrix()`
    ///
    /// The rotation block uses [`So3::matrix`]; the final column is translation.
    pub fn matrix(&self) -> Matrix4<S> {
        let mut res: Matrix4<S> = Matrix4::zeros();
        res.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&self.rotation.matrix());
        res.fixed_view_mut::<3, 1>(0, 3)
            .copy_from(&self.translation);
        res[(3, 3)] = S::one();
        res
    }

    /// The affine 3x4 matrix, `Sophus::SE3::matrix3x4()`
    /// `[R | t]`.
    pub fn matrix3x4(&self) -> Matrix3x4<S> {
        let mut res: Matrix3x4<S> = Matrix3x4::zeros();
        res.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&self.rotation.matrix());
        res.fixed_view_mut::<3, 1>(0, 3)
            .copy_from(&self.translation);
        res
    }

    /// The inverse transform.
    pub fn inverse(&self) -> Self {
        let rotation: So3<S> = self.rotation.inverse();
        Self {
            translation: -(rotation * self.translation),
            rotation,
        }
    }

    /// Adjoint for translation-first tangents: `[[R, hat(t) R], [0, R]]`.
    /// It satisfies `Adj(T) xi = log(T exp(xi) T^-1)`.
    pub fn adjoint(&self) -> Matrix6<S> {
        let r: Matrix3<S> = self.rotation.matrix();
        let mut res: Matrix6<S> = Matrix6::zeros();
        res.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        res.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
        res.fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(So3::hat(&self.translation) * r));
        res
    }

    /// Pose increment: `t += inc[0..3]`, `R = exp(inc[3..6]) R`.
    /// State blocks, Jacobians and priors all require this left-multiplied rotation convention.
    pub fn apply_inc(&mut self, inc: &Vector6<S>) {
        self.translation += inc.fixed_rows::<3>(0);
        self.rotation = So3::exp(&inc.fixed_rows::<3>(3).into_owned()) * self.rotation;
    }
}

impl<S: LieScalar> std::ops::Mul for Se3<S> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self {
            rotation: self.rotation * rhs.rotation,
            translation: self.translation + self.rotation * rhs.translation,
        }
    }
}

impl<S: LieScalar> std::ops::Mul<Vector3<S>> for Se3<S> {
    type Output = Vector3<S>;

    fn mul(self, rhs: Vector3<S>) -> Vector3<S> {
        self.rotation * rhs + self.translation
    }
}

/// Right SO(3) Jacobian: `exp(phi + eps) ~ exp(phi) exp(J eps)`.
pub fn right_jacobian_so3<S: LieScalar>(phi: &Vector3<S>) -> Matrix3<S> {
    let phi_norm2: S = phi.norm_squared();
    let phi_hat: Matrix3<S> = So3::hat(phi);
    let phi_hat2: Matrix3<S> = phi_hat * phi_hat;

    let mut j: Matrix3<S> = Matrix3::identity();
    if phi_norm2 > S::SOPHUS_EPSILON {
        let phi_norm: S = phi_norm2.sqrt();
        let phi_norm3: S = phi_norm2 * phi_norm;
        j -= phi_hat * ((c::<S>(1.0) - phi_norm.cos()) / phi_norm2);
        j += phi_hat2 * ((phi_norm - phi_norm.sin()) / phi_norm3);
    } else {
        // Taylor expansion around 0.
        j -= phi_hat / c::<S>(2.0);
        j += phi_hat2 / c::<S>(6.0);
    }
    j
}

/// Inverse right SO(3) Jacobian: `log(exp(phi) exp(eps)) ~ phi + J eps`.
/// Over-rotated inputs use the same zeroth-order branch as pi, without panic (D32).
pub fn right_jacobian_inv_so3<S: LieScalar>(phi: &Vector3<S>) -> Matrix3<S> {
    let phi_norm2: S = phi.norm_squared();
    let phi_hat: Matrix3<S> = So3::hat(phi);
    let phi_hat2: Matrix3<S> = phi_hat * phi_hat;

    let mut j: Matrix3<S> = Matrix3::identity() + phi_hat / c::<S>(2.0);
    j += inverse_jacobian_second_order_term(&phi_hat2, phi_norm2);
    j
}

/// Inverse left SO(3) Jacobian: `log(exp(eps) exp(phi)) ~ phi + J eps`.
pub fn left_jacobian_inv_so3<S: LieScalar>(phi: &Vector3<S>) -> Matrix3<S> {
    let phi_norm2: S = phi.norm_squared();
    let phi_hat: Matrix3<S> = So3::hat(phi);
    let phi_hat2: Matrix3<S> = phi_hat * phi_hat;

    let mut j: Matrix3<S> = Matrix3::identity() - phi_hat / c::<S>(2.0);
    j += inverse_jacobian_second_order_term(&phi_hat2, phi_norm2);
    j
}

/// The `hat(phi)^2` term in both inverse SO(3) Jacobians.
/// Use the closed form on `(0, pi)`, a zeroth-order expansion at pi where sine
/// vanishes, and `1/12` at zero. Thresholds and denominators use the input scalar.
fn inverse_jacobian_second_order_term<S: LieScalar>(
    phi_hat2: &Matrix3<S>,
    phi_norm2: S,
) -> Matrix3<S> {
    if phi_norm2 <= S::SOPHUS_EPSILON {
        // Taylor expansion around 0.
        return phi_hat2 / c::<S>(12.0);
    }
    let phi_norm: S = phi_norm2.sqrt();
    let pi: S = c(std::f64::consts::PI);
    let threshold: S = pi - S::sophus_epsilon_sqrt();
    if phi_norm < threshold {
        // Regular case on (0, pi).
        phi_hat2
            * (c::<S>(1.0) / phi_norm2
                - (c::<S>(1.0) + phi_norm.cos()) / (c::<S>(2.0) * phi_norm * phi_norm.sin()))
    } else {
        // 0th-order Taylor expansion around pi.
        phi_hat2 / (pi * pi)
    }
}

/// Left Jacobian for coupled [`Se3::exp`]. Its Taylor branch stops at
/// `I + Omega/2`, unlike the standalone Jacobian's additional `Omega^2/6` term.
fn sophus_left_jacobian_so3<S: LieScalar>(omega: &Vector3<S>, theta: S) -> Matrix3<S> {
    let theta_sq: S = theta * theta;
    let big_omega: Matrix3<S> = So3::hat(omega);
    let epsilon: S = S::SOPHUS_EPSILON;

    if theta_sq < epsilon * epsilon {
        Matrix3::identity() + big_omega * c::<S>(0.5)
    } else {
        Matrix3::identity()
            + big_omega * ((c::<S>(1.0) - theta.cos()) / theta_sq)
            + big_omega * big_omega * ((theta - theta.sin()) / (theta_sq * theta))
    }
}

/// Sophus's own inverse left Jacobian, `SO3::leftJacobianInverse`
/// Used only by [`Se3::log`].
fn sophus_left_jacobian_inv_so3<S: LieScalar>(omega: &Vector3<S>, theta: S) -> Matrix3<S> {
    let theta_sq: S = theta * theta;
    let big_omega: Matrix3<S> = So3::hat(omega);
    let epsilon: S = S::SOPHUS_EPSILON;

    let identity: Matrix3<S> = Matrix3::identity();
    if theta_sq < epsilon * epsilon {
        identity - big_omega * c::<S>(0.5) + big_omega * big_omega * c::<S>(1.0 / 12.0)
    } else {
        let half_theta: S = c::<S>(0.5) * theta;
        let factor: S = (c::<S>(1.0) - c::<S>(0.5) * theta * half_theta.cos() / half_theta.sin())
            / (theta * theta);
        identity - big_omega * c::<S>(0.5) + big_omega * big_omega * factor
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    /// Keeps the whole Rust suite in the "runs in seconds" band.
    const CASES: u32 = 256;

    #[test]
    fn eigen_maxi_keeps_a_nan_on_the_left() {
        assert!(eigen_maxi(f64::NAN, 1.0).is_nan());
        assert_eq!(eigen_maxi(1.0f64, f64::NAN), 1.0);
        assert_eq!(f64::NAN.max(1.0), 1.0, "std::f64::max is the other way");
    }

    fn config() -> ProptestConfig {
        ProptestConfig::with_cases(CASES)
    }

    /// Tangent vectors up to about 1.7 rad, well inside the `|phi| < pi` domain
    /// the inverse Jacobians are defined on.
    fn tangent3() -> impl Strategy<Value = Vector3<f64>> {
        (-1.0f64..1.0, -1.0f64..1.0, -1.0f64..1.0).prop_map(|(x, y, z)| Vector3::new(x, y, z))
    }

    fn tangent6() -> impl Strategy<Value = Vector6<f64>> {
        (
            -2.0f64..2.0,
            -2.0f64..2.0,
            -2.0f64..2.0,
            -1.0f64..1.0,
            -1.0f64..1.0,
            -1.0f64..1.0,
        )
            .prop_map(|(a, b, c, d, e, f)| Vector6::new(a, b, c, d, e, f))
    }

    /// Central-difference Jacobian of a map from R^3 to R^3.
    fn numeric_jacobian3<F>(at: &Vector3<f64>, f: F) -> Matrix3<f64>
    where
        F: Fn(&Vector3<f64>) -> Vector3<f64>,
    {
        let h: f64 = 1e-6;
        let mut j: Matrix3<f64> = Matrix3::zeros();
        for i in 0..3 {
            let mut plus: Vector3<f64> = *at;
            let mut minus: Vector3<f64> = *at;
            plus[i] += h;
            minus[i] -= h;
            let column: Vector3<f64> = (f(&plus) - f(&minus)) / (2.0 * h);
            j.set_column(i, &column);
        }
        j
    }

    /// The one value in this module checked by hand rather than against a
    /// finite difference: `SO3::exp([0.1, 0.2, 0.3])` under Sophus's formula
    /// `w = cos(theta/2)`, `v = sin(theta/2)/theta * omega`
    ///  with `theta = |omega|`.
    #[test]
    fn exp_matches_the_hand_computed_sophus_quaternion() {
        let rotation: So3<f64> = So3::exp(&Vector3::new(0.1, 0.2, 0.3));
        let [qx, qy, qz, qw] = rotation.quaternion_xyzw();
        assert_abs_diff_eq!(qw, 0.982_550_982_155_258_9, epsilon = 1e-15);
        assert_abs_diff_eq!(qx, 0.049_708_843_324_859_475, epsilon = 1e-15);
        assert_abs_diff_eq!(qy, 0.099_417_686_649_718_95, epsilon = 1e-15);
        assert_abs_diff_eq!(qz, 0.149_126_529_974_578_43, epsilon = 1e-15);
    }

    /// The rotation is a quarter turn about z, so the matrix is exact.
    #[test]
    fn exp_of_a_quarter_turn_about_z_is_the_expected_matrix() {
        let rotation: So3<f64> = So3::exp(&Vector3::new(0.0, 0.0, std::f64::consts::FRAC_PI_2));
        let m: Matrix3<f64> = rotation.matrix();
        assert_abs_diff_eq!(m[(0, 0)], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(m[(0, 1)], -1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(m[(1, 0)], 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(m[(2, 2)], 1.0, epsilon = 1e-15);
    }

    #[test]
    fn exp_and_log_agree_at_exactly_zero() {
        let zero: Vector3<f64> = Vector3::zeros();
        let rotation: So3<f64> = So3::exp(&zero);
        assert_abs_diff_eq!(rotation.log().norm(), 0.0, epsilon = 1e-18);
        assert_abs_diff_eq!(rotation.quaternion_xyzw()[3], 1.0, epsilon = 1e-18);
    }

    /// The Taylor branches must join the closed forms: just under the threshold
    /// and just over it, the results agree.
    #[test]
    fn the_small_angle_branches_join_the_closed_forms() {
        let below: Vector3<f64> = Vector3::new(1e-9, 0.0, 0.0);
        let above: Vector3<f64> = Vector3::new(1e-3, 0.0, 0.0);
        for phi in [below, above] {
            let j_right: Matrix3<f64> = right_jacobian_so3(&phi);
            let j_right_inv: Matrix3<f64> = right_jacobian_inv_so3(&phi);
            assert_abs_diff_eq!(j_right_inv * j_right, Matrix3::identity(), epsilon = 1e-12);
        }
    }

    /// [`So3::log_and_theta`]'s three branches, which S33 re-derived.
    ///
    /// The tangent is `kornia-algebra`'s now, but `theta` is still Sophus's and
    /// [`Se3::log`] feeds it to the inverse left Jacobian, so its two
    /// conventions have to be pinned rather than inferred from the tangent:
    ///
    /// * **past pi** (`w < 0`) `theta` is *negative* — Sophus wraps rather than
    ///   reporting the reflex angle — and the tangent is the wrapped one, so
    ///   `theta = -|log|`;
    /// * **near zero** `theta` is `2 n^2 / w`, second order in the vector part
    ///   where the tangent is first order, so it is emphatically **not**
    ///   `|log|`: at `|omega| = 1e-12` the tangent has norm 1e-12 and `theta` is
    ///   5e-25.
    ///
    /// Everywhere else `theta` is `+|log|`, including exactly at pi, where
    /// `w == 0` and Sophus's `w < 0` test is false.
    #[test]
    fn log_and_theta_keeps_sophus_two_conventions() {
        let axis: Vector3<f64> = Vector3::new(0.0, 0.0, 1.0);

        // Just under pi: theta is the angle, and positive.
        let (log, theta) = So3::exp(&(axis * (std::f64::consts::PI - 1e-6))).log_and_theta();
        assert_abs_diff_eq!(theta, std::f64::consts::PI - 1e-6, epsilon = 1e-12);
        assert_abs_diff_eq!(theta, log.norm(), epsilon = 1e-15);

        // Exactly at pi: w is zero, so the wrap does not fire.
        let (log, theta) = So3::exp(&(axis * std::f64::consts::PI)).log_and_theta();
        assert!(theta > 0.0, "at pi theta stays positive, got {theta}");
        assert_abs_diff_eq!(theta, log.norm(), epsilon = 1e-15);

        // Past pi: w < 0, the tangent wraps to |log| <= pi and theta is its
        // negative.
        let past: So3<f64> = So3::exp(&(axis * (std::f64::consts::PI + 0.5)));
        assert!(past.quaternion_xyzw()[3] < 0.0, "the fixture must wrap");
        let (log, theta) = past.log_and_theta();
        assert_abs_diff_eq!(log.norm(), std::f64::consts::PI - 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(theta, -log.norm(), epsilon = 1e-15);

        // Near zero: theta is second order where the tangent is first.
        let (log, theta) = So3::exp(&(axis * 1e-12)).log_and_theta();
        assert_abs_diff_eq!(log.norm(), 1e-12, epsilon = 1e-24);
        assert_abs_diff_eq!(theta, 5e-25, epsilon = 1e-27);
    }

    /// A long chain of compositions must not leak unit length.
    ///
    /// `nalgebra`'s `UnitQuaternion * UnitQuaternion` skips the renormalization
    /// Sophus does on every product, and
    /// 100,000 `f32` compositions of one small rotation take the norm to
    /// 1.0010456 without it — a rotation matrix off by 7e-3, which the
    /// estimator's window would carry straight into the residuals.
    #[test]
    fn a_hundred_thousand_compositions_stay_unit_length() {
        let step32: So3<f32> = So3::exp(&Vector3::new(0.001, 0.002, -0.0015));
        let mut chain32: So3<f32> = So3::identity();
        let step64: So3<f64> = So3::exp(&Vector3::new(0.001, 0.002, -0.0015));
        let mut chain64: So3<f64> = So3::identity();
        for _ in 0..100_000 {
            chain32 = chain32 * step32;
            chain64 = chain64 * step64;
        }

        let norm32: f32 = chain32
            .quaternion_xyzw()
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        assert_abs_diff_eq!(norm32, 1.0, epsilon = 1e-6);
        let m: Matrix3<f32> = chain32.matrix();
        assert!((m.transpose() * m - Matrix3::identity()).norm() < 1e-5);

        let norm64: f64 = chain64
            .quaternion_xyzw()
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert_abs_diff_eq!(norm64, 1.0, epsilon = 1e-12);
        let m: Matrix3<f64> = chain64.matrix();
        assert!((m.transpose() * m - Matrix3::identity()).norm() < 1e-12);
    }

    /// Check accuracy on both sides of the f32 branch boundary and at pi.
    #[test]
    fn the_inverse_jacobians_are_accurate_near_pi_in_f32() {
        let boundary: f32 = std::f32::consts::PI - f32::sophus_epsilon_sqrt();
        for angle in [
            f32::from_bits(boundary.to_bits() - 1),
            boundary,
            f32::from_bits(boundary.to_bits() + 1),
            std::f32::consts::PI,
        ] {
            let phi: Vector3<f32> = Vector3::new(angle, 0.0, 0.0);
            let reference: Vector3<f64> = phi.map(f64::from);
            for (actual, expected) in [
                (
                    right_jacobian_inv_so3(&phi),
                    right_jacobian_inv_so3(&reference),
                ),
                (
                    left_jacobian_inv_so3(&phi),
                    left_jacobian_inv_so3(&reference),
                ),
            ] {
                assert!(actual.iter().all(|v| v.is_finite()));
                assert!(
                    (actual.map(f64::from) - expected).norm() < 1e-3,
                    "angle={angle}"
                );
            }
        }
    }

    /// Near pi, use a zeroth-order expansion to avoid the vanishing sine denominator.
    #[test]
    fn the_inverse_jacobians_stay_finite_at_pi() {
        let phi: Vector3<f64> = Vector3::new(std::f64::consts::PI, 0.0, 0.0);
        let j: Matrix3<f64> = right_jacobian_inv_so3(&phi);
        assert!(j.iter().all(|v| v.is_finite()));
        let j: Matrix3<f64> = left_jacobian_inv_so3(&phi);
        assert!(j.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn se3_exp_of_a_pure_translation_is_that_translation() {
        let tangent: Vector6<f64> = Vector6::new(1.0, 2.0, 3.0, 0.0, 0.0, 0.0);
        let pose: Se3<f64> = Se3::exp(&tangent);
        assert_abs_diff_eq!(
            pose.translation,
            Vector3::new(1.0, 2.0, 3.0),
            epsilon = 1e-15
        );
    }

    /// The decoupled exponential puts the tangent head into the translation
    /// untouched, which the coupled one does not.
    #[test]
    fn the_decoupled_exponential_differs_from_the_coupled_one() {
        let tangent: Vector6<f64> = Vector6::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let coupled: Se3<f64> = Se3::exp(&tangent);
        let decoupled: Se3<f64> = Se3::exp_decoupled(&tangent);
        assert_abs_diff_eq!(
            decoupled.translation,
            Vector3::new(1.0, 0.0, 0.0),
            epsilon = 1e-15
        );
        assert!((coupled.translation - decoupled.translation).norm() > 0.1);
    }

    /// `apply_inc` is neither exponential: the rotation multiplies from the left
    /// and the translation is added in the world frame
    #[test]
    fn apply_inc_left_multiplies_the_rotation() {
        let base: Se3<f64> = Se3::new(
            So3::exp(&Vector3::new(0.3, -0.2, 0.1)),
            Vector3::new(1.0, 2.0, 3.0),
        );
        let inc: Vector6<f64> = Vector6::new(0.1, 0.2, 0.3, 0.01, -0.02, 0.03);
        let mut moved: Se3<f64> = base;
        moved.apply_inc(&inc);

        assert_abs_diff_eq!(
            moved.translation,
            base.translation + Vector3::new(0.1, 0.2, 0.3),
            epsilon = 1e-15
        );
        let expected: So3<f64> = So3::exp(&Vector3::new(0.01, -0.02, 0.03)) * base.rotation;
        assert_abs_diff_eq!(moved.rotation.matrix(), expected.matrix(), epsilon = 1e-15);
        // A right-multiplied increment would give a different rotation.
        let right: So3<f64> = base.rotation * So3::exp(&Vector3::new(0.01, -0.02, 0.03));
        assert!((moved.rotation.matrix() - right.matrix()).norm() > 1e-4);
    }

    /// f32 carries the same code path with Sophus's larger epsilon.
    #[test]
    fn exp_and_log_round_trip_in_f32() {
        for phi in [
            Vector3::<f32>::new(0.1, 0.2, 0.3),
            Vector3::<f32>::new(0.0, 0.0, 0.0),
            Vector3::<f32>::new(1e-7, 0.0, 0.0),
            Vector3::<f32>::new(0.0, -1.5, 0.0),
        ] {
            let back: Vector3<f32> = So3::exp(&phi).log();
            assert_abs_diff_eq!(back, phi, epsilon = 1e-5);
        }
    }

    #[test]
    fn se3_exp_and_log_round_trip_in_f32() {
        let tangent: Vector6<f32> = Vector6::new(0.5, -0.25, 1.5, 0.1, 0.2, 0.3);
        assert_abs_diff_eq!(Se3::exp(&tangent).log(), tangent, epsilon = 1e-5);
        assert_abs_diff_eq!(
            Se3::exp_decoupled(&tangent).log_decoupled(),
            tangent,
            epsilon = 1e-5
        );
    }

    proptest! {
        #![proptest_config(config())]

        #[test]
        fn inverse_so3_jacobians_are_accurate_near_pi_in_f32(
            axis in tangent3(), gap in 0.0f32..0.01,
        ) {
            prop_assume!(axis.norm() > 0.1);
            let phi: Vector3<f32> = axis.normalize().map(|v| v as f32)
                * (std::f32::consts::PI - gap);
            let reference: Vector3<f64> = phi.map(f64::from);
            for (actual, expected) in [
                (right_jacobian_inv_so3(&phi), right_jacobian_inv_so3(&reference)),
                (left_jacobian_inv_so3(&phi), left_jacobian_inv_so3(&reference)),
            ] {
                prop_assert!(actual.iter().all(|v| v.is_finite()));
                prop_assert!((actual.map(f64::from) - expected).norm() < 1e-3);
            }
        }

        #[test]
        fn so3_exp_then_log_is_the_identity(phi in tangent3()) {
            let back: Vector3<f64> = So3::exp(&phi).log();
            prop_assert!((back - phi).norm() < 1e-12);
        }

        #[test]
        fn so3_inverse_undoes_the_rotation(phi in tangent3(), v in tangent3()) {
            let rotation: So3<f64> = So3::exp(&phi);
            let round_trip: Vector3<f64> = rotation.inverse() * (rotation * v);
            prop_assert!((round_trip - v).norm() < 1e-12);
        }

        #[test]
        fn so3_adjoint_is_conjugation(phi in tangent3(), v in tangent3()) {
            let rotation: So3<f64> = So3::exp(&phi);
            let lhs: Matrix3<f64> = rotation.matrix() * So3::hat(&v) * rotation.matrix().transpose();
            let rhs: Matrix3<f64> = So3::hat(&(rotation * v));
            prop_assert!((lhs - rhs).norm() < 1e-12);
        }

        #[test]
        fn se3_exp_then_log_is_the_identity(tangent in tangent6()) {
            let back: Vector6<f64> = Se3::exp(&tangent).log();
            prop_assert!((back - tangent).norm() < 1e-10);
        }

        #[test]
        fn se3_decoupled_exp_then_log_is_the_identity(tangent in tangent6()) {
            let back: Vector6<f64> = Se3::exp_decoupled(&tangent).log_decoupled();
            prop_assert!((back - tangent).norm() < 1e-12);
        }

        #[test]
        fn se3_inverse_undoes_the_transform(tangent in tangent6(), v in tangent3()) {
            let pose: Se3<f64> = Se3::exp(&tangent);
            let round_trip: Vector3<f64> = pose.inverse() * (pose * v);
            prop_assert!((round_trip - v).norm() < 1e-11);
        }

        /// `Adj(T) xi = log(T exp(xi) T^-1)`, the defining property.
        #[test]
        fn se3_adjoint_is_conjugation(pose_tangent in tangent6(), xi in tangent6()) {
            let pose: Se3<f64> = Se3::exp(&pose_tangent);
            let small: Vector6<f64> = xi * 1e-5;
            let conjugated: Vector6<f64> = (pose * Se3::exp(&small) * pose.inverse()).log();
            let predicted: Vector6<f64> = pose.adjoint() * small;
            prop_assert!((conjugated - predicted).norm() < 1e-12);
        }

        /// `J_r(phi) = d/d eps log(exp(phi)^-1 exp(phi + eps))` at `eps = 0`.
        #[test]
        fn right_jacobian_so3_matches_finite_differences(phi in tangent3()) {
            let base_inverse: So3<f64> = So3::exp(&phi).inverse();
            let numeric: Matrix3<f64> = numeric_jacobian3(&phi, |p| (base_inverse * So3::exp(p)).log());
            prop_assert!((numeric - right_jacobian_so3(&phi)).norm() < 1e-7);
        }

        #[test]
        fn the_inverse_so3_jacobians_invert(phi in tangent3()) {
            let identity: Matrix3<f64> = Matrix3::identity();
            prop_assert!((right_jacobian_inv_so3(&phi) * right_jacobian_so3(&phi) - identity).norm() < 1e-11);
            prop_assert!((right_jacobian_so3(&phi) * right_jacobian_inv_so3(&phi) - identity).norm() < 1e-11);
        }

        /// Check `J_l(phi) = J_r(phi)^T` alongside inverse and finite-difference identities.
        #[test]
        fn the_left_jacobian_is_the_right_jacobian_transposed(phi in tangent3()) {
            prop_assert!(
                (left_jacobian_inv_so3(&phi) - right_jacobian_inv_so3(&phi).transpose()).norm() < 1e-14
            );
        }

        /// The same identities in f32, where Sophus's epsilon is 1e-5.
        #[test]
        fn so3_exp_then_log_is_the_identity_in_f32(phi in tangent3()) {
            let phi32: Vector3<f32> = Vector3::new(phi.x as f32, phi.y as f32, phi.z as f32);
            let back: Vector3<f32> = So3::exp(&phi32).log();
            prop_assert!((back - phi32).norm() < 1e-5);
        }

        #[test]
        fn se3_decoupled_exp_then_log_is_the_identity_in_f32(tangent in tangent6()) {
            let mut tangent32: Vector6<f32> = Vector6::zeros();
            for i in 0..6 {
                tangent32[i] = tangent[i] as f32;
            }
            let back: Vector6<f32> = Se3::exp_decoupled(&tangent32).log_decoupled();
            prop_assert!((back - tangent32).norm() < 1e-5);
        }
    }
}
