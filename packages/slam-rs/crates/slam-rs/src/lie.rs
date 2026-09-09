//! SO(3) and SE(3) with basalt's exact conventions.
//!
//! basalt builds on Sophus, so the port reproduces Sophus's `exp`/`log`
//! (quaternion, atan-based log, `Constants<Scalar>::epsilon()` small-angle
//! branches) and the group Jacobians basalt adds on top in
//! `thirdparty/basalt-headers/include/basalt/utils/sophus_utils.hpp`. No Rust
//! crate carries the two inverse right/left Jacobians or the *decoupled* SE(3)
//! convention, which is why this module exists (decision D06).
//!
//! Two conventions are load-bearing and easy to get wrong:
//!
//! * **Tangent order is translation first.** Sophus's `SE3::Tangent` is
//!   `(upsilon, omega)` (`Sophus/sophus/se3.hpp:850`), and so is basalt's
//!   decoupled pair `se3_expd`/`se3_logd` (`sophus_utils.hpp:62,82`).
//! * **The pose increment is left-multiplied.** `PoseState::incPose` does
//!   `t += inc[0..3]; R = exp(inc[3..6]) * R`
//!   (`thirdparty/basalt-headers/include/basalt/imu/imu_types.h:96-99`), which
//!   is neither the coupled nor the decoupled exponential — see
//!   [`Se3::apply_inc`].
//!
//! Two *different* small-angle branches live here on purpose. [`Se3::exp`] and
//! [`Se3::log`] need Sophus's own `SO3::leftJacobian`/`leftJacobianInverse`
//! (`Sophus/sophus/so3.hpp:570,594`), whose Taylor branch stops one term
//! earlier than basalt's own left Jacobian (`sophus_utils.hpp:312`), which
//! [`left_jacobian_inv_so3`] inverts. The
//! difference is below the branch threshold, but reproducing each at its own
//! call site keeps the port literal.

use nalgebra::{
    Matrix3, Matrix3x4, Matrix4, Matrix6, Quaternion, RealField, UnitQuaternion, Vector3, Vector6,
};

/// A scalar the Lie module can run in: `f32` and `f64`.
///
/// The extra methods carry `Sophus::Constants<Scalar>` (`Sophus/sophus/common.hpp:165-196`),
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

    /// `Eigen::NumTraits<Scalar>::dummy_precision()`: `1e-12` in double and
    /// `1e-5` in float (`Eigen/src/Core/NumTraits.h`).
    ///
    /// Eigen's own fuzzy-comparison tolerance, several orders coarser than the
    /// machine epsilon `RealField::default_epsilon` reports. It decides the
    /// anti-parallel branch of `Quaternion::setFromTwoVectors`, which is how
    /// basalt initialises orientation from the first accelerometer sample.
    fn eigen_dummy_precision() -> Self;

    /// `std::numeric_limits<Scalar>::min()`: the smallest positive normal value.
    ///
    /// Not `RealField::min_value()`, which is the most *negative* finite value
    /// despite its doc comment. basalt compares an LDLT pivot against this
    /// constant to decide whether a direction is rank deficient
    /// (`basalt-headers/include/basalt/imu/preintegration.h:314`), so the two
    /// must mean the same thing.
    fn min_positive() -> Self;

    /// `std::numeric_limits<Scalar>::max()`: the largest finite value.
    ///
    /// basalt seeds both keyframe-eviction scores with it
    /// (`sqrt_keypoint_vio.cpp:788`, `:842`) so the first candidate always
    /// wins the `score < min_score` test.
    fn largest() -> Self;

    /// Eigen's summation order for a **three**-coefficient reduction, such as
    /// `v.head<3>().squaredNorm()`.
    ///
    /// The order is decided by `redux_traits` (`Eigen/src/Core/Redux.h:29-67`)
    /// comparing `find_best_packet<Scalar, 3>` against the three coefficients,
    /// so it is **not** the same in the two precisions:
    ///
    /// * `f64` — `Packet2d` is two doubles, so `LinearVectorizedTraversal`
    ///   reduces one packet and folds the remainder in: `(a + b) + c`.
    /// * `f32` — `Packet4f` is four floats, wider than the expression, so the
    ///   aligned part is empty and the scalar `redux_novec_unroller` runs, which
    ///   splits at `Length / 2` (`Redux.h:98-108`): `a + (b + c)`.
    ///
    /// A sweep of 200,000 random vectors through the fork's own Eigen agrees
    /// with exactly one order in each precision on every discriminating case
    /// (48,359 of 200,000 in `f64`, 48,336 in `f32`) and with the other on none.
    /// The packet widths come from the fork's build flags, where a trailing
    /// `-march=nocona` overrides the earlier `-march=native` — so the reference
    /// binary is SSE3, on every host, and this is not a property of the machine
    /// the port happens to run on. The fixture `lmdb/lmdb_oracle.json` pins both
    /// orders bit for bit.
    ///
    /// It matters: with the wrong order in `f64`, a landmark exactly 1/3 m away
    /// normalises to `2.9999999999999996` instead of `3.0` and basalt's
    /// `inv_dist < 3` gate (`sqrt_keypoint_vio.cpp:534`) accepts a landmark C++
    /// rejects.
    fn eigen_redux3(a: Self, b: Self, c: Self) -> Self;

    /// The width of the widest Eigen packet for this scalar in the reference
    /// build: `unpacket_traits<packet_traits<Scalar>::type>::size`.
    ///
    /// The reference `libbasalt.so` is SSE3 on every host — conda's `CXXFLAGS`
    /// append `-march=nocona` after the CMakeLists' `-march=native`, and the
    /// last `-march` wins — so a `Packet2d` holds two doubles and a `Packet4f`
    /// four floats. `tools/marg_norm_probe.cpp` on the fork prints both sizes
    /// from `packet_traits` itself rather than trusting the flag reading.
    ///
    /// This is the loop bound of `crate::eigen::qr::contiguous_squared_norm`,
    /// which is Eigen's vectorised `redux` for a contiguous column segment.
    const EIGEN_PACKET_SIZE: usize;

    /// `predux` on a full packet of this scalar
    /// (`Eigen/src/Core/arch/SSE/Reductions.h`).
    ///
    /// Not a plain fold: each architecture pairs the lanes the way its shuffle
    /// instructions do, and the pairing is a different rounding.
    ///
    /// * `Packet2d` (`:265-272`) — `a + unpackhi(a, a)`, lane 0: `a₀ + a₁`.
    /// * `Packet4f` (`:205-217`) — `a + movehl(a, a)` puts `(a₀+a₂, a₁+a₃)` in
    ///   the low lanes, then SSE3's `movehdup` adds lane 1 into lane 0:
    ///   `(a₀ + a₂) + (a₁ + a₃)`.
    ///
    /// **Contract: `lanes` is at least [`Self::EIGEN_PACKET_SIZE`] long**, and
    /// an implementation reads exactly that many of them. The pairing is not
    /// free choice: the package README records how a left fold over the lanes
    /// scores against the fork's own Eigen on the shape sweep.
    fn eigen_predux(lanes: &[Self]) -> Self;

    /// Exact-as-possible conversion of a literal, standing in for C++'s `Scalar(x)`.
    fn from_literal(value: f64) -> Self;

    /// Widen to `f64`, for the comparisons C++ promotes to `double`.
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

    /// One `Packet2d` plus the scalar remainder.
    fn eigen_redux3(a: Self, b: Self, c: Self) -> Self {
        (a + b) + c
    }

    const EIGEN_PACKET_SIZE: usize = 2;

    fn eigen_predux(lanes: &[Self]) -> Self {
        // `Packet2d`: two lanes, so the sum is the only order there is.
        lanes[0] + lanes[1]
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

    /// `Packet4f` is wider than three floats, so the scalar unroller runs.
    fn eigen_redux3(a: Self, b: Self, c: Self) -> Self {
        a + (b + c)
    }

    const EIGEN_PACKET_SIZE: usize = 4;

    fn eigen_predux(lanes: &[Self]) -> Self {
        // `Packet4f`: `movehl` pairs lane 0 with lane 2 and lane 1 with lane 3
        // before the two halves meet.
        (lanes[0] + lanes[2]) + (lanes[1] + lanes[3])
    }

    fn from_literal(value: f64) -> Self {
        value as Self
    }

    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

/// `Scalar(x)` in C++: a literal in the caller's scalar type.
///
/// The crate's one spelling of it. `estimator/*`, `marg/*` and
/// `linearize/abs_qr.rs` say `S::from_literal(...)` directly, which is the same
/// call; everything with enough literals for the noise to matter imports this.
#[inline]
pub(crate) fn c<S: LieScalar>(value: f64) -> S {
    S::from_literal(value)
}

/// `numext::maxi(a, b)` (`Core/MathFunctions.h`), which is what `cwiseMax`
/// applies coefficient by coefficient.
///
/// `(a < b ? b : a)`, so a NaN on the left survives and `f32::max`'s
/// NaN-suppressing behaviour is wrong here. `sqrt_keypoint_vio.cpp:1415` sends
/// the result straight into the damped diagonal, so a NaN that Eigen keeps and
/// Rust would drop changes whether the solve retries.
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

    /// Normalize `(x, y, z, w)` into a rotation.
    ///
    /// basalt's cereal reader writes the four coefficients straight into the
    /// quaternion without normalizing (`serialization/eigen_io.h:148-153`); the
    /// port normalizes, so a calibration file that is a few ulps off unit length
    /// still yields a valid rotation. Returns `None` for a zero-norm input,
    /// which would otherwise produce NaNs.
    pub fn from_quaternion_xyzw(x: S, y: S, z: S, w: S) -> Option<Self> {
        let quaternion: Quaternion<S> = Quaternion::new(w, x, y, z);
        if !quaternion.norm().is_finite() || quaternion.norm() <= S::zero() {
            return None;
        }
        Some(Self {
            quaternion: UnitQuaternion::new_normalize(quaternion),
        })
    }

    /// `Sophus::SO3::cast` (`Sophus/sophus/so3.hpp`): the same rotation in
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

    /// The four coefficients in basalt's on-disk order, `(qx, qy, qz, qw)`.
    ///
    /// Eigen stores a quaternion `xyzw`, and cereal writes `so3().data()[0..3]`
    /// followed by `[3]` (`serialization/eigen_io.h:150-153`).
    pub fn quaternion_xyzw(&self) -> [S; 4] {
        let q = self.quaternion.as_ref();
        [q.i, q.j, q.k, q.w]
    }

    /// The 3x3 skew-symmetric matrix of a vector, `Sophus::SO3::hat`.
    pub fn hat(v: &Vector3<S>) -> Matrix3<S> {
        let zero: S = S::zero();
        Matrix3::new(zero, -v.z, v.y, v.z, zero, -v.x, -v.y, v.x, zero)
    }

    /// The exponential map, ported from `Sophus/sophus/so3.hpp:716-751`.
    pub fn exp(omega: &Vector3<S>) -> Self {
        Self::exp_and_theta(omega).0
    }

    /// [`So3::exp`] plus `theta = |omega|`, which is zero on the Taylor branch.
    ///
    /// `Sophus::SO3::expAndTheta` (`Sophus/sophus/so3.hpp:716`) reports the same
    /// `theta` it branched on, and [`Se3::exp`] feeds it straight into the left
    /// Jacobian, so the two must be computed together.
    pub fn exp_and_theta(omega: &Vector3<S>) -> (Self, S) {
        let theta_sq: S = omega.norm_squared();
        let epsilon: S = S::SOPHUS_EPSILON;

        let (theta, imag_factor, real_factor): (S, S, S) = if theta_sq < epsilon * epsilon {
            let theta_po4: S = theta_sq * theta_sq;
            (
                S::zero(),
                c::<S>(0.5) - c::<S>(1.0 / 48.0) * theta_sq + c::<S>(1.0 / 3840.0) * theta_po4,
                c::<S>(1.0) - c::<S>(1.0 / 8.0) * theta_sq + c::<S>(1.0 / 384.0) * theta_po4,
            )
        } else {
            let theta: S = theta_sq.sqrt();
            let half_theta: S = c::<S>(0.5) * theta;
            (theta, half_theta.sin() / theta, half_theta.cos())
        };

        let quaternion: Quaternion<S> = Quaternion::new(
            real_factor,
            imag_factor * omega.x,
            imag_factor * omega.y,
            imag_factor * omega.z,
        );
        (
            Self {
                // The factors above already make this unit length; Sophus asserts
                // it rather than normalizing (`so3.hpp:747-750`).
                quaternion: UnitQuaternion::new_unchecked(quaternion),
            },
            theta,
        )
    }

    /// The logarithm, ported from `Sophus/sophus/so3.hpp:281-334`.
    pub fn log(&self) -> Vector3<S> {
        self.log_and_theta().0
    }

    /// [`So3::log`] plus Sophus's `theta`, which is **not** `|log|` on the
    /// Taylor branch.
    ///
    /// `Sophus::SO3::logAndTheta` returns `2 n^2 / w` for a near-identity
    /// rotation (`Sophus/sophus/so3.hpp:311`), i.e. second order in the vector
    /// part, while the tangent itself is first order. [`Se3::log`] passes that
    /// `theta` to the inverse left Jacobian, and substituting `|log|` there
    /// moves it onto the ill-conditioned branch, so the pair travels together.
    pub fn log_and_theta(&self) -> (Vector3<S>, S) {
        let q = self.quaternion.as_ref();
        let squared_n: S = q.vector().norm_squared();
        let w: S = q.w;
        let epsilon: S = S::SOPHUS_EPSILON;

        let (two_atan_nbyw_by_n, theta): (S, S) = if squared_n < epsilon * epsilon {
            // A unit quaternion with a vanishing vector part has |w| ~ 1, so the
            // division below is safe; Sophus asserts the same thing (`so3.hpp:306`).
            let squared_w: S = w * w;
            (
                c::<S>(2.0) / w - c::<S>(2.0 / 3.0) * squared_n / (w * squared_w),
                c::<S>(2.0) * squared_n / w,
            )
        } else {
            let n: S = squared_n.sqrt();
            // w < 0 means theta > pi; folding the wrap into atan2 keeps |theta| <= pi.
            let atan_nbyw: S = if w < S::zero() {
                (-n).atan2(-w)
            } else {
                n.atan2(w)
            };
            let two_atan_nbyw_by_n: S = c::<S>(2.0) * atan_nbyw / n;
            (two_atan_nbyw_by_n, two_atan_nbyw_by_n * n)
        };

        (q.vector() * two_atan_nbyw_by_n, theta)
    }

    /// The inverse rotation.
    ///
    /// Sophus builds it as `SO3(unit_quaternion().conjugate())`
    /// (`Sophus/sophus/so3.hpp:267-269`), and that constructor renormalizes
    /// (`:548-553`), so this one does too. Conjugating only flips signs, so the
    /// renormalization is a no-op here — it is kept for the same reason Sophus
    /// keeps it: every path that produces an `So3` leaves it unit length.
    pub fn inverse(&self) -> Self {
        Self {
            quaternion: normalized(self.quaternion.conjugate().into_inner()),
        }
    }

    /// The rotation as a 3x3 matrix.
    ///
    /// `Sophus::SO3::matrix()` is `unit_quaternion().toRotationMatrix()`
    /// (`Sophus/sophus/so3.hpp:257`), so this is Eigen's
    /// `QuaternionBase::toRotationMatrix`
    /// (`thirdparty/basalt-headers/thirdparty/eigen/Eigen/src/Geometry/Quaternion.h:646-678`)
    /// written out in its own operation order rather than nalgebra's
    /// `to_rotation_matrix`.
    ///
    /// The two agree mathematically and disagree in the last bits: Eigen builds
    /// each diagonal entry as `1 - (tyy + tzz)` from the doubled coefficients,
    /// nalgebra as `ww + ii - jj - kk`, and the off-diagonal triple products
    /// associate differently (`(2y)·x` against `(x·y)·2`). That is enough to
    /// move a cancellation residue across zero: on one 5 ms `f32` IMU sample the
    /// preintegrated covariance is rank deficient in a different *direction*
    /// under the two roundings, and the whitening then puts its `1.15e18` weight
    /// on a different axis (`crates/slam-rs/tests/imu_oracle.rs`,
    /// `rotating_singular_f32`).
    pub fn matrix(&self) -> Matrix3<S> {
        let q = self.quaternion.as_ref();
        let two: S = c::<S>(2.0);
        let tx: S = two * q.i;
        let ty: S = two * q.j;
        let tz: S = two * q.k;
        let twx: S = tx * q.w;
        let twy: S = ty * q.w;
        let twz: S = tz * q.w;
        let txx: S = tx * q.i;
        let txy: S = ty * q.i;
        let txz: S = tz * q.i;
        let tyy: S = ty * q.j;
        let tyz: S = tz * q.j;
        let tzz: S = tz * q.k;
        let one: S = S::one();
        Matrix3::new(
            one - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            one - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            one - (txx + tyy),
        )
    }
}

impl<S: LieScalar> std::ops::Mul for So3<S> {
    type Output = Self;

    /// Compose two rotations, renormalizing the product.
    ///
    /// Sophus's `operator*` hands the raw quaternion product to the `SO3`
    /// quaternion constructor (`Sophus/sophus/so3.hpp:378-389`), which calls
    /// `normalize()` (`:548-553`, `:339-345`). `nalgebra`'s
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

/// `Sophus::SO3::normalize()` (`Sophus/sophus/so3.hpp:339-345`).
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

    /// Rotate a point, `Sophus::SO3::operator*(Point)`
    /// (`Sophus/sophus/so3.hpp:408-417`).
    ///
    /// Sophus writes this out rather than calling Eigen's `_transformVector`,
    /// and the port writes out Sophus's:
    ///
    /// ```text
    /// uv = 2 (q.vec x p);  result = p + q.w uv + q.vec x uv
    /// ```
    ///
    /// nalgebra's `UnitQuaternion * Vector3` forms the same three terms but adds
    /// them as `(uv w + cross) + p`, one association away from Sophus's
    /// `(p + uv w) + cross`. That is a last-ulp difference in every rotated
    /// point, and it reaches a threshold: `computeRelPose` rotates the baseline
    /// (`ba_utils.h:50`) into the relative pose the DLT triangulates from, and
    /// basalt accepts a landmark only when the result satisfies
    /// `0 < inv_dist < 3` (`sqrt_keypoint_vio.cpp:534`). Decision D44's rule
    /// applies, so the order is Sophus's.
    fn mul(self, rhs: Vector3<S>) -> Vector3<S> {
        let q: &Quaternion<S> = self.quaternion.as_ref();
        let vec: Vector3<S> = q.vector().into_owned();
        let mut uv: Vector3<S> = vec.cross(&rhs);
        uv += uv;
        rhs + uv * q.w + vec.cross(&uv)
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

    /// The coupled exponential map, ported from `Sophus/sophus/se3.hpp:850-859`.
    ///
    /// The tangent is `(upsilon, omega)` and the translation is `V(omega)
    /// upsilon`, so translation and rotation are **not** independent. basalt's
    /// state increments use [`Se3::exp_decoupled`] and [`Se3::apply_inc`]
    /// instead.
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

    /// The coupled logarithm, ported from `Sophus/sophus/se3.hpp:237-252`.
    pub fn log(&self) -> Vector6<S> {
        let (omega, theta) = self.rotation.log_and_theta();
        let v_inv: Matrix3<S> = sophus_left_jacobian_inv_so3(&omega, theta);
        let upsilon: Vector3<S> = v_inv * self.translation;
        Vector6::new(upsilon.x, upsilon.y, upsilon.z, omega.x, omega.y, omega.z)
    }

    /// basalt's decoupled exponential, `Sophus::se3_expd`
    /// (`basalt-headers/include/basalt/utils/sophus_utils.hpp:82-89`).
    ///
    /// The head of the tangent becomes the translation directly; only the tail
    /// goes through `SO3::exp`.
    pub fn exp_decoupled(tangent: &Vector6<S>) -> Self {
        Self {
            rotation: So3::exp(&tangent.fixed_rows::<3>(3).into_owned()),
            translation: tangent.fixed_rows::<3>(0).into_owned(),
        }
    }

    /// basalt's decoupled logarithm, `Sophus::se3_logd`
    /// (`basalt-headers/include/basalt/utils/sophus_utils.hpp:62-68`).
    ///
    /// No production caller: the estimator moves states with
    /// [`Self::exp_decoupled`] and [`Self::apply_inc`] and never reads a
    /// tangent back. It is kept as the ported pair's other half, and it is what
    /// pins `exp_decoupled` — the round-trip proptests here and the
    /// finite-difference Jacobians in `ba_base`'s tests take their tangents
    /// through it.
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
    /// (`Sophus/sophus/se3.hpp:275-280`).
    ///
    /// The rotation block goes through [`So3::matrix`], i.e. Eigen's
    /// `toRotationMatrix` operation order (decision D44), because this matrix
    /// is what the reprojection residual multiplies its landmark by
    /// (`ba_utils.h:94`).
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
    /// (`Sophus/sophus/se3.hpp:285-290`): `[R | t]`.
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

    /// The adjoint, ported from `Sophus/sophus/se3.hpp:105-113`.
    ///
    /// With the `(upsilon, omega)` tangent order the blocks are
    /// `[[R, hat(t) R], [0, R]]`, and `Adj(T) xi = log(T exp(xi) T^-1)`.
    pub fn adjoint(&self) -> Matrix6<S> {
        let r: Matrix3<S> = self.rotation.matrix();
        let mut res: Matrix6<S> = Matrix6::zeros();
        res.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        res.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
        res.fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(So3::hat(&self.translation) * r));
        res
    }

    /// basalt's pose increment, `PoseState::incPose`
    /// (`basalt-headers/include/basalt/imu/imu_types.h:96-99`).
    ///
    /// `t += inc[0..3]` and `R = exp(inc[3..6]) R`: translation first, and the
    /// rotation increment multiplies from the **left**. This is the single most
    /// copied convention in the estimator — every state block, every Jacobian
    /// column and the marginalization prior all assume it.
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

/// Right Jacobian of SO(3): `exp(phi + eps) ~ exp(phi) exp(J eps)`.
///
/// Ported from `basalt-headers/include/basalt/utils/sophus_utils.hpp:145-168`.
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

/// Inverse right Jacobian of SO(3): `log(exp(phi) exp(eps)) ~ phi + J eps`.
///
/// Ported from `basalt-headers/include/basalt/utils/sophus_utils.hpp:180-216`.
/// basalt asserts `|phi| <= pi` there; the port drops the assert (decision D32,
/// the core never panics on data) and lets an over-rotated input take the same
/// zeroth-order branch the C++ takes at exactly pi.
pub fn right_jacobian_inv_so3<S: LieScalar>(phi: &Vector3<S>) -> Matrix3<S> {
    let phi_norm2: S = phi.norm_squared();
    let phi_hat: Matrix3<S> = So3::hat(phi);
    let phi_hat2: Matrix3<S> = phi_hat * phi_hat;

    let mut j: Matrix3<S> = Matrix3::identity() + phi_hat / c::<S>(2.0);
    j += inverse_jacobian_second_order_term(&phi_hat2, phi_norm2);
    j
}

/// Inverse left Jacobian of SO(3): `log(exp(eps) exp(phi)) ~ phi + J eps`.
///
/// Ported from `basalt-headers/include/basalt/utils/sophus_utils.hpp:348-384`,
/// with the same dropped assert as [`right_jacobian_inv_so3`].
pub fn left_jacobian_inv_so3<S: LieScalar>(phi: &Vector3<S>) -> Matrix3<S> {
    let phi_norm2: S = phi.norm_squared();
    let phi_hat: Matrix3<S> = So3::hat(phi);
    let phi_hat2: Matrix3<S> = phi_hat * phi_hat;

    let mut j: Matrix3<S> = Matrix3::identity() - phi_hat / c::<S>(2.0);
    j += inverse_jacobian_second_order_term(&phi_hat2, phi_norm2);
    j
}

/// The `hat(phi)^2` term both inverse SO(3) Jacobians add.
///
/// The three branches are basalt's (`sophus_utils.hpp:191-215`): the closed
/// form on `(0, pi)`, a zeroth-order expansion at pi where `sin` vanishes, and
/// the Taylor value `1/12` at zero.
///
/// Two details are about arithmetic width rather than mathematics, and both
/// change which branch an `f32` input lands in:
///
/// * **The pi comparison happens in `f64`.** C++ writes
///   `phi_norm < M_PI - Sophus::Constants<Scalar>::epsilonSqrt()`
///   (`sophus_utils.hpp:202`). `M_PI` is a `double`, so the whole comparison is
///   promoted to `double` even when `Scalar` is `float`. Doing it in `f32`
///   moves the threshold by about 2e-8, and the single `f32` value
///   `3.13843035697937` — which is exactly `pi_f32 - epsilonSqrt_f32` — then
///   takes the pi branch where basalt takes the closed form, changing the
///   result from 0.0024845 to 0.0020120.
/// * **`M_PI * M_PI` is a `double` product** that Eigen rounds to `Scalar`
///   only when it divides (`sophus_utils.hpp:214`). Squaring `pi_f32` instead
///   gives a different last bit.
///
/// The matrix is divided rather than multiplied by a reciprocal in the two
/// constant branches, because that is what the C++ does and the two differ by
/// an ulp in `f32`.
fn inverse_jacobian_second_order_term<S: LieScalar>(
    phi_hat2: &Matrix3<S>,
    phi_norm2: S,
) -> Matrix3<S> {
    if phi_norm2 <= S::SOPHUS_EPSILON {
        // Taylor expansion around 0.
        return phi_hat2 / c::<S>(12.0);
    }
    let phi_norm: S = phi_norm2.sqrt();
    let threshold: f64 = std::f64::consts::PI - S::sophus_epsilon_sqrt().to_f64();
    if phi_norm.to_f64() < threshold {
        // Regular case on (0, pi).
        phi_hat2
            * (c::<S>(1.0) / phi_norm2
                - (c::<S>(1.0) + phi_norm.cos()) / (c::<S>(2.0) * phi_norm * phi_norm.sin()))
    } else {
        // 0th-order Taylor expansion around pi.
        phi_hat2 / c::<S>(std::f64::consts::PI * std::f64::consts::PI)
    }
}

/// Sophus's own left Jacobian, `SO3::leftJacobian`
/// (`Sophus/sophus/so3.hpp:570-591`).
///
/// Used only by [`Se3::exp`], because that is what `SE3::exp` calls. It differs
/// from basalt's own left Jacobian (`sophus_utils.hpp:312-337`) in the Taylor
/// branch, which stops at `I + Omega/2` instead of adding `Omega^2/6`.
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
/// (`Sophus/sophus/so3.hpp:594-620`). Used only by [`Se3::log`].
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

    /// The packet widths are the fork's SSE3 ones, whatever this machine is.
    #[test]
    fn the_packet_widths_are_the_forks() {
        assert_eq!(f32::EIGEN_PACKET_SIZE, 4);
        assert_eq!(f64::EIGEN_PACKET_SIZE, 2);
    }

    /// The two `predux` trees, spelled out on values whose sum is
    /// order-dependent in `f32`: `1 + 2^-24` rounds away against `1` but
    /// survives against `2^-24`.
    #[test]
    fn eigen_predux_pairs_the_lanes_across_the_halves() {
        let tiny: f32 = f32::EPSILON / 2.0;
        // (1 + 1) + (tiny + tiny) keeps both tiny terms; a left fold
        // ((1 + 1) + tiny) + tiny loses them.
        assert_eq!(
            f32::eigen_predux(&[1.0, tiny, 1.0, tiny]),
            2.0 + (tiny + tiny)
        );
        assert_eq!(
            [1.0f32, tiny, 1.0f32, tiny]
                .iter()
                .copied()
                .fold(0.0f32, |a, b| a + b),
            2.0
        );
        assert_eq!(f64::eigen_predux(&[1.0, 2.0]), 3.0);
    }

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
    /// (`Sophus/sophus/so3.hpp:735-741`) with `theta = |omega|`.
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

    /// A long chain of compositions must not leak unit length.
    ///
    /// `nalgebra`'s `UnitQuaternion * UnitQuaternion` skips the renormalization
    /// Sophus does on every product (`Sophus/sophus/so3.hpp:378-389`), and
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

    /// The pi branch is selected in `f64`, as C++ does through `M_PI`.
    ///
    /// `3.13843035697937_f32` is exactly `pi_f32 - epsilonSqrt_f32`, so an
    /// `f32` comparison sends it to the pi branch while basalt's promoted
    /// `double` comparison keeps it on the closed form. The two expected values
    /// are the ones the C++ produces at this input and at the next
    /// representable `f32` above it (`sophus_utils.hpp:202-214`).
    #[test]
    fn the_inverse_jacobian_pi_boundary_is_compared_in_f64() {
        // 3.1384304_f32 is 3.138430356979370... exactly, which is exactly
        // pi_f32 - epsilonSqrt_f32; the next f32 up is 3.138430595397949...
        let below: Vector3<f32> = Vector3::new(3.138_430_4, 0.0, 0.0);
        let above: Vector3<f32> = Vector3::new(f32::from_bits(below.x.to_bits() + 1), 0.0, 0.0);
        assert_eq!(f64::from(below.x), 3.138_430_356_979_37);
        assert_eq!(f64::from(above.x), 3.138_430_595_397_949);

        // Closed form on (0, pi), the value the C++ produces here.
        let expected_closed_form: f64 = 0.002_484_500_408_172_607_4;
        assert_abs_diff_eq!(
            f64::from(right_jacobian_inv_so3(&below)[(1, 1)]),
            expected_closed_form,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            f64::from(left_jacobian_inv_so3(&below)[(1, 1)]),
            expected_closed_form,
            epsilon = 1e-12
        );
        // 0th-order expansion around pi, one ulp of input later.
        let expected_near_pi: f64 = 0.002_011_954_784_393_310_5;
        assert_abs_diff_eq!(
            f64::from(right_jacobian_inv_so3(&above)[(1, 1)]),
            expected_near_pi,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            f64::from(left_jacobian_inv_so3(&above)[(1, 1)]),
            expected_near_pi,
            epsilon = 1e-12
        );
        // The two branches really are far apart here, so this is a branch test,
        // not a rounding test.
        assert!((expected_closed_form - expected_near_pi).abs() > 4e-4);
    }

    /// Near pi the closed form's `sin` denominator vanishes; basalt swaps in a
    /// zeroth-order expansion (`sophus_utils.hpp:212-214`) and so do we.
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
    /// untouched, which the coupled one does not (`sophus_utils.hpp:82-89`).
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
    /// (`basalt-headers/include/basalt/imu/imu_types.h:96-99`).
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

        /// basalt's `J_l(phi) = J_r(phi)^T` (`sophus_utils.hpp:145` vs `:312`),
        /// which is what pins [`left_jacobian_inv_so3`]: the right-hand pair is
        /// checked against finite differences and against each other above.
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
