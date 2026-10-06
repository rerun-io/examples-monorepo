use kornia_algebra::{Vec3AF32, Vec3F64, SO3F32, SO3F64};

mod sealed {
    pub trait Sealed: Sized {
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
    }
}

/// Supported f32 and f64 precisions for camera and Lie geometry.
pub trait Scalar: sealed::Sealed + nalgebra::RealField + Copy {
    /// Sophus small-angle tolerance.
    const SOPHUS_EPSILON: Self;
    /// Smallest positive normal scalar.
    const MIN_POSITIVE_NORMAL: Self;
    /// Largest finite scalar.
    fn largest() -> Self;
    /// Convert a constant or f64 angle into this precision.
    fn from_literal(value: f64) -> Self;
    /// Widen for the KB4 angle computation and cached branch search.
    fn to_f64(self) -> f64;
    /// Square root of the Sophus small-angle tolerance.
    fn sophus_epsilon_sqrt() -> Self;
}
impl Scalar for f64 {
    const SOPHUS_EPSILON: Self = 1e-10;
    const MIN_POSITIVE_NORMAL: Self = Self::MIN_POSITIVE;
    #[inline]
    fn largest() -> Self {
        Self::MAX
    }
    #[inline]
    fn from_literal(value: f64) -> Self {
        value
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn sophus_epsilon_sqrt() -> Self {
        1e-5
    }
}
impl Scalar for f32 {
    const SOPHUS_EPSILON: Self = 1e-5;
    const MIN_POSITIVE_NORMAL: Self = Self::MIN_POSITIVE;
    #[inline]
    fn largest() -> Self {
        Self::MAX
    }
    #[inline]
    fn from_literal(value: f64) -> Self {
        value as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn sophus_epsilon_sqrt() -> Self {
        1e-5_f32.sqrt()
    }
}

impl sealed::Sealed for f64 {
    #[inline]
    fn so3_exp(omega: &[Self; 3]) -> [Self; 4] {
        SO3F64::exp(Vec3F64::from_array(*omega)).to_array()
    }

    #[inline]
    fn so3_log(quaternion_xyzw: &[Self; 4]) -> [Self; 3] {
        SO3F64::from_array(*quaternion_xyzw).log().to_array()
    }

    #[inline]
    fn so3_matrix(quaternion_xyzw: &[Self; 4]) -> [Self; 9] {
        SO3F64::from_array(*quaternion_xyzw)
            .matrix()
            .to_cols_array()
    }

    #[inline]
    fn so3_inverse(quaternion_xyzw: &[Self; 4]) -> [Self; 4] {
        SO3F64::from_array(*quaternion_xyzw).inverse().to_array()
    }

    #[inline]
    fn so3_act(quaternion_xyzw: &[Self; 4], point: &[Self; 3]) -> [Self; 3] {
        (SO3F64::from_array(*quaternion_xyzw) * Vec3F64::from_array(*point)).to_array()
    }
}

impl sealed::Sealed for f32 {
    #[inline]
    fn so3_exp(omega: &[Self; 3]) -> [Self; 4] {
        SO3F32::exp(Vec3AF32::from_array(*omega)).to_array()
    }

    #[inline]
    fn so3_log(quaternion_xyzw: &[Self; 4]) -> [Self; 3] {
        SO3F32::from_array(*quaternion_xyzw).log().to_array()
    }

    #[inline]
    fn so3_matrix(quaternion_xyzw: &[Self; 4]) -> [Self; 9] {
        SO3F32::from_array(*quaternion_xyzw)
            .matrix()
            .to_cols_array()
    }

    #[inline]
    fn so3_inverse(quaternion_xyzw: &[Self; 4]) -> [Self; 4] {
        SO3F32::from_array(*quaternion_xyzw).inverse().to_array()
    }

    #[inline]
    fn so3_act(quaternion_xyzw: &[Self; 4], point: &[Self; 3]) -> [Self; 3] {
        (SO3F32::from_array(*quaternion_xyzw) * Vec3AF32::from_array(*point)).to_array()
    }
}
