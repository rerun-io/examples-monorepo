//! basalt's camera models: projection, unprojection and their analytic Jacobians.
//!
//! Ported line by line from
//! `thirdparty/basalt-headers/include/basalt/camera/{pinhole_camera,kannala_brandt_camera4,pinhole_radtan8_camera}.hpp`.
//! Every formula and every domain check below names the C++ line it comes from,
//! because the accuracy gate cannot tell a transcription slip from an estimator
//! bug (decision D14).
//!
//! ## The signature
//!
//! basalt projects a **4-D homogeneous** point and writes the result through an
//! out parameter, returning a validity flag:
//!
//! ```cpp
//! // pinhole_radtan8_camera.hpp:301-302
//! bool project(const Eigen::MatrixBase<DerivedPoint3D>& p3d, Eigen::MatrixBase<DerivedPoint2D>& proj,
//!              DerivedJ3D d_proj_d_p3d = nullptr, DerivedJparam d_proj_d_param = nullptr) const;
//! ```
//!
//! The port keeps that shape: [`Camera::project`] fills a `Vector2` and returns
//! `bool`, and [`Camera::project_with_jacobians`] takes the two Jacobians as
//! `Option<&mut …>`. C++ selects the Jacobian blocks with `if constexpr` on a
//! template parameter; the Rust equivalent is a monomorphized `Option` that the
//! optimizer folds away, so neither form allocates and neither is a `Box<dyn>`
//! (decision D11).
//!
//! Layouts follow basalt exactly: `d_proj_d_p3d` is 2x4 with a zero fourth
//! column (the homogeneous coordinate never enters the projection),
//! `d_proj_d_param` is 2xN with N = 4 for pinhole, 8 for kb4 and 12 for
//! pinhole-radtan8, `d_p3d_d_proj` is 4x2 and `d_p3d_d_param` is 4xN, both with
//! a zero fourth row. [`Camera::unproject`] returns a unit-norm bearing whose
//! fourth component is zero, which is what `p3d.setZero()` followed by three
//! assignments produces in C++ (`kannala_brandt_camera4.hpp:366-369`).
//!
//! ## What is here and what is not
//!
//! V0 ships `pinhole`, `kb4` and `pinhole-radtan8`: the models the reference
//! datasets use (msd-index and RoboCap are kb4, msd-g2 is pinhole-radtan8) plus
//! the pinhole the VIT binding maps `DISTORTION_NONE` onto (decision D11).
//! `ds`, `eucm` and `ucm` parse in [`crate::calib`] and are rejected here with
//! [`CameraError::UnsupportedModel`].
//!
//! Two pieces of the C++ are deliberately absent:
//!
//! * **`PinholeRadtan8Camera::computeRpmax()`** (`:131-242`), the gradient-ascent
//!   estimate of the valid radius. Every calibration this port reads carries
//!   `rpmax` on disk, and the ABS_QR VIO path does not optimize intrinsics, so
//!   the estimate has no caller. [`PinholeRadtan8::apply_inc`] therefore leaves
//!   `rpmax` alone where C++ recomputes it (`:688-691`) — which is also what
//!   makes a finite-difference check of `d_proj_d_param` meaningful, since the
//!   analytic Jacobian treats `rpmax` as a constant.
//! * **`d_p3d_d_proj` for pinhole-radtan8**, which C++ does not have either: it
//!   asserts (`:657-659`). The port returns
//!   [`CameraError::UnprojectJacobianUnsupported`] rather than asserting.
//!
//! ## Scalars
//!
//! Generic over [`LieScalar`] (`f32` and `f64`) because the frontend runs `f32`
//! and the estimator is instantiated at both (decision D05). Two rules, and they
//! point in opposite directions:
//!
//! * **No domain check is promoted.** Every one compares a `Scalar` against
//!   `Sophus::Constants<Scalar>::epsilonSqrt()`, which is `sqrt(1e-10)` in double
//!   and `sqrt(1e-5)` in float, so the branch is genuinely precision-dependent
//!   and the port reproduces it as it stands.
//! * **`atan2` is promoted**, in kb4's projection only. The header imports `cos`,
//!   `sin` and `sqrt` from `std` (`kannala_brandt_camera4.hpp:48-50`) but not
//!   `atan2`, so the unqualified call at `:153` takes the `double` overload from
//!   `<math.h>` and the float instantiation computes its angle in double. The
//!   port does the same; see the comment at the call site for the measurement.
//!
//! Both are decision D37's rule — do in `f64` exactly what the C++ does in
//! `double`, and no more — read off the C++ rather than assumed.

use nalgebra::{Matrix2, Matrix2x4, Matrix4x2, SMatrix, SVector, Vector2, Vector4};

use crate::calib::{Calibration, CameraModel};
use crate::lie::{LieScalar, c};

/// Something a camera model cannot do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CameraError {
    /// A model that [`crate::calib`] parses but this module does not project with.
    #[error("camera model {model} is parsed but its projection is not implemented")]
    UnsupportedModel {
        /// basalt's name for the model, as written in `camera_type`.
        model: &'static str,
    },
    /// basalt has no analytic unprojection Jacobian for this model.
    #[error("camera model {model} has no analytic unprojection Jacobian")]
    UnprojectJacobianUnsupported {
        /// basalt's name for the model, as written in `camera_type`.
        model: &'static str,
    },
    /// The calibration does not carry one resolution per camera.
    #[error("calibration has {intrinsics} intrinsics but {resolutions} resolutions")]
    RaggedCalibration {
        /// Number of camera models.
        intrinsics: usize,
        /// Number of resolutions.
        resolutions: usize,
    },
}

/// One camera model: basalt's `project`/`unproject` pair.
///
/// The associated `Params` vector is basalt's `param_` in `getParam()` order,
/// which is also the order `operator+=` increments and the column order of
/// `d_proj_d_param`.
pub trait Camera<S: LieScalar>: Copy {
    /// basalt's `N`, the number of intrinsic parameters.
    const NUM_PARAMS: usize;

    /// `VecN`: the intrinsic parameters in `getParam()` order.
    type Params: Copy;

    /// `Mat2N`: the projection Jacobian with respect to the intrinsics.
    type ParamJacobian: Copy;

    /// basalt's `getName()`, the string that appears in `camera_type`.
    fn name(&self) -> &'static str;

    /// `getParam()`.
    fn params(&self) -> Self::Params;

    /// `operator+=`: increment the intrinsics, as the calibration optimizer does.
    fn apply_inc(&mut self, inc: &Self::Params);

    /// Project a homogeneous point, filling `proj`; `false` outside the valid domain.
    ///
    /// `proj` is written even when the point is invalid, exactly as the C++
    /// does: the caller checks the flag, not the pixel.
    #[inline]
    fn project(&self, p3d: &Vector4<S>, proj: &mut Vector2<S>) -> bool {
        self.project_with_jacobians(p3d, proj, None, None)
    }

    /// Project a homogeneous point and fill whichever Jacobians are asked for.
    fn project_with_jacobians(
        &self,
        p3d: &Vector4<S>,
        proj: &mut Vector2<S>,
        d_proj_d_p3d: Option<&mut Matrix2x4<S>>,
        d_proj_d_param: Option<&mut Self::ParamJacobian>,
    ) -> bool;

    /// Unproject a pixel into a unit-norm bearing with a zero fourth component.
    fn unproject(&self, proj: &Vector2<S>, p3d: &mut Vector4<S>) -> bool;
}

/// The models whose unprojection basalt differentiates: `pinhole` and `kb4`.
///
/// `pinhole-radtan8` is not one of them (`pinhole_radtan8_camera.hpp:657-659`
/// asserts), and the C++ test file has the matching tests commented out
/// (`test/src/test_camera.cpp:374-379`).
pub trait UnprojectJacobians<S: LieScalar>: Camera<S> {
    /// `Mat4N`: the unprojection Jacobian with respect to the intrinsics.
    type UnprojectParamJacobian: Copy;

    /// Unproject and fill whichever Jacobians are asked for.
    fn unproject_with_jacobians(
        &self,
        proj: &Vector2<S>,
        p3d: &mut Vector4<S>,
        d_p3d_d_proj: Option<&mut Matrix4x2<S>>,
        d_p3d_d_param: Option<&mut Self::UnprojectParamJacobian>,
    ) -> bool;
}

// ─── pinhole ──────────────────────────────────────────────────────────────

/// `basalt::PinholeCamera` (`pinhole_camera.hpp:55`), N = 4.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pinhole<S: LieScalar> {
    param: SVector<S, 4>,
}

impl<S: LieScalar> Pinhole<S> {
    /// Build from `[fx, fy, cx, cy]` (`pinhole_camera.hpp:77`).
    pub fn new(param: SVector<S, 4>) -> Self {
        Self { param }
    }
}

impl<S: LieScalar> Camera<S> for Pinhole<S> {
    const NUM_PARAMS: usize = 4;
    type Params = SVector<S, 4>;
    type ParamJacobian = SMatrix<S, 2, 4>;

    fn name(&self) -> &'static str {
        "pinhole"
    }

    fn params(&self) -> Self::Params {
        self.param
    }

    fn apply_inc(&mut self, inc: &Self::Params) {
        self.param += inc;
    }

    /// `pinhole_camera.hpp:119-166`.
    fn project_with_jacobians(
        &self,
        p3d: &Vector4<S>,
        proj: &mut Vector2<S>,
        d_proj_d_p3d: Option<&mut Matrix2x4<S>>,
        d_proj_d_param: Option<&mut Self::ParamJacobian>,
    ) -> bool {
        let fx: S = self.param[0];
        let fy: S = self.param[1];
        let cx: S = self.param[2];
        let cy: S = self.param[3];

        let x: S = p3d[0];
        let y: S = p3d[1];
        let z: S = p3d[2];

        // :134-135
        proj[0] = fx * x / z + cx;
        proj[1] = fy * y / z + cy;

        // :137
        let is_valid: bool = z >= S::sophus_epsilon_sqrt();

        if let Some(jacobian) = d_proj_d_p3d {
            // :142-149
            jacobian.fill(S::zero());
            let z2: S = z * z;
            jacobian[(0, 0)] = fx / z;
            jacobian[(0, 2)] = -fx * x / z2;
            jacobian[(1, 1)] = fy / z;
            jacobian[(1, 2)] = -fy * y / z2;
        }

        if let Some(jacobian) = d_proj_d_param {
            // :156-160
            jacobian.fill(S::zero());
            jacobian[(0, 0)] = x / z;
            jacobian[(0, 2)] = S::one();
            jacobian[(1, 1)] = y / z;
            jacobian[(1, 3)] = S::one();
        }

        is_valid
    }

    /// `pinhole_camera.hpp:190-254`.
    fn unproject(&self, proj: &Vector2<S>, p3d: &mut Vector4<S>) -> bool {
        self.unproject_with_jacobians(proj, p3d, None, None)
    }
}

impl<S: LieScalar> UnprojectJacobians<S> for Pinhole<S> {
    type UnprojectParamJacobian = SMatrix<S, 4, 4>;

    /// `pinhole_camera.hpp:190-254`.
    fn unproject_with_jacobians(
        &self,
        proj: &Vector2<S>,
        p3d: &mut Vector4<S>,
        d_p3d_d_proj: Option<&mut Matrix4x2<S>>,
        d_p3d_d_param: Option<&mut Self::UnprojectParamJacobian>,
    ) -> bool {
        let fx: S = self.param[0];
        let fy: S = self.param[1];
        let cx: S = self.param[2];
        let cy: S = self.param[3];

        // :201-212
        let mx: S = (proj[0] - cx) / fx;
        let my: S = (proj[1] - cy) / fy;
        let r2: S = mx * mx + my * my;
        let norm: S = (S::one() + r2).sqrt();
        let norm_inv: S = S::one() / norm;

        p3d.fill(S::zero());
        p3d[0] = mx * norm_inv;
        p3d[1] = my * norm_inv;
        p3d[2] = norm_inv;

        if d_p3d_d_proj.is_some() || d_p3d_d_param.is_some() {
            // :215-228
            let d_norm_inv_d_r2: S = -c::<S>(0.5) * norm_inv * norm_inv * norm_inv;
            let two: S = c(2.0);

            let mut c0: Vector4<S> = Vector4::zeros();
            c0[0] = (norm_inv + two * mx * mx * d_norm_inv_d_r2) / fx;
            c0[1] = (two * my * mx * d_norm_inv_d_r2) / fx;
            c0[2] = two * mx * d_norm_inv_d_r2 / fx;

            let mut c1: Vector4<S> = Vector4::zeros();
            c1[0] = (two * my * mx * d_norm_inv_d_r2) / fy;
            c1[1] = (norm_inv + two * my * my * d_norm_inv_d_r2) / fy;
            c1[2] = two * my * d_norm_inv_d_r2 / fy;

            if let Some(jacobian) = d_p3d_d_proj {
                // :232-233
                jacobian.set_column(0, &c0);
                jacobian.set_column(1, &c1);
            }

            if let Some(jacobian) = d_p3d_d_param {
                // :240-244
                jacobian.set_column(2, &(-c0));
                jacobian.set_column(3, &(-c1));
                jacobian.set_column(0, &(-c0 * mx));
                jacobian.set_column(1, &(-c1 * my));
            }
        }

        // :253
        true
    }
}

// ─── kb4 ──────────────────────────────────────────────────────────────────

/// `basalt::KannalaBrandtCamera4` (`kannala_brandt_camera4.hpp:63`), N = 8.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KannalaBrandt4<S: LieScalar> {
    param: SVector<S, 8>,
}

impl<S: LieScalar> KannalaBrandt4<S> {
    /// Build from `[fx, fy, cx, cy, k1, k2, k3, k4]` (`kannala_brandt_camera4.hpp:86`).
    pub fn new(param: SVector<S, 8>) -> Self {
        Self { param }
    }

    /// `solveTheta<ITER>` (`kannala_brandt_camera4.hpp:274-308`).
    ///
    /// Three Newton steps on `d(theta) - r_theta = 0`, with no convergence test
    /// and no guard on `d_func_d_theta`: the count is fixed at the call site
    /// (`:359`, `solveTheta<3>`) and that is the whole stopping rule.
    fn solve_theta(&self, r_theta: S, d_func_d_theta: &mut S) -> S {
        let k1: S = self.param[4];
        let k2: S = self.param[5];
        let k3: S = self.param[6];
        let k4: S = self.param[7];

        let mut theta: S = r_theta;
        for _ in 0..3 {
            let theta2: S = theta * theta;

            // :284-292
            let mut func: S = k4 * theta2;
            func += k3;
            func *= theta2;
            func += k2;
            func *= theta2;
            func += k1;
            func *= theta2;
            func += S::one();
            func *= theta;

            // :294-301
            let mut derivative: S = c::<S>(9.0) * k4 * theta2;
            derivative += c::<S>(7.0) * k3;
            derivative *= theta2;
            derivative += c::<S>(5.0) * k2;
            derivative *= theta2;
            derivative += c::<S>(3.0) * k1;
            derivative *= theta2;
            derivative += S::one();
            *d_func_d_theta = derivative;

            // :304
            theta += (r_theta - func) / derivative;
        }

        theta
    }
}

impl<S: LieScalar> Camera<S> for KannalaBrandt4<S> {
    const NUM_PARAMS: usize = 8;
    type Params = SVector<S, 8>;
    type ParamJacobian = SMatrix<S, 2, 8>;

    fn name(&self) -> &'static str {
        "kb4"
    }

    fn params(&self) -> Self::Params {
        self.param
    }

    fn apply_inc(&mut self, inc: &Self::Params) {
        self.param += inc;
    }

    /// `kannala_brandt_camera4.hpp:129-260`.
    fn project_with_jacobians(
        &self,
        p3d: &Vector4<S>,
        proj: &mut Vector2<S>,
        d_proj_d_p3d: Option<&mut Matrix2x4<S>>,
        d_proj_d_param: Option<&mut Self::ParamJacobian>,
    ) -> bool {
        let fx: S = self.param[0];
        let fy: S = self.param[1];
        let cx: S = self.param[2];
        let cy: S = self.param[3];
        let k1: S = self.param[4];
        let k2: S = self.param[5];
        let k3: S = self.param[6];
        let k4: S = self.param[7];

        let x: S = p3d[0];
        let y: S = p3d[1];
        let z: S = p3d[2];

        // :148-149
        let r2: S = x * x + y * y;
        let r: S = r2.sqrt();

        let mut is_valid: bool = true;
        if r > S::sophus_epsilon_sqrt() {
            // :153-167. atan2(r, z) accepts z < 0: a fisheye sees behind its
            // own plane, which is why this branch never invalidates a point.
            //
            // The angle is computed in `f64` and rounded back even in the `f32`
            // instantiation, because that is what the C++ does. The header's
            // `using` declarations (`:48-50`) cover `cos`, `sin` and `sqrt` but
            // **not** `atan2`, so the unqualified call at `:153` resolves to
            // `::atan2(double, double)` from `<math.h>` and the float arguments
            // are promoted. Measured on this host: `atan2f(4.123105526f, -1.0f)`
            // is `0x3fe784b5` while the promoted call gives `0x3fe784b6`, and
            // that one bit of angle is 6e-5 px on basalt's own kb4 test camera.
            // Same rule as decision D37, applied to a call instead of a
            // comparison.
            let theta: S = S::from_literal(r.to_f64().atan2(z.to_f64()));
            let theta2: S = theta * theta;

            let mut r_theta: S = k4 * theta2;
            r_theta += k3;
            r_theta *= theta2;
            r_theta += k2;
            r_theta *= theta2;
            r_theta += k1;
            r_theta *= theta2;
            r_theta += S::one();
            r_theta *= theta;

            let mx: S = x * r_theta / r;
            let my: S = y * r_theta / r;

            // :169-170
            proj[0] = fx * mx + cx;
            proj[1] = fy * my + cy;

            if let Some(jacobian) = d_proj_d_p3d {
                // :174-201
                let d_r_d_x: S = x / r;
                let d_r_d_y: S = y / r;

                let tmp: S = z * z + r2;
                let d_theta_d_x: S = d_r_d_x * z / tmp;
                let d_theta_d_y: S = d_r_d_y * z / tmp;
                let d_theta_d_z: S = -r / tmp;

                let mut d_r_theta_d_theta: S = c::<S>(9.0) * k4 * theta2;
                d_r_theta_d_theta += c::<S>(7.0) * k3;
                d_r_theta_d_theta *= theta2;
                d_r_theta_d_theta += c::<S>(5.0) * k2;
                d_r_theta_d_theta *= theta2;
                d_r_theta_d_theta += c::<S>(3.0) * k1;
                d_r_theta_d_theta *= theta2;
                d_r_theta_d_theta += S::one();

                jacobian.fill(S::zero());

                jacobian[(0, 0)] = fx
                    * (r_theta * r + x * r * d_r_theta_d_theta * d_theta_d_x - x * x * r_theta / r)
                    / r2;
                jacobian[(1, 0)] =
                    fy * y * (d_r_theta_d_theta * d_theta_d_x * r - x * r_theta / r) / r2;

                jacobian[(0, 1)] =
                    fx * x * (d_r_theta_d_theta * d_theta_d_y * r - y * r_theta / r) / r2;

                jacobian[(1, 1)] = fy
                    * (r_theta * r + y * r * d_r_theta_d_theta * d_theta_d_y - y * y * r_theta / r)
                    / r2;

                jacobian[(0, 2)] = fx * x * d_r_theta_d_theta * d_theta_d_z / r;
                jacobian[(1, 2)] = fy * y * d_r_theta_d_theta * d_theta_d_z / r;
            }

            if let Some(jacobian) = d_proj_d_param {
                // :208-219
                jacobian.fill(S::zero());
                jacobian[(0, 0)] = mx;
                jacobian[(0, 2)] = S::one();
                jacobian[(1, 1)] = my;
                jacobian[(1, 3)] = S::one();

                jacobian[(0, 4)] = fx * x * theta * theta2 / r;
                jacobian[(1, 4)] = fy * y * theta * theta2 / r;

                for column in 5..8 {
                    jacobian[(0, column)] = jacobian[(0, column - 1)] * theta2;
                    jacobian[(1, column)] = jacobian[(1, column - 1)] * theta2;
                }
            }
        } else {
            // :225-256. Too close to the optical axis for `atan2` to be stable,
            // so the model degenerates to a pinhole and the sign of z decides.
            if z < S::sophus_epsilon_sqrt() {
                is_valid = false;
            }

            proj[0] = fx * x / z + cx;
            proj[1] = fy * y / z + cy;

            if let Some(jacobian) = d_proj_d_p3d {
                jacobian.fill(S::zero());
                let z2: S = z * z;
                jacobian[(0, 0)] = fx / z;
                jacobian[(0, 2)] = -fx * x / z2;
                jacobian[(1, 1)] = fy / z;
                jacobian[(1, 2)] = -fy * y / z2;
            }

            if let Some(jacobian) = d_proj_d_param {
                jacobian.fill(S::zero());
                jacobian[(0, 0)] = x / z;
                jacobian[(0, 2)] = S::one();
                jacobian[(1, 1)] = y / z;
                jacobian[(1, 3)] = S::one();
            }
        }

        is_valid
    }

    /// `kannala_brandt_camera4.hpp:337-455`.
    fn unproject(&self, proj: &Vector2<S>, p3d: &mut Vector4<S>) -> bool {
        self.unproject_with_jacobians(proj, p3d, None, None)
    }
}

impl<S: LieScalar> UnprojectJacobians<S> for KannalaBrandt4<S> {
    type UnprojectParamJacobian = SMatrix<S, 4, 8>;

    /// `kannala_brandt_camera4.hpp:337-455`.
    fn unproject_with_jacobians(
        &self,
        proj: &Vector2<S>,
        p3d: &mut Vector4<S>,
        d_p3d_d_proj: Option<&mut Matrix4x2<S>>,
        d_p3d_d_param: Option<&mut Self::UnprojectParamJacobian>,
    ) -> bool {
        let fx: S = self.param[0];
        let fy: S = self.param[1];
        let cx: S = self.param[2];
        let cy: S = self.param[3];

        // :348-369
        let mx: S = (proj[0] - cx) / fx;
        let my: S = (proj[1] - cy) / fy;

        let mut theta: S = S::zero();
        let mut sin_theta: S = S::zero();
        let mut cos_theta: S = S::one();
        let thetad: S = (mx * mx + my * my).sqrt();
        let mut scaling: S = S::one();
        let mut d_func_d_theta: S = S::zero();

        if thetad > S::sophus_epsilon_sqrt() {
            theta = self.solve_theta(thetad, &mut d_func_d_theta);
            sin_theta = theta.sin();
            cos_theta = theta.cos();
            scaling = sin_theta / thetad;
        }

        p3d.fill(S::zero());
        p3d[0] = mx * scaling;
        p3d[1] = my * scaling;
        p3d[2] = cos_theta;

        if d_p3d_d_proj.is_some() || d_p3d_d_param.is_some() {
            // :372-417
            let mut d_thetad_d_mx: S = S::zero();
            let mut d_thetad_d_my: S = S::zero();
            let mut d_scaling_d_thetad: S = S::zero();
            let mut d_cos_d_thetad: S = S::zero();
            let mut d_scaling_d_k1: S = S::zero();
            let mut d_cos_d_k1: S = S::zero();
            let mut theta2: S = S::zero();

            if thetad > S::sophus_epsilon_sqrt() {
                d_thetad_d_mx = mx / thetad;
                d_thetad_d_my = my / thetad;

                theta2 = theta * theta;

                d_scaling_d_thetad =
                    (thetad * cos_theta / d_func_d_theta - sin_theta) / (thetad * thetad);

                d_cos_d_thetad = sin_theta / d_func_d_theta;

                d_scaling_d_k1 = -cos_theta * theta * theta2 / (d_func_d_theta * thetad);

                d_cos_d_k1 = d_cos_d_thetad * theta * theta2;
            }

            let d_res0_d_mx: S = scaling + mx * d_scaling_d_thetad * d_thetad_d_mx;
            let d_res0_d_my: S = mx * d_scaling_d_thetad * d_thetad_d_my;

            let d_res1_d_mx: S = my * d_scaling_d_thetad * d_thetad_d_mx;
            let d_res1_d_my: S = scaling + my * d_scaling_d_thetad * d_thetad_d_my;

            let d_res2_d_mx: S = -d_cos_d_thetad * d_thetad_d_mx;
            let d_res2_d_my: S = -d_cos_d_thetad * d_thetad_d_my;

            let mut c0: Vector4<S> = Vector4::zeros();
            c0[0] = d_res0_d_mx / fx;
            c0[1] = d_res1_d_mx / fx;
            c0[2] = d_res2_d_mx / fx;

            let mut c1: Vector4<S> = Vector4::zeros();
            c1[0] = d_res0_d_my / fy;
            c1[1] = d_res1_d_my / fy;
            c1[2] = d_res2_d_my / fy;

            if let Some(jacobian) = d_p3d_d_proj {
                // :421-422
                jacobian.set_column(0, &c0);
                jacobian.set_column(1, &c1);
            }

            if let Some(jacobian) = d_p3d_d_param {
                // :429-443
                jacobian.fill(S::zero());

                jacobian.set_column(2, &(-c0));
                jacobian.set_column(3, &(-c1));
                jacobian.set_column(0, &(-c0 * mx));
                jacobian.set_column(1, &(-c1 * my));

                jacobian[(0, 4)] = mx * d_scaling_d_k1;
                jacobian[(1, 4)] = my * d_scaling_d_k1;
                jacobian[(2, 4)] = d_cos_d_k1;

                for column in 5..8 {
                    for row in 0..4 {
                        jacobian[(row, column)] = jacobian[(row, column - 1)] * theta2;
                    }
                }
            }
        }

        // :454
        true
    }
}

// ─── pinhole-radtan8 ──────────────────────────────────────────────────────

/// `basalt::PinholeRadtan8Camera` (`pinhole_radtan8_camera.hpp:69`), N = 12 plus `rpmax`.
///
/// `rpmax` bounds the radius in the z = 1 plane where the rational distortion
/// is still injective (`:116-130`); zero means unbounded. It is stored beside
/// the twelve optimized parameters, not inside them (`:729-733`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PinholeRadtan8<S: LieScalar> {
    param: SVector<S, 12>,
    rpmax: S,
}

impl<S: LieScalar> PinholeRadtan8<S> {
    /// Build from `[fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6]` and `rpmax`
    /// (`pinhole_radtan8_camera.hpp:100-103`).
    ///
    /// Unlike the C++ constructor there is no `rpmax = -1` sentinel that
    /// triggers `computeRpmax()`: the port never estimates the radius, it reads
    /// it from the calibration.
    pub fn new(param: SVector<S, 12>, rpmax: S) -> Self {
        Self { param, rpmax }
    }

    /// `getRpmax()` (`pinhole_radtan8_camera.hpp:702`).
    pub fn rpmax(&self) -> S {
        self.rpmax
    }

    /// `distort` (`pinhole_radtan8_camera.hpp:505-563`): the normalized point
    /// `(x', y')` mapped to `(x'', y'')`, with an optional 2x2 Jacobian.
    pub fn distort(
        &self,
        undist: &Vector2<S>,
        dist: &mut Vector2<S>,
        d_dist_d_undist: Option<&mut Matrix2<S>>,
    ) {
        let k1: S = self.param[4];
        let k2: S = self.param[5];
        let p1: S = self.param[6];
        let p2: S = self.param[7];
        let k3: S = self.param[8];
        let k4: S = self.param[9];
        let k5: S = self.param[10];
        let k6: S = self.param[11];

        let s1: S = S::one();
        let s2: S = c(2.0);
        let s3: S = c(3.0);

        // :515-524
        let xp: S = undist[0];
        let yp: S = undist[1];
        let rp2: S = xp * xp + yp * yp;
        let cdist: S =
            (s1 + rp2 * (k1 + rp2 * (k2 + rp2 * k3))) / (s1 + rp2 * (k4 + rp2 * (k5 + rp2 * k6)));
        let delta_x: S = s2 * p1 * xp * yp + p2 * (rp2 + s2 * xp * xp);
        let delta_y: S = s2 * p2 * xp * yp + p1 * (rp2 + s2 * yp * yp);
        dist[0] = xp * cdist + delta_x;
        dist[1] = yp * cdist + delta_y;

        if let Some(jacobian) = d_dist_d_undist {
            // :530-559, sympy-derived in the C++ and transcribed verbatim.
            let v0: S = xp * xp;
            let v1: S = yp * yp;
            let v2: S = v0 + v1;
            let v3: S = k6 * v2;
            let v4: S = k4 + v2 * (k5 + v3);
            let v5: S = v2 * v4 + s1;
            let v6: S = v5 * v5;
            let v7: S = s1 / v6;
            let v8: S = p1 * yp;
            let v9: S = p2 * xp;
            let v10: S = s2 * v6;
            let v11: S = k3 * v2;
            let v12: S = k1 + v2 * (k2 + v11);
            let v13: S = v12 * v2 + s1;
            let v14: S = v13 * (v2 * (k5 + s2 * v3) + v4);
            let v15: S = s2 * v14;
            let v16: S = v12 + v2 * (k2 + s2 * v11);
            let v17: S = s2 * v16;
            let v18: S = xp * yp;
            let v19: S = s2 * v7 * (-v14 * v18 + v16 * v18 * v5 + v6 * (p1 * xp + p2 * yp));

            jacobian[(0, 0)] = v7 * (-v0 * v15 + v10 * (v8 + s3 * v9) + v5 * (v0 * v17 + v13));
            jacobian[(0, 1)] = v19;
            jacobian[(1, 0)] = v19;
            jacobian[(1, 1)] = v7 * (-v1 * v15 + v10 * (s3 * v8 + v9) + v5 * (v1 * v17 + v13));
        }
    }
}

impl<S: LieScalar> Camera<S> for PinholeRadtan8<S> {
    const NUM_PARAMS: usize = 12;
    type Params = SVector<S, 12>;
    type ParamJacobian = SMatrix<S, 2, 12>;

    fn name(&self) -> &'static str {
        "pinhole-radtan8"
    }

    fn params(&self) -> Self::Params {
        self.param
    }

    /// `operator+=` (`pinhole_radtan8_camera.hpp:688-691`) **without** the
    /// `rpmax_ = computeRpmax()` that follows it: see the module docs.
    fn apply_inc(&mut self, inc: &Self::Params) {
        self.param += inc;
    }

    /// `pinhole_radtan8_camera.hpp:301-494`.
    fn project_with_jacobians(
        &self,
        p3d: &Vector4<S>,
        proj: &mut Vector2<S>,
        d_proj_d_p3d: Option<&mut Matrix2x4<S>>,
        d_proj_d_param: Option<&mut Self::ParamJacobian>,
    ) -> bool {
        let fx: S = self.param[0];
        let fy: S = self.param[1];
        let cx: S = self.param[2];
        let cy: S = self.param[3];
        let k1: S = self.param[4];
        let k2: S = self.param[5];
        let p1: S = self.param[6];
        let p2: S = self.param[7];
        let k3: S = self.param[8];
        let k4: S = self.param[9];
        let k5: S = self.param[10];
        let k6: S = self.param[11];

        let x: S = p3d[0];
        let y: S = p3d[1];
        let z: S = p3d[2];

        let s0: S = S::zero();
        let s1: S = S::one();
        let s2: S = c(2.0);
        let s3: S = c(3.0);

        // :324-336
        let xp: S = x / z;
        let yp: S = y / z;
        let rp2: S = xp * xp + yp * yp;
        let cdist: S =
            (s1 + rp2 * (k1 + rp2 * (k2 + rp2 * k3))) / (s1 + rp2 * (k4 + rp2 * (k5 + rp2 * k6)));
        let delta_x: S = s2 * p1 * xp * yp + p2 * (rp2 + s2 * xp * xp);
        let delta_y: S = s2 * p2 * xp * yp + p1 * (rp2 + s2 * yp * yp);
        let xpp: S = xp * cdist + delta_x;
        let ypp: S = yp * cdist + delta_y;
        proj[0] = fx * xpp + cx;
        proj[1] = fy * ypp + cy;

        // :338-340
        let positive_z: bool = z >= S::sophus_epsilon_sqrt();
        let in_injective_area: bool = if self.rpmax == s0 {
            true
        } else {
            rp2 <= self.rpmax * self.rpmax
        };
        let is_valid: bool = positive_z && in_injective_area;

        if let Some(jacobian) = d_proj_d_p3d {
            // :348-398, sympy-derived in the C++ and transcribed verbatim.
            jacobian.fill(s0);

            let v0: S = p1 * y;
            let v1: S = p2 * x;
            let v2: S = z * z * z * z * z * z;
            let v3: S = x * x;
            let v4: S = y * y;
            let v5: S = v3 + v4;
            let v6: S = z * z * z * z;
            let v7: S = z * z;
            let v8: S = k5 * v7;
            let v9: S = k6 * v5;
            let v10: S = k4 * v6 + v5 * (v8 + v9);
            let v11: S = v10 * v5 + v2;
            let v12: S = v11 * v11;
            let v13: S = s2 * v12;
            let v14: S = k2 * v7;
            let v15: S = k3 * v5;
            let v16: S = k1 * v6 + v5 * (v14 + v15);
            let v17: S = v16 * v5 + v2;
            let v18: S = v17 * z * (v10 + v5 * (v8 + s2 * v9));
            let v19: S = s2 * v18;
            let v20: S = v16 + v5 * (v14 + s2 * v15);
            let v21: S = s2 * v20;
            let v22: S = v11 * z;
            let v23: S = s1 / v7;
            let v24: S = s1 / v12;
            let v25: S = fx * v24;
            let v26: S = v23 * v25;
            let v27: S = p2 * y;
            let v28: S = x * y;
            let v29: S = s2 * v12 * (p1 * x + v27) - s2 * v18 * v28 + s2 * v20 * v22 * v28;
            let v30: S = s1 / (z * z * z);
            let v31: S = s2 * x;
            let v32: S = v22 * (v17 + v21 * v5);
            let v33: S = fy * v24;
            let v34: S = v23 * v33;

            jacobian[(0, 0)] = v26 * (v13 * (v0 + s3 * v1) - v19 * v3 + v22 * (v17 + v21 * v3));
            jacobian[(0, 1)] = v26 * v29;
            jacobian[(0, 2)] =
                -v25 * v30 * (v13 * (p2 * (s3 * v3 + v4) + v0 * v31) - v18 * v31 * v5 + v32 * x);
            jacobian[(1, 0)] = v29 * v34;
            jacobian[(1, 1)] = v34 * (v13 * (s3 * v0 + v1) - v19 * v4 + v22 * (v17 + v21 * v4));
            jacobian[(1, 2)] =
                -v30 * v33 * (v13 * (p1 * (v3 + s3 * v4) + v27 * v31) - v19 * v5 * y + v32 * y);
        }

        if let Some(jacobian) = d_proj_d_param {
            // :405-488, sympy-derived in the C++ and transcribed verbatim.
            jacobian.fill(s0);

            let w0: S = z * z * z * z * z * z;
            let w1: S = x * x;
            let w2: S = y * y;
            let w3: S = w1 + w2;
            let w4: S = z * z * z * z;
            let w5: S = z * z;
            let w6: S = w0 + w3 * (k1 * w4 + w3 * (k2 * w5 + k3 * w3));
            let w7: S = w6 * z;
            let w8: S = w7 * x;
            let w9: S = s2 * x * y;
            let w10: S = s3 * w1 + w2;
            let w11: S = w0 + w3 * (k4 * w4 + w3 * (k5 * w5 + k6 * w3));
            let w12: S = s1 / w5;
            let w13: S = s1 / w11;
            let w14: S = w12 * w13;
            let w15: S = w3 * (z * z * z);
            let w16: S = fx * x;
            let w17: S = w13 * w16;
            let w18: S = w3 * w3;
            let w19: S = w18 * z;
            let w20: S = fx * w12;
            let w21: S = (w3 * w3 * w3) * s1 / z;
            let w22: S = s1 / (w11 * w11);
            let w23: S = w22 * w6;
            let w24: S = w16 * w23;
            let w25: S = w18 * w22;
            let w26: S = w7 * y;
            let w27: S = w1 + s3 * w2;
            let w28: S = fy * y;
            let w29: S = w13 * w28;
            let w30: S = fy * w12;
            let w31: S = w23 * w28;

            jacobian[(0, 0)] = w14 * (w11 * (p1 * w9 + p2 * w10) + w8);
            jacobian[(0, 1)] = s0;
            jacobian[(0, 2)] = s1;
            jacobian[(0, 3)] = s0;
            jacobian[(0, 4)] = w15 * w17;
            jacobian[(0, 5)] = w17 * w19;
            jacobian[(0, 6)] = w20 * w9;
            jacobian[(0, 7)] = w10 * w20;
            jacobian[(0, 8)] = w17 * w21;
            jacobian[(0, 9)] = -w15 * w24;
            jacobian[(0, 10)] = -fx * w25 * w8;
            jacobian[(0, 11)] = -w21 * w24;
            jacobian[(1, 0)] = s0;
            jacobian[(1, 1)] = w14 * (w11 * (p1 * w27 + p2 * w9) + w26);
            jacobian[(1, 2)] = s0;
            jacobian[(1, 3)] = s1;
            jacobian[(1, 4)] = w15 * w29;
            jacobian[(1, 5)] = w19 * w29;
            jacobian[(1, 6)] = w27 * w30;
            jacobian[(1, 7)] = w30 * w9;
            jacobian[(1, 8)] = w21 * w29;
            jacobian[(1, 9)] = -w15 * w31;
            jacobian[(1, 10)] = -fy * w25 * w26;
            jacobian[(1, 11)] = -w21 * w31;
        }

        is_valid
    }

    /// `pinhole_radtan8_camera.hpp:597-669`: five Newton steps on `distort`,
    /// stopping early when the residual falls under `epsilonSqrt`.
    ///
    /// The 2x2 inverse inside the loop is written out because it has to round
    /// like Eigen's, not like nalgebra's: Eigen takes **one** reciprocal of the
    /// determinant and multiplies each cofactor by it
    /// (`eigen/Eigen/src/LU/InverseImpl.h:66-83`, determinant at
    /// `Determinant.h:40-44`), while `Matrix2::try_inverse` divides each
    /// coefficient by the determinant. The two differ by a rounding step, which
    /// is worth 4e-5 of bearing in `f32` on the msd-g2 cam2 calibration.
    ///
    /// A singular Jacobian is not special-cased either. `1 / 0` is an infinity,
    /// the iterate becomes NaN, and the final `rp2 <= rpmax^2` comparison is
    /// false, so the pixel is rejected — which is exactly what the C++ does with
    /// `J.inverse()` (`:628`). Dividing by zero in floating point is not a
    /// panic, so decision D32 is not in play, and returning the last finite
    /// iterate instead would report success for a pixel that reprojects 50 px
    /// away.
    fn unproject(&self, proj: &Vector2<S>, p3d: &mut Vector4<S>) -> bool {
        let fx: S = self.param[0];
        let fy: S = self.param[1];
        let cx: S = self.param[2];
        let cy: S = self.param[3];

        // :610-611
        let dist: Vector2<S> = Vector2::new((proj[0] - cx) / fx, (proj[1] - cy) / fy);

        // :619-630
        let mut undist: Vector2<S> = dist;
        let eps: S = S::sophus_epsilon_sqrt();
        for _ in 0..5 {
            let mut jacobian: Matrix2<S> = Matrix2::zeros();
            let mut fundist: Vector2<S> = Vector2::zeros();
            self.distort(&undist, &mut fundist, Some(&mut jacobian));
            let residual: Vector2<S> = fundist - dist;
            let determinant: S =
                jacobian[(0, 0)] * jacobian[(1, 1)] - jacobian[(1, 0)] * jacobian[(0, 1)];
            let invdet: S = S::one() / determinant;
            let inverse: Matrix2<S> = Matrix2::new(
                jacobian[(1, 1)] * invdet,
                -jacobian[(0, 1)] * invdet,
                -jacobian[(1, 0)] * invdet,
                jacobian[(0, 0)] * invdet,
            );
            undist -= inverse * residual;
            if residual.norm() < eps {
                break;
            }
        }
        let xp: S = undist[0];
        let yp: S = undist[1];

        // :651-655
        let norm_inv: S = S::one() / (xp * xp + yp * yp + S::one()).sqrt();
        p3d.fill(S::zero());
        p3d[0] = xp * norm_inv;
        p3d[1] = yp * norm_inv;
        p3d[2] = norm_inv;

        // :664-666
        let rp2: S = xp * xp + yp * yp;
        if self.rpmax == S::zero() {
            true
        } else {
            rp2 <= self.rpmax * self.rpmax
        }
    }
}

// ─── the variant ──────────────────────────────────────────────────────────

/// `basalt::GenericCamera` (`generic_camera.hpp:62`), narrowed to V0's models.
///
/// basalt dispatches with `std::visit` and warns that a per-point visit is
/// "**SLOW** … requires vtable lookup for every projection"
/// (`generic_camera.hpp:127-129`); a Rust enum matched inside the loop is a
/// jump table over three monomorphized bodies, with no vtable and no `Box<dyn>`.
///
/// `ds`, `eucm` and `ucm` parse in [`crate::calib::CameraModel`] but have no
/// variant here: [`CameraEnum::from_model`] rejects them (decision D11).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CameraEnum<S: LieScalar> {
    /// `pinhole`.
    Pinhole(Pinhole<S>),
    /// `kb4`.
    Kb4(KannalaBrandt4<S>),
    /// `pinhole-radtan8`.
    PinholeRadtan8(PinholeRadtan8<S>),
}

impl<S: LieScalar> CameraEnum<S> {
    /// Build from a parsed calibration entry.
    pub fn from_model(model: &CameraModel<S>) -> Result<Self, CameraError> {
        match model {
            CameraModel::Pinhole(p) => Ok(Self::Pinhole(Pinhole::new(SVector::<S, 4>::from([
                p.fx, p.fy, p.cx, p.cy,
            ])))),
            CameraModel::Kb4(p) => Ok(Self::Kb4(KannalaBrandt4::new(SVector::<S, 8>::from([
                p.fx, p.fy, p.cx, p.cy, p.k1, p.k2, p.k3, p.k4,
            ])))),
            CameraModel::PinholeRadtan8(p) => Ok(Self::PinholeRadtan8(PinholeRadtan8::new(
                SVector::<S, 12>::from([
                    p.fx, p.fy, p.cx, p.cy, p.k1, p.k2, p.p1, p.p2, p.k3, p.k4, p.k5, p.k6,
                ]),
                p.rpmax,
            ))),
            CameraModel::DoubleSphere(_)
            | CameraModel::ExtendedUnified(_)
            | CameraModel::Unified(_) => Err(CameraError::UnsupportedModel {
                model: model.name(),
            }),
        }
    }

    /// basalt's `getName()` (`generic_camera.hpp:96`).
    pub fn name(&self) -> &'static str {
        match self {
            Self::Pinhole(cam) => cam.name(),
            Self::Kb4(cam) => cam.name(),
            Self::PinholeRadtan8(cam) => cam.name(),
        }
    }

    /// `[fx, fy, cx, cy]`, which every model's `getParam()` starts with.
    ///
    /// Read off the optimized parameters rather than the parsed calibration.
    /// Every projection
    /// ends in `f * m + c`, so `c` is also the scale a rounding argument about a
    /// pixel has to use: near the principal point the two terms cancel and the
    /// error of the sum is set by `c`, not by the pixel.
    pub fn focal_and_principal_point(&self) -> [S; 4] {
        match self {
            Self::Pinhole(cam) => {
                let p: SVector<S, 4> = cam.params();
                [p[0], p[1], p[2], p[3]]
            }
            Self::Kb4(cam) => {
                let p: SVector<S, 8> = cam.params();
                [p[0], p[1], p[2], p[3]]
            }
            Self::PinholeRadtan8(cam) => {
                let p: SVector<S, 12> = cam.params();
                [p[0], p[1], p[2], p[3]]
            }
        }
    }

    /// Project a homogeneous point; `false` outside the model's valid domain.
    #[inline]
    pub fn project(&self, p3d: &Vector4<S>, proj: &mut Vector2<S>) -> bool {
        match self {
            Self::Pinhole(cam) => cam.project(p3d, proj),
            Self::Kb4(cam) => cam.project(p3d, proj),
            Self::PinholeRadtan8(cam) => cam.project(p3d, proj),
        }
    }

    /// Project and fill the 2x4 point Jacobian, the only one
    /// `GenericCamera::project` exposes (`generic_camera.hpp:135`).
    #[inline]
    pub fn project_with_jacobian(
        &self,
        p3d: &Vector4<S>,
        proj: &mut Vector2<S>,
        d_proj_d_p3d: &mut Matrix2x4<S>,
    ) -> bool {
        match self {
            Self::Pinhole(cam) => cam.project_with_jacobians(p3d, proj, Some(d_proj_d_p3d), None),
            Self::Kb4(cam) => cam.project_with_jacobians(p3d, proj, Some(d_proj_d_p3d), None),
            Self::PinholeRadtan8(cam) => {
                cam.project_with_jacobians(p3d, proj, Some(d_proj_d_p3d), None)
            }
        }
    }

    /// Unproject a pixel into a unit-norm bearing with a zero fourth component.
    #[inline]
    pub fn unproject(&self, proj: &Vector2<S>, p3d: &mut Vector4<S>) -> bool {
        match self {
            Self::Pinhole(cam) => cam.unproject(proj, p3d),
            Self::Kb4(cam) => cam.unproject(proj, p3d),
            Self::PinholeRadtan8(cam) => cam.unproject(proj, p3d),
        }
    }

    /// Unproject and fill the 4x2 Jacobian (`generic_camera.hpp:157`).
    ///
    /// `pinhole-radtan8` has none: C++ asserts, the port returns
    /// [`CameraError::UnprojectJacobianUnsupported`].
    pub fn unproject_with_jacobian(
        &self,
        proj: &Vector2<S>,
        p3d: &mut Vector4<S>,
        d_p3d_d_proj: &mut Matrix4x2<S>,
    ) -> Result<bool, CameraError> {
        match self {
            Self::Pinhole(cam) => {
                Ok(cam.unproject_with_jacobians(proj, p3d, Some(d_p3d_d_proj), None))
            }
            Self::Kb4(cam) => Ok(cam.unproject_with_jacobians(proj, p3d, Some(d_p3d_d_proj), None)),
            Self::PinholeRadtan8(_) => {
                Err(CameraError::UnprojectJacobianUnsupported { model: self.name() })
            }
        }
    }
}

/// One camera of a rig: a projection model plus the size of the images it produces.
///
/// basalt reads `resolution[0]` for every camera
/// (`frame_to_frame_optical_flow.h:108-109`); the port carries a resolution per
/// camera because the msd-g2 recordings are stored rotated into portrait and
/// the cameras of one rig then disagree (decision D30).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigCamera<S: LieScalar> {
    /// The projection model.
    pub model: CameraEnum<S>,
    /// `[width, height]` in pixels.
    pub resolution: [u32; 2],
}

impl<S: LieScalar> RigCamera<S> {
    /// Build one camera per entry of a parsed calibration.
    pub fn from_calibration(calibration: &Calibration<S>) -> Result<Vec<Self>, CameraError> {
        if calibration.intrinsics.len() != calibration.resolution.len() {
            return Err(CameraError::RaggedCalibration {
                intrinsics: calibration.intrinsics.len(),
                resolutions: calibration.resolution.len(),
            });
        }
        calibration
            .intrinsics
            .iter()
            .zip(calibration.resolution.iter())
            .map(|(model, resolution)| {
                Ok(Self {
                    model: CameraEnum::from_model(model)?,
                    resolution: *resolution,
                })
            })
            .collect()
    }

    /// Image width in pixels.
    pub fn width(&self) -> u32 {
        self.resolution[0]
    }

    /// Image height in pixels.
    pub fn height(&self) -> u32 {
        self.resolution[1]
    }

    /// `Image::InBounds(p, border)` for floating-point coordinates
    /// (`image/image.h:695-705`).
    ///
    /// The `offset` of one that the C++ adds for floating-point scalars is what
    /// keeps `interp` from reading the row past the last one, so a keypoint at
    /// `border` is in and a keypoint at `height - border - 1` is out.
    ///
    /// No production caller: the frontend asks
    /// [`crate::image::ImageU16::in_bounds`], the same C++ predicate over the
    /// buffer it is about to read. This one answers from the *calibrated*
    /// resolution, which is what `tests/camera_jacobians.rs` needs to say where
    /// a projection lands on the sensor.
    pub fn in_bounds(&self, uv: &Vector2<S>, border: S) -> bool {
        let width: S = c(f64::from(self.resolution[0]));
        let height: S = c(f64::from(self.resolution[1]));
        let offset: S = S::one();
        border <= uv[0]
            && uv[0] < (width - border - offset)
            && border <= uv[1]
            && uv[1] < (height - border - offset)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use approx::assert_abs_diff_eq;

    /// `KannalaBrandtCamera4::getTestProjections()` (`kannala_brandt_camera4.hpp:491`).
    fn kb4_test_camera<S: LieScalar>() -> KannalaBrandt4<S> {
        KannalaBrandt4::new(SVector::<S, 8>::from([
            c(379.045),
            c(379.008),
            c(505.512),
            c(509.969),
            c(0.00693023),
            c(-0.0013828),
            c(-0.000272596),
            c(-0.000452646),
        ]))
    }

    #[test]
    fn a_pinhole_projects_and_unprojects_a_known_point() {
        // Euroc intrinsics, `pinhole_camera.hpp:287`.
        let camera: Pinhole<f64> = Pinhole::new(SVector::<f64, 4>::from([
            460.76484651566468,
            459.4051018049483,
            365.8937161309615,
            249.33499869752445,
        ]));
        let point: Vector4<f64> = Vector4::new(0.5, -0.25, 2.0, 1.0);
        let mut uv: Vector2<f64> = Vector2::zeros();
        assert!(camera.project(&point, &mut uv));
        assert_abs_diff_eq!(
            uv[0],
            460.76484651566468 * 0.25 + 365.8937161309615,
            epsilon = 1e-12
        );

        let mut bearing: Vector4<f64> = Vector4::zeros();
        assert!(camera.unproject(&uv, &mut bearing));
        assert_abs_diff_eq!(bearing[3], 0.0, epsilon = 0.0);
        assert_abs_diff_eq!(bearing.norm(), 1.0, epsilon = 1e-15);
        let expected: Vector4<f64> = Vector4::new(0.5, -0.25, 2.0, 0.0).normalize();
        assert_abs_diff_eq!((bearing - expected).norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn a_pinhole_rejects_points_behind_the_camera() {
        let camera: Pinhole<f64> =
            Pinhole::new(SVector::<f64, 4>::from([400.0, 400.0, 320.0, 240.0]));
        let mut uv: Vector2<f64> = Vector2::zeros();
        // `pinhole_camera.hpp:137`: the bound is epsilonSqrt, not zero.
        assert!(!camera.project(&Vector4::new(0.1, 0.1, -1.0, 1.0), &mut uv));
        assert!(uv.iter().all(|value| value.is_finite()));
        assert!(!camera.project(&Vector4::new(0.1, 0.1, 1e-8, 1.0), &mut uv));
        assert!(camera.project(&Vector4::new(0.1, 0.1, 1e-4, 1.0), &mut uv));
    }

    #[test]
    fn kb4_accepts_a_point_behind_its_own_plane_but_not_on_the_negative_axis() {
        let camera: KannalaBrandt4<f64> = kb4_test_camera();
        let mut uv: Vector2<f64> = Vector2::zeros();
        // r > epsilonSqrt: `kannala_brandt_camera4.hpp:152` never invalidates,
        // which is how a 190-degree fisheye sees behind itself.
        assert!(camera.project(&Vector4::new(1.0, 0.0, -1.0, 1.0), &mut uv));
        // r <= epsilonSqrt and z < epsilonSqrt: `:226`.
        assert!(!camera.project(&Vector4::new(0.0, 0.0, -1.0, 1.0), &mut uv));
        assert!(uv.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn radtan8_rejects_a_point_outside_the_valid_radius() {
        // cam0 of the msd-g2 calibration, whose rpmax is 2.72763729095459.
        let calibration: Calibration<f64> =
            Calibration::from_json_str(include_str!("../tests/fixtures/msdmg_calib.json")).unwrap();
        let CameraEnum::PinholeRadtan8(camera) =
            CameraEnum::from_model(&calibration.intrinsics[0]).unwrap()
        else {
            panic!("msdmg cam0 is pinhole-radtan8");
        };
        let rpmax: f64 = camera.rpmax();
        assert_abs_diff_eq!(rpmax, 2.72763729095459, epsilon = 0.0);

        // rp2 = (x/z)^2 + (y/z)^2 is compared against rpmax^2 (`:339`), so a
        // point on the x axis at z = 1 is in or out by its x alone.
        let mut uv: Vector2<f64> = Vector2::zeros();
        let inside: Vector4<f64> = Vector4::new(rpmax - 1e-6, 0.0, 1.0, 1.0);
        let outside: Vector4<f64> = Vector4::new(rpmax + 1e-6, 0.0, 1.0, 1.0);
        assert!(camera.project(&inside, &mut uv));
        assert!(uv.iter().all(|value| value.is_finite()));
        assert!(!camera.project(&outside, &mut uv));
        assert!(uv.iter().all(|value| value.is_finite()));

        // The bound is inclusive: `rp2 <= rpmax * rpmax`.
        assert!(camera.project(&Vector4::new(rpmax, 0.0, 1.0, 1.0), &mut uv));
    }

    /// A pixel the distortion never reaches sends the Newton solve onto a
    /// singular Jacobian, and the pixel must be **rejected**.
    ///
    /// `fx = fy = 100`, `cx = 320`, `cy = 240`, `k4 = 1`: the distortion is
    /// `xp / (1 + rp^2)`, which peaks at 0.5 and has a vanishing derivative
    /// there. Pixel (420, 240) asks for `xpp = 1`. C++ divides by the zero
    /// determinant, carries NaNs through and fails the `rp2 <= rpmax^2` check
    /// (`pinhole_radtan8_camera.hpp:628, :664-666`); so does this port. Stopping
    /// the iteration and keeping the last finite iterate instead — which an
    /// earlier version of this module did — reports success for a bearing that
    /// reprojects fifty pixels away, which is the failure this test exists to
    /// catch. `tests/camera_oracle.rs` pins the same case against the C++
    /// numbers, in both scalars.
    #[test]
    fn a_singular_newton_step_rejects_the_pixel() {
        let camera: PinholeRadtan8<f64> = PinholeRadtan8::new(
            SVector::<f64, 12>::from([
                100.0, 100.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            ]),
            1.0,
        );
        let pixel: Vector2<f64> = Vector2::new(420.0, 240.0);
        let mut bearing: Vector4<f64> = Vector4::zeros();
        assert!(!camera.unproject(&pixel, &mut bearing));
        assert!(bearing[0].is_nan() && bearing[1].is_nan() && bearing[2].is_nan());

        // The same solve on a pixel the distortion does reach is unremarkable.
        let mut inside: Vector4<f64> = Vector4::zeros();
        assert!(camera.unproject(&Vector2::new(340.0, 250.0), &mut inside));
        assert!(inside.iter().all(|value| value.is_finite()));
        let mut reprojected: Vector2<f64> = Vector2::zeros();
        assert!(camera.project(&inside, &mut reprojected));
        assert_abs_diff_eq!(reprojected[0], 340.0, epsilon = 1e-9);
        assert_abs_diff_eq!(reprojected[1], 250.0, epsilon = 1e-9);
    }

    #[test]
    fn radtan8_without_a_valid_radius_is_unbounded() {
        // `:339`: rpmax == 0 means "injective everywhere", not "nothing is valid".
        let camera: PinholeRadtan8<f64> = PinholeRadtan8::new(
            SVector::<f64, 12>::from([
                269.06, 269.16, 324.33, 245.22, 0.6257, 0.4661, -0.000185, -4.288e-5, 0.00417,
                0.8943, 0.5425, 0.0662,
            ]),
            0.0,
        );
        let mut uv: Vector2<f64> = Vector2::zeros();
        assert!(camera.project(&Vector4::new(100.0, 0.0, 1.0, 1.0), &mut uv));
    }

    #[test]
    fn the_variant_rejects_the_three_deferred_models() {
        let calibration: Calibration<f64> =
            Calibration::from_json_str(include_str!("../tests/fixtures/euroc_ds_calib.json"))
                .unwrap();
        assert_eq!(
            CameraEnum::from_model(&calibration.intrinsics[0]),
            Err(CameraError::UnsupportedModel { model: "ds" })
        );
    }

    #[test]
    fn radtan8_has_no_unprojection_jacobian() {
        let calibration: Calibration<f64> =
            Calibration::from_json_str(include_str!("../tests/fixtures/msdmg_calib.json")).unwrap();
        let camera: CameraEnum<f64> = CameraEnum::from_model(&calibration.intrinsics[0]).unwrap();
        let mut bearing: Vector4<f64> = Vector4::zeros();
        let mut jacobian: Matrix4x2<f64> = Matrix4x2::zeros();
        assert_eq!(
            camera.unproject_with_jacobian(
                &Vector2::new(320.0, 240.0),
                &mut bearing,
                &mut jacobian
            ),
            Err(CameraError::UnprojectJacobianUnsupported {
                model: "pinhole-radtan8"
            })
        );
    }

    #[test]
    fn the_variant_reports_the_focal_length_and_principal_point() {
        let calibration: Calibration<f64> =
            Calibration::from_json_str(include_str!("../tests/fixtures/msdmg_calib.json")).unwrap();
        let model: CameraEnum<f64> = CameraEnum::from_model(&calibration.intrinsics[0]).unwrap();
        assert_abs_diff_eq!(
            model.focal_and_principal_point()[2],
            322.5578605887897,
            epsilon = 0.0
        );
    }

    #[test]
    fn a_rig_carries_one_resolution_per_camera() {
        let calibration: Calibration<f64> =
            Calibration::from_json_str(include_str!("../tests/fixtures/msdmi_calib.json")).unwrap();
        let rig: Vec<RigCamera<f64>> = RigCamera::from_calibration(&calibration).unwrap();
        assert_eq!(rig.len(), 2);
        assert_eq!(rig[0].model.name(), "kb4");
        assert_eq!([rig[0].width(), rig[0].height()], [960, 960]);

        // `image/image.h:704`: border <= u < w - border - 1.
        assert!(rig[0].in_bounds(&Vector2::new(0.0, 0.0), 0.0));
        assert!(rig[0].in_bounds(&Vector2::new(958.999, 958.999), 0.0));
        assert!(!rig[0].in_bounds(&Vector2::new(959.0, 0.0), 0.0));
        assert!(!rig[0].in_bounds(&Vector2::new(-0.001, 0.0), 0.0));
        assert!(rig[0].in_bounds(&Vector2::new(2.0, 2.0), 2.0));
        assert!(!rig[0].in_bounds(&Vector2::new(1.999, 2.0), 2.0));
        assert!(!rig[0].in_bounds(&Vector2::new(957.0, 2.0), 2.0));
    }
}
