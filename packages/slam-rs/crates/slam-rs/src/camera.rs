//! Homogeneous SLAM camera policy over staged calibration models.

use kornia_staging_algebra::Scalar;
use kornia_staging_3d::camera as staged;
use nalgebra::{Matrix2x4, Vector2, Vector4};

use crate::calib::{BasaltCamera, Calibration};
use crate::lie::{c};

/// Something a camera model cannot do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CameraError {
    /// Non-finite or invalid calibration parameters.
    #[error("invalid camera calibration")]
    InvalidCalibration,
    /// A model that [`crate::calib`] parses but this module does not project with.
    #[error("camera model {model} is parsed but its projection is not implemented")]
    UnsupportedModel {
        /// Model name stored in `camera_type`.
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

/// Homogeneous SLAM projection policy over a validated staged lens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlamCamera<S: Scalar> {
    /// The staged calibration and projection model.
    pub inner: staged::CameraModelKind<S>,
}

impl<S: Scalar> SlamCamera<S> {
    /// Build from a parsed calibration entry.
    pub fn from_model(model: &BasaltCamera<S>) -> Result<Self, CameraError> {
        match model.to_staged() {
            Ok(inner) => Ok(Self { inner }),
            Err(
                staged::formats::ConversionError::Invalid(_)
                | staged::formats::ConversionError::Calibration(_),
            ) => Err(CameraError::InvalidCalibration),
            _ => Err(CameraError::UnsupportedModel {
                model: model.name(),
            }),
        }
    }

    /// The calibration model name.
    pub fn name(&self) -> &'static str {
        match self.inner {
            staged::CameraModelKind::Pinhole(_) => "pinhole",
            staged::CameraModelKind::Kb4(_) => "kb4",
            staged::CameraModelKind::BrownConrady(_) => "pinhole-radtan8",
            staged::CameraModelKind::Fisheye624(_) => "fisheye624",
        }
    }

    /// The staged calibration's focal lengths and principal point.
    pub fn focal_and_principal_point(&self) -> [S; 4] {
        match self.inner {
            staged::CameraModelKind::Pinhole(v) => v.params(),
            staged::CameraModelKind::Kb4(v) => {
                let [fx, fy, cx, cy, ..] = v.params();
                [fx, fy, cx, cy]
            }
            staged::CameraModelKind::BrownConrady(v) => {
                let [fx, fy, cx, cy, ..] = v.params();
                [fx, fy, cx, cy]
            }
            staged::CameraModelKind::Fisheye624(v) => {
                let [fx, fy, cx, cy, ..] = v.params();
                [fx, fy, cx, cy]
            }
        }
    }

    /// Project once, writing the same unchecked pixel on rejection.
    /// The caller must check validity before using the output in geometry.
    #[inline]
    pub fn project_point(
        &self,
        point: &Vector4<S>,
        pixel: &mut Vector2<S>,
        jacobian: Option<&mut Matrix2x4<S>>,
    ) -> bool {
        let mut derivative = [[S::zero(); 3]; 2];
        let (projected, status) = self.inner.project_with_status(
            [point[0], point[1], point[2]],
            jacobian.as_ref().map(|_| &mut derivative),
        );
        *pixel = Vector2::from(projected);
        if let Some(j) = jacobian {
            j.fill(S::zero());
            if status.is_ok() {
                j.fixed_view_mut::<2, 3>(0, 0)
                    .copy_from(&nalgebra::Matrix2x3::from_row_slice(
                        derivative.as_flattened(),
                    ));
            }
        }
        status.is_ok()
    }

    /// Unproject to a unit bearing with homogeneous coordinate zero.
    #[inline]
    pub fn unproject(&self, pixel: &Vector2<S>, point: &mut Vector4<S>) -> bool {
        match self.inner.unproject([pixel[0], pixel[1]]) {
            Ok(p) => {
                *point = Vector4::new(p[0], p[1], p[2], S::zero());
                true
            }
            Err(_) => {
                *point = Vector4::new(c(f64::NAN), c(f64::NAN), c(f64::NAN), S::zero());
                false
            }
        }
    }
}

/// A projection model and its camera's own image size.
/// Per-camera resolutions support rigs whose stored orientations differ (D30).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigCamera<S: Scalar> {
    /// The projection model.
    pub model: SlamCamera<S>,
    /// `[width, height]` in pixels.
    pub resolution: [u32; 2],
}

impl<S: Scalar> RigCamera<S> {
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
                    model: SlamCamera::from_model(model)?,
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

    /// Floating-point image bounds with a one-pixel interpolation margin.
    /// A coordinate at `border` is in; one at `height - border - 1` is out.
    /// Tests use the calibrated resolution here. The frontend uses
    /// [`kornia_staging_imgproc::interpolation::in_bounds_u16`] on the actual buffer.
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
    use nalgebra::SVector;
    fn pinhole<S: Scalar>(
        params: SVector<S, 4>,
    ) -> Result<SlamCamera<S>, staged::InvalidCalibration> {
        Ok(SlamCamera {
            inner: staged::CameraModelKind::Pinhole(staged::Pinhole::new(params.into())?),
        })
    }
    fn kb4<S: Scalar>(
        params: SVector<S, 8>,
    ) -> Result<SlamCamera<S>, staged::InvalidCalibration> {
        Ok(SlamCamera {
            inner: staged::CameraModelKind::Kb4(staged::KannalaBrandt4::new(params.into())?),
        })
    }
    fn brown<S: Scalar>(
        params: SVector<S, 12>,
        radius: S,
    ) -> Result<SlamCamera<S>, staged::InvalidCalibration> {
        let mut full = [S::zero(); 18];
        full[..12].copy_from_slice(params.as_slice());
        Ok(SlamCamera {
            inner: staged::CameraModelKind::BrownConrady(staged::BrownConrady::new(
                full,
                (radius != S::zero()).then_some(radius),
            )?),
        })
    }
    use approx::assert_abs_diff_eq;

    /// `KannalaBrandtCamera4::getTestProjections()`.
    fn kb4_test_camera<S: Scalar>() -> SlamCamera<S> {
        kb4(SVector::<S, 8>::from([
            c(379.045),
            c(379.008),
            c(505.512),
            c(509.969),
            c(0.00693023),
            c(-0.0013828),
            c(-0.000272596),
            c(-0.000452646),
        ]))
        .unwrap()
    }

    #[test]
    fn a_pinhole_projects_and_unprojects_a_known_point() {
        // Euroc intrinsics.
        let camera: SlamCamera<f64> = pinhole(SVector::<f64, 4>::from([
            460.76484651566468,
            459.4051018049483,
            365.8937161309615,
            249.33499869752445,
        ]))
        .unwrap();
        let point: Vector4<f64> = Vector4::new(0.5, -0.25, 2.0, 1.0);
        let mut uv: Vector2<f64> = Vector2::zeros();
        assert!(camera.project_point(&point, &mut uv, None));
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
        let camera: SlamCamera<f64> =
            pinhole(SVector::<f64, 4>::from([400.0, 400.0, 320.0, 240.0])).unwrap();
        let mut uv: Vector2<f64> = Vector2::zeros();
        // the bound is epsilonSqrt, not zero.
        assert!(!camera.project_point(&Vector4::new(0.1, 0.1, -1.0, 1.0), &mut uv, None));
        assert!(uv.iter().all(|value| value.is_finite()));
        assert!(!camera.project_point(&Vector4::new(0.1, 0.1, 1e-8, 1.0), &mut uv, None));
        assert!(camera.project_point(&Vector4::new(0.1, 0.1, 1e-4, 1.0), &mut uv, None));
    }

    #[test]
    fn kb4_accepts_a_point_behind_its_own_plane_but_not_on_the_negative_axis() {
        let camera: SlamCamera<f64> = kb4_test_camera();
        let mut uv: Vector2<f64> = Vector2::zeros();
        // The calibrated polynomial ends before this angle; a monotone KB4
        // still admits rear rays, as a greater-than-180-degree lens requires.
        assert!(!camera.project_point(&Vector4::new(1.0, 0.0, -1.0, 1.0), &mut uv, None));
        let camera = kb4(SVector::<f64, 8>::from([
            400.0, 400.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0,
        ]))
        .unwrap();
        assert!(camera.project_point(&Vector4::new(1.0, 0.0, -1.0, 1.0), &mut uv, None));
        // The negative optical axis has no unique azimuth.
        assert!(!camera.project_point(&Vector4::new(0.0, 0.0, -1.0, 1.0), &mut uv, None));
        assert!(uv.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn radtan8_rejects_a_point_outside_the_valid_radius() {
        // cam0 of the msd-g2 calibration, whose rpmax is 2.72763729095459.
        let calibration: Calibration<f64> =
            Calibration::from_json_str(include_str!("../tests/fixtures/msdmg_calib.json")).unwrap();
        let camera = SlamCamera::from_model(&calibration.intrinsics[0]).unwrap();
        let staged::CameraModelKind::BrownConrady(inner) = camera.inner else {
            panic!("msdmg cam0 is pinhole-radtan8");
        };
        let rpmax: f64 = inner.valid_radius().unwrap();
        assert_abs_diff_eq!(rpmax, 2.72763729095459, epsilon = 0.0);

        // rp2 = (x/z)^2 + (y/z)^2 is compared against rpmax^2, so a
        // point on the x axis at z = 1 is in or out by its x alone.
        let mut uv: Vector2<f64> = Vector2::zeros();
        let inside: Vector4<f64> = Vector4::new(rpmax - 1e-6, 0.0, 1.0, 1.0);
        let outside: Vector4<f64> = Vector4::new(rpmax + 1e-6, 0.0, 1.0, 1.0);
        assert!(camera.project_point(&inside, &mut uv, None));
        assert!(uv.iter().all(|value| value.is_finite()));
        assert!(!camera.project_point(&outside, &mut uv, None));
        assert!(uv.iter().all(|value| value.is_finite()));

        // The bound is inclusive: `rp2 <= rpmax * rpmax`.
        assert!(camera.project_point(&Vector4::new(rpmax, 0.0, 1.0, 1.0), &mut uv, None));
    }

    /// A pixel the distortion never reaches sends the Newton solve onto a
    /// singular Jacobian, and the pixel must be **rejected**.
    ///
    /// `fx = fy = 100`, `cx = 320`, `cy = 240`, `k4 = 1`: the distortion is
    /// `xp / (1 + rp^2)`, which peaks at 0.5 and has a vanishing derivative
    /// there. Pixel (420, 240) asks for `xpp = 1`. The singular solve must reject
    /// the pixel and mark the bearing invalid. Accepting the last finite iterate
    /// would return a bearing that reprojects fifty pixels away.
    #[test]
    fn a_singular_newton_step_rejects_the_pixel() {
        let camera: SlamCamera<f64> = brown(
            SVector::<f64, 12>::from([
                100.0, 100.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            ]),
            1.0,
        )
        .unwrap();
        let pixel: Vector2<f64> = Vector2::new(420.0, 240.0);
        let mut bearing: Vector4<f64> = Vector4::zeros();
        assert!(!camera.unproject(&pixel, &mut bearing));
        assert!(bearing[0].is_nan() && bearing[1].is_nan() && bearing[2].is_nan());

        // The same solve on a pixel the distortion does reach is unremarkable.
        let mut inside: Vector4<f64> = Vector4::zeros();
        assert!(camera.unproject(&Vector2::new(340.0, 250.0), &mut inside));
        assert!(inside.iter().all(|value| value.is_finite()));
        let mut reprojected: Vector2<f64> = Vector2::zeros();
        assert!(camera.project_point(&inside, &mut reprojected, None));
        assert_abs_diff_eq!(reprojected[0], 340.0, epsilon = 1e-9);
        assert_abs_diff_eq!(reprojected[1], 250.0, epsilon = 1e-9);
    }

    #[test]
    fn radtan8_without_a_valid_radius_is_unbounded() {
        // rpmax == 0 means "injective everywhere", not "nothing is valid".
        let camera: SlamCamera<f64> = brown(
            SVector::<f64, 12>::from([
                269.06, 269.16, 324.33, 245.22, 0.6257, 0.4661, -0.000185, -4.288e-5, 0.00417,
                0.8943, 0.5425, 0.0662,
            ]),
            0.0,
        )
        .unwrap();
        let mut uv: Vector2<f64> = Vector2::zeros();
        assert!(camera.project_point(&Vector4::new(100.0, 0.0, 1.0, 1.0), &mut uv, None));
    }

    #[test]
    fn the_variant_rejects_the_three_deferred_models() {
        let calibration: Calibration<f64> =
            Calibration::from_json_str(include_str!("../tests/fixtures/euroc_ds_calib.json"))
                .unwrap();
        assert_eq!(
            SlamCamera::from_model(&calibration.intrinsics[0]),
            Err(CameraError::UnsupportedModel { model: "ds" })
        );
    }

    #[test]
    fn the_variant_reports_the_focal_length_and_principal_point() {
        let calibration: Calibration<f64> =
            Calibration::from_json_str(include_str!("../tests/fixtures/msdmg_calib.json")).unwrap();
        let model: SlamCamera<f64> = SlamCamera::from_model(&calibration.intrinsics[0]).unwrap();
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

        // border <= u < w - border - 1.
        assert!(rig[0].in_bounds(&Vector2::new(0.0, 0.0), 0.0));
        assert!(rig[0].in_bounds(&Vector2::new(958.999, 958.999), 0.0));
        assert!(!rig[0].in_bounds(&Vector2::new(959.0, 0.0), 0.0));
        assert!(!rig[0].in_bounds(&Vector2::new(-0.001, 0.0), 0.0));
        assert!(rig[0].in_bounds(&Vector2::new(2.0, 2.0), 2.0));
        assert!(!rig[0].in_bounds(&Vector2::new(1.999, 2.0), 2.0));
        assert!(!rig[0].in_bounds(&Vector2::new(957.0, 2.0), 2.0));
    }
}
