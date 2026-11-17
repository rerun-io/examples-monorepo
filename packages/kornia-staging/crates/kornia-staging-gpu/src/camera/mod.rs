//! Device cameras using the staged CPU calibration and validity contract.
//!
//! Each model module holds inline `#[cube]` functions over a packed f32 parameter
//! slice: `project` (a comptime flag adds the point Jacobian; Brown8 keeps a
//! separate `project_with_point_jacobian`) and `unproject`, with the status codes
//! in [`brown8`] mapped 1:1 to the CPU rejections. The packers below
//! take every derived constant from the validated CPU f32 model, so CPU and device
//! apply the same domain bounds.
pub mod brown8;
pub mod dispatch;
pub mod double_sphere;
pub mod equirect;
pub mod eucm;
pub mod kb4;
pub mod mei;
use kornia_staging_3d::camera::{BrownConrady, CameraModelKind, InvalidCalibration, MeiDistortion};

/// Number of device parameters: OpenCV Brown8's twelve values plus valid radius.
pub const BROWN8_PARAMETERS: usize = 13;

/// Validated f32 Brown8 coefficients for inline GPU projection and inversion.
/// Zero device radius means no calibrated radius bound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Brown8 {
    parameters: [f32; BROWN8_PARAMETERS],
}
impl Brown8 {
    /// Convert a validated Brown camera when every prism and tilt term is zero.
    /// Returns `None` when the camera needs the general projection path.
    pub fn from_brown_conrady(model: &BrownConrady<f32>) -> Option<Self> {
        let parameters = model.params();
        if parameters[12..].iter().any(|&term| term != 0.0) {
            return None;
        }
        let mut device = [0.0; BROWN8_PARAMETERS];
        device[..12].copy_from_slice(&parameters[..12]);
        device[12] = model.valid_radius().unwrap_or(0.0);
        Some(Self { parameters: device })
    }

    /// Validate OpenCV-order `[fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6]` parameters.
    ///
    /// # Arguments
    /// * `parameters` - Finite coefficients with positive focal lengths.
    /// * `valid_radius` - Optional positive undistorted normalized radius.
    ///
    /// # Errors
    /// Returns the staged CPU calibration error for invalid values.
    ///
    /// ```
    /// use kornia_staging_gpu::camera::Brown8;
    /// let camera = Brown8::new([400.,400.,320.,240.,0.,0.,0.,0.,0.,0.,0.,0.], None)?;
    /// assert_eq!(camera.device_parameters()[12], 0.0);
    /// # Ok::<(), kornia_staging_3d::camera::InvalidCalibration>(())
    /// ```
    pub fn new(
        parameters: [f32; 12],
        valid_radius: Option<f32>,
    ) -> Result<Self, InvalidCalibration> {
        let mut full = [0.0; 18];
        full[..12].copy_from_slice(&parameters);
        BrownConrady::new(full, valid_radius)?;
        let mut device = [0.0; BROWN8_PARAMETERS];
        device[..12].copy_from_slice(&parameters);
        device[12] = valid_radius.unwrap_or(0.0);
        Ok(Self { parameters: device })
    }
    /// Validated device coefficients for inline use by a larger compute kernel.
    pub fn device_parameters(&self) -> [f32; BROWN8_PARAMETERS] {
        self.parameters
    }
}

/// Model tag of [`DeviceCamera::Brown8`].
pub const MODEL_BROWN8: u32 = 0;
/// Model tag of [`DeviceCamera::Kb4`].
pub const MODEL_KB4: u32 = 1;
/// Model tag of [`DeviceCamera::DoubleSphere`].
pub const MODEL_DOUBLE_SPHERE: u32 = 2;
/// Model tag of [`DeviceCamera::Eucm`].
pub const MODEL_EUCM: u32 = 3;
/// Model tag of [`DeviceCamera::Equirect`].
pub const MODEL_EQUIRECT: u32 = 4;
/// Model tag of [`DeviceCamera::Mei`].
pub const MODEL_MEI: u32 = 5;

/// Packed parameter count of the model tagged `model` (`MODEL_*`).
///
/// # Panics
/// On a tag that names no model, so a host layout never sizes an unknown camera as another model.
pub const fn parameter_count(model: u32) -> usize {
    match model {
        MODEL_BROWN8 => BROWN8_PARAMETERS,
        MODEL_KB4 => kb4::PARAMETERS,
        MODEL_DOUBLE_SPHERE => double_sphere::PARAMETERS,
        MODEL_EUCM => eucm::PARAMETERS,
        MODEL_EQUIRECT => equirect::PARAMETERS,
        MODEL_MEI => mei::PARAMETERS,
        _ => panic!("unknown device camera model tag"),
    }
}

/// A camera the device functions can evaluate, with its packed parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeviceCamera {
    /// [`brown8`]: OpenCV twelve plus valid radius.
    Brown8(Brown8),
    /// [`kb4`]: `fx fy cx cy k1 k2 k3 k4 theta_max r_max`.
    Kb4([f32; kb4::PARAMETERS]),
    /// [`double_sphere`]: `fx fy cx cy xi alpha w2`.
    DoubleSphere([f32; double_sphere::PARAMETERS]),
    /// [`eucm`]: `fx fy cx cy alpha beta w`.
    Eucm([f32; eucm::PARAMETERS]),
    /// [`equirect`]: `fx fy cx cy lon_max`.
    Equirect([f32; equirect::PARAMETERS]),
    /// [`mei`]: `fx fy cx cy xi cos_half_field d0..d12`, V3 packed into the V6 terms.
    Mei([f32; mei::PARAMETERS]),
}
impl DeviceCamera {
    /// Pack a validated CPU camera, or `None` when it needs the general CPU path
    /// (pinhole, Fisheye624, Brown with prism or tilt terms: no device kernel for those).
    pub fn from_model(model: &CameraModelKind<f32>) -> Option<Self> {
        Some(match model {
            CameraModelKind::BrownConrady(model) => {
                Self::Brown8(Brown8::from_brown_conrady(model)?)
            }
            CameraModelKind::Kb4(model) => {
                let mut packed = [0.0; kb4::PARAMETERS];
                packed[..8].copy_from_slice(&model.params());
                packed[8] = model.max_angle();
                packed[9] = model.max_radius();
                Self::Kb4(packed)
            }
            CameraModelKind::DoubleSphere(model) => {
                let mut packed = [0.0; double_sphere::PARAMETERS];
                packed[..6].copy_from_slice(&model.params());
                packed[6] = model.domain_bound();
                Self::DoubleSphere(packed)
            }
            CameraModelKind::Eucm(model) => {
                let mut packed = [0.0; eucm::PARAMETERS];
                packed[..6].copy_from_slice(&model.params());
                packed[6] = model.domain_bound();
                Self::Eucm(packed)
            }
            CameraModelKind::Equirectangular(model) => Self::Equirect(model.params()),
            CameraModelKind::Mei(model) => {
                let mut packed = [0.0; mei::PARAMETERS];
                packed[..5].copy_from_slice(&model.params());
                packed[5] = model.cos_half_field();
                let d = &mut packed[6..];
                match model.distortion() {
                    MeiDistortion::V3 { k, p: [p1, p2] } => {
                        // V6's tangential `a`, `b` are V3's `p2`, `p1`.
                        d[..3].copy_from_slice(&k);
                        d[5] = p2;
                        d[6] = p1;
                    }
                    MeiDistortion::V6 { d: coefficients } => d.copy_from_slice(&coefficients),
                }
                Self::Mei(packed)
            }
            CameraModelKind::Pinhole(_) | CameraModelKind::Fisheye624(_) => return None,
        })
    }
    /// Model tag for comptime or in-kernel dispatch (`MODEL_*`).
    pub fn model_id(&self) -> u32 {
        match self {
            Self::Brown8(_) => MODEL_BROWN8,
            Self::Kb4(_) => MODEL_KB4,
            Self::DoubleSphere(_) => MODEL_DOUBLE_SPHERE,
            Self::Eucm(_) => MODEL_EUCM,
            Self::Equirect(_) => MODEL_EQUIRECT,
            Self::Mei(_) => MODEL_MEI,
        }
    }
    /// Packed parameters, in the layout the model module reads at its offset.
    pub fn device_parameters(&self) -> &[f32] {
        match self {
            Self::Brown8(camera) => &camera.parameters,
            Self::Kb4(packed) => packed,
            Self::DoubleSphere(packed) | Self::Eucm(packed) => packed,
            Self::Equirect(packed) => packed,
            Self::Mei(packed) => packed,
        }
    }
}

#[cfg(all(test, feature = "wgpu"))]
mod tests;

#[cfg(test)]
mod conversion_tests {
    use super::*;
    #[test]
    fn parameter_count_names_every_model_and_refuses_other_tags() {
        assert_eq!(
            [
                MODEL_BROWN8,
                MODEL_KB4,
                MODEL_DOUBLE_SPHERE,
                MODEL_EUCM,
                MODEL_EQUIRECT,
                MODEL_MEI
            ]
            .map(parameter_count),
            [
                BROWN8_PARAMETERS,
                kb4::PARAMETERS,
                double_sphere::PARAMETERS,
                eucm::PARAMETERS,
                equirect::PARAMETERS,
                mei::PARAMETERS
            ]
        );
        assert!(std::panic::catch_unwind(|| parameter_count(MODEL_MEI + 1)).is_err());
    }
    #[test]
    fn mei_packs_v3_into_the_v6_terms_and_the_cpu_field_bound() {
        use kornia_staging_3d::camera::Mei;
        let param = [1834.6, 1834.2, 960.242, 965.681, 2.31494];
        let distortion = MeiDistortion::V3 {
            k: [0.96, -2.06, 4.31],
            p: [0.0017, -0.0012],
        };
        let model = Mei::new(param, 100f32.to_radians(), distortion).unwrap();
        let device = DeviceCamera::from_model(&CameraModelKind::Mei(model)).unwrap();
        assert_eq!(device.model_id(), MODEL_MEI);
        let packed = device.device_parameters();
        assert_eq!(packed.len(), parameter_count(MODEL_MEI));
        assert_eq!(packed[..5], param);
        assert_eq!(packed[5], model.cos_half_field());
        assert_eq!(
            packed[6..],
            [0.96, -2.06, 4.31, 0.0, 0.0, -0.0012, 0.0017, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
        let d: [f32; 13] = std::array::from_fn(|i| i as f32 * 1e-3);
        let model = Mei::new(param, 1.5, MeiDistortion::V6 { d }).unwrap();
        let device = DeviceCamera::from_model(&CameraModelKind::Mei(model)).unwrap();
        assert_eq!(device.device_parameters()[6..], d);
    }
    #[test]
    fn prism_and_tilt_require_the_general_camera() {
        let mut parameters = [0.0; 18];
        parameters[..6].copy_from_slice(&[400.0, 410.0, 320.0, 240.0, 0.01, -0.002]);
        let model = BrownConrady::new(parameters, Some(2.0)).unwrap();
        let device = Brown8::from_brown_conrady(&model)
            .unwrap()
            .device_parameters();
        assert_eq!(device[..12], parameters[..12]);
        assert_eq!(device[12], 2.0);
        for term in 12..18 {
            let mut extra = parameters;
            extra[term] = 1e-4;
            let model = BrownConrady::new(extra, None).unwrap();
            assert!(
                Brown8::from_brown_conrady(&model).is_none(),
                "nonzero parameter {term}"
            );
        }
    }
}
