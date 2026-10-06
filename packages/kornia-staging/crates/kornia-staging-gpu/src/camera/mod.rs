//! Brown8 cameras using the staged CPU calibration and validity contract.
pub mod brown8;
use kornia_staging_3d::camera::{BrownConrady, InvalidCalibration};

/// Number of device parameters: OpenCV Brown8's twelve values plus valid radius.
pub const BROWN8_PARAMETERS: usize = 13;

/// Validated f32 Brown8 coefficients for inline GPU projection and inversion.
/// Zero device radius means no calibrated radius bound.
#[derive(Debug, Clone, Copy)]
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

#[cfg(all(test, feature = "wgpu"))]
mod tests;

#[cfg(test)]
mod conversion_tests {
    use super::*;
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
