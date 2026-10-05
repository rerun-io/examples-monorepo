//! Catalog statics and their conversion to estimator calibration.
use super::{CameraParts, ImuParts};

/// A catalog input that cannot be converted or associated.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct CatalogError(pub String);

/// Untouched camera statics: matrices are column-major and the pose is camera-from-IMU.
pub struct CameraStatics {
    pub distortion_model: String,
    pub distortion_coefficients: Vec<f64>,
    pub image_from_camera: [f64; 9],
    pub resolution_wh: [f64; 2],
    pub transform_mat3x3: [f64; 9],
    pub transform_translation: [f64; 3],
    pub transform_relation: i64,
    pub distortion_valid_radius: Option<f64>,
}

/// Catalog noise densities, random walks, frequency, and the already-applied clock shift.
pub struct ImuStatics {
    pub rate_hz: Option<f64>,
    pub gyro_noise_density: Option<f64>,
    pub accel_noise_density: Option<f64>,
    pub gyro_bias_random_walk: Option<f64>,
    pub accel_bias_random_walk: Option<f64>,
    pub applied_time_shift_ns: i64,
}

impl CameraParts<f64> {
    /// Convert raw catalog matrices, intrinsics and distortion coefficients once.
    pub fn from_catalog_statics(
        index: usize,
        raw: &CameraStatics,
        downscale: i64,
    ) -> Result<Self, CatalogError> {
        let invalid = |message: String| CatalogError(format!("cam_{index:02}: {message}"));
        if downscale < 1 {
            return Err(invalid(format!(
                "downscale must be at least 1; got {downscale}"
            )));
        }
        let [w, h] = raw.resolution_wh.map(|v| v as u32);
        let d = downscale as u64;
        if u64::from(w) / d < 1 || u64::from(h) / d < 1 {
            return Err(invalid(format!(
                "downscale {downscale} leaves nothing of the {w}x{h} frame"
            )));
        }
        let (model, count) = match raw.distortion_model.as_str() {
            "kannala_brandt" => ("kb4", 4),
            "brown_conrady" => ("radtan8", 8),
            other => return Err(invalid(format!("unsupported distortion model {other:?}"))),
        };
        if raw.distortion_coefficients.len() < count {
            return Err(invalid(format!(
                "{model} needs {count} coefficients, got {}",
                raw.distortion_coefficients.len()
            )));
        }
        if raw.distortion_coefficients[count..]
            .iter()
            .any(|v| !v.is_finite() || v.abs() > 1e-8)
        {
            return Err(invalid(format!(
                "{model} uses {count} coefficients but the tail is non-zero"
            )));
        }
        if raw.transform_relation != 2 {
            return Err(invalid(format!(
                "Transform3D relation {} is not ChildFromParent(2)",
                raw.transform_relation
            )));
        }
        let mut pose = [0.0; 16];
        for row in 0..3 {
            for col in 0..3 {
                pose[row * 4 + col] = raw.transform_mat3x3[row * 3 + col];
            }
            pose[row * 4 + 3] = -(0..3)
                .map(|col| pose[row * 4 + col] * raw.transform_translation[col])
                .sum::<f64>();
        }
        pose[15] = 1.0;
        let d = downscale as f64;
        let k = raw.image_from_camera;
        Ok(Self {
            width: w / downscale as u32,
            height: h / downscale as u32,
            fx: k[0] / d,
            fy: k[4] / d,
            cx: (k[6] + 0.5) / d - 0.5,
            cy: (k[7] + 0.5) / d - 0.5,
            model: model.into(),
            distortion: raw.distortion_coefficients[..count].to_vec(),
            distortion_valid_radius: raw.distortion_valid_radius,
            imu_t_cam_row_major: pose,
        })
    }
}

impl ImuParts<f64> {
    /// Undo the ingestion shift once; all streams use the returned common offset.
    pub fn from_catalog_statics(raw: &ImuStatics) -> Result<Self, CatalogError> {
        let (
            Some(frequency_hz),
            Some(gyro_noise_std),
            Some(accel_noise_std),
            Some(gyro_bias_std),
            Some(accel_bias_std),
        ) = (
            raw.rate_hz,
            raw.gyro_noise_density,
            raw.accel_noise_density,
            raw.gyro_bias_random_walk,
            raw.accel_bias_random_walk,
        )
        else {
            return Err(CatalogError("missing IMU calibration: require rate_hz, gyro_noise_density, accel_noise_density, gyro_bias_random_walk and accel_bias_random_walk; ingest or backfill the catalog calibration".into()));
        };
        Ok(Self {
            frequency_hz,
            gyro_noise_std,
            accel_noise_std,
            gyro_bias_std,
            accel_bias_std,
            cam_time_offset_ns: raw
                .applied_time_shift_ns
                .checked_neg()
                .ok_or_else(|| CatalogError("IMU time shift overflow".into()))?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_camera_inverts_column_major_extrinsics_and_scales_pixel_centres()
    -> Result<(), CatalogError> {
        let raw = CameraStatics {
            distortion_model: "kannala_brandt".into(),
            distortion_coefficients: vec![0.1, 0.2, 0.3, 0.4, 0.0],
            image_from_camera: [600.0, 0.0, 0.0, 0.0, 603.0, 0.0, 959.5, 539.5, 1.0],
            resolution_wh: [1920.0, 1080.0],
            transform_mat3x3: [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            transform_translation: [2.0, 3.0, 4.0],
            transform_relation: 2,
            distortion_valid_radius: None,
        };
        let camera = CameraParts::from_catalog_statics(0, &raw, 3)?;
        assert_eq!(
            (
                camera.width,
                camera.height,
                camera.fx,
                camera.fy,
                camera.cx,
                camera.cy
            ),
            (640, 360, 200.0, 201.0, 319.5, 179.5)
        );
        assert_eq!(
            camera.imu_t_cam_row_major,
            [
                0.0, 1.0, 0.0, -3.0, -1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, -4.0, 0.0, 0.0, 0.0, 1.0
            ]
        );
        let mut bad = raw;
        bad.distortion_coefficients[4] = 1.0;
        assert!(CameraParts::from_catalog_statics(0, &bad, 3).is_err());
        Ok(())
    }
}
