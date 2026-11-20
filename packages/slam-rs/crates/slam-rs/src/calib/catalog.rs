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
    /// The camera node's `rolling_shutter_readout_s`, when the camera states one.
    pub rolling_shutter_readout_s: Option<f64>,
}

/// Catalog noise densities, random walks, frequency, and the already-applied clock shift.
pub struct ImuStatics {
    pub rate_hz: Option<f64>,
    pub gyro_noise_density: Option<f64>,
    pub accel_noise_density: Option<f64>,
    pub gyro_bias_random_walk: Option<f64>,
    pub accel_bias_random_walk: Option<f64>,
    pub applied_time_shift_ns: i64,
    /// The IMU node's `gravity_ms2`, when the recording states one.
    pub gravity_ms2: Option<f64>,
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
            // fx fy cx cy come from the pinhole matrix; the coefficients follow them.
            "double_sphere" => ("ds", 2),
            "eucm" => ("eucm", 2),
            "equirectangular" => ("equirectangular", 1),
            // xi, half_field_rad, then the factory distortion (CONTRACT.md order).
            "mei_v3" => ("mei_v3", 7),
            "mei_v6" => ("mei_v6", 15),
            // gamma_x, gamma_y, then as above: the model's own focal rides in the coefficients, and the Pinhole
            // holds the paraxial gamma / (1 + xi) as the viewer's approximation only (exoego:v2 section 4).
            "mei_v3_unified" => ("mei_v3", 9),
            "mei_v6_unified" => ("mei_v6", 17),
            other => return Err(invalid(format!("unsupported distortion model {other:?}"))),
        };
        let unified = raw.distortion_model.ends_with("_unified");
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
        let (fx, fy, first) = if unified {
            (raw.distortion_coefficients[0], raw.distortion_coefficients[1], 2)
        } else {
            (k[0], k[4], 0)
        };
        Ok(Self {
            width: w / downscale as u32,
            height: h / downscale as u32,
            fx: fx / d,
            fy: fy / d,
            cx: (k[6] + 0.5) / d - 0.5,
            cy: (k[7] + 0.5) / d - 0.5,
            model: model.into(),
            distortion: raw.distortion_coefficients[first..count].to_vec(),
            distortion_valid_radius: raw.distortion_valid_radius,
            readout_s: raw.rolling_shutter_readout_s,
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
            gravity_ms2: raw.gravity_ms2,
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
            rolling_shutter_readout_s: None,
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

    #[test]
    fn both_mei_names_give_the_same_camera() -> Result<(), CatalogError> {
        // The X4 Air's lens 0: gamma 1835.37 / 1835.01, xi 2.31494, half field 100 deg, mei_v6 factory terms.
        let (gx, gy, xi) = (1835.37, 1835.01, 2.31494);
        let sphere = vec![
            xi, 1.745_329_2, 0.94, -1.68, 2.19, 3.86, 0.0, -0.005, 0.0016, -0.0033, 0.004, 0.0083, -0.0013,
            0.024, -0.0014,
        ];
        let statics = |model: &str, coefficients: Vec<f64>, fx: f64, fy: f64| CameraStatics {
            distortion_model: model.into(),
            distortion_coefficients: coefficients,
            image_from_camera: [fx, 0.0, 0.0, 0.0, fy, 0.0, 958.29, 965.87, 1.0],
            resolution_wh: [1920.0, 1920.0],
            transform_mat3x3: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            transform_translation: [0.0; 3],
            transform_relation: 2,
            distortion_valid_radius: None,
            rolling_shutter_readout_s: None,
        };
        let old = statics("mei_v6", sphere.clone(), gx, gy);
        let mut unified_coefficients = vec![gx, gy];
        unified_coefficients.extend(&sphere);
        // The Pinhole states the paraxial focal; it must not reach the estimator.
        let unified = statics("mei_v6_unified", unified_coefficients, gx / (1.0 + xi), gy / (1.0 + xi));
        for downscale in [1, 2] {
            let a = CameraParts::from_catalog_statics(0, &old, downscale)?;
            let b = CameraParts::from_catalog_statics(0, &unified, downscale)?;
            assert_eq!((a.model.as_str(), b.model.as_str()), ("mei_v6", "mei_v6"));
            assert_eq!((a.fx, a.fy, a.cx, a.cy), (b.fx, b.fy, b.cx, b.cy));
            assert_eq!((b.fx, b.fy), (gx / downscale as f64, gy / downscale as f64));
            assert_eq!(a.distortion, b.distortion);
            assert_eq!(b.distortion, sphere);
        }
        let mut short = unified;
        short.distortion_coefficients.pop();
        assert!(CameraParts::from_catalog_statics(0, &short, 1).is_err());
        Ok(())
    }

    #[test]
    fn catalog_wide_angle_and_panorama_cameras_reach_the_staged_models() -> Result<(), CatalogError>
    {
        let pi = std::f64::consts::PI;
        for (name, coefficients, model) in [
            ("double_sphere", vec![-0.17, 0.59, 0.0, 0.0], "ds"),
            ("eucm", vec![0.63, 1.04, 0.0], "eucm"),
            ("equirectangular", vec![pi, 0.0], "equirectangular"),
        ] {
            let raw = CameraStatics {
                distortion_model: name.into(),
                distortion_coefficients: coefficients.clone(),
                image_from_camera: [305.0, 0.0, 0.0, 0.0, 305.0, 0.0, 959.5, 479.5, 1.0],
                resolution_wh: [1920.0, 960.0],
                transform_mat3x3: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                transform_translation: [0.0; 3],
                transform_relation: 2,
                distortion_valid_radius: None,
                rolling_shutter_readout_s: None,
            };
            let camera = CameraParts::from_catalog_statics(0, &raw, 2)?;
            assert_eq!((camera.model.as_str(), camera.cx), (model, 479.5));
            let failed = |e: &dyn std::error::Error| CatalogError(e.to_string());
            let basalt =
                super::super::camera_model_from_parts(&camera, 0).map_err(|e| failed(&e))?;
            let slam = crate::camera::SlamCamera::from_model(&basalt).map_err(|e| failed(&e))?;
            assert_eq!(slam.name(), model);
            let mut live = raw;
            live.distortion_coefficients[coefficients.len() - 1] = 0.5;
            assert!(CameraParts::from_catalog_statics(0, &live, 2).is_err());
        }
        Ok(())
    }
}
