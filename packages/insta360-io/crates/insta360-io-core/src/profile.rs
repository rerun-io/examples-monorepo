//! Device profiles: the calibration of one Insta360 body in one video mode that the file does not carry.
//!
//! The profiles are measured records in `insta360_profiles.toml` (next to the workspace manifest), built into the
//! crate. A clip takes the profile whose camera type, lens type, frame rate and rolling-shutter readout match its own;
//! a clip outside every profile's measured mode is refused, naming the property that differs.
use nalgebra::{Matrix3, Vector3};
use serde::Deserialize;

/// The profile file, as built into the crate.
pub const PROFILES_TOML: &str = include_str!("../../../insta360_profiles.toml");

/// An IMU noise model.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ImuNoise {
    /// rad/s/sqrt(Hz).
    pub gyro_noise_density: f64,
    /// m/s^2/sqrt(Hz).
    pub accel_noise_density: f64,
    /// rad/s^2/sqrt(Hz).
    pub gyro_bias_random_walk: f64,
    /// m/s^3/sqrt(Hz).
    pub accel_bias_random_walk: f64,
    /// Sample rate the densities are stated for.
    pub rate_hz: f64,
    /// Where the numbers come from.
    pub source: String,
}

/// The calibration of one body in one video mode that the file does not carry.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DeviceProfile {
    /// Profile id, written into every recording read with it.
    pub name: String,
    /// Bumped whenever a measured number changes.
    pub version: u32,
    /// Where the numbers were measured.
    pub provenance: String,
    /// The clip's camera type (`Insta360 X4 Air`).
    pub camera_type: String,
    /// Lens type of the factory lens strings (131 X4 Air, 113 X5). The lens size is not a property of the profile:
    /// the parser states the factory lenses at the track resolution, so one profile serves every resolution.
    pub lens_type: u32,
    /// Frame rate of the measured mode.
    pub frame_rate_hz: f64,
    /// Largest difference from `frame_rate_hz` a clip may have.
    pub frame_rate_tolerance_hz: f64,
    /// Rolling-shutter readout of one lens image, seconds (first row to last).
    pub readout_s: f64,
    /// Largest difference between a clip's own readout and `readout_s`.
    pub readout_tolerance_s: f64,
    /// `R_P_imu` as fitted (rows: P axes in IMU coordinates); [`DeviceProfile::imu_from_pano`] orthonormalises it.
    pub pano_from_imu_measured: [[f64; 3]; 3],
    /// Vector from the IMU to the lens midpoint, metres, in P.
    pub lever_arm_m: [f64; 3],
    /// `a` of the per-frame offset `a * shutter_i + b` (shutter = exposure record 4 duration, s).
    pub frame_time_shutter_slope: f64,
    /// `b` of the same model, seconds.
    pub frame_time_constant_s: f64,
    /// The IMU noise model.
    pub imu_noise: ImuNoise,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileFile {
    profile: Vec<DeviceProfile>,
}

/// Why profiles could not be read or none fits a clip.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ProfileError {
    /// The profile text is not TOML of the profile schema, or states impossible numbers.
    #[error("not an Insta360 device profile file: {0}")]
    Malformed(String),
    /// No profile was measured for the clip's camera type.
    #[error("no device profile for a {camera_type:?} (profiles: {names:?})")]
    UnknownCamera {
        /// The clip's camera type.
        camera_type: String,
        /// Every profile name, sorted.
        names: Vec<String>,
    },
    /// Profiles of the camera type exist, but the clip lies outside each one's measured mode.
    #[error("the clip is outside every measured {camera_type:?} mode: {}", reasons.join(" | "))]
    OutsideMode {
        /// The clip's camera type.
        camera_type: String,
        /// `<profile>: <difference>; ...` per profile of that camera type.
        reasons: Vec<String>,
    },
}

/// Parse a profile file.
///
/// # Errors
/// [`ProfileError::Malformed`] on text that is not the schema, a negative tolerance, or a non-positive rate or readout.
pub fn parse_profiles(text: &str) -> Result<Vec<DeviceProfile>, ProfileError> {
    let file: ProfileFile =
        toml::from_str(text).map_err(|e| ProfileError::Malformed(e.to_string()))?;
    for p in &file.profile {
        if p.frame_rate_tolerance_hz < 0.0 || p.readout_tolerance_s < 0.0 {
            return Err(ProfileError::Malformed(format!(
                "profile {}: a tolerance is negative",
                p.name
            )));
        }
        if p.frame_rate_hz <= 0.0 || p.readout_s <= 0.0 {
            return Err(ProfileError::Malformed(format!(
                "profile {}: frame rate and readout must be positive",
                p.name
            )));
        }
    }
    Ok(file.profile)
}

/// The profiles built into the crate.
pub fn builtin_profiles() -> Vec<DeviceProfile> {
    parse_profiles(PROFILES_TOML).expect("insta360_profiles.toml is checked by the crate's tests")
}

/// What a clip states about the mode it was recorded in: the properties a profile is selected by.
#[derive(Clone, Debug, PartialEq)]
pub struct CaptureMode {
    /// The clip's camera type.
    pub camera_type: String,
    /// The lens type both emitted lenses state (V6 lenses when the clip has them, else V3).
    pub lens_type: u32,
    /// From the median exposure-record-4 stamp interval.
    pub frame_rate_hz: f64,
    /// The clip's own rolling-shutter readout, seconds; `None` when the clip does not state one.
    pub readout_s: Option<f64>,
}

impl DeviceProfile {
    /// Each property in which `mode` lies outside what this profile was measured on, as a readable reason.
    pub fn differences(&self, mode: &CaptureMode) -> Vec<String> {
        let mut out = Vec::new();
        if mode.camera_type != self.camera_type {
            out.push(format!(
                "camera type {:?} is not {:?}",
                mode.camera_type, self.camera_type
            ));
        }
        if mode.lens_type != self.lens_type {
            out.push(format!(
                "lens type {} is not {}",
                mode.lens_type, self.lens_type
            ));
        }
        if (mode.frame_rate_hz - self.frame_rate_hz).abs() > self.frame_rate_tolerance_hz {
            out.push(format!(
                "frame rate {:.4} Hz is not {} +- {} Hz",
                mode.frame_rate_hz, self.frame_rate_hz, self.frame_rate_tolerance_hz
            ));
        }
        if let Some(readout) = mode.readout_s {
            if (readout - self.readout_s).abs() > self.readout_tolerance_s {
                out.push(format!(
                    "rolling-shutter readout {:.4} ms is not {:.4} +- {:.4} ms",
                    readout * 1e3,
                    self.readout_s * 1e3,
                    self.readout_tolerance_s * 1e3
                ));
            }
        }
        out
    }

    /// `R_imu_P`: the transpose of the rotation nearest to `pano_from_imu_measured` (SVD polar factor, det +1).
    pub fn imu_from_pano(&self) -> Matrix3<f64> {
        let m = Matrix3::from_fn(|r, c| self.pano_from_imu_measured[r][c]);
        orthonormal(&m).transpose()
    }

    /// The lever arm as a vector in P, metres.
    pub fn lever_arm(&self) -> Vector3<f64> {
        Vector3::from(self.lever_arm_m)
    }

    /// Each frame's effective time on the IMU clock, ns: the record-4 stamp plus `slope * shutter + constant`.
    ///
    /// # Errors
    /// The model makes the times non-increasing.
    pub fn frame_times_ns(
        &self,
        record_t_us: &[i64],
        shutter_s: &[f64],
    ) -> Result<Vec<i64>, String> {
        let times: Vec<i64> = record_t_us
            .iter()
            .zip(shutter_s)
            .map(|(&t, &s)| {
                t * 1000
                    + ((self.frame_time_shutter_slope * s + self.frame_time_constant_s) * 1e9)
                        .round() as i64
            })
            .collect();
        if times.windows(2).any(|w| w[1] <= w[0]) {
            return Err("the per-frame time model makes frame times non-increasing".into());
        }
        Ok(times)
    }
}

/// The profile measured in `mode`.
///
/// # Errors
/// [`ProfileError::UnknownCamera`] without a profile of the camera type, else [`ProfileError::OutsideMode`] naming,
/// per profile of that type, what differs.
pub fn select_profile<'a>(
    profiles: &'a [DeviceProfile],
    mode: &CaptureMode,
) -> Result<&'a DeviceProfile, ProfileError> {
    let mut reasons = Vec::new();
    for profile in profiles {
        let differences = profile.differences(mode);
        if differences.is_empty() {
            return Ok(profile);
        }
        if profile.camera_type == mode.camera_type {
            reasons.push(format!("{}: {}", profile.name, differences.join("; ")));
        }
    }
    if reasons.is_empty() {
        let mut names: Vec<String> = profiles.iter().map(|p| p.name.clone()).collect();
        names.sort();
        return Err(ProfileError::UnknownCamera {
            camera_type: mode.camera_type.clone(),
            names,
        });
    }
    Err(ProfileError::OutsideMode {
        camera_type: mode.camera_type.clone(),
        reasons,
    })
}

/// The rotation nearest to `m` (SVD polar factor, determinant +1).
pub fn orthonormal(m: &Matrix3<f64>) -> Matrix3<f64> {
    let svd = m.svd(true, true);
    let (u, vt) = (svd.u.expect("u"), svd.v_t.expect("v_t"));
    let r = u * vt;
    if r.determinant() < 0.0 {
        u * Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, -1.0)) * vt
    } else {
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn mode(camera_type: &str, lens_type: u32, rate: f64, readout: Option<f64>) -> CaptureMode {
        CaptureMode {
            camera_type: camera_type.into(),
            lens_type,
            frame_rate_hz: rate,
            readout_s: readout,
        }
    }

    #[test]
    fn the_builtin_profiles_parse_and_name_x4_air_and_x5() {
        let names: Vec<String> = builtin_profiles().into_iter().map(|p| p.name).collect();
        assert_eq!(names, ["x4-air", "x5"]);
    }

    #[test]
    fn a_clip_selects_its_profile_by_camera_lens_rate_and_readout() {
        let profiles = builtin_profiles();
        let x4 = select_profile(
            &profiles,
            &mode("Insta360 X4 Air", 131, 29.9697, Some(0.015883)),
        )
        .unwrap();
        assert_eq!(x4.name, "x4-air");
        let x5 = select_profile(
            &profiles,
            &mode("Insta360 X5", 113, 29.9688, Some(0.021244)),
        )
        .unwrap();
        assert_eq!((x5.name.as_str(), x5.readout_s), ("x5", 0.021244));
        // A clip that states no readout is not refused for it.
        assert!(select_profile(&profiles, &mode("Insta360 X5", 113, 29.9688, None)).is_ok());
    }

    #[test]
    fn a_clip_outside_every_mode_is_refused_naming_the_difference() {
        let profiles = builtin_profiles();
        let err = select_profile(
            &profiles,
            &mode("Insta360 X4 Air", 131, 59.94, Some(0.015883)),
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("x4-air: frame rate 59.9400 Hz is not 29.97 +- 0.05 Hz"),
            "{err}"
        );
        let err = select_profile(&profiles, &mode("Insta360 X4 Air", 113, 29.97, Some(0.02)))
            .unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains("lens type 113 is not 131")
                && text.contains("rolling-shutter readout 20.0000 ms"),
            "{text}"
        );
        let err = select_profile(&profiles, &mode("Insta360 X3", 1, 29.97, None)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "no device profile for a \"Insta360 X3\" (profiles: [\"x4-air\", \"x5\"])"
        );
    }

    #[test]
    fn malformed_profile_text_is_refused() {
        assert!(matches!(
            parse_profiles("profile = 3"),
            Err(ProfileError::Malformed(_))
        ));
        let negative = PROFILES_TOML.replacen(
            "frame_rate_tolerance_hz = 0.05",
            "frame_rate_tolerance_hz = -0.05",
            1,
        );
        assert_eq!(
            parse_profiles(&negative).unwrap_err(),
            ProfileError::Malformed("profile x4-air: a tolerance is negative".into())
        );
        let unknown = PROFILES_TOML.replacen("version = 1", "version = 1\ncolour = 3", 1);
        assert!(matches!(
            parse_profiles(&unknown),
            Err(ProfileError::Malformed(_))
        ));
    }

    #[test]
    fn the_x4_air_rotation_is_the_measured_permutation() {
        let profile = &builtin_profiles()[0];
        let r = profile.imu_from_pano();
        // x_P = -z_imu, y_P = +x_imu, z_P = -y_imu, within 0.2 deg.
        let nominal = Matrix3::new(0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0).transpose();
        let angle = ((r.transpose() * nominal).trace() - 1.0).clamp(-2.0, 2.0) / 2.0;
        assert!(angle.acos().to_degrees() < 0.2);
    }

    #[test]
    fn a_hand_written_integer_widens_to_a_float() {
        let text = PROFILES_TOML.replacen(
            "frame_time_constant_s = -0.0001",
            "frame_time_constant_s = 0",
            1,
        );
        assert_eq!(parse_profiles(&text).unwrap()[0].frame_time_constant_s, 0.0);
    }

    #[test]
    fn frames_land_on_the_middle_row_time() {
        // -0.44 * shutter - 0.1 ms, on the stamp.
        let times = builtin_profiles()[0]
            .frame_times_ns(&[1_000_000, 1_033_367, 1_066_733], &[0.01, 0.002, 0.0])
            .unwrap();
        assert_eq!(
            times,
            [
                1_000_000_000 - 4_500_000,
                1_033_367_000 - 980_000,
                1_066_733_000 - 100_000
            ]
        );
    }

    #[test]
    fn non_increasing_frame_times_are_refused() {
        let profile = &builtin_profiles()[0];
        // A 40 ms shutter moves the second frame 17.6 ms back, before the first, whose stamp is 1 ms earlier.
        assert!(profile.frame_times_ns(&[0, 1000], &[0.0, 0.04]).is_err());
        assert_eq!(
            profile
                .frame_times_ns(&[0, 33_367], &[0.002, 0.002])
                .unwrap()
                .len(),
            2
        );
    }

    proptest! {
        /// The polar factor is a rotation and leaves a rotation unchanged.
        #[test]
        fn orthonormal_is_a_rotation(values in prop::array::uniform9(-1.0f64..1.0)) {
            let m = Matrix3::from_row_slice(&values);
            prop_assume!(m.determinant().abs() > 1e-3);
            let r = orthonormal(&m);
            prop_assert!((r.transpose() * r - Matrix3::identity()).norm() < 1e-9);
            prop_assert!((r.determinant() - 1.0).abs() < 1e-9);
            prop_assert!((orthonormal(&r) - r).norm() < 1e-9);
        }

        /// Frame times move each stamp by exactly the model's shift, rounded to the nanosecond.
        #[test]
        fn frame_times_are_stamp_plus_model(shutter in prop::collection::vec(0.0001f64..0.01, 2..50)) {
            let profile = &builtin_profiles()[0];
            let stamps: Vec<i64> = (0..shutter.len() as i64).map(|i| 1_000_000 + i * 33_367).collect();
            let times = profile.frame_times_ns(&stamps, &shutter).unwrap();
            for ((t, s), stamp) in times.iter().zip(&shutter).zip(&stamps) {
                let shift = (profile.frame_time_shutter_slope * s + profile.frame_time_constant_s) * 1e9;
                prop_assert_eq!(*t, stamp * 1000 + shift.round() as i64);
            }
        }
    }
}
