//! basalt's camera-IMU calibration, read with serde.
//!
//! Ported from `thirdparty/basalt-headers/include/basalt/calibration/calibration.hpp`
//! and `calib_bias.hpp`. The on-disk shape is cereal's again: a `value0`
//! wrapper, `T_imu_cam` as `{px, py, pz, qx, qy, qz, qw}`
//! (`serialization/eigen_io.h:148-153`) and `intrinsics` as
//! `{"camera_type": ..., "intrinsics": {...}}`
//! (`serialization/headers_serialization.h:55-72`), so every shipped
//! `*_calib.json` is a free test fixture (decision D18).
//!
//! Projection math is **not** here. This PR stores the camera parameters in the
//! shape basalt stores them and stops; `project`/`unproject` and their analytic
//! Jacobians land with the camera module.
//!
//! ## Two deliberate divergences
//!
//! * **The quaternion is normalized on read.** cereal writes the four
//!   coefficients straight into Eigen's storage without normalizing; a
//!   calibration a few ulps off unit length would then be a non-rotation. The
//!   port normalizes and rejects a zero-norm quaternion instead of producing
//!   NaNs (decision D32).
//! * **Per-camera resolution is real data.** basalt assumes every camera shares
//!   `resolution[0]` (`frame_to_frame_optical_flow.h:108-109`); the port keeps
//!   one entry per camera because the msd-g2 recordings are stored rotated into
//!   portrait (decision D30).
//!
//! ## The fixtures
//!
//! `msdmi_calib.json` (2 kb4 cameras, Valve Index), `msdmg_calib.json` (4
//! pinhole-radtan8, HP Reverb G2), `euroc_ds_calib.json` (2 double-sphere, and
//! the only shipped file with a vignette spline and mocap keys) and
//! `robocap-basalt-calib.json` (4 kb4 at 960x540, the only one with non-default
//! IMU noise).

use std::collections::BTreeMap;

use nalgebra::{Matrix3, Vector3};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::lie::{LieScalar, Se3, So3};

/// Something that went wrong reading a calibration.
#[derive(Debug, thiserror::Error)]
pub enum CalibError {
    /// The text is not the JSON the struct expects.
    #[error("could not parse the calibration: {0}")]
    Parse(#[from] serde_json::Error),
    /// A stored quaternion has no length, so it names no rotation.
    #[error("camera {index}: the stored quaternion has zero length")]
    DegenerateQuaternion {
        /// Index of the offending camera.
        index: usize,
    },
    /// `T_imu_cam`, `intrinsics` and `resolution` must have one entry per camera.
    #[error(
        "calibration is ragged: {extrinsics} extrinsics, {intrinsics} intrinsics, {resolutions} resolutions"
    )]
    RaggedCameraLists {
        /// Number of `T_imu_cam` entries.
        extrinsics: usize,
        /// Number of `intrinsics` entries.
        intrinsics: usize,
        /// Number of `resolution` entries.
        resolutions: usize,
    },
    /// A camera model name that is not one of the six basalt ships in its
    /// calibration files.
    #[error("camera {index}: unknown camera model {model:?}")]
    UnknownCameraModel {
        /// Index of the offending camera.
        index: usize,
        /// The name that was given.
        model: String,
    },
    /// A model was given the wrong number of distortion coefficients.
    #[error("camera {index}: model {model} wants {expected} distortion coefficients, got {actual}")]
    WrongCoefficientCount {
        /// Index of the offending camera.
        index: usize,
        /// The model that was asked for.
        model: &'static str,
        /// How many coefficients it needs.
        expected: usize,
        /// How many were supplied.
        actual: usize,
    },
    /// A 4x4 pose whose rotation block is not a rotation.
    #[error("camera {index}: the rotation block of imu_T_cam is not orthonormal")]
    NonRotation {
        /// Index of the offending camera.
        index: usize,
    },
    /// A 4x4 pose whose rotation block is orthonormal but mirrors the frame.
    #[error("camera {index}: the rotation block of imu_T_cam is a reflection, not a rotation")]
    ReflectedRotation {
        /// Index of the offending camera.
        index: usize,
    },
}

/// cereal's outer wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Value0<T> {
    #[serde(rename = "value0")]
    value0: T,
}

/// A rigid transform in basalt's on-disk form.
///
/// The rotation is stored `xyzw`, which is Eigen's internal quaternion order
/// (`serialization/eigen_io.h:150-153`), not the `wxyz` a maths text would use.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct PoseJson<S> {
    px: S,
    py: S,
    pz: S,
    qx: S,
    qy: S,
    qz: S,
    qw: S,
}

impl<S: LieScalar> PoseJson<S> {
    fn to_se3(self, index: usize) -> Result<Se3<S>, CalibError> {
        let rotation: So3<S> = So3::from_quaternion_xyzw(self.qx, self.qy, self.qz, self.qw)
            .ok_or(CalibError::DegenerateQuaternion { index })?;
        Ok(Se3::new(rotation, Vector3::new(self.px, self.py, self.pz)))
    }

    fn from_se3(pose: &Se3<S>) -> Self {
        let [qx, qy, qz, qw] = pose.rotation.quaternion_xyzw();
        Self {
            px: pose.translation.x,
            py: pose.translation.y,
            pz: pose.translation.z,
            qx,
            qy,
            qz,
            qw,
        }
    }
}

/// Serde for a `Vector3` stored as a three-element JSON array.
mod vector3_json {
    use super::{DeserializeOwned, Serialize, Vector3};
    use nalgebra::Scalar;
    use serde::{Deserializer, Serializer};

    pub(super) fn serialize<S, Ser>(value: &Vector3<S>, ser: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        S: Scalar + Copy + Serialize,
        Ser: Serializer,
    {
        [value.x, value.y, value.z].serialize(ser)
    }

    pub(super) fn deserialize<'de, S, De>(de: De) -> Result<Vector3<S>, De::Error>
    where
        S: Scalar + Copy + DeserializeOwned,
        De: Deserializer<'de>,
    {
        let [x, y, z]: [S; 3] = serde::Deserialize::deserialize(de)?;
        Ok(Vector3::new(x, y, z))
    }
}

/// Pinhole, `fx fy cx cy` (`camera/pinhole_camera.hpp:88`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PinholeParams<S> {
    /// Focal length along image x, pixels.
    pub fx: S,
    /// Focal length along image y, pixels.
    pub fy: S,
    /// Principal point x, pixels.
    pub cx: S,
    /// Principal point y, pixels.
    pub cy: S,
}

/// Kannala-Brandt with four radial terms (`camera/kannala_brandt_camera4.hpp:91`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Kb4Params<S> {
    /// Focal length along image x, pixels.
    pub fx: S,
    /// Focal length along image y, pixels.
    pub fy: S,
    /// Principal point x, pixels.
    pub cx: S,
    /// Principal point y, pixels.
    pub cy: S,
    /// First radial coefficient.
    pub k1: S,
    /// Second radial coefficient.
    pub k2: S,
    /// Third radial coefficient.
    pub k3: S,
    /// Fourth radial coefficient.
    pub k4: S,
}

/// Pinhole with the eight-term rational Brown-Conrady distortion
/// (`camera/pinhole_radtan8_camera.hpp:114`).
///
/// The parameter order on disk is `k1 k2 p1 p2 k3 k4 k5 k6`: the two tangential
/// terms sit **between** the radial ones, matching OpenCV's layout and
/// `serialization/headers_serialization.h:144-153`. `rpmax` is stored beside
/// the twelve optimized parameters rather than inside them.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Radtan8Params<S> {
    /// Focal length along image x, pixels.
    pub fx: S,
    /// Focal length along image y, pixels.
    pub fy: S,
    /// Principal point x, pixels.
    pub cx: S,
    /// Principal point y, pixels.
    pub cy: S,
    /// First numerator radial coefficient.
    pub k1: S,
    /// Second numerator radial coefficient.
    pub k2: S,
    /// First tangential coefficient.
    pub p1: S,
    /// Second tangential coefficient.
    pub p2: S,
    /// Third numerator radial coefficient.
    pub k3: S,
    /// First denominator radial coefficient.
    pub k4: S,
    /// Second denominator radial coefficient.
    pub k5: S,
    /// Third denominator radial coefficient.
    pub k6: S,
    /// Largest projectable radius; beyond it the rational model turns over.
    pub rpmax: S,
}

/// Double sphere (`camera/double_sphere_camera.hpp:89`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DoubleSphereParams<S> {
    /// Focal length along image x, pixels.
    pub fx: S,
    /// Focal length along image y, pixels.
    pub fy: S,
    /// Principal point x, pixels.
    pub cx: S,
    /// Principal point y, pixels.
    pub cy: S,
    /// Offset between the two projection spheres.
    pub xi: S,
    /// Blend between the two spheres.
    pub alpha: S,
}

/// Extended unified (`camera/extended_camera.hpp:90`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ExtendedUnifiedParams<S> {
    /// Focal length along image x, pixels.
    pub fx: S,
    /// Focal length along image y, pixels.
    pub fy: S,
    /// Principal point x, pixels.
    pub cx: S,
    /// Principal point y, pixels.
    pub cy: S,
    /// Blend parameter.
    pub alpha: S,
    /// Ellipsoid shape parameter.
    pub beta: S,
}

/// Unified / Mei (`camera/unified_camera.hpp:89`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UnifiedParams<S> {
    /// Focal length along image x, pixels.
    pub fx: S,
    /// Focal length along image y, pixels.
    pub fy: S,
    /// Principal point x, pixels.
    pub cx: S,
    /// Principal point y, pixels.
    pub cy: S,
    /// Blend parameter.
    pub alpha: S,
}

/// One camera's projection model, `basalt::GenericCamera`
/// (`camera/generic_camera.hpp:75-77`).
///
/// The six variants are the ones that appear in shipped calibration files.
/// basalt's variant also holds `fisheye624`, which no reference calibration
/// uses; it is left out until a dataset needs it.
///
/// **Three of the six parse and are then refused.** `ds`, `eucm` and `ucm`
/// have no projection in this port — D13 keeps the pinhole, kb4 and
/// pinhole-radtan8 the shipped rigs use — so
/// [`crate::camera::CameraEnum::from_model`] answers them with
/// `CameraError::UnsupportedModel`. They are modelled here rather than left to
/// serde so that a EuRoC-shaped calibration reports *which* model cannot be
/// used, instead of a parse error naming a field; `euroc_ds_calib.json` is the
/// fixture that pins it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "camera_type", content = "intrinsics")]
pub enum CameraModel<S> {
    /// `pinhole`.
    #[serde(rename = "pinhole")]
    Pinhole(PinholeParams<S>),
    /// `kb4`.
    #[serde(rename = "kb4")]
    Kb4(Kb4Params<S>),
    /// `pinhole-radtan8`.
    #[serde(rename = "pinhole-radtan8")]
    PinholeRadtan8(Radtan8Params<S>),
    /// `ds`.
    #[serde(rename = "ds")]
    DoubleSphere(DoubleSphereParams<S>),
    /// `eucm`.
    #[serde(rename = "eucm")]
    ExtendedUnified(ExtendedUnifiedParams<S>),
    /// `ucm`.
    #[serde(rename = "ucm")]
    Unified(UnifiedParams<S>),
}

impl<S: Copy> CameraModel<S> {
    /// The model name basalt writes into `camera_type`, `getName()`.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Pinhole(_) => "pinhole",
            Self::Kb4(_) => "kb4",
            Self::PinholeRadtan8(_) => "pinhole-radtan8",
            Self::DoubleSphere(_) => "ds",
            Self::ExtendedUnified(_) => "eucm",
            Self::Unified(_) => "ucm",
        }
    }

    /// The optimized parameters in basalt's `getParam()` order.
    ///
    /// `fx fy cx cy` first in every model, then the model's own terms; the
    /// distortion coefficients alone are `params()[4..]`. For
    /// `pinhole-radtan8` this is the twelve-vector basalt optimizes, and
    /// `rpmax` is not in it: it is a fixed bound, which
    /// [`CameraModel::valid_radius`] reports instead
    /// (`serialization/headers_serialization.h:144-153`).
    pub fn params(&self) -> Vec<S> {
        match self {
            Self::Pinhole(p) => vec![p.fx, p.fy, p.cx, p.cy],
            Self::Kb4(p) => vec![p.fx, p.fy, p.cx, p.cy, p.k1, p.k2, p.k3, p.k4],
            Self::PinholeRadtan8(p) => vec![
                p.fx, p.fy, p.cx, p.cy, p.k1, p.k2, p.p1, p.p2, p.k3, p.k4, p.k5, p.k6,
            ],
            Self::DoubleSphere(p) => vec![p.fx, p.fy, p.cx, p.cy, p.xi, p.alpha],
            Self::ExtendedUnified(p) => vec![p.fx, p.fy, p.cx, p.cy, p.alpha, p.beta],
            Self::Unified(p) => vec![p.fx, p.fy, p.cx, p.cy, p.alpha],
        }
    }

    /// `rpmax` for `pinhole-radtan8`, which is the only model that has one.
    ///
    /// No production caller — `rpmax` reaches the projection through
    /// [`crate::camera::CameraEnum::from_model`] — and the only reader of the
    /// parsed field on this type, which is what makes a calibration's `rpmax`
    /// checkable where it is parsed.
    pub fn valid_radius(&self) -> Option<S> {
        match self {
            Self::PinholeRadtan8(p) => Some(p.rpmax),
            _ => None,
        }
    }
}

impl<S: LieScalar> CameraModel<S> {
    /// `GenericCamera::cast` (`camera/generic_camera.hpp`): the same model in
    /// another scalar.
    pub fn cast<T: LieScalar>(&self) -> CameraModel<T> {
        let convert = |value: S| -> T { T::from_literal(value.to_f64()) };
        match self {
            Self::Pinhole(p) => CameraModel::Pinhole(PinholeParams {
                fx: convert(p.fx),
                fy: convert(p.fy),
                cx: convert(p.cx),
                cy: convert(p.cy),
            }),
            Self::Kb4(p) => CameraModel::Kb4(Kb4Params {
                fx: convert(p.fx),
                fy: convert(p.fy),
                cx: convert(p.cx),
                cy: convert(p.cy),
                k1: convert(p.k1),
                k2: convert(p.k2),
                k3: convert(p.k3),
                k4: convert(p.k4),
            }),
            Self::PinholeRadtan8(p) => CameraModel::PinholeRadtan8(Radtan8Params {
                fx: convert(p.fx),
                fy: convert(p.fy),
                cx: convert(p.cx),
                cy: convert(p.cy),
                k1: convert(p.k1),
                k2: convert(p.k2),
                p1: convert(p.p1),
                p2: convert(p.p2),
                k3: convert(p.k3),
                k4: convert(p.k4),
                k5: convert(p.k5),
                k6: convert(p.k6),
                rpmax: convert(p.rpmax),
            }),
            Self::DoubleSphere(p) => CameraModel::DoubleSphere(DoubleSphereParams {
                fx: convert(p.fx),
                fy: convert(p.fy),
                cx: convert(p.cx),
                cy: convert(p.cy),
                xi: convert(p.xi),
                alpha: convert(p.alpha),
            }),
            Self::ExtendedUnified(p) => CameraModel::ExtendedUnified(ExtendedUnifiedParams {
                fx: convert(p.fx),
                fy: convert(p.fy),
                cx: convert(p.cx),
                cy: convert(p.cy),
                alpha: convert(p.alpha),
                beta: convert(p.beta),
            }),
            Self::Unified(p) => CameraModel::Unified(UnifiedParams {
                fx: convert(p.fx),
                fy: convert(p.fy),
                cx: convert(p.cx),
                cy: convert(p.cy),
                alpha: convert(p.alpha),
            }),
        }
    }
}

/// Static accelerometer calibration: bias plus a lower-triangular scale
/// (`calibration/calib_bias.hpp:44-125`, 9 parameters).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CalibAccelBias<S> {
    /// `[b_x, b_y, b_z, s_1 ... s_6]`, zero meaning "identity mapping".
    pub params: [S; 9],
}

impl<S: LieScalar> Default for CalibAccelBias<S> {
    fn default() -> Self {
        Self {
            params: [S::zero(); 9],
        }
    }
}

impl<S: LieScalar> CalibAccelBias<S> {
    /// The bias and the scale matrix (`calib_bias.hpp:86-94`).
    ///
    /// The scale is lower triangular: column 0 is `(s1, s2, s3)`, then
    /// `(1,1) = s4`, `(2,1) = s5`, `(2,2) = s6`. Everything above the diagonal
    /// stays zero, which is what makes the accelerometer frame the reference the
    /// gyroscope is aligned to.
    pub fn bias_and_scale(&self) -> (Vector3<S>, Matrix3<S>) {
        let p: &[S; 9] = &self.params;
        let bias: Vector3<S> = Vector3::new(p[0], p[1], p[2]);
        let mut scale: Matrix3<S> = Matrix3::zeros();
        scale[(0, 0)] = p[3];
        scale[(1, 0)] = p[4];
        scale[(2, 0)] = p[5];
        scale[(1, 1)] = p[6];
        scale[(2, 1)] = p[7];
        scale[(2, 2)] = p[8];
        (bias, scale)
    }

    /// `a_c = (I + S) a_r - b` (`calib_bias.hpp:101-107`).
    pub fn calibrated(&self, raw: &Vector3<S>) -> Vector3<S> {
        let (bias, scale) = self.bias_and_scale();
        raw + scale * raw - bias
    }

    /// The inverse map, `invertCalibration` (`calib_bias.hpp:114-122`).
    ///
    /// Returns `None` when `I + S` is singular, where the C++ would hand back
    /// an Eigen inverse full of infinities.
    pub fn raw(&self, calibrated: &Vector3<S>) -> Option<Vector3<S>> {
        let (bias, scale) = self.bias_and_scale();
        let inverse: Matrix3<S> = (Matrix3::identity() + scale).try_inverse()?;
        Some(inverse * (calibrated + bias))
    }
}

/// Static gyroscope calibration: bias plus a full 3x3 scale
/// (`calibration/calib_bias.hpp:128-205`, 12 parameters).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CalibGyroBias<S> {
    /// `[b_x, b_y, b_z, s_1 ... s_9]`, zero meaning "identity mapping".
    pub params: [S; 12],
}

impl<S: LieScalar> Default for CalibGyroBias<S> {
    fn default() -> Self {
        Self {
            params: [S::zero(); 12],
        }
    }
}

impl<S: LieScalar> CalibGyroBias<S> {
    /// The bias and the scale matrix (`calib_bias.hpp:165-170`).
    ///
    /// Column-major: `(s1, s2, s3)`, `(s4, s5, s6)`, `(s7, s8, s9)`. Unlike the
    /// accelerometer's, this one is full, because it also absorbs the rotation
    /// between the two sensor frames.
    pub fn bias_and_scale(&self) -> (Vector3<S>, Matrix3<S>) {
        let p: &[S; 12] = &self.params;
        let bias: Vector3<S> = Vector3::new(p[0], p[1], p[2]);
        let scale: Matrix3<S> = Matrix3::from_columns(&[
            Vector3::new(p[3], p[4], p[5]),
            Vector3::new(p[6], p[7], p[8]),
            Vector3::new(p[9], p[10], p[11]),
        ]);
        (bias, scale)
    }

    /// `w_c = (I + S) w_r - b` (`calib_bias.hpp:176-182`).
    pub fn calibrated(&self, raw: &Vector3<S>) -> Vector3<S> {
        let (bias, scale) = self.bias_and_scale();
        raw + scale * raw - bias
    }

    /// The inverse map, `invertCalibration` (`calib_bias.hpp:189-197`).
    pub fn raw(&self, calibrated: &Vector3<S>) -> Option<Vector3<S>> {
        let (bias, scale) = self.bias_and_scale();
        let inverse: Matrix3<S> = (Matrix3::identity() + scale).try_inverse()?;
        Some(inverse * (calibrated + bias))
    }
}

/// One camera's vignetting curve, a uniform B-spline over radius
/// (`calibration.hpp:158`, `RdSpline<1, 4, Scalar>`).
///
/// Parsed so a calibration round-trips; nothing reads it. cereal writes the
/// three members unnamed (`serialization/headers_serialization.h:243-249`), so
/// on disk they are `value0`, `value1` and `value2`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VignetteSpline<S> {
    /// Start of the spline's domain. basalt reuses the time axis for radius in
    /// pixels times 1e9 (`calibration.hpp:152-157`).
    #[serde(rename = "value0")]
    pub start_t_ns: i64,
    /// Knot spacing on the same axis.
    #[serde(rename = "value1")]
    pub dt_ns: i64,
    /// The one-dimensional knots.
    #[serde(rename = "value2")]
    pub knots: Vec<[S; 1]>,
}

/// The camera-IMU calibration, `basalt::Calibration`
/// (`calibration/calibration.hpp:51-186`).
#[derive(Debug, Clone, PartialEq)]
pub struct Calibration<S: LieScalar> {
    /// Camera pose in the IMU frame, one per camera: `p_i = T_i_c p_c`.
    pub t_i_c: Vec<Se3<S>>,
    /// Projection model, one per camera.
    pub intrinsics: Vec<CameraModel<S>>,
    /// `[width, height]` per camera, in pixels.
    pub resolution: Vec<[u32; 2]>,
    /// Vignetting splines, empty in every reference calibration but EuRoC's.
    pub vignette: Vec<VignetteSpline<S>>,
    /// Added to a raw camera timestamp to reach the IMU clock.
    pub cam_time_offset_ns: i64,
    /// Static accelerometer calibration.
    pub calib_accel_bias: CalibAccelBias<S>,
    /// Static gyroscope calibration.
    pub calib_gyro_bias: CalibGyroBias<S>,
    /// IMU sample rate, Hz.
    pub imu_update_rate: S,
    /// Continuous-time gyroscope noise density.
    pub gyro_noise_std: Vector3<S>,
    /// Continuous-time accelerometer noise density.
    pub accel_noise_std: Vector3<S>,
    /// Continuous-time gyroscope bias random walk.
    pub gyro_bias_std: Vector3<S>,
    /// Continuous-time accelerometer bias random walk.
    pub accel_bias_std: Vector3<S>,
    /// Keys the struct does not model, such as EuRoC's mocap block.
    pub unknown: BTreeMap<String, serde_json::Value>,
}

/// The wire form: what cereal actually writes, before the quaternions are
/// normalized and the lists checked against each other.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, bound = "S: LieScalar + Serialize + DeserializeOwned")]
struct CalibrationJson<S: LieScalar> {
    #[serde(rename = "T_imu_cam")]
    t_imu_cam: Vec<PoseJson<S>>,
    intrinsics: Vec<CameraModel<S>>,
    resolution: Vec<[u32; 2]>,
    vignette: Vec<VignetteSpline<S>>,
    calib_accel_bias: CalibAccelBias<S>,
    calib_gyro_bias: CalibGyroBias<S>,
    imu_update_rate: S,
    #[serde(with = "vector3_json")]
    gyro_noise_std: Vector3<S>,
    #[serde(with = "vector3_json")]
    accel_noise_std: Vector3<S>,
    #[serde(with = "vector3_json")]
    gyro_bias_std: Vector3<S>,
    #[serde(with = "vector3_json")]
    accel_bias_std: Vector3<S>,
    cam_time_offset_ns: i64,
    #[serde(flatten)]
    unknown: BTreeMap<String, serde_json::Value>,
}

impl<S: LieScalar> Default for CalibrationJson<S> {
    /// `Calibration::Calibration()` (`calibration.hpp:60-70`): the reasonable
    /// defaults basalt starts from before a file overwrites them.
    fn default() -> Self {
        Self {
            t_imu_cam: Vec::new(),
            intrinsics: Vec::new(),
            resolution: Vec::new(),
            vignette: Vec::new(),
            calib_accel_bias: CalibAccelBias::default(),
            calib_gyro_bias: CalibGyroBias::default(),
            imu_update_rate: S::from_literal(200.0),
            gyro_noise_std: Vector3::repeat(S::from_literal(0.000_282)),
            accel_noise_std: Vector3::repeat(S::from_literal(0.016)),
            gyro_bias_std: Vector3::repeat(S::from_literal(0.0001)),
            accel_bias_std: Vector3::repeat(S::from_literal(0.001)),
            cam_time_offset_ns: 0,
            unknown: BTreeMap::new(),
        }
    }
}

impl<S: LieScalar> Default for Calibration<S> {
    fn default() -> Self {
        Self {
            t_i_c: Vec::new(),
            intrinsics: Vec::new(),
            resolution: Vec::new(),
            vignette: Vec::new(),
            cam_time_offset_ns: 0,
            calib_accel_bias: CalibAccelBias::default(),
            calib_gyro_bias: CalibGyroBias::default(),
            imu_update_rate: S::from_literal(200.0),
            gyro_noise_std: Vector3::repeat(S::from_literal(0.000_282)),
            accel_noise_std: Vector3::repeat(S::from_literal(0.016)),
            gyro_bias_std: Vector3::repeat(S::from_literal(0.0001)),
            accel_bias_std: Vector3::repeat(S::from_literal(0.001)),
            unknown: BTreeMap::new(),
        }
    }
}

/// One camera as a catalog recording describes it, for
/// [`Calibration::from_catalog_parts`].
///
/// Mirrors `slam_rs.catalog_feed.CameraCalib` field for field, so PR 5 can hand
/// the dataclass across the PyO3 boundary without a second translation.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraParts<S> {
    /// Decoded frame width in pixels.
    pub width: u32,
    /// Decoded frame height in pixels.
    pub height: u32,
    /// Focal length along image x, pixels.
    pub fx: S,
    /// Focal length along image y, pixels.
    pub fy: S,
    /// Principal point x, pixels.
    pub cx: S,
    /// Principal point y, pixels.
    pub cy: S,
    /// The catalog's model name. `"kb4"` and `"radtan8"` are what
    /// `catalog_feed` produces; basalt's own `"pinhole-radtan8"`, `"pinhole"`,
    /// `"ds"`, `"eucm"` and `"ucm"` are accepted too.
    pub model: String,
    /// Exactly the coefficients the model uses: four for `kb4`, eight for
    /// `radtan8` in the order `k1 k2 p1 p2 k3 k4 k5 k6`.
    pub distortion: Vec<S>,
    /// basalt's `rpmax`, when the recording carries one. Required by
    /// `radtan8`; ignored by every other model.
    pub distortion_valid_radius: Option<S>,
    /// The camera pose in the IMU frame as a 4x4 matrix, flattened in C order
    /// (row by row) — what `numpy.ndarray.ravel()` gives for the
    /// `CameraCalib.imu_T_cam` field.
    pub imu_t_cam_row_major: [S; 16],
}

/// The IMU as a catalog recording plus the reference manifest describes it,
/// mirroring `slam_rs.catalog_feed.ImuCalib`.
///
/// The catalog carries intrinsics and extrinsics but not the noise model, the
/// update rate or the time offset, which is why these five numbers are frozen
/// in `reference_segments.toml` (decision D29).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuParts<S> {
    /// Nominal update rate, Hz.
    pub frequency_hz: S,
    /// Gyroscope noise density, isotropic.
    pub gyro_noise_std: S,
    /// Accelerometer noise density, isotropic.
    pub accel_noise_std: S,
    /// Gyroscope bias random walk, isotropic.
    pub gyro_bias_std: S,
    /// Accelerometer bias random walk, isotropic.
    pub accel_bias_std: S,
    /// Added to a camera timestamp to reach the IMU clock.
    pub cam_time_offset_ns: i64,
}

impl<S: LieScalar + Serialize + DeserializeOwned> Calibration<S> {
    /// Read one of basalt's calibration files.
    ///
    /// Unknown keys are collected and logged rather than rejected: EuRoC's
    /// calibration carries a mocap block and msd-g2's carries a `comment`, and
    /// neither belongs to the VIO struct.
    pub fn from_json_str(text: &str) -> Result<Self, CalibError> {
        let wrapper: Value0<CalibrationJson<S>> = serde_json::from_str(text)?;
        let raw: CalibrationJson<S> = wrapper.value0;

        if raw.t_imu_cam.len() != raw.intrinsics.len()
            || raw.t_imu_cam.len() != raw.resolution.len()
        {
            return Err(CalibError::RaggedCameraLists {
                extrinsics: raw.t_imu_cam.len(),
                intrinsics: raw.intrinsics.len(),
                resolutions: raw.resolution.len(),
            });
        }
        if !raw.unknown.is_empty() {
            let names: Vec<&str> = raw.unknown.keys().map(String::as_str).collect();
            log::warn!(
                "calibration: ignoring {} unmodelled key(s): {}",
                names.len(),
                names.join(", ")
            );
        }

        let mut t_i_c: Vec<Se3<S>> = Vec::with_capacity(raw.t_imu_cam.len());
        for (index, pose) in raw.t_imu_cam.iter().enumerate() {
            t_i_c.push(pose.to_se3(index)?);
        }

        Ok(Self {
            t_i_c,
            intrinsics: raw.intrinsics,
            resolution: raw.resolution,
            vignette: raw.vignette,
            cam_time_offset_ns: raw.cam_time_offset_ns,
            calib_accel_bias: raw.calib_accel_bias,
            calib_gyro_bias: raw.calib_gyro_bias,
            imu_update_rate: raw.imu_update_rate,
            gyro_noise_std: raw.gyro_noise_std,
            accel_noise_std: raw.accel_noise_std,
            gyro_bias_std: raw.gyro_bias_std,
            accel_bias_std: raw.accel_bias_std,
            unknown: raw.unknown,
        })
    }

    /// Write the calibration back in basalt's shape, wrapper and all.
    pub fn to_json_string(&self) -> Result<String, CalibError> {
        let raw: CalibrationJson<S> = CalibrationJson {
            t_imu_cam: self.t_i_c.iter().map(PoseJson::from_se3).collect(),
            intrinsics: self.intrinsics.clone(),
            resolution: self.resolution.clone(),
            vignette: self.vignette.clone(),
            calib_accel_bias: self.calib_accel_bias,
            calib_gyro_bias: self.calib_gyro_bias,
            imu_update_rate: self.imu_update_rate,
            gyro_noise_std: self.gyro_noise_std,
            accel_noise_std: self.accel_noise_std,
            gyro_bias_std: self.gyro_bias_std,
            accel_bias_std: self.accel_bias_std,
            cam_time_offset_ns: self.cam_time_offset_ns,
            unknown: self.unknown.clone(),
        };
        Ok(serde_json::to_string_pretty(&Value0 { value0: raw })?)
    }
}

impl<S: LieScalar> Calibration<S> {
    /// Number of cameras on the rig.
    pub fn camera_count(&self) -> usize {
        self.intrinsics.len()
    }

    /// `Calibration::cast` (`calibration.hpp:113-136`): the whole rig in
    /// another scalar.
    ///
    /// basalt builds its frontend from `cal.template cast<Scalar>()`
    /// (`optical_flow.h:204`), so a `Calibration<f64>` read from the JSON is what
    /// the file gives and a `Calibration<f32>` is what the frontend runs on
    /// (decision D05). `unknown` is carried across unchanged; the C++ has no
    /// such field.
    pub fn cast<T: LieScalar>(&self) -> Calibration<T> {
        let convert = |value: S| -> T { T::from_literal(value.to_f64()) };
        let convert3 = |value: &Vector3<S>| -> Vector3<T> {
            Vector3::new(convert(value.x), convert(value.y), convert(value.z))
        };
        Calibration {
            t_i_c: self.t_i_c.iter().map(Se3::cast).collect(),
            intrinsics: self.intrinsics.iter().map(CameraModel::cast).collect(),
            resolution: self.resolution.clone(),
            vignette: self
                .vignette
                .iter()
                .map(|spline| VignetteSpline {
                    start_t_ns: spline.start_t_ns,
                    dt_ns: spline.dt_ns,
                    knots: spline.knots.iter().map(|knot| [convert(knot[0])]).collect(),
                })
                .collect(),
            cam_time_offset_ns: self.cam_time_offset_ns,
            calib_accel_bias: CalibAccelBias {
                params: std::array::from_fn(|i| convert(self.calib_accel_bias.params[i])),
            },
            calib_gyro_bias: CalibGyroBias {
                params: std::array::from_fn(|i| convert(self.calib_gyro_bias.params[i])),
            },
            imu_update_rate: convert(self.imu_update_rate),
            gyro_noise_std: convert3(&self.gyro_noise_std),
            accel_noise_std: convert3(&self.accel_noise_std),
            gyro_bias_std: convert3(&self.gyro_bias_std),
            accel_bias_std: convert3(&self.accel_bias_std),
            unknown: self.unknown.clone(),
        }
    }

    /// Discrete-time gyroscope noise, `sigma_c sqrt(rate)`
    /// (`calibration.hpp:186`).
    pub fn discrete_time_gyro_noise_std(&self) -> Vector3<S> {
        self.gyro_noise_std * self.imu_update_rate.sqrt()
    }

    /// Discrete-time accelerometer noise, `sigma_c sqrt(rate)`
    /// (`calibration.hpp:193`).
    pub fn discrete_time_accel_noise_std(&self) -> Vector3<S> {
        self.accel_noise_std * self.imu_update_rate.sqrt()
    }

    /// Build a calibration from what the catalog feed reports.
    ///
    /// The Python side already turns a recording's statics into
    /// `CameraCalib`/`ImuCalib` dataclasses; this takes the same fields so the
    /// PyO3 layer copies numbers rather than re-deriving them. The noise
    /// densities are isotropic there and become the three equal components
    /// basalt stores.
    pub fn from_catalog_parts(
        cameras: &[CameraParts<S>],
        imu: &ImuParts<S>,
    ) -> Result<Self, CalibError> {
        let mut t_i_c: Vec<Se3<S>> = Vec::with_capacity(cameras.len());
        let mut intrinsics: Vec<CameraModel<S>> = Vec::with_capacity(cameras.len());
        let mut resolution: Vec<[u32; 2]> = Vec::with_capacity(cameras.len());

        for (index, camera) in cameras.iter().enumerate() {
            t_i_c.push(pose_from_row_major(&camera.imu_t_cam_row_major, index)?);
            intrinsics.push(camera_model_from_parts(camera, index)?);
            resolution.push([camera.width, camera.height]);
        }

        Ok(Self {
            t_i_c,
            intrinsics,
            resolution,
            vignette: Vec::new(),
            cam_time_offset_ns: imu.cam_time_offset_ns,
            calib_accel_bias: CalibAccelBias::default(),
            calib_gyro_bias: CalibGyroBias::default(),
            imu_update_rate: imu.frequency_hz,
            gyro_noise_std: Vector3::repeat(imu.gyro_noise_std),
            accel_noise_std: Vector3::repeat(imu.accel_noise_std),
            gyro_bias_std: Vector3::repeat(imu.gyro_bias_std),
            accel_bias_std: Vector3::repeat(imu.accel_bias_std),
            unknown: BTreeMap::new(),
        })
    }
}

/// A rigid transform from a row-major 4x4, rejecting anything that is not one.
///
/// Sophus's rotation-matrix constructor requires **both** orthogonality and a
/// positive determinant (`Sophus/sophus/so3.hpp:536-541`). Checking only the
/// first lets a reflection through, and Eigen's matrix-to-quaternion conversion
/// then returns a rotation that is not the input at all — `diag(-1, 1, 1)`
/// becomes the identity, silently discarding the camera's geometry.
fn pose_from_row_major<S: LieScalar>(m: &[S; 16], index: usize) -> Result<Se3<S>, CalibError> {
    let rotation: Matrix3<S> = Matrix3::new(m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]);
    let residual: Matrix3<S> = rotation.transpose() * rotation - Matrix3::identity();
    // Catalog extrinsics arrive as float32 promoted to f64, so the tolerance has
    // to sit above single-precision noise while still catching a scaled or
    // mirrored matrix.
    if !residual.norm().is_finite() || residual.norm() > S::from_literal(1e-4) {
        return Err(CalibError::NonRotation { index });
    }
    // A NaN determinant cannot reach here: the orthogonality check above rejects
    // any non-finite entry first.
    if rotation.determinant() <= S::zero() {
        return Err(CalibError::ReflectedRotation { index });
    }
    let quaternion: nalgebra::UnitQuaternion<S> = nalgebra::UnitQuaternion::from_rotation_matrix(
        &nalgebra::Rotation3::from_matrix_unchecked(rotation),
    );
    let rotation: So3<S> =
        So3::from_quaternion_xyzw(quaternion.i, quaternion.j, quaternion.k, quaternion.w)
            .ok_or(CalibError::NonRotation { index })?;
    Ok(Se3::new(rotation, Vector3::new(m[3], m[7], m[11])))
}

/// One camera model from a name and a coefficient list.
fn camera_model_from_parts<S: LieScalar>(
    camera: &CameraParts<S>,
    index: usize,
) -> Result<CameraModel<S>, CalibError> {
    let (fx, fy, cx, cy): (S, S, S, S) = (camera.fx, camera.fy, camera.cx, camera.cy);
    let d: &[S] = &camera.distortion;

    let expect = |wanted: usize, model: &'static str| -> Result<(), CalibError> {
        if d.len() == wanted {
            Ok(())
        } else {
            Err(CalibError::WrongCoefficientCount {
                index,
                model,
                expected: wanted,
                actual: d.len(),
            })
        }
    };

    match camera.model.as_str() {
        "pinhole" => {
            expect(0, "pinhole")?;
            Ok(CameraModel::Pinhole(PinholeParams { fx, fy, cx, cy }))
        }
        "kb4" => {
            expect(4, "kb4")?;
            Ok(CameraModel::Kb4(Kb4Params {
                fx,
                fy,
                cx,
                cy,
                k1: d[0],
                k2: d[1],
                k3: d[2],
                k4: d[3],
            }))
        }
        // `catalog_feed` calls the model `radtan8`; basalt calls the same thing
        // `pinhole-radtan8`. Both names land here.
        "radtan8" | "pinhole-radtan8" => {
            expect(8, "pinhole-radtan8")?;
            Ok(CameraModel::PinholeRadtan8(Radtan8Params {
                fx,
                fy,
                cx,
                cy,
                k1: d[0],
                k2: d[1],
                p1: d[2],
                p2: d[3],
                k3: d[4],
                k4: d[5],
                k5: d[6],
                k6: d[7],
                // basalt's own default when a calibration omits it
                // (`camera/pinhole_radtan8_camera.hpp`): no radial cut-off.
                rpmax: camera.distortion_valid_radius.unwrap_or_else(S::zero),
            }))
        }
        "ds" => {
            expect(2, "ds")?;
            Ok(CameraModel::DoubleSphere(DoubleSphereParams {
                fx,
                fy,
                cx,
                cy,
                xi: d[0],
                alpha: d[1],
            }))
        }
        "eucm" => {
            expect(2, "eucm")?;
            Ok(CameraModel::ExtendedUnified(ExtendedUnifiedParams {
                fx,
                fy,
                cx,
                cy,
                alpha: d[0],
                beta: d[1],
            }))
        }
        "ucm" => {
            expect(1, "ucm")?;
            Ok(CameraModel::Unified(UnifiedParams {
                fx,
                fy,
                cx,
                cy,
                alpha: d[0],
            }))
        }
        other => Err(CalibError::UnknownCameraModel {
            index,
            model: other.to_owned(),
        }),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use approx::assert_abs_diff_eq;

    const MSDMI: &str = include_str!("../tests/fixtures/msdmi_calib.json");
    const MSDMG: &str = include_str!("../tests/fixtures/msdmg_calib.json");
    const EUROC: &str = include_str!("../tests/fixtures/euroc_ds_calib.json");
    const ROBOCAP: &str = include_str!("../tests/fixtures/robocap-basalt-calib.json");

    fn every_fixture() -> [(&'static str, &'static str); 4] {
        [
            ("msdmi_calib.json", MSDMI),
            ("msdmg_calib.json", MSDMG),
            ("euroc_ds_calib.json", EUROC),
            ("robocap-basalt-calib.json", ROBOCAP),
        ]
    }

    /// Two calibrations agree to within a tolerance. Exact equality would be
    /// hostage to the quaternion renormalization, which is not idempotent to the
    /// last bit.
    fn assert_close(left: &Calibration<f64>, right: &Calibration<f64>, name: &str) {
        assert_eq!(left.resolution, right.resolution, "{name}: resolution");
        assert_eq!(
            left.cam_time_offset_ns, right.cam_time_offset_ns,
            "{name}: time offset"
        );
        assert_eq!(left.unknown, right.unknown, "{name}: unknown keys");
        assert_eq!(left.vignette, right.vignette, "{name}: vignette");
        assert_eq!(
            left.calib_accel_bias, right.calib_accel_bias,
            "{name}: accel bias"
        );
        assert_eq!(
            left.calib_gyro_bias, right.calib_gyro_bias,
            "{name}: gyro bias"
        );
        assert_eq!(
            left.imu_update_rate, right.imu_update_rate,
            "{name}: update rate"
        );
        assert_eq!(left.intrinsics, right.intrinsics, "{name}: intrinsics");
        assert_abs_diff_eq!(left.gyro_noise_std, right.gyro_noise_std, epsilon = 0.0);
        assert_abs_diff_eq!(left.accel_noise_std, right.accel_noise_std, epsilon = 0.0);
        assert_eq!(left.t_i_c.len(), right.t_i_c.len(), "{name}: camera count");
        for (a, b) in left.t_i_c.iter().zip(&right.t_i_c) {
            assert_abs_diff_eq!(a.translation, b.translation, epsilon = 0.0);
            let (qa, qb) = (a.rotation.quaternion_xyzw(), b.rotation.quaternion_xyzw());
            for i in 0..4 {
                assert_abs_diff_eq!(qa[i], qb[i], epsilon = 1e-15);
            }
        }
    }

    #[test]
    fn every_shipped_calibration_parses() {
        for (name, text) in every_fixture() {
            let calib: Calibration<f64> = Calibration::from_json_str(text)
                .unwrap_or_else(|e| panic!("{name} failed to parse: {e}"));
            assert!(calib.camera_count() >= 2, "{name}: fewer than two cameras");
            assert_eq!(calib.t_i_c.len(), calib.camera_count());
            assert_eq!(calib.resolution.len(), calib.camera_count());
        }
    }

    #[test]
    fn every_shipped_calibration_round_trips() {
        for (name, text) in every_fixture() {
            let once: Calibration<f64> = Calibration::from_json_str(text).unwrap();
            let written: String = once.to_json_string().unwrap();
            let twice: Calibration<f64> = Calibration::from_json_str(&written).unwrap();
            assert_close(&once, &twice, name);
        }
    }

    #[test]
    fn msd_index_is_two_kb4_cameras() {
        let calib: Calibration<f64> = Calibration::from_json_str(MSDMI).unwrap();
        assert_eq!(calib.camera_count(), 2);
        for model in &calib.intrinsics {
            assert_eq!(model.name(), "kb4");
            // kb4 is `fx fy cx cy` plus exactly four radial terms.
            assert_eq!(model.params().len(), 8);
            assert_eq!(model.valid_radius(), None);
        }
        assert_eq!(calib.resolution, vec![[960, 960], [960, 960]]);
        assert_eq!(calib.imu_update_rate, 1000.0);
        assert_eq!(calib.cam_time_offset_ns, 0);
        assert_abs_diff_eq!(
            calib.gyro_noise_std,
            Vector3::repeat(0.000_282),
            epsilon = 0.0
        );
    }

    #[test]
    fn msd_g2_is_four_radtan8_cameras() {
        let calib: Calibration<f64> = Calibration::from_json_str(MSDMG).unwrap();
        assert_eq!(calib.camera_count(), 4);
        for model in &calib.intrinsics {
            assert_eq!(model.name(), "pinhole-radtan8");
            // radtan8 is `fx fy cx cy`, eight distortion terms and `rpmax`.
            assert_eq!(model.params().len(), 12);
            assert!(model.valid_radius().is_some());
        }
        assert_eq!(calib.resolution, vec![[640, 480]; 4]);
        // msd-g2's calibration carries a free-text `comment` the struct does not
        // model, which must not be fatal.
        assert_eq!(calib.unknown.keys().collect::<Vec<_>>(), ["comment"]);
    }

    #[test]
    fn robocap_is_four_kb4_cameras_with_its_own_noise() {
        let calib: Calibration<f64> = Calibration::from_json_str(ROBOCAP).unwrap();
        assert_eq!(calib.camera_count(), 4);
        assert!(calib.intrinsics.iter().all(|m| m.name() == "kb4"));
        assert_eq!(calib.resolution, vec![[960, 540]; 4]);
        assert_eq!(calib.imu_update_rate, 200.0);
        // Kalibr numbers, not basalt's MSD defaults.
        assert_abs_diff_eq!(
            calib.gyro_noise_std,
            Vector3::repeat(0.000_730_044_281_254_7),
            epsilon = 0.0
        );
    }

    /// EuRoC is the only shipped calibration with a vignette spline and with
    /// mocap keys the VIO struct does not model.
    #[test]
    fn euroc_carries_a_vignette_and_mocap_keys() {
        let calib: Calibration<f64> = Calibration::from_json_str(EUROC).unwrap();
        assert_eq!(calib.camera_count(), 2);
        assert!(calib.intrinsics.iter().all(|m| m.name() == "ds"));
        assert_eq!(calib.intrinsics[0].params().len(), 6);
        assert_eq!(calib.vignette.len(), 2);
        assert!(!calib.vignette[0].knots.is_empty());
        assert_eq!(
            calib.unknown.keys().collect::<Vec<_>>(),
            [
                "T_imu_marker",
                "T_mocap_world",
                "mocap_time_offset_ns",
                "mocap_to_imu_offset_ns"
            ]
        );
        // EuRoC is the only fixture with a non-zero accelerometer bias.
        assert!(calib.calib_accel_bias.params[0] != 0.0);
    }

    #[test]
    fn stored_quaternions_come_back_normalized() {
        for (name, text) in every_fixture() {
            let calib: Calibration<f64> = Calibration::from_json_str(text).unwrap();
            for pose in &calib.t_i_c {
                let q: [f64; 4] = pose.rotation.quaternion_xyzw();
                let norm: f64 = q.iter().map(|v| v * v).sum::<f64>().sqrt();
                assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-15);
                assert!(norm.is_finite(), "{name}");
            }
        }
    }

    /// A quaternion that is off unit length by a percent still yields a
    /// rotation; a zero one is an error, not a NaN.
    #[test]
    fn a_denormalized_quaternion_is_normalized_and_a_zero_one_rejected() {
        let text: &str = r#"{"value0": {
            "T_imu_cam": [{"px": 1.0, "py": 2.0, "pz": 3.0,
                           "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.01}],
            "intrinsics": [{"camera_type": "pinhole",
                            "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5}}],
            "resolution": [[2, 2]]
        }}"#;
        let calib: Calibration<f64> = Calibration::from_json_str(text).unwrap();
        let q: [f64; 4] = calib.t_i_c[0].rotation.quaternion_xyzw();
        assert_abs_diff_eq!(q[3], 1.0, epsilon = 1e-15);
        // Everything absent falls back to basalt's constructor defaults.
        assert_eq!(calib.imu_update_rate, 200.0);
        assert_abs_diff_eq!(calib.accel_noise_std, Vector3::repeat(0.016), epsilon = 0.0);

        let zero: &str = r#"{"value0": {
            "T_imu_cam": [{"px": 0.0, "py": 0.0, "pz": 0.0,
                           "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 0.0}],
            "intrinsics": [{"camera_type": "pinhole",
                            "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5}}],
            "resolution": [[2, 2]]
        }}"#;
        assert!(matches!(
            Calibration::<f64>::from_json_str(zero),
            Err(CalibError::DegenerateQuaternion { index: 0 })
        ));
    }

    #[test]
    fn a_ragged_calibration_is_rejected() {
        let text: &str = r#"{"value0": {
            "T_imu_cam": [{"px": 0.0, "py": 0.0, "pz": 0.0,
                           "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}],
            "intrinsics": [],
            "resolution": [[2, 2]]
        }}"#;
        assert!(matches!(
            Calibration::<f64>::from_json_str(text),
            Err(CalibError::RaggedCameraLists { .. })
        ));
    }

    #[test]
    fn discrete_time_noise_scales_with_the_square_root_of_the_rate() {
        let calib: Calibration<f64> = Calibration::from_json_str(MSDMI).unwrap();
        assert_abs_diff_eq!(
            calib.discrete_time_gyro_noise_std(),
            calib.gyro_noise_std * 1000.0f64.sqrt(),
            epsilon = 1e-18
        );
        assert_abs_diff_eq!(
            calib.discrete_time_accel_noise_std(),
            calib.accel_noise_std * 1000.0f64.sqrt(),
            epsilon = 1e-18
        );
    }

    /// A zero calibration is the identity mapping, and the two `getCalibrated`
    /// implementations invert cleanly.
    #[test]
    fn the_bias_calibrations_round_trip() {
        let raw: Vector3<f64> = Vector3::new(0.3, -1.2, 9.7);

        let identity: CalibAccelBias<f64> = CalibAccelBias::default();
        assert_abs_diff_eq!(identity.calibrated(&raw), raw, epsilon = 1e-15);

        let accel: CalibAccelBias<f64> = CalibAccelBias {
            params: [0.01, -0.02, 0.03, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006],
        };
        let (bias, scale) = accel.bias_and_scale();
        assert_eq!(bias, Vector3::new(0.01, -0.02, 0.03));
        // Lower triangular: the two entries above the diagonal stay zero.
        assert_eq!(scale[(0, 1)], 0.0);
        assert_eq!(scale[(0, 2)], 0.0);
        assert_eq!(scale[(1, 2)], 0.0);
        assert_eq!(scale[(0, 0)], 0.001);
        assert_eq!(scale[(1, 1)], 0.004);
        assert_eq!(scale[(2, 2)], 0.006);
        let back: Vector3<f64> = accel.raw(&accel.calibrated(&raw)).unwrap();
        assert_abs_diff_eq!(back, raw, epsilon = 1e-12);

        let identity: CalibGyroBias<f64> = CalibGyroBias::default();
        assert_abs_diff_eq!(identity.calibrated(&raw), raw, epsilon = 1e-15);

        let gyro: CalibGyroBias<f64> = CalibGyroBias {
            params: [
                0.01, -0.02, 0.03, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
            ],
        };
        let (_, scale) = gyro.bias_and_scale();
        // Full 3x3, filled column by column.
        assert_eq!(scale[(0, 0)], 0.001);
        assert_eq!(scale[(1, 0)], 0.002);
        assert_eq!(scale[(0, 1)], 0.004);
        assert_eq!(scale[(0, 2)], 0.007);
        let back: Vector3<f64> = gyro.raw(&gyro.calibrated(&raw)).unwrap();
        assert_abs_diff_eq!(back, raw, epsilon = 1e-12);
    }

    fn a_camera(model: &str, distortion: Vec<f64>) -> CameraParts<f64> {
        // A 90-degree rotation about z with a translation, row-major.
        let imu_t_cam_row_major: [f64; 16] = [
            0.0, -1.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.2, 0.0, 0.0, 1.0, 0.3, 0.0, 0.0, 0.0, 1.0,
        ];
        CameraParts {
            width: 960,
            height: 540,
            fx: 420.5,
            fy: 421.0,
            cx: 480.0,
            cy: 270.0,
            model: model.to_owned(),
            distortion,
            distortion_valid_radius: None,
            imu_t_cam_row_major,
        }
    }

    fn an_imu() -> ImuParts<f64> {
        ImuParts {
            frequency_hz: 200.0,
            gyro_noise_std: 0.000_730_044_281_254_7,
            accel_noise_std: 0.005_955_224_218_014,
            gyro_bias_std: 3.445_397_083_168e-5,
            accel_bias_std: 0.000_196_315_048_921_8,
            cam_time_offset_ns: 14_902_432,
        }
    }

    #[test]
    fn a_calibration_can_be_built_from_catalog_parts() {
        let cameras: Vec<CameraParts<f64>> = vec![
            a_camera("kb4", vec![0.19, 0.04, -0.23, 0.09]),
            a_camera("kb4", vec![0.19, 0.05, -0.24, 0.10]),
        ];
        let calib: Calibration<f64> = Calibration::from_catalog_parts(&cameras, &an_imu()).unwrap();

        assert_eq!(calib.camera_count(), 2);
        assert_eq!(calib.resolution, vec![[960, 540], [960, 540]]);
        assert_eq!(calib.cam_time_offset_ns, 14_902_432);
        assert_eq!(calib.imu_update_rate, 200.0);
        assert_abs_diff_eq!(
            calib.accel_bias_std,
            Vector3::repeat(0.000_196_315_048_921_8),
            epsilon = 0.0
        );
        assert_eq!(calib.intrinsics[0].name(), "kb4");
        assert_eq!(calib.intrinsics[0].params().len(), 8);

        // The 4x4 came in row-major, so the translation is the last column.
        assert_abs_diff_eq!(
            calib.t_i_c[0].translation,
            Vector3::new(0.1, 0.2, 0.3),
            epsilon = 1e-15
        );
        // And the rotation is the quarter turn the matrix describes.
        assert_abs_diff_eq!(
            calib.t_i_c[0].rotation * Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            epsilon = 1e-15
        );

        // The result serializes into basalt's own shape and reads back.
        let text: String = calib.to_json_string().unwrap();
        let reread: Calibration<f64> = Calibration::from_json_str(&text).unwrap();
        assert_close(&calib, &reread, "from_catalog_parts");
    }

    /// The catalog spells the eight-coefficient model `radtan8`; basalt spells
    /// it `pinhole-radtan8`. Both must produce the same variant.
    #[test]
    fn the_catalog_radtan8_name_is_accepted() {
        let coefficients: Vec<f64> = vec![0.1, 0.2, 0.001, 0.002, 0.3, 0.4, 0.5, 0.6];
        let mut camera: CameraParts<f64> = a_camera("radtan8", coefficients.clone());
        camera.distortion_valid_radius = Some(1.5);
        let calib: Calibration<f64> =
            Calibration::from_catalog_parts(&[camera], &an_imu()).unwrap();
        assert_eq!(calib.intrinsics[0].name(), "pinhole-radtan8");
        assert_eq!(calib.intrinsics[0].params()[4..], coefficients);
        assert_eq!(calib.intrinsics[0].valid_radius(), Some(1.5));

        let basalt_name: Calibration<f64> = Calibration::from_catalog_parts(
            &[{
                let mut c: CameraParts<f64> = a_camera("pinhole-radtan8", coefficients);
                c.distortion_valid_radius = Some(1.5);
                c
            }],
            &an_imu(),
        )
        .unwrap();
        assert_eq!(basalt_name.intrinsics, calib.intrinsics);
    }

    #[test]
    fn catalog_parts_reject_the_wrong_coefficient_count() {
        let camera: CameraParts<f64> = a_camera("kb4", vec![0.1, 0.2, 0.3]);
        assert!(matches!(
            Calibration::from_catalog_parts(&[camera], &an_imu()),
            Err(CalibError::WrongCoefficientCount {
                index: 0,
                model: "kb4",
                expected: 4,
                actual: 3
            })
        ));

        let camera: CameraParts<f64> = a_camera("radtan8", vec![0.1; 5]);
        assert!(matches!(
            Calibration::from_catalog_parts(&[camera], &an_imu()),
            Err(CalibError::WrongCoefficientCount { expected: 8, .. })
        ));
    }

    #[test]
    fn catalog_parts_reject_an_unknown_model_and_a_non_rotation() {
        let camera: CameraParts<f64> = a_camera("fisheye624", vec![0.0; 12]);
        assert!(matches!(
            Calibration::from_catalog_parts(&[camera], &an_imu()),
            Err(CalibError::UnknownCameraModel { index: 0, .. })
        ));

        let mut camera: CameraParts<f64> = a_camera("kb4", vec![0.1; 4]);
        // Scale one axis: still a valid matrix, no longer a rotation.
        camera.imu_t_cam_row_major[4] = 2.0;
        assert!(matches!(
            Calibration::from_catalog_parts(&[camera], &an_imu()),
            Err(CalibError::NonRotation { index: 0 })
        ));
    }

    /// A reflection is orthonormal, so the orthogonality check alone lets it
    /// through, and Eigen's conversion then turns `diag(-1, 1, 1)` into the
    /// identity — a camera silently pointing somewhere else. Sophus rejects it
    /// on the determinant (`Sophus/sophus/so3.hpp:539-540`) and so does this.
    #[test]
    fn catalog_parts_reject_a_reflected_rotation() {
        let mut camera: CameraParts<f64> = a_camera("kb4", vec![0.1; 4]);
        camera.imu_t_cam_row_major = [
            -1.0, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.2, 0.0, 0.0, 1.0, 0.3, 0.0, 0.0, 0.0, 1.0,
        ];
        let rotation: Matrix3<f64> = Matrix3::new(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        // Orthonormal, so only the determinant tells it apart from a rotation.
        assert_abs_diff_eq!(
            rotation.transpose() * rotation,
            Matrix3::identity(),
            epsilon = 0.0
        );
        assert_eq!(rotation.determinant(), -1.0);
        assert!(matches!(
            Calibration::from_catalog_parts(&[camera], &an_imu()),
            Err(CalibError::ReflectedRotation { index: 0 })
        ));
    }

    /// The estimator is generic over the scalar (decision D05), so a
    /// calibration reads in `f32` too.
    #[test]
    fn a_calibration_reads_in_f32() {
        let calib: Calibration<f32> = Calibration::from_json_str(MSDMI).unwrap();
        assert_eq!(calib.camera_count(), 2);
        assert_eq!(calib.imu_update_rate, 1000.0);
        let q: [f32; 4] = calib.t_i_c[0].rotation.quaternion_xyzw();
        let norm: f32 = q.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn malformed_json_is_an_error_not_a_panic() {
        assert!(Calibration::<f64>::from_json_str("{").is_err());
        assert!(
            Calibration::<f64>::from_json_str(r#"{"value0": {"imu_update_rate": "fast"}}"#)
                .is_err()
        );
        // An unmodelled camera model name in a file is a parse error, since the
        // enum cannot represent it.
        assert!(
            Calibration::<f64>::from_json_str(
                r#"{"value0": {"intrinsics": [{"camera_type": "fov",
                    "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "w": 0.9}}]}}"#
            )
            .is_err()
        );
    }
}
