//! Lossless calibration boundaries. COLMAP principal points shift by half a pixel.
#![doc = include_str!("crosswalk.md")]

use super::*;
use serde::{Deserialize, Serialize};

/// Pinhole, `fx fy cx cy`.
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

/// Kannala-Brandt with four radial terms.
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

/// Pinhole with eight-term rational Brown-Conrady distortion.
/// The disk order is `k1 k2 p1 p2 k3 k4 k5 k6`, matching OpenCV.
/// `rpmax` is stored beside the twelve optimized parameters.
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
    /// Brown y-axis tangential coefficient (p1).
    pub p1: S,
    /// Brown x-axis tangential coefficient (p2).
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

/// Basalt schemas for the three supported camera families.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "camera_type", content = "intrinsics")]
pub enum BasaltCamera<S> {
    /// `pinhole`.
    #[serde(rename = "pinhole")]
    Pinhole(PinholeParams<S>),
    /// `kb4`.
    #[serde(rename = "kb4")]
    Kb4(Kb4Params<S>),
    /// `pinhole-radtan8`.
    #[serde(rename = "pinhole-radtan8")]
    PinholeRadtan8(Radtan8Params<S>),
}

/// A format conversion would change or cannot evaluate the camera model.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ConversionError {
    /// Calibration failed model construction.
    #[error(transparent)]
    Calibration(#[from] InvalidCalibration),
    /// A model has no staged implementation.
    #[error("unsupported camera model {0}")]
    Unsupported(String),
    /// Wrong parameter count, invalid values, or an unrepresentable restriction.
    #[error("invalid or lossy camera conversion: {0}")]
    Invalid(&'static str),
}

/// The model-dependent portion of a COLMAP cameras record.
///
/// Camera id, dimensions and rig pose belong to the caller. `params` follow
/// COLMAP's exact per-id order in the module crosswalk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColmapCamera {
    /// Numeric COLMAP model id.
    pub model_id: u32,
    /// COLMAP parameters; principal point uses a (0.5,0.5) top-left pixel centre.
    pub params: Vec<f64>,
}

const COLMAP_MODELS: [(u32, usize, bool); 10] = [
    (0, 3, true),
    (1, 4, false),
    (2, 4, true),
    (3, 5, true),
    (4, 8, false),
    (5, 8, false),
    (6, 12, false),
    (8, 4, true),
    (9, 5, true),
    (11, 16, false),
];
fn colmap_layout(id: u32) -> Result<(usize, bool), ConversionError> {
    COLMAP_MODELS
        .iter()
        .find(|m| m.0 == id)
        .map(|m| (m.1, m.2))
        .ok_or_else(|| ConversionError::Unsupported(format!("COLMAP {id}")))
}

impl TryFrom<&ColmapCamera> for CameraModelKind<f64> {
    type Error = ConversionError;
    fn try_from(source: &ColmapCamera) -> Result<Self, Self::Error> {
        let id = source.model_id;
        let (count, tied) = colmap_layout(id)?;
        let p = &source.params;
        if p.len() != count {
            return Err(ConversionError::Invalid(
                "parameter count or non-finite coefficient",
            ));
        }
        let head = if tied {
            [p[0], p[0], p[1] - 0.5, p[2] - 0.5]
        } else {
            [p[0], p[1], p[2] - 0.5, p[3] - 0.5]
        };
        match id {
            0 | 1 => Ok(Self::Pinhole(Pinhole::new(head)?)),
            2 | 3 | 4 | 6 => {
                let mut full = [0.0; 18];
                full[..4].copy_from_slice(&head);
                let offset = if tied { 3 } else { 4 };
                full[4..4 + p.len() - offset].copy_from_slice(&p[offset..]);
                Ok(Self::BrownConrady(BrownConrady::new(full, None)?))
            }
            5 | 8 | 9 => {
                let mut full = [0.0; 8];
                full[..4].copy_from_slice(&head);
                let offset = if tied { 3 } else { 4 };
                full[4..4 + p.len() - offset].copy_from_slice(&p[offset..]);
                Ok(Self::Kb4(KannalaBrandt4::new(full)?))
            }
            11 => {
                let mut full = [0.0; 16];
                full.copy_from_slice(p);
                full[..4].copy_from_slice(&head);
                Ok(Self::Fisheye624(Fisheye624::new(full)?))
            }
            _ => unreachable!(),
        }
    }
}
impl CameraModelKind<f64> {
    /// Export to a requested COLMAP family, refusing lost coefficients or radius metadata.
    /// # Arguments
    /// * `model_id` - Supported COLMAP id, including restricted tied-focal models.
    /// # Errors
    /// Returns `Unsupported` for unavailable models and `Invalid` for a lossy restriction.
    pub fn to_colmap(&self, model_id: u32) -> Result<ColmapCamera, ConversionError> {
        let (count, tied) = colmap_layout(model_id)?;
        let mut full = match (self, model_id) {
            (Self::Pinhole(v), 0 | 1) => v.params().to_vec(),
            (Self::BrownConrady(v), 2 | 3 | 4 | 6) if v.valid_radius().is_none() => {
                v.params().to_vec()
            }
            (Self::Kb4(v), 5 | 8 | 9) => v.params().to_vec(),
            (Self::Fisheye624(v), 11) => v.params().to_vec(),
            _ => {
                return Err(ConversionError::Invalid(
                    "family or valid radius cannot be represented",
                ))
            }
        };
        full[2] += 0.5;
        full[3] += 0.5;
        if tied {
            if full[0] != full[1] {
                return Err(ConversionError::Invalid(
                    "requested model ties focal lengths",
                ));
            }
            full.remove(1);
        }
        if full[count..].iter().any(|v| *v != 0.0) {
            return Err(ConversionError::Invalid(
                "requested model drops nonzero coefficients",
            ));
        }
        full.truncate(count);
        if full.iter().any(|v| !v.is_finite()) || full[0] <= 0.0 {
            return Err(ConversionError::Invalid("invalid calibration"));
        }
        Ok(ColmapCamera {
            model_id,
            params: full,
        })
    }
}

impl<S: Scalar> TryFrom<&BasaltCamera<S>> for CameraModelKind<S> {
    type Error = ConversionError;
    fn try_from(source: &BasaltCamera<S>) -> Result<Self, Self::Error> {
        match source {
            BasaltCamera::Pinhole(v) => Ok(Self::Pinhole(Pinhole::new([v.fx, v.fy, v.cx, v.cy])?)),
            BasaltCamera::Kb4(v) => Ok(Self::Kb4(KannalaBrandt4::new([
                v.fx, v.fy, v.cx, v.cy, v.k1, v.k2, v.k3, v.k4,
            ])?)),
            BasaltCamera::PinholeRadtan8(v) => {
                if !v.rpmax.is_finite() || v.rpmax < S::zero() {
                    return Err(ConversionError::Invalid("invalid rpmax"));
                }
                let mut full = [S::zero(); 18];
                full[..12].copy_from_slice(&[
                    v.fx, v.fy, v.cx, v.cy, v.k1, v.k2, v.p1, v.p2, v.k3, v.k4, v.k5, v.k6,
                ]);
                Ok(Self::BrownConrady(BrownConrady::new(
                    full,
                    if v.rpmax == S::zero() {
                        None
                    } else {
                        Some(v.rpmax)
                    },
                )?))
            }
        }
    }
}
impl<S: Scalar> TryFrom<CameraModelKind<S>> for BasaltCamera<S> {
    type Error = ConversionError;
    fn try_from(camera: CameraModelKind<S>) -> Result<Self, Self::Error> {
        match camera {
            CameraModelKind::Pinhole(v) => {
                let [fx, fy, cx, cy] = v.params();
                Ok(Self::Pinhole(PinholeParams { fx, fy, cx, cy }))
            }
            CameraModelKind::Kb4(v) => {
                let [fx, fy, cx, cy, k1, k2, k3, k4] = v.params();
                Ok(Self::Kb4(Kb4Params {
                    fx,
                    fy,
                    cx,
                    cy,
                    k1,
                    k2,
                    k3,
                    k4,
                }))
            }
            CameraModelKind::BrownConrady(v) => {
                let p = v.params();
                if p[12..].iter().any(|x| *x != S::zero()) {
                    return Err(ConversionError::Invalid(
                        "Basalt radtan8 cannot carry prism or tilt",
                    ));
                }
                let [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6] = p[..12].try_into().unwrap();
                Ok(Self::PinholeRadtan8(Radtan8Params {
                    fx,
                    fy,
                    cx,
                    cy,
                    k1,
                    k2,
                    p1,
                    p2,
                    k3,
                    k4,
                    k5,
                    k6,
                    rpmax: v.valid_radius().unwrap_or(S::zero()),
                }))
            }
            CameraModelKind::Fisheye624(_) => {
                Err(ConversionError::Unsupported("fisheye624 in Basalt".into()))
            }
        }
    }
}
