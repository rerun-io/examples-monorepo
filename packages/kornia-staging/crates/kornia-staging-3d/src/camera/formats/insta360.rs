//! The Insta360 factory lens string (`.insv` metadata fields 54 and 111): one [`Mei`] camera per lens and the
//! lens's pose in the vendor rig frame.
//!
//! Tokens: the lens count, then per lens `xi fx fy cx cy yaw pitch roll tx ty tz <coefficients> W H type`, then one
//! tail token `(version << 16) | (mirror << 8) | lens class`. `W x H` is the calibration frame holding every lens
//! side by side. The window crop maps a lens's frame onto the recorded window about pixel centres (telemetry-parser
//! `rig.rs`, `Offset::windowed`), then the window is scaled to the track: `u = (u' + 0.5) s - 0.5`, `s = track/dst`.
//! Rotations use the vendor's `R_lens<-world = Ry(pitch + 90) Rz(yaw) Rx(roll)`, with an extra `Rz(180)` on lens 0
//! of a two-lens body; the translations are taken as already in lens 0's frame (telemetry-parser `rig.rs`).

use super::super::{Mei, MeiDistortion};
use nalgebra::{IsometryMatrix3, Matrix3, Rotation3, Translation3, Vector3};

/// Full optical field, degrees, per lens type (the lens class in the string's `type` token), after telemetry-parser's
/// `rig.rs` `class_half_field_deg`; a type without an entry has a 190 degree field. Half of it is the largest angle
/// off the axis whose pixels a camera uses. Checked here on two types: 131 (X4 Air), and 113 (X5), whose image circle
/// on the clip VID_20261113_145021_00_003 keeps full brightness to 100 degrees through its V3 model and falls to half
/// at 102.5-103.5 degrees (the vignetted rim).
const FIELD_DEG: [(&[u32], f64); 2] = [
    (
        &[
            19, 33, 38, 40, 41, 44, 70, 71, 85, 113, 131, 142, 149, 150, 193,
        ],
        200.0,
    ),
    (&[140, 141, 156], 195.0),
];
/// Full field of a lens type without an entry in [`FIELD_DEG`], degrees.
const DEFAULT_FIELD_DEG: f64 = 190.0;

/// Half field, degrees, of a lens type: [`FIELD_DEG`] or the 190 degree default.
fn half_field_deg(lens_type: f64) -> f64 {
    FIELD_DEG
        .iter()
        .find(|(types, _)| types.iter().any(|&t| f64::from(t) == lens_type))
        .map_or(DEFAULT_FIELD_DEG, |(_, field)| *field)
        / 2.0
}

/// `window_crop_info`: the sensor window a calibration string describes and the part of it the picture holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowCrop {
    /// Source window width and height, pixels.
    pub src: [u32; 2],
    /// Recorded window width and height, pixels.
    pub dst: [u32; 2],
    /// Crop offset, pixels.
    pub offset: [i64; 2],
}

/// One lens of the factory calibration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FactoryLens {
    /// The lens camera, stated for the video track's pixels (pixel centres at integers).
    pub camera: Mei<f64>,
    /// `T_P_lens`: rotation `R_P_lens` and the lens centre in the vendor rig frame `P` (telemetry-parser
    /// `rig.rs::rotation`: `+Z` the nominal lens-0 axis, `+Y` image-down of lens 0, toward the handle, `+X` right),
    /// metres, with the origin at the mean of the lens centres. `R_P_lens = Rv(0, 0, 90; lens 0) Rv(ypr_i; i)^T`
    /// with the vendor rotation `Rv`; the factory relative rotation, not the nominal 180 deg, puts the two seams
    /// within 0.25 px on the X4 Air clip. On the X4 Air the two centres sit about 16.56 mm either side of the
    /// origin along `Z`.
    pub pano_from_lens: IsometryMatrix3<f64>,
    /// The lens class: the lens's `type` token.
    pub lens_type: u32,
}

/// A lens string the parser refuses, with the reason.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[error("Insta360 lens string: {0}")]
pub struct LensStringError(pub String);

fn refuse(what: impl Into<String>) -> LensStringError {
    LensStringError(what.into())
}

/// Parse a V3 or V6 lens string and state it for a `track_size` picture.
///
/// # Errors
/// A token that is not a finite number, a version other than 3 and 6, a lens count that is not a whole number from
/// 1 to 8 or does not match the token count, or values that make no valid camera.
pub fn parse_lens_string(
    text: &str,
    crop: Option<WindowCrop>,
    track_size: [u32; 2],
) -> Result<Vec<FactoryLens>, LensStringError> {
    let tokens: Vec<f64> = text
        .split('_')
        .map(|t| {
            t.parse::<f64>()
                .ok()
                .filter(|v| v.is_finite())
                .ok_or_else(|| refuse(format!("token {t:?} is not a finite number")))
        })
        .collect::<Result<_, _>>()?;
    let (&count, rest) = tokens.split_first().ok_or_else(|| refuse("empty"))?;
    let (&tail, body) = rest.split_last().ok_or_else(|| refuse("no tail token"))?;
    let version = (tail as u64 >> 16) as u8;
    let n_coeffs = match version {
        3 => 5,
        6 => 13,
        v => {
            return Err(refuse(format!(
                "version {v} has no typed reader (V3 and V6 do)"
            )))
        }
    };
    let per_lens = 11 + n_coeffs + 3;
    if count.fract() != 0.0 || !(1.0..=8.0).contains(&count) {
        return Err(refuse(format!(
            "V{version} states {count} lenses, not a whole number from 1 to 8"
        )));
    }
    let count = count as usize;
    if body.len() != count * per_lens {
        return Err(refuse(format!(
            "V{version}: {} tokens for {count} lenses of {per_lens}",
            body.len()
        )));
    }
    let mut lenses = Vec::with_capacity(count);
    let mut types = Vec::with_capacity(count);
    let mut rotations = Vec::with_capacity(count);
    let mut translations = Vec::with_capacity(count);
    for (i, t) in body.chunks_exact(per_lens).enumerate() {
        let half_field_deg = half_field_deg(t[per_lens - 1]);
        types.push(t[per_lens - 1] as u32);
        let frame = [t[per_lens - 3] / count as f64, t[per_lens - 2]];
        let (mut cx, mut cy, mut fx, mut fy) = (t[3] - frame[0] * i as f64, t[4], t[1], t[2]);
        let mut window = frame;
        if let Some(c) = crop {
            let src = [f64::from(c.src[0]), f64::from(c.src[1])];
            let dst = [f64::from(c.dst[0]), f64::from(c.dst[1])];
            let scale = src[0] / frame[0];
            cx = scale * (cx + 0.5) - 0.5 - (src[0] - dst[0]) * 0.5 - c.offset[0] as f64;
            cy = scale * (cy + 0.5) - 0.5 - (src[1] - dst[1]) * 0.5 - c.offset[1] as f64;
            (fx, fy, window) = (fx * scale, fy * scale, dst);
        }
        let s = [
            f64::from(track_size[0]) / window[0],
            f64::from(track_size[1]) / window[1],
        ];
        rotations.push(vendor_rotation([t[5], t[6], t[7]], count == 2 && i == 0));
        translations.push([t[8], t[9], t[10]]);
        let c = &t[11..11 + n_coeffs];
        let distortion = if version == 3 {
            MeiDistortion::V3 {
                k: [c[0], c[1], c[2]],
                p: [c[3], c[4]],
            }
        } else {
            MeiDistortion::V6 {
                d: std::array::from_fn(|k| c[k]),
            }
        };
        let camera = Mei::new(
            [
                fx * s[0],
                fy * s[1],
                (cx + 0.5) * s[0] - 0.5,
                (cy + 0.5) * s[1] - 0.5,
                t[0],
            ],
            // As `100.0 * PI / 180.0`, the expression the X4 Air tables were built with, bit for bit.
            half_field_deg * std::f64::consts::PI / 180.0,
            distortion,
        )
        .map_err(|_| refuse(format!("lens {i} makes no valid camera")))?;
        lenses.push(camera);
    }
    let origin: [f64; 3] =
        std::array::from_fn(|k| translations.iter().map(|t| t[k]).sum::<f64>() / count as f64);
    let pano_base = vendor_rotation([0.0, 0.0, 90.0], count == 2);
    let pano_from_lens: Vec<Matrix3<f64>> = rotations
        .iter()
        .map(|r| pano_base * r.transpose())
        .collect();
    Ok(lenses
        .into_iter()
        .zip(&pano_from_lens)
        .zip(&translations)
        .zip(types)
        .map(|(((camera, rotation), t), lens_type)| {
            let d = Vector3::from_fn(|k, _| t[k] - origin[k]);
            FactoryLens {
                camera,
                pano_from_lens: IsometryMatrix3::from_parts(
                    Translation3::from(pano_from_lens[0] * d),
                    Rotation3::from_matrix_unchecked(*rotation),
                ),
                lens_type,
            }
        })
        .collect())
}

/// Active rotation by `deg` degrees about axis 0 (x), 1 (y) or 2 (z).
fn axis_rotation(axis: usize, deg: f64) -> Matrix3<f64> {
    let (s, c) = deg.to_radians().sin_cos();
    let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
    let mut r = Matrix3::zeros();
    r[(axis, axis)] = 1.0;
    r[(a, a)] = c;
    r[(b, b)] = c;
    r[(a, b)] = -s;
    r[(b, a)] = s;
    r
}

/// The vendor's world-to-lens rotation `Ry(pitch + 90) Rz(yaw) Rx(roll)`, times `Rz(180)` for lens 0 of two.
fn vendor_rotation(ypr_deg: [f64; 3], first_of_two: bool) -> Matrix3<f64> {
    let r = axis_rotation(1, ypr_deg[1] + 90.0)
        * axis_rotation(2, ypr_deg[0])
        * axis_rotation(0, ypr_deg[2]);
    if first_of_two {
        r * axis_rotation(2, 180.0)
    } else {
        r
    }
}
