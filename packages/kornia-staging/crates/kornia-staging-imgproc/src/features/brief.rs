//! Basalt's oriented 256-bit BRIEF descriptor at given keypoints of a u16 image (RootSLAM's loop-closing
//! descriptor, `computeAngles` + `computeDescriptors` in Basalt's `keypoints.cpp`).
//!
//! The angle is the intensity centroid over a radius-15 disc; the descriptor compares 256 point pairs of ORB's
//! learned `bit_pattern_31_`, rotated by that angle and rounded to the nearest pixel. With
//! [`BriefConfig::basalt`] the output is bit-identical to Basalt.
//!
//! Smoothing (RootSLAM port bug B16, a behaviour knob): Basalt compares raw pixels; ORB smooths before its tests.
//! The default compares the sums of the 5x5 boxes around the two points instead, following the port's rule that the
//! fixed behaviour is the default. The tests here show only that the bits then differ from Basalt's; they do not
//! show that smoothing makes retrieval or matching more robust. That is measured end to end by the knob A/B (PR k
//! of the port), which keeps or flips this default. The `rootslam-parity` profile selects [`BriefConfig::basalt`].

use kornia_image::Image;

use super::hamming::Descriptor256;
use crate::interpolation::in_bounds_u16;

/// Radius of the disc the orientation is measured over (Basalt `HALF_PATCH_SIZE`).
pub const BRIEF_HALF_PATCH: i32 = 15;

/// Border Basalt's loop closer keeps keypoints away from before describing them (`EDGE_THRESHOLD`).
pub const BRIEF_BASALT_BORDER: f32 = 19.0;

/// Largest pixel offset of a rotated test point: `round(|(13, 13)|)`.
const MAX_ROTATED_OFFSET: i32 = 18;

/// ORB's `bit_pattern_31_` as `[x_a, y_a, x_b, y_b]` per test. Stored as `i8`: Basalt's `char` tables broke on
/// aarch64, where `char` is unsigned. Equal to kornia-imgproc 0.2's `orb::pattern::{POS0, POS1}`.
const PATTERN_31: [[i8; 4]; 256] = [
    [8, -3, 9, 5],
    [4, 2, 7, -12],
    [-11, 9, -8, 2],
    [7, -12, 12, -13],
    [2, -13, 2, 12],
    [1, -7, 1, 6],
    [-2, -10, -2, -4],
    [-13, -13, -11, -8],
    [-13, -3, -12, -9],
    [10, 4, 11, 9],
    [-13, -8, -8, -9],
    [-11, 7, -9, 12],
    [7, 7, 12, 6],
    [-4, -5, -3, 0],
    [-13, 2, -12, -3],
    [-9, 0, -7, 5],
    [12, -6, 12, -1],
    [-3, 6, -2, 12],
    [-6, -13, -4, -8],
    [11, -13, 12, -8],
    [4, 7, 5, 1],
    [5, -3, 10, -3],
    [3, -7, 6, 12],
    [-8, -7, -6, -2],
    [-2, 11, -1, -10],
    [-13, 12, -8, 10],
    [-7, 3, -5, -3],
    [-4, 2, -3, 7],
    [-10, -12, -6, 11],
    [5, -12, 6, -7],
    [5, -6, 7, -1],
    [1, 0, 4, -5],
    [9, 11, 11, -13],
    [4, 7, 4, 12],
    [2, -1, 4, 4],
    [-4, -12, -2, 7],
    [-8, -5, -7, -10],
    [4, 11, 9, 12],
    [0, -8, 1, -13],
    [-13, -2, -8, 2],
    [-3, -2, -2, 3],
    [-6, 9, -4, -9],
    [8, 12, 10, 7],
    [0, 9, 1, 3],
    [7, -5, 11, -10],
    [-13, -6, -11, 0],
    [10, 7, 12, 1],
    [-6, -3, -6, 12],
    [10, -9, 12, -4],
    [-13, 8, -8, -12],
    [-13, 0, -8, -4],
    [3, 3, 7, 8],
    [5, 7, 10, -7],
    [-1, 7, 1, -12],
    [3, -10, 5, 6],
    [2, -4, 3, -10],
    [-13, 0, -13, 5],
    [-13, -7, -12, 12],
    [-13, 3, -11, 8],
    [-7, 12, -4, 7],
    [6, -10, 12, 8],
    [-9, -1, -7, -6],
    [-2, -5, 0, 12],
    [-12, 5, -7, 5],
    [3, -10, 8, -13],
    [-7, -7, -4, 5],
    [-3, -2, -1, -7],
    [2, 9, 5, -11],
    [-11, -13, -5, -13],
    [-1, 6, 0, -1],
    [5, -3, 5, 2],
    [-4, -13, -4, 12],
    [-9, -6, -9, 6],
    [-12, -10, -8, -4],
    [10, 2, 12, -3],
    [7, 12, 12, 12],
    [-7, -13, -6, 5],
    [-4, 9, -3, 4],
    [7, -1, 12, 2],
    [-7, 6, -5, 1],
    [-13, 11, -12, 5],
    [-3, 7, -2, -6],
    [7, -8, 12, -7],
    [-13, -7, -11, -12],
    [1, -3, 12, 12],
    [2, -6, 3, 0],
    [-4, 3, -2, -13],
    [-1, -13, 1, 9],
    [7, 1, 8, -6],
    [1, -1, 3, 12],
    [9, 1, 12, 6],
    [-1, -9, -1, 3],
    [-13, -13, -10, 5],
    [7, 7, 10, 12],
    [12, -5, 12, 9],
    [6, 3, 7, 11],
    [5, -13, 6, 10],
    [2, -12, 2, 3],
    [3, 8, 4, -6],
    [2, 6, 12, -13],
    [9, -12, 10, 3],
    [-8, 4, -7, 9],
    [-11, 12, -4, -6],
    [1, 12, 2, -8],
    [6, -9, 7, -4],
    [2, 3, 3, -2],
    [6, 3, 11, 0],
    [3, -3, 8, -8],
    [7, 8, 9, 3],
    [-11, -5, -6, -4],
    [-10, 11, -5, 10],
    [-5, -8, -3, 12],
    [-10, 5, -9, 0],
    [8, -1, 12, -6],
    [4, -6, 6, -11],
    [-10, 12, -8, 7],
    [4, -2, 6, 7],
    [-2, 0, -2, 12],
    [-5, -8, -5, 2],
    [7, -6, 10, 12],
    [-9, -13, -8, -8],
    [-5, -13, -5, -2],
    [8, -8, 9, -13],
    [-9, -11, -9, 0],
    [1, -8, 1, -2],
    [7, -4, 9, 1],
    [-2, 1, -1, -4],
    [11, -6, 12, -11],
    [-12, -9, -6, 4],
    [3, 7, 7, 12],
    [5, 5, 10, 8],
    [0, -4, 2, 8],
    [-9, 12, -5, -13],
    [0, 7, 2, 12],
    [-1, 2, 1, 7],
    [5, 11, 7, -9],
    [3, 5, 6, -8],
    [-13, -4, -8, 9],
    [-5, 9, -3, -3],
    [-4, -7, -3, -12],
    [6, 5, 8, 0],
    [-7, 6, -6, 12],
    [-13, 6, -5, -2],
    [1, -10, 3, 10],
    [4, 1, 8, -4],
    [-2, -2, 2, -13],
    [2, -12, 12, 12],
    [-2, -13, 0, -6],
    [4, 1, 9, 3],
    [-6, -10, -3, -5],
    [-3, -13, -1, 1],
    [7, 5, 12, -11],
    [4, -2, 5, -7],
    [-13, 9, -9, -5],
    [7, 1, 8, 6],
    [7, -8, 7, 6],
    [-7, -4, -7, 1],
    [-8, 11, -7, -8],
    [-13, 6, -12, -8],
    [2, 4, 3, 9],
    [10, -5, 12, 3],
    [-6, -5, -6, 7],
    [8, -3, 9, -8],
    [2, -12, 2, 8],
    [-11, -2, -10, 3],
    [-12, -13, -7, -9],
    [-11, 0, -10, -5],
    [5, -3, 11, 8],
    [-2, -13, -1, 12],
    [-1, -8, 0, 9],
    [-13, -11, -12, -5],
    [-10, -2, -10, 11],
    [-3, 9, -2, -13],
    [2, -3, 3, 2],
    [-9, -13, -4, 0],
    [-4, 6, -3, -10],
    [-4, 12, -2, -7],
    [-6, -11, -4, 9],
    [6, -3, 6, 11],
    [-13, 11, -5, 5],
    [11, 11, 12, 6],
    [7, -5, 12, -2],
    [-1, 12, 0, 7],
    [-4, -8, -3, -2],
    [-7, 1, -6, 7],
    [-13, -12, -8, -13],
    [-7, -2, -6, -8],
    [-8, 5, -6, -9],
    [-5, -1, -4, 5],
    [-13, 7, -8, 10],
    [1, 5, 5, -13],
    [1, 0, 10, -13],
    [9, 12, 10, -1],
    [5, -8, 10, -9],
    [-1, 11, 1, -13],
    [-9, -3, -6, 2],
    [-1, -10, 1, 12],
    [-13, 1, -8, -10],
    [8, -11, 10, -6],
    [2, -13, 3, -6],
    [7, -13, 12, -9],
    [-10, -10, -5, -7],
    [-10, -8, -8, -13],
    [4, -6, 8, 5],
    [3, 12, 8, -13],
    [-4, 2, -3, -3],
    [5, -13, 10, -12],
    [4, -13, 5, -1],
    [-9, 9, -4, 3],
    [0, 3, 3, -9],
    [-12, 1, -6, 1],
    [3, 2, 4, -8],
    [-10, -10, -10, 9],
    [8, -13, 12, 12],
    [-8, -12, -6, -5],
    [2, 2, 3, 7],
    [10, 6, 11, -8],
    [6, 8, 8, -12],
    [-7, 10, -6, 5],
    [-3, -9, -3, 9],
    [-1, -13, -1, 5],
    [-3, -7, -3, 4],
    [-8, -2, -8, 3],
    [4, 2, 12, 12],
    [2, -5, 3, 11],
    [6, -9, 11, -13],
    [3, -1, 7, 12],
    [11, -1, 12, 4],
    [-3, 0, -3, 6],
    [4, -11, 4, 12],
    [2, -4, 2, 1],
    [-10, -6, -8, 1],
    [-13, 7, -11, 1],
    [-13, 12, -11, -13],
    [6, 0, 11, -13],
    [0, -1, 1, 4],
    [-13, 3, -9, -2],
    [-9, 8, -6, -3],
    [-13, -6, -8, -2],
    [5, -9, 8, 10],
    [2, 7, 3, -9],
    [-1, -6, -1, -1],
    [9, 5, 11, -2],
    [11, -3, 12, -8],
    [3, 0, 3, 5],
    [-1, 4, 0, 10],
    [3, -6, 4, 5],
    [-13, 0, -10, 5],
    [5, 8, 12, 11],
    [8, 9, 9, -6],
    [7, -4, 8, -12],
    [-10, 4, -10, 9],
    [7, 3, 12, 4],
    [9, -7, 10, -2],
    [7, 0, 12, -2],
    [-1, -6, 0, -11],
];

/// How each BRIEF test reads the image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BriefSmoothing {
    /// Compare raw pixels, as Basalt and RootSLAM do.
    None,
    /// Compare the sums of the `(2 radius + 1)^2` boxes around the two points.
    Box {
        /// Half-width of the box, in pixels.
        radius: u8,
    },
}

impl BriefSmoothing {
    fn radius(self) -> i32 {
        match self {
            Self::None => 0,
            Self::Box { radius } => i32::from(radius),
        }
    }
}

/// Descriptor options.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BriefConfig {
    /// Smoothing of the tests (B16). The default smooths; [`BriefConfig::basalt`] (the `rootslam-parity` setting)
    /// does not.
    pub smoothing: BriefSmoothing,
}

impl Default for BriefConfig {
    /// 5x5 boxes, about ORB's smoothing scale (ORB blurs with a 7x7 Gaussian of sigma 2 before its tests). The
    /// bits differ from Basalt's; the knob A/B of the RootSLAM port decides whether this stays the default.
    fn default() -> Self {
        Self {
            smoothing: BriefSmoothing::Box { radius: 2 },
        }
    }
}

impl BriefConfig {
    /// Basalt's descriptor, bit for bit: raw pixel tests. The `rootslam-parity` profile uses this.
    pub const fn basalt() -> Self {
        Self {
            smoothing: BriefSmoothing::None,
        }
    }

    /// Keypoints closer than this to the image edge (in Basalt's `InBounds` sense) get no descriptor: Basalt's 19
    /// pixels, widened when the smoothing box would leave the image.
    pub fn border(&self) -> f32 {
        BRIEF_BASALT_BORDER.max((MAX_ROTATED_OFFSET + self.smoothing.radius()) as f32)
    }
}

/// A keypoint's orientation and descriptor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrientedDescriptor {
    /// Intensity-centroid angle in radians, `atan2(m01, m10)`.
    pub angle: f64,
    /// The 256 rotated BRIEF tests.
    pub descriptor: Descriptor256,
}

/// Intensity-centroid angle of the radius-15 disc around pixel `(cx, cy)`, Basalt's `computeAngles`.
///
/// The moments are integer sums, so the result does not depend on summation order.
///
/// # Panics
///
/// If the disc leaves the image.
pub fn intensity_centroid_angle(image: &Image<u16, 1>, cx: usize, cy: usize) -> f64 {
    let width = image.width();
    let pixels = image.as_slice();
    let (mut m01, mut m10) = (0i64, 0i64);
    for x in -BRIEF_HALF_PATCH..=BRIEF_HALF_PATCH {
        for y in -BRIEF_HALF_PATCH..=BRIEF_HALF_PATCH {
            if x * x + y * y <= BRIEF_HALF_PATCH * BRIEF_HALF_PATCH {
                let value = i64::from(
                    pixels[(cy as i64 + i64::from(y)) as usize * width
                        + (cx as i64 + i64::from(x)) as usize],
                );
                m01 += i64::from(y) * value;
                m10 += i64::from(x) * value;
            }
        }
    }
    (m01 as f64).atan2(m10 as f64)
}

/// Orientation and descriptor of each keypoint, in input order.
///
/// A keypoint outside [`BriefConfig::border`] (or non-finite) gets `None`: callers never see a placeholder
/// descriptor for it. The pixel used is `(x as i32, y as i32)`, Basalt's truncation.
///
/// # Arguments
///
/// * `image` - The u16 image the keypoints were detected in.
/// * `keypoints` - Keypoint positions `[x, y]` in pixels.
/// * `config` - Smoothing; [`BriefConfig::basalt`] reproduces Basalt.
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_staging_imgproc::features::{describe_keypoints_u16, BriefConfig};
/// let size = ImageSize { width: 64, height: 64 };
/// let pixels: Vec<u16> = (0..64 * 64).map(|i| ((i % 64) * 1000 + (i / 64) * 7) as u16).collect();
/// let image = Image::new(size, pixels)?;
/// let out = describe_keypoints_u16(&image, &[[32.0, 32.0], [5.0, 32.0]], BriefConfig::basalt());
/// assert!(out[0].is_some() && out[1].is_none());
/// assert!(out[0].unwrap().angle.abs() < 0.01); // brightness grows along +x
/// # Ok::<(), kornia_image::ImageError>(())
/// ```
pub fn describe_keypoints_u16(
    image: &Image<u16, 1>,
    keypoints: &[[f32; 2]],
    config: BriefConfig,
) -> Vec<Option<OrientedDescriptor>> {
    let border = config.border();
    keypoints
        .iter()
        .map(|&[x, y]| {
            in_bounds_u16(image, x, y, border).then(|| {
                let (cx, cy) = (x as usize, y as usize);
                let angle = intensity_centroid_angle(image, cx, cy);
                OrientedDescriptor {
                    angle,
                    descriptor: rotated_brief(image, cx, cy, angle, config.smoothing.radius()),
                }
            })
        })
        .collect()
}

/// Basalt's `computeDescriptors` for one keypoint, with an optional box around each test point.
fn rotated_brief(
    image: &Image<u16, 1>,
    cx: usize,
    cy: usize,
    angle: f64,
    radius: i32,
) -> Descriptor256 {
    let width = image.width();
    let pixels = image.as_slice();
    let (sin, cos) = angle.sin_cos();
    let at = |dx: f64, dy: f64| -> u32 {
        // Eigen's `(R * v).array().round()`: R = [[cos, -sin], [sin, cos]], rounding half away from zero.
        let x = (cos * dx - sin * dy).round() as i64 + cx as i64;
        let y = (sin * dx + cos * dy).round() as i64 + cy as i64;
        let mut sum = 0u32;
        for yy in y - i64::from(radius)..=y + i64::from(radius) {
            let row = &pixels[yy as usize * width..];
            for xx in x - i64::from(radius)..=x + i64::from(radius) {
                sum += u32::from(row[xx as usize]);
            }
        }
        sum
    };
    let mut words = [0u64; 4];
    for (index, &[xa, ya, xb, yb]) in PATTERN_31.iter().enumerate() {
        if at(f64::from(xa), f64::from(ya)) < at(f64::from(xb), f64::from(yb)) {
            words[index / 64] |= 1 << (index % 64);
        }
    }
    Descriptor256(words)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::random_image;
    use kornia_image::ImageSize;

    #[test]
    fn border_follows_the_smoothing_box() {
        assert_eq!(BriefConfig::basalt().border(), 19.0);
        assert_eq!(BriefConfig::default().border(), 20.0);
        let wide = BriefConfig {
            smoothing: BriefSmoothing::Box { radius: 4 },
        };
        assert_eq!(wide.border(), 22.0);
    }

    #[test]
    fn keypoints_outside_the_border_get_no_descriptor() {
        // B7: no placeholder for skipped keypoints, so no all-zero descriptor reaches matching or the BoW.
        let image = random_image(64, 48, 7);
        let points = [
            [19.0, 19.0],
            [18.99, 30.0],
            [43.99, 27.99],
            [44.0, 20.0],
            [30.0, f32::NAN],
        ];
        let out = describe_keypoints_u16(&image, &points, BriefConfig::basalt());
        assert_eq!(
            out.iter().map(Option::is_some).collect::<Vec<_>>(),
            [true, false, true, false, false]
        );
        let smoothed = describe_keypoints_u16(&image, &points[..1], BriefConfig::default());
        assert!(smoothed[0].is_none());
    }

    #[test]
    fn a_flat_patch_has_angle_zero_and_no_set_bit() {
        let image = Image::from_size_val(
            ImageSize {
                width: 64,
                height: 64,
            },
            1234u16,
        )
        .unwrap();
        let out =
            describe_keypoints_u16(&image, &[[32.0, 32.0]], BriefConfig::default())[0].unwrap();
        assert_eq!(out.angle, 0.0);
        assert_eq!(out.descriptor, Descriptor256::default());
    }

    #[test]
    fn the_pixel_is_the_truncated_position() {
        let image = random_image(80, 80, 3);
        let out = describe_keypoints_u16(
            &image,
            &[[40.0, 41.0], [40.999, 41.5]],
            BriefConfig::basalt(),
        );
        assert_eq!(out[0], out[1]);
    }

    #[test]
    fn smoothing_changes_the_bits_of_a_noisy_patch() {
        let image = random_image(96, 96, 11);
        let raw =
            describe_keypoints_u16(&image, &[[48.0, 48.0]], BriefConfig::basalt())[0].unwrap();
        let smooth =
            describe_keypoints_u16(&image, &[[48.0, 48.0]], BriefConfig::default())[0].unwrap();
        assert_eq!(raw.angle, smooth.angle);
        assert!(raw.descriptor.hamming(&smooth.descriptor) > 20);
    }

    #[test]
    fn brightness_ramp_points_the_angle_up_the_gradient() {
        let size = ImageSize {
            width: 64,
            height: 64,
        };
        let ramp_y: Vec<u16> = (0..64 * 64).map(|i| (i / 64) as u16 * 900).collect();
        let image = Image::new(size, ramp_y).unwrap();
        let angle = intensity_centroid_angle(&image, 32, 32);
        assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }
}
