//! KeyNet output decoding (handtrack `labels/heatmaps.py` and `labels/keypoint_input.py`): heatmap peaks to crop pixels,
//! distance bins to millimetres, and the 63-value keypoint input.
#![deny(missing_docs)]

use nalgebra::{Vector2, Vector3};

use crate::kornia_ext::heatmap::{argmax_first, decode_peak_2d, refine_peak_log_quadratic};
use crate::nets::{DISTANCE_BINS, DISTANCE_LEN, HEATMAP_LEN, HEATMAP_SIDE, KEYNET_CROP, NUM_LANDMARKS};

/// Relative distances span `[-130, 130]` mm over the 18 bins (handtrack `DISTANCE_RANGE_MM`).
pub const DISTANCE_RANGE_MM: f32 = 130.0;

/// 18-pixel heatmap centres into 96-pixel crop centres: `(p + 0.5) * 96 / 18 - 0.5`.
pub fn heatmap_to_crop(p: f32) -> f32 {
    (p + 0.5) * (KEYNET_CROP as f32 / HEATMAP_SIDE as f32) - 0.5
}

/// 96-pixel crop centres into 18-pixel heatmap centres (the inverse of [`heatmap_to_crop`]).
pub fn crop_to_heatmap(p: f32) -> f32 {
    (p + 0.5) * (HEATMAP_SIDE as f32 / KEYNET_CROP as f32) - 0.5
}

/// `decode_heatmaps` for one crop: each landmark's heatmap peak, refined separably (log-quadratic), in crop pixels.
///
/// # Arguments
///
/// * `heatmaps` - KeyNet's 21 row-major 18x18 heatmaps, landmark-major.
///
/// # Returns
///
/// The keypoints in 96-pixel crop pixel centres and each one's sampled peak value (its confidence); `None` when `heatmaps`
/// does not hold 21 x 18 x 18 values.
///
/// # Example
///
/// ```
/// use robocap_live::hands::heatmaps::decode_heatmaps;
/// let mut heatmaps = vec![0.0f32; 21 * 18 * 18];
/// heatmaps[9 * 18 + 9] = 1.0; // landmark 0's peak at heatmap pixel (9, 9)
/// let (points, peaks) = decode_heatmaps(&heatmaps).unwrap_or_default();
/// // Heatmap pixel 9 is crop pixel (9 + 0.5) * 96 / 18 - 0.5.
/// assert!(points[0].iter().all(|v| (v - 50.1667).abs() < 1e-3) && peaks[0] == 1.0);
/// ```
pub fn decode_heatmaps(heatmaps: &[f32]) -> Option<([[f32; 2]; NUM_LANDMARKS], [f32; NUM_LANDMARKS])> {
    let side = HEATMAP_SIDE;
    if heatmaps.len() != HEATMAP_LEN {
        return None;
    }
    let mut points = [[0f32; 2]; NUM_LANDMARKS];
    let mut peaks = [0f32; NUM_LANDMARKS];
    for (landmark, heatmap) in heatmaps.chunks_exact(side * side).enumerate() {
        let ([x, y], peak) = decode_peak_2d(heatmap, side, side)?;
        points[landmark] = [heatmap_to_crop(x), heatmap_to_crop(y)];
        peaks[landmark] = peak;
    }
    Some((points, peaks))
}

/// `decode_distance` for one crop: each landmark's distance-bin peak, refined (log-quadratic), as a relative distance.
///
/// # Arguments
///
/// * `distance` - KeyNet's 21 x 18 distance bins, landmark-major, spanning `[-130, 130]` mm.
///
/// # Returns
///
/// The relative distances in millimetres of the generic hand; `None` when `distance` does not hold 21 x 18 values.
pub fn decode_distance(distance: &[f32]) -> Option<[f32; NUM_LANDMARKS]> {
    if distance.len() != DISTANCE_LEN {
        return None;
    }
    let step = 2.0 * DISTANCE_RANGE_MM / (DISTANCE_BINS as f32 - 1.0);
    let mut out = [0f32; NUM_LANDMARKS];
    for (landmark, bins) in distance.chunks_exact(DISTANCE_BINS).enumerate() {
        let peak = argmax_first(bins)?;
        let index = refine_peak_log_quadratic(bins, peak).clamp(0.0, DISTANCE_BINS as f32 - 1.0);
        out[landmark] = index * step - DISTANCE_RANGE_MM;
    }
    Some(out)
}

/// `relative_distances`: radial distances to centred, scale-normalised millimetres, `(d - mean d) * 1000 / phi`.
///
/// # Arguments
///
/// * `points_cam` - The 21 landmarks in the camera frame, metres.
/// * `phi` - The hand scale.
///
/// # Returns
///
/// Each landmark's distance from the camera minus their mean, in millimetres of the generic hand.
pub fn relative_distances(points_cam: &[Vector3<f64>; NUM_LANDMARKS], phi: f64) -> [f64; NUM_LANDMARKS] {
    let norms = points_cam.map(|p| p.norm());
    let mean = norms.iter().sum::<f64>() / NUM_LANDMARKS as f64;
    norms.map(|d| (d - mean) * 1000.0 / phi)
}

/// torch's `nan_to_num` for f32: NaN -> 0, +inf -> f32::MAX, -inf -> f32::MIN.
pub fn nan_to_num(value: f32) -> f32 {
    if value.is_nan() {
        0.0
    } else if value == f32::INFINITY {
        f32::MAX
    } else if value == f32::NEG_INFINITY {
        f32::MIN
    } else {
        value
    }
}

/// `keypoint_input`: KeyNet's 63-value keypoint prior.
///
/// # Arguments
///
/// * `points_crop` - The planning pose's 21 keypoints in crop pixels (mirror included).
/// * `d_rel_mm` - Their relative distances, millimetres.
///
/// # Returns
///
/// Interleaved `((u + 0.5) / 96, (v + 0.5) / 96, d / 130)` per landmark; non-finite values made finite as torch's
/// `nan_to_num` (the estimator applies it).
pub fn keypoint_input(points_crop: &[Vector2<f64>; NUM_LANDMARKS], d_rel_mm: &[f64; NUM_LANDMARKS]) -> [f32; 3 * NUM_LANDMARKS] {
    let mut out = [0f32; 3 * NUM_LANDMARKS];
    for landmark in 0..NUM_LANDMARKS {
        let uv = points_crop[landmark];
        out[3 * landmark] = nan_to_num(((uv.x + 0.5) / KEYNET_CROP as f64) as f32);
        out[3 * landmark + 1] = nan_to_num(((uv.y + 0.5) / KEYNET_CROP as f64) as f32);
        out[3 * landmark + 2] = nan_to_num((d_rel_mm[landmark] / DISTANCE_RANGE_MM as f64) as f32);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_and_heatmap_pixel_centres_invert() {
        for p in [-0.5f32, 0.0, 47.5, 95.0] {
            assert!((heatmap_to_crop(crop_to_heatmap(p)) - p).abs() < 1e-4);
        }
        assert!((heatmap_to_crop(8.5) - 47.5).abs() < 1e-5);
    }

    #[test]
    fn rendered_labels_decode_to_their_points_and_distances() {
        // handtrack's render_heatmaps / render_distance (sigma 1), decoded back.
        let truth = [[30.2f32, 61.7], [5.0, 90.4]];
        let mut heatmaps = vec![0f32; NUM_LANDMARKS * 324];
        let mut distance = vec![0f32; NUM_LANDMARKS * 18];
        for landmark in 0..NUM_LANDMARKS {
            let [u, v] = truth[landmark % 2];
            let (cx, cy) = (crop_to_heatmap(u), crop_to_heatmap(v));
            for r in 0..18 {
                for c in 0..18 {
                    heatmaps[landmark * 324 + r * 18 + c] = (-((c as f32 - cx).powi(2)) / 2.0).exp() * (-((r as f32 - cy).powi(2)) / 2.0).exp();
                }
            }
            let centre = (40.0f32 + DISTANCE_RANGE_MM) * (17.0 / (2.0 * DISTANCE_RANGE_MM));
            for b in 0..18 {
                distance[landmark * 18 + b] = (-((b as f32 - centre).powi(2)) * 0.5).exp();
            }
        }
        let (points, peaks) = decode_heatmaps(&heatmaps).unwrap_or(([[0.0; 2]; 21], [0.0; 21]));
        for landmark in 0..NUM_LANDMARKS {
            let [u, v] = truth[landmark % 2];
            assert!((points[landmark][0] - u).abs() < 2e-3 && (points[landmark][1] - v).abs() < 2e-3, "{landmark}: {:?}", points[landmark]);
            assert!(peaks[landmark] > 0.5);
        }
        let d = decode_distance(&distance).unwrap_or([0.0; 21]);
        assert!(d.iter().all(|mm| (mm - 40.0).abs() < 1e-2), "{d:?}");
        assert!(decode_heatmaps(&heatmaps[1..]).is_none() && decode_distance(&distance[1..]).is_none());
    }

    #[test]
    fn the_keypoint_input_interleaves_scaled_uv_and_distance() {
        let points = [Vector2::new(47.5, -0.5); NUM_LANDMARKS];
        let mut d = [65.0; NUM_LANDMARKS];
        d[3] = f64::NAN;
        let input = keypoint_input(&points, &d);
        assert_eq!(&input[0..3], &[0.5, 0.0, 0.5]);
        assert_eq!(input[3 * 3 + 2], 0.0);
    }
}
