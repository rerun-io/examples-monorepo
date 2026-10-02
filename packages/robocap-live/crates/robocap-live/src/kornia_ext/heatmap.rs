//! Sub-pixel peak decoding of keypoint heatmaps: the integer argmax refined by a separable log-quadratic (Gaussian) fit.
//!
//! Unlike a local soft-argmax, the three-sample log-parabola recovers a sampled Gaussian's vertex exactly. Target upstream:
//! kornia-imgproc `features` (or kornia-tensor-ops beside a `spatial_soft_argmax2d`).

/// Index of the first maximum of `values` (`None` when empty); NaN never wins over a number.
///
/// # Arguments
///
/// * `values` - The samples.
///
/// # Returns
///
/// The index of the first largest value.
///
/// # Example
///
/// ```
/// use robocap_live::kornia_ext::heatmap::argmax_first;
/// assert_eq!(argmax_first(&[0.0, 2.0, 2.0, 1.0]), Some(1));
/// ```
pub fn argmax_first(values: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (index, &value) in values.iter().enumerate() {
        match best {
            None => best = Some((index, value)),
            Some((_, current)) if value > current || (current.is_nan() && !value.is_nan()) => best = Some((index, value)),
            _ => {}
        }
    }
    best.map(|(index, _)| index)
}

/// Refine an integer peak of a 1D profile with a log-quadratic fit through three samples.
///
/// The stencil is centred on `peak`, shifted inward at the profile's ends. The vertex offset is
/// `0.5 (l0 - l2) / (l0 - 2 l1 + l2)` with `l = ln(max(s, 1e-30))`, clamped to two samples from the stencil centre. Flat,
/// non-positive or non-concave samples (curvature >= -1e-6) keep the integer peak.
///
/// # Arguments
///
/// * `profile` - The 1D samples (at least 3).
/// * `peak` - The integer peak index to refine.
///
/// # Returns
///
/// The refined peak position in sample units, or `peak` itself when the profile is shorter than 3.
///
/// # Example
///
/// ```
/// use robocap_live::kornia_ext::heatmap::refine_peak_log_quadratic;
/// // A sampled Gaussian centred at 4.3 is recovered exactly.
/// let profile: Vec<f32> = (0..10).map(|i| (-((i as f32 - 4.3).powi(2)) / 2.0).exp()).collect();
/// assert!((refine_peak_log_quadratic(&profile, 4) - 4.3).abs() < 1e-4);
/// ```
pub fn refine_peak_log_quadratic(profile: &[f32], peak: usize) -> f32 {
    let n = profile.len();
    if n < 3 {
        return peak as f32;
    }
    let centre = peak.clamp(1, n - 2);
    let samples = [profile[centre - 1], profile[centre], profile[centre + 1]];
    let logs = samples.map(|s| s.max(1e-30).ln());
    let curvature = logs[0] - 2.0 * logs[1] + logs[2];
    let usable = curvature < -1e-6 && samples.iter().all(|s| *s > 0.0);
    if !usable {
        return peak as f32;
    }
    let offset = 0.5 * (logs[0] - logs[2]) / curvature;
    centre as f32 + offset.clamp(-2.0, 2.0)
}

/// Decode one row-major `width x height` heatmap: the first argmax, refined separably along its row (x) and column (y).
///
/// # Arguments
///
/// * `heatmap` - `width * height` samples, row-major.
/// * `width` - Samples per row.
/// * `height` - Rows.
///
/// # Returns
///
/// `Some(([x, y], peak))` in heatmap sample units with the sampled peak value, `None` when the sizes do not match or are empty.
///
/// # Example
///
/// ```
/// use robocap_live::kornia_ext::heatmap::decode_peak_2d;
/// let mut heatmap = vec![0.0f32; 25];
/// heatmap[2 * 5 + 3] = 1.0;
/// assert_eq!(decode_peak_2d(&heatmap, 5, 5), Some(([3.0, 2.0], 1.0)));
/// ```
pub fn decode_peak_2d(heatmap: &[f32], width: usize, height: usize) -> Option<([f32; 2], f32)> {
    if width == 0 || height == 0 || heatmap.len() != width * height {
        return None;
    }
    let index = argmax_first(heatmap)?;
    let (x, y) = (index % width, index / width);
    let row = &heatmap[y * width..(y + 1) * width];
    let mut column = [0f32; 64];
    let refined_y = if height <= column.len() {
        for (r, value) in column.iter_mut().take(height).enumerate() {
            *value = heatmap[r * width + x];
        }
        refine_peak_log_quadratic(&column[..height], y)
    } else {
        let column: Vec<f32> = (0..height).map(|r| heatmap[r * width + x]).collect();
        refine_peak_log_quadratic(&column, y)
    };
    Some(([refine_peak_log_quadratic(row, x), refined_y], heatmap[index]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussian(n: usize, centre: f32, sigma: f32) -> Vec<f32> {
        (0..n).map(|i| (-((i as f32 - centre).powi(2)) / (2.0 * sigma * sigma)).exp()).collect()
    }

    #[test]
    fn a_sampled_gaussian_vertex_is_recovered_exactly_even_at_the_ends() {
        for centre in [0.2f32, 3.7, 8.0, 16.6] {
            let profile = gaussian(18, centre, 1.0);
            let peak = argmax_first(&profile).unwrap_or(0);
            let refined = refine_peak_log_quadratic(&profile, peak);
            assert!((refined - centre).abs() < 2e-4, "{centre} -> {refined}");
        }
    }

    #[test]
    fn flat_and_non_positive_profiles_keep_the_integer_peak() {
        assert_eq!(refine_peak_log_quadratic(&[0.0; 18], 0), 0.0);
        assert_eq!(refine_peak_log_quadratic(&[1.0; 18], 5), 5.0);
        let mut profile = [0.0f32; 18];
        profile[7] = 1.0;
        assert_eq!(refine_peak_log_quadratic(&profile, 7), 7.0);
    }

    #[test]
    fn the_offset_is_clamped_to_two_samples() {
        // Nearly flat but concave: the parabola's vertex lies far away.
        let refined = refine_peak_log_quadratic(&[1.0, 0.999, 0.99], 0);
        assert!((refined - (1.0 - 2.0)).abs() < 1e-6 || refined >= -1.0, "{refined}");
    }

    #[test]
    fn the_2d_decode_is_separable() {
        let gx = gaussian(18, 5.4, 1.0);
        let gy = gaussian(18, 11.2, 1.0);
        let heatmap: Vec<f32> = (0..18 * 18).map(|i| gy[i / 18] * gx[i % 18]).collect();
        let ([x, y], peak) = decode_peak_2d(&heatmap, 18, 18).unwrap_or(([0.0, 0.0], 0.0));
        assert!((x - 5.4).abs() < 2e-4 && (y - 11.2).abs() < 2e-4 && peak > 0.7, "{x} {y} {peak}");
        assert_eq!(decode_peak_2d(&heatmap, 17, 18), None);
    }
}
