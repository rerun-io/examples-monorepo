//! Spherical harmonics evaluation (CPU path).
//!
//! Evaluates real spherical harmonic basis functions up to degree 4 (25
//! coefficients per channel).  The math matches `gaussian_project.wgsl` and
//! the [Brush](https://github.com/ArthurBrussee/brush) renderer exactly —
//! same basis constants, same evaluation order.

use glam::Vec3;

use super::constants::SH_C0;

/// Map the number of SH coefficients per channel to the SH degree.
///
/// | Coefficients | Degree | Bands |
/// |-------------|--------|-------|
/// | 1           | 0      | DC only |
/// | 4           | 1      | + 3 first-order terms |
/// | 9           | 2      | + 5 second-order terms |
/// | 16          | 3      | + 7 third-order terms |
/// | 25          | 4      | + 9 fourth-order terms |
pub fn sh_degree_from_coeffs(coeffs_per_channel: usize) -> Option<u32> {
    match coeffs_per_channel {
        1 => Some(0),
        4 => Some(1),
        9 => Some(2),
        16 => Some(3),
        25 => Some(4),
        _ => None,
    }
}

/// Evaluate spherical harmonics for a given view direction and return the
/// final activated RGB color.
///
/// This is the CPU-side equivalent of `evaluate_sh_rgb` in `gaussian_project.wgsl`.
/// The GPU shader evaluates the same math per-splat in parallel.
pub fn evaluate_sh_rgb(
    coefficients: &[f32],
    coeffs_per_channel: usize,
    view_direction: Vec3,
) -> Option<[f32; 3]> {
    evaluate_sh_rgb_raw(coefficients, coeffs_per_channel, view_direction).map(activate_sh_rgb)
}

/// Apply the SH activation function: `max(raw + 0.5, 0.0)`.
/// The `+0.5` bias shifts the DC baseline so that a zero-valued SH field
/// produces a mid-gray color rather than black.
fn activate_sh_rgb(raw_rgb: [f32; 3]) -> [f32; 3] {
    [
        (raw_rgb[0] + 0.5).max(0.0),
        (raw_rgb[1] + 0.5).max(0.0),
        (raw_rgb[2] + 0.5).max(0.0),
    ]
}

/// Evaluate the raw (pre-activation) SH expansion up to the given degree.
///
/// The mathematical expansion uses real spherical harmonic basis functions
/// evaluated at the normalized view direction.  Coefficients are packed as
/// `[coeff_index * 3 + channel]` so `coeff(i)` returns the RGB triplet for
/// basis function `i`.
///
/// The magic constants are the real SH basis function normalizations.
/// They match the values used in 3DGS, Brush, and gsplat.
fn evaluate_sh_rgb_raw(
    coefficients: &[f32],
    coeffs_per_channel: usize,
    view_direction: Vec3,
) -> Option<[f32; 3]> {
    let degree = sh_degree_from_coeffs(coeffs_per_channel)?;
    if coefficients.len() < coeffs_per_channel * 3 {
        return None;
    }

    let viewdir = if view_direction.length_squared() > 1e-12 {
        view_direction.normalize()
    } else {
        Vec3::Z
    };
    // Helper closure: read the RGB triplet for SH basis function `basis_index`.
    let coeff = |basis_index: usize| -> Vec3 {
        let offset = basis_index * 3;
        Vec3::new(
            coefficients[offset],
            coefficients[offset + 1],
            coefficients[offset + 2],
        )
    };

    // ── Degree 0 (DC): 1 coefficient ─────────────────────────────────
    let mut color = SH_C0 * coeff(0);
    if degree == 0 {
        return Some(color.to_array());
    }

    let x = viewdir.x;
    let y = viewdir.y;
    let z = viewdir.z;

    // ── Degree 1: 3 additional coefficients (Y₁⁻¹, Y₁⁰, Y₁¹) ───────
    let f_tmp_0a = 0.488_602_52_f32;
    color += f_tmp_0a * (-y * coeff(1) + z * coeff(2) - x * coeff(3));
    if degree == 1 {
        return Some(color.to_array());
    }

    // ── Degree 2: 5 additional coefficients ──────────────────────────
    let z2 = z * z;

    let f_tmp_0b = -1.092_548_5_f32 * z;
    let f_tmp_1a = 0.546_274_24_f32;
    let f_c1 = x * x - y * y;
    let f_s1 = 2.0 * x * y;
    let p_sh_6 = 0.946_174_7_f32 * z2 - 0.315_391_57_f32;
    let p_sh_7 = f_tmp_0b * x;
    let p_sh_5 = f_tmp_0b * y;
    let p_sh_8 = f_tmp_1a * f_c1;
    let p_sh_4 = f_tmp_1a * f_s1;

    color += p_sh_4 * coeff(4)
        + p_sh_5 * coeff(5)
        + p_sh_6 * coeff(6)
        + p_sh_7 * coeff(7)
        + p_sh_8 * coeff(8);
    if degree == 2 {
        return Some(color.to_array());
    }

    // ── Degree 3: 7 additional coefficients ──────────────────────────
    let f_tmp_0c = -2.285_229_f32 * z2 + 0.457_045_8_f32;
    let f_tmp_1b = 1.445_305_7_f32 * z;
    let f_tmp_2a = -0.590_043_6_f32;
    let f_c2 = x * f_c1 - y * f_s1;
    let f_s2 = x * f_s1 + y * f_c1;
    let p_sh_12 = z * (1.865_881_7_f32 * z2 - 1.119_529_f32);
    let p_sh_13 = f_tmp_0c * x;
    let p_sh_11 = f_tmp_0c * y;
    let p_sh_14 = f_tmp_1b * f_c1;
    let p_sh_10 = f_tmp_1b * f_s1;
    let p_sh_15 = f_tmp_2a * f_c2;
    let p_sh_9 = f_tmp_2a * f_s2;

    color += p_sh_9 * coeff(9)
        + p_sh_10 * coeff(10)
        + p_sh_11 * coeff(11)
        + p_sh_12 * coeff(12)
        + p_sh_13 * coeff(13)
        + p_sh_14 * coeff(14)
        + p_sh_15 * coeff(15);
    if degree == 3 {
        return Some(color.to_array());
    }

    // ── Degree 4: 9 additional coefficients ──────────────────────────
    let f_tmp_0d = z * (-4.683_326_f32 * z2 + 2.007_139_7_f32);
    let f_tmp_1c = 3.311_611_4_f32 * z2 - 0.473_087_34_f32;
    let f_tmp_2b = -1.770_130_8_f32 * z;
    let f_tmp_3a = 0.625_835_7_f32;
    let f_c3 = x * f_c2 - y * f_s2;
    let f_s3 = x * f_s2 + y * f_c2;
    let p_sh_20 = 1.984_313_5_f32 * z * p_sh_12 - 1.006_230_6_f32 * p_sh_6;
    let p_sh_21 = f_tmp_0d * x;
    let p_sh_19 = f_tmp_0d * y;
    let p_sh_22 = f_tmp_1c * f_c1;
    let p_sh_18 = f_tmp_1c * f_s1;
    let p_sh_23 = f_tmp_2b * f_c2;
    let p_sh_17 = f_tmp_2b * f_s2;
    let p_sh_24 = f_tmp_3a * f_c3;
    let p_sh_16 = f_tmp_3a * f_s3;

    color += p_sh_16 * coeff(16)
        + p_sh_17 * coeff(17)
        + p_sh_18 * coeff(18)
        + p_sh_19 * coeff(19)
        + p_sh_20 * coeff(20)
        + p_sh_21 * coeff(21)
        + p_sh_22 * coeff(22)
        + p_sh_23 * coeff(23)
        + p_sh_24 * coeff(24);
    Some(color.to_array())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sh_degree_lookup() {
        assert_eq!(sh_degree_from_coeffs(1), Some(0));
        assert_eq!(sh_degree_from_coeffs(4), Some(1));
        assert_eq!(sh_degree_from_coeffs(9), Some(2));
        assert_eq!(sh_degree_from_coeffs(16), Some(3));
        assert_eq!(sh_degree_from_coeffs(25), Some(4));
        assert_eq!(sh_degree_from_coeffs(5), None);
        assert_eq!(sh_degree_from_coeffs(0), None);
    }

    #[test]
    fn sh_degree_0_produces_dc_color() {
        // SH degree 0 with DC = [1.0, 0.0, -1.0] per channel.
        // After activation: max(SH_C0 * dc + 0.5, 0.0)
        let coeffs = [1.0_f32, 0.0, -1.0];
        let result = evaluate_sh_rgb(&coeffs, 1, Vec3::Z).unwrap();
        let expected_r = (SH_C0 * 1.0 + 0.5).max(0.0);
        let expected_g = (SH_C0 * 0.0 + 0.5).max(0.0);
        let expected_b = (-SH_C0 + 0.5).max(0.0);
        assert!((result[0] - expected_r).abs() < 1e-6);
        assert!((result[1] - expected_g).abs() < 1e-6);
        assert!((result[2] - expected_b).abs() < 1e-6);
    }

    #[test]
    fn sh_returns_none_for_too_short_coefficients() {
        let coeffs = [1.0_f32, 0.0]; // Need at least 3 for degree 0
        assert!(evaluate_sh_rgb(&coeffs, 1, Vec3::Z).is_none());
    }

    #[test]
    fn sh_returns_none_for_invalid_degree() {
        let coeffs = [0.0_f32; 15]; // 5 coefficients per channel, invalid
        assert!(evaluate_sh_rgb(&coeffs, 5, Vec3::Z).is_none());
    }
}
