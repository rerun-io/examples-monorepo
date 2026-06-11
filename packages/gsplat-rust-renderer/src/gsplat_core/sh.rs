//! Spherical harmonics metadata helpers.
//!
//! The SH *evaluation* itself runs entirely on the GPU — see
//! `evaluate_sh_rgb` in `gaussian_project.wgsl` (which matches the
//! [Brush](https://github.com/ArthurBrussee/brush) renderer exactly).  The
//! CPU side only needs to map coefficient counts to SH degrees for the
//! shader uniforms and loader validation.

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
}
