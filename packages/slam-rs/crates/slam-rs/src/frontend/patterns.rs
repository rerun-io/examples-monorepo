//! basalt's four sampling patterns, ported from `optical_flow/patterns.h`.
//!
//! Each pattern is a fixed set of 2-D offsets from a keypoint. The C++ stores
//! them as `constexpr Scalar pattern_raw[][2]` and maps them into a column-major
//! `Eigen::Matrix<Scalar, 2, PATTERN_SIZE>`, so column `i` of `pattern2` is
//! exactly `(pattern_raw[i][0], pattern_raw[i][1])` (`patterns.h:75-77`). Here
//! that is one `[[f32; 2]]` slice in the same order, so index `i` is column `i`.
//!
//! * [`Pattern52`] — 52 taps at unit half-spacing, `patterns.h:107-126`.
//! * [`Pattern51`] — `0.5 * Pattern52`, `patterns.h:146`.
//!
//! **Every shipped config sets `optical_flow_pattern = 51`**
//! (`data/default_config.json`, `data/msd/*_config.json`), i.e. 52 taps at half
//! spacing, so every sample of the shipped frontend is a genuine bilinear
//! interpolation rather than a pixel read.
//! [`crate::frontend::flow::FrameToFrameOpticalFlow`] refuses any other value,
//! and the binding hard-codes `Pattern51`.
//!
//! The C++ has two more tables that are **not** ported, because no config
//! selects them and nothing here could exercise them: `Pattern24`
//! (`patterns.h:44-79`, 24 taps) and `Pattern50` (`patterns.h:148-158`,
//! `0.75 * Pattern52`). Adding either back is a `Pattern` impl over its raw
//! table beside `Pattern51`'s.
//!
//! The port is `f32` only (decision D05: the frontend runs `f32` on `u16`
//! pixels). The scale factors are exact binary fractions and every raw offset is
//! a small integer, so `0.5 *` and `0.75 *` are exact in both `f32` and the
//! `double` Eigen evaluates them in.

/// Taps in the largest pattern, and therefore the capacity of every per-patch buffer.
///
/// A fixed capacity plus a per-pattern count is what the GPU seam asks for
/// (`cubecl-portability.md` §12.2: preallocated fixed-capacity outputs with
/// counts, never a growing buffer), and it lets one buffer layout serve all four
/// patterns.
pub const MAX_PATTERN_SIZE: usize = 52;

/// One of basalt's sampling patterns, as a compile-time choice.
///
/// The C++ passes the pattern as a template template parameter
/// (`OpticalFlowTyped<Scalar, Pattern>`, `optical_flow.h:184`); the port passes
/// it as a type parameter with the same effect: no indirection at a tap.
pub trait Pattern: Copy + Clone + Send + Sync + 'static {
    /// `Pattern::PATTERN_SIZE`.
    const SIZE: usize;

    /// The number `optical_flow_pattern` carries in the config JSON.
    const CODE: i32;

    /// The offsets, index `i` being column `i` of the C++ `pattern2`.
    const OFFSETS: &'static [[f32; 2]];
}

/// `Pattern52` (`patterns.h:81-131`), 52 taps at spacing 2.
///
/// ```text
///          00  01  02  03
///      04  05  06  07  08  09
///  10  11  12  13  14  15  16  17
///  18  19  20  21  22  23  24  25
///  26  27  28  29  30  31  32  33
///  34  35  36  37  38  39  40  41
///      42  43  44  45  46  47
///          48  49  50  51
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pattern52;

/// `patterns.h:107-126`, verbatim.
const PATTERN52_RAW: [[f32; 2]; 52] = [
    [-3.0, 7.0],
    [-1.0, 7.0],
    [1.0, 7.0],
    [3.0, 7.0],
    [-5.0, 5.0],
    [-3.0, 5.0],
    [-1.0, 5.0],
    [1.0, 5.0],
    [3.0, 5.0],
    [5.0, 5.0],
    [-7.0, 3.0],
    [-5.0, 3.0],
    [-3.0, 3.0],
    [-1.0, 3.0],
    [1.0, 3.0],
    [3.0, 3.0],
    [5.0, 3.0],
    [7.0, 3.0],
    [-7.0, 1.0],
    [-5.0, 1.0],
    [-3.0, 1.0],
    [-1.0, 1.0],
    [1.0, 1.0],
    [3.0, 1.0],
    [5.0, 1.0],
    [7.0, 1.0],
    [-7.0, -1.0],
    [-5.0, -1.0],
    [-3.0, -1.0],
    [-1.0, -1.0],
    [1.0, -1.0],
    [3.0, -1.0],
    [5.0, -1.0],
    [7.0, -1.0],
    [-7.0, -3.0],
    [-5.0, -3.0],
    [-3.0, -3.0],
    [-1.0, -3.0],
    [1.0, -3.0],
    [3.0, -3.0],
    [5.0, -3.0],
    [7.0, -3.0],
    [-5.0, -5.0],
    [-3.0, -5.0],
    [-1.0, -5.0],
    [1.0, -5.0],
    [3.0, -5.0],
    [5.0, -5.0],
    [-3.0, -7.0],
    [-1.0, -7.0],
    [1.0, -7.0],
    [3.0, -7.0],
];

impl Pattern for Pattern52 {
    const SIZE: usize = 52;
    const CODE: i32 = 52;
    const OFFSETS: &'static [[f32; 2]] = &PATTERN52_RAW;
}

/// `factor * Pattern52`, the way `patterns.h:146` and `:158` build the other two.
const fn scaled_pattern52(factor: f32) -> [[f32; 2]; 52] {
    let mut out: [[f32; 2]; 52] = [[0.0; 2]; 52];
    let mut i: usize = 0;
    while i < 52 {
        out[i] = [PATTERN52_RAW[i][0] * factor, PATTERN52_RAW[i][1] * factor];
        i += 1;
    }
    out
}

/// `Pattern51` = `0.5 * Pattern52` (`patterns.h:133-146`), the shipped pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pattern51;

/// `0.5 * Pattern52::pattern2` (`patterns.h:146`).
const PATTERN51_RAW: [[f32; 2]; 52] = scaled_pattern52(0.5);

impl Pattern for Pattern51 {
    const SIZE: usize = 52;
    const CODE: i32 = 51;
    const OFFSETS: &'static [[f32; 2]] = &PATTERN51_RAW;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_pattern_fits_the_shared_capacity() {
        assert_eq!(Pattern52::SIZE, MAX_PATTERN_SIZE);
        assert_eq!(Pattern51::SIZE, Pattern52::SIZE);
        assert_eq!(Pattern52::OFFSETS.len(), Pattern52::SIZE);
        assert_eq!(Pattern51::OFFSETS.len(), Pattern51::SIZE);
    }

    #[test]
    fn pattern51_is_half_of_pattern52() {
        for i in 0..Pattern52::SIZE {
            for axis in 0..2 {
                assert_eq!(
                    Pattern51::OFFSETS[i][axis],
                    0.5 * Pattern52::OFFSETS[i][axis]
                );
            }
        }
    }

    /// The layout of the ASCII art in `patterns.h:82-101`: eight rows of
    /// descending `y`, and each row's `x` ascending.
    #[test]
    fn pattern52_rows_run_from_the_top_down_and_left_to_right() {
        let rows: [(f32, usize); 8] = [
            (7.0, 4),
            (5.0, 6),
            (3.0, 8),
            (1.0, 8),
            (-1.0, 8),
            (-3.0, 8),
            (-5.0, 6),
            (-7.0, 4),
        ];
        let mut index: usize = 0;
        for (y, count) in rows {
            let mut previous_x: f32 = f32::NEG_INFINITY;
            for _ in 0..count {
                let tap: [f32; 2] = Pattern52::OFFSETS[index];
                assert_eq!(tap[1], y, "tap {index} is not on row y = {y}");
                assert!(tap[0] > previous_x, "tap {index} is not left to right");
                previous_x = tap[0];
                index += 1;
            }
        }
        assert_eq!(index, Pattern52::SIZE);
    }

    /// Both patterns are symmetric about the origin, which is what makes the
    /// SE(2) rotation column of `Jw_se2` (`patch.h:114-115`) mean-free before
    /// normalisation.
    #[test]
    fn the_patterns_are_symmetric_about_the_origin() {
        for offsets in [Pattern51::OFFSETS, Pattern52::OFFSETS] {
            let sum_x: f32 = offsets.iter().map(|tap| tap[0]).sum();
            let sum_y: f32 = offsets.iter().map(|tap| tap[1]).sum();
            assert_eq!(sum_x, 0.0);
            assert_eq!(sum_y, 0.0);
        }
    }
}
