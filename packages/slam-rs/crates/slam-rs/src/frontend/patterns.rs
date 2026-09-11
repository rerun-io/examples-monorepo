//! Fixed 2-D sampling offsets for optical-flow patches.
//! [`Pattern52`] has 52 taps; [`Pattern51`] scales them by one half.
//! Shipped configs select Pattern51 and the binding requires it, so sampling
//! uses bilinear interpolation. Offsets and the binary scale are exact in f32.

/// Taps in the largest pattern, and therefore the capacity of every per-patch buffer.
///
/// A fixed capacity plus a per-pattern count is what the GPU seam asks for
/// (`cubecl-portability.md` §12.2: preallocated fixed-capacity outputs with
/// counts, never a growing buffer), and it lets one buffer layout serve all four
/// patterns.
pub const MAX_PATTERN_SIZE: usize = 52;

/// Compile-time sampling pattern; the type parameter avoids per-tap indirection.
pub trait Pattern: Copy + Clone + Send + Sync + 'static {
    /// `Pattern::PATTERN_SIZE`.
    const SIZE: usize;

    /// The number `optical_flow_pattern` carries in the config JSON.
    const CODE: i32;

    /// Tap offsets in sampling order.
    const OFFSETS: &'static [[f32; 2]];
}

/// `Pattern52`, 52 taps at spacing 2.
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

/// `factor * Pattern52`, the way and build the other two.
const fn scaled_pattern52(factor: f32) -> [[f32; 2]; 52] {
    let mut out: [[f32; 2]; 52] = [[0.0; 2]; 52];
    let mut i: usize = 0;
    while i < 52 {
        out[i] = [PATTERN52_RAW[i][0] * factor, PATTERN52_RAW[i][1] * factor];
        i += 1;
    }
    out
}

/// `Pattern51` = `0.5 * Pattern52`, the shipped pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pattern51;

/// `0.5 * Pattern52::pattern2`.
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

    /// The layout of the ASCII art in : eight rows of
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
    /// SE(2) rotation column of `Jw_se2` mean-free before
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
