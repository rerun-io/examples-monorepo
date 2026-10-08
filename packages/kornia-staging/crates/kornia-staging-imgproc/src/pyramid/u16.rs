//! Exact integer Gaussian downsampling with reusable row scratch.
use kornia_image::{Image, ImageError};

const KERNEL: [i32; 5] = [1, 4, 6, 4, 1];

/// Filter with `[1,4,6,4,1]` in both axes and floor-half the dimensions.
/// Borders use reflect-101; i32 intermediates retain all precision until the
/// final `(sum + 128) >> 8`. This convenience entry point allocates one temporary row. Use
/// [`super::PyramidPlanU16`] to reuse storage across frames.
///
/// # Arguments
/// * `src` - Dense, single-channel u16 image, at least 3 by 3.
/// * `dst` - Destination with dimensions `(src.width()/2, src.height()/2)`.
///
/// # Errors
/// Rejects incompatible geometry before writing output.
///
/// # Examples
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_staging_imgproc::pyramid::pyrdown_floor_u16;
/// let src = Image::from_size_val(ImageSize { width: 9, height: 7 }, 4242u16)?;
/// let mut dst = Image::from_size_val(ImageSize { width: 4, height: 3 }, 0u16)?;
/// pyrdown_floor_u16(&src, &mut dst)?;
/// assert!(dst.as_slice().iter().all(|&pixel| pixel == 4242));
/// # Ok::<(), kornia_image::ImageError>(())
/// ```
pub fn pyrdown_floor_u16(src: &Image<u16, 1>, dst: &mut Image<u16, 1>) -> Result<(), ImageError> {
    if src.width() < 3 || src.height() < 3 {
        return Err(ImageError::InvalidImageSize(
            src.width(),
            src.height(),
            3,
            3,
        ));
    }
    if dst.width() != src.width() / 2 || dst.height() != src.height() / 2 {
        return Err(ImageError::InvalidImageSize(
            dst.width(),
            dst.height(),
            src.width() / 2,
            src.height() / 2,
        ));
    }
    let mut scratch = vec![0; src.width()];
    pyrdown_floor_u16_with_scratch(src, dst, &mut scratch);
    Ok(())
}

/// Filter a level whose geometry and scratch have already been checked.
///
/// # Arguments
/// As [`pyrdown_floor_u16`]. Source sides must be at least three, destination sides
/// must be floor-halved, and scratch must cover the source width.
///
/// # Panics
/// Indexing may panic if these preconditions are not met.
#[inline]
pub(super) fn pyrdown_floor_u16_with_scratch(
    src: &Image<u16, 1>,
    dst: &mut Image<u16, 1>,
    scratch: &mut [i32],
) {
    let src_width: usize = src.width();
    let src_height: usize = src.height();
    let dst_width: usize = dst.width();
    debug_assert_eq!(dst_width, src_width >> 1);
    debug_assert_eq!(dst.height(), src_height >> 1);
    let pixels = src.as_slice();
    let output = dst.as_slice_mut();

    // Vertical convolution, one accumulator row per destination row.
    for (r, out_row) in output.chunks_exact_mut(dst_width).enumerate() {
        let row2: i64 = 2 * r as i64;
        // `std::abs(2 * r - 2)` and `std::abs(2 * r - 1)`, not `border101`.
        let rows: [usize; 5] = [
            (row2 - 2).unsigned_abs() as usize,
            (row2 - 1).unsigned_abs() as usize,
            row2 as usize,
            border101(row2 + 1, src_height as i64) as usize,
            border101(row2 + 2, src_height as i64) as usize,
        ];
        let [row_m2, row_m1, row_0, row_p1, row_p2] =
            rows.map(|row| &pixels[row * src_width..][..src_width]);
        // `tmp(r, c)`, one contiguous run of `c` rather than one column of it.
        let band: &mut [i32] = &mut scratch[..src_width];
        for c in 0..src_width {
            band[c] = KERNEL[0] * i32::from(row_m2[c])
                + KERNEL[1] * i32::from(row_m1[c])
                + KERNEL[2] * i32::from(row_0[c])
                + KERNEL[3] * i32::from(row_p1[c])
                + KERNEL[4] * i32::from(row_p2[c]);
        }
        // Consume the vertical row immediately. Reflection is about the source
        // width, and rounding still occurs only after both integer passes.
        for (c, pixel) in out_row.iter_mut().enumerate() {
            // Interior five-tap windows are contiguous. Peel low/high border columns to keep
            // reflection arithmetic out of the large interior loop.
            let value: i32 = match (2 * c)
                .checked_sub(2)
                .and_then(|first| band.get(first..)?.first_chunk::<5>())
            {
                Some(window) => {
                    KERNEL[0] * window[0]
                        + KERNEL[1] * window[1]
                        + KERNEL[2] * window[2]
                        + KERNEL[3] * window[3]
                        + KERNEL[4] * window[4]
                }
                None => {
                    let col2: i64 = 2 * c as i64;
                    let columns: [usize; 5] = [
                        (col2 - 2).unsigned_abs() as usize,
                        (col2 - 1).unsigned_abs() as usize,
                        col2 as usize,
                        border101(col2 + 1, src_width as i64) as usize,
                        border101(col2 + 2, src_width as i64) as usize,
                    ];
                    KERNEL[0] * band[columns[0]]
                        + KERNEL[1] * band[columns[1]]
                        + KERNEL[2] * band[columns[2]]
                        + KERNEL[3] * band[columns[3]]
                        + KERNEL[4] * band[columns[4]]
                }
            };
            // `T val = ((val_int + (1 << 7)) >> 8)`. The
            // accumulator peaks at 65535 * 16 * 16, so the shift lands back in
            // `u16` exactly and the cast never truncates.
            *pixel = ((value + (1 << 7)) >> 8) as u16;
        }
    }
}
/// High-end reflect-101: `h - 1 - |h - 1 - x|` for non-negative x.
/// Negative indices instead reflect with absolute value; the formulas are not
/// interchangeable outside their domains.
#[inline]
fn border101(x: i64, h: i64) -> i64 {
    h - 1 - (h - 1 - x).abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    use crate::test_fixtures::{random_image, zeros};

    #[test]
    fn checked_boundary_rejects_invalid_geometry_before_writing() {
        let source = zeros(9, 7);
        let mut wrong = Image::from_size_val(
            kornia_image::ImageSize {
                width: 5,
                height: 3,
            },
            123,
        )
        .unwrap();
        assert!(pyrdown_floor_u16(&source, &mut wrong).is_err());
        assert!(wrong.as_slice().iter().all(|&value| value == 123));
        assert!(pyrdown_floor_u16(&zeros(2, 7), &mut zeros(1, 3)).is_err());
    }
    /// Independent reflect-101 implementation that repeatedly mirrors into `[0, n)`.
    fn reflect101_naive(mut index: i64, n: i64) -> i64 {
        assert!(n >= 2, "reflect-101 needs at least two samples");
        loop {
            if index < 0 {
                index = -index;
            } else if index >= n {
                index = 2 * (n - 1) - index;
            } else {
                return index;
            }
        }
    }

    /// Independent direct 5x5 convolution with reflect-101 borders and one final rounding.
    fn subsample_naive(src: &Image<u16, 1>) -> Image<u16, 1> {
        let width: usize = src.width() >> 1;
        let height: usize = src.height() >> 1;
        let mut dst: Image<u16, 1> = zeros(width, height);
        for r in 0..height {
            for c in 0..width {
                let mut sum: i64 = 0;
                for (dy, ky) in KERNEL.iter().enumerate() {
                    let y: i64 =
                        reflect101_naive(2 * r as i64 + dy as i64 - 2, src.height() as i64);
                    for (dx, kx) in KERNEL.iter().enumerate() {
                        let x: i64 =
                            reflect101_naive(2 * c as i64 + dx as i64 - 2, src.width() as i64);
                        let pixel: i64 =
                            i64::from(src.get_pixel(x as usize, y as usize, 0).copied().unwrap());
                        sum += i64::from(*ky) * i64::from(*kx) * pixel;
                    }
                }
                dst.set_pixel(c, r, 0, ((sum + 128) >> 8) as u16).unwrap();
            }
        }
        dst
    }

    #[test]
    fn border101_matches_the_naive_reflection_over_its_whole_domain() {
        for n in 2i64..24 {
            // Check high-end reflection on `[0, 2*(n-1)]` and absolute-value reflection below
            // zero against the independent implementation.
            for x in 0..=2 * (n - 1) {
                assert_eq!(
                    border101(x, n),
                    reflect101_naive(x, n),
                    "border101({x}, {n})"
                );
            }
            for x in -(n - 1)..0 {
                assert_eq!(x.abs(), reflect101_naive(x, n), "abs({x}) for n = {n}");
                assert_ne!(
                    border101(x, n),
                    reflect101_naive(x, n),
                    "border101({x}, {n})"
                );
            }
        }
    }

    /// Compare with kornia's independent integer pyramid filter.
    /// Both use `[1,4,6,4,1]`, reflect-101 and one rounding `(sum + 128) >> 8`.
    /// Use unshifted values 0–255: filtering widened `v << 8` and narrowing afterwards
    /// rounds at a different magnitude and need not agree. Even dimensions avoid
    /// kornia's ceil-half versus this crate's floor-half geometry difference.
    #[test]
    fn subsample_matches_kornia_pyrdown_u8_on_byte_valued_pixels() {
        use kornia_image::{Image, ImageSize};

        for (width, height, seed) in [(16usize, 12usize, 11u64), (64, 64, 12), (34, 18, 13)] {
            let bytes: Vec<u8> = {
                let mut state: u64 = seed | 1;
                (0..width * height)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1);
                        (state >> 33) as u8
                    })
                    .collect()
            };

            let source: Image<u8, 1> =
                Image::new(ImageSize { width, height }, bytes.clone()).unwrap();
            let mut kornia_out: Image<u8, 1> = Image::from_size_val(
                ImageSize {
                    width: width / 2,
                    height: height / 2,
                },
                0u8,
            )
            .unwrap();
            kornia_imgproc::pyramid::pyrdown_u8(&source, &mut kornia_out).unwrap();

            // Our own subsample over the same values, held in `u16` with no shift.
            let mut ours: Image<u16, 1> = zeros(width, height);
            for (pixel, byte) in ours.as_slice_mut().iter_mut().zip(&bytes) {
                *pixel = u16::from(*byte);
            }
            let mut got: Image<u16, 1> = zeros(width / 2, height / 2);
            pyrdown_floor_u16(&ours, &mut got).unwrap();

            let expected: Vec<u16> = kornia_out
                .as_slice()
                .iter()
                .map(|byte| u16::from(*byte))
                .collect();
            assert_eq!(got.as_slice(), expected.as_slice(), "{width}x{height}");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]

        /// The separable, `abs`/`border101` implementation with its
        /// row-major accumulator and a direct 5x5 convolution with true
        /// reflect-101 borders agree bit for bit, on even and odd sizes alike.
        #[test]
        fn subsample_matches_a_naive_5x5_convolution(
            width in 3usize..40,
            height in 3usize..40,
            seed in any::<u64>(),
        ) {
            let image: Image<u16, 1> = random_image(width, height, seed);
            let expected: Image<u16, 1> = subsample_naive(&image);
            let mut got: Image<u16, 1> = zeros(width >> 1, height >> 1);
            pyrdown_floor_u16(&image, &mut got).unwrap();
            prop_assert_eq!(got.as_slice(), expected.as_slice());
        }

        /// Every level of a whole pyramid, not just the first subsample.
        #[test]
        fn every_pyramid_level_matches_the_naive_reference(
            width in 24usize..70,
            height in 24usize..70,
            seed in any::<u64>(),
        ) {
            let image: Image<u16, 1> = random_image(width, height, seed);
            let mut pyramid = super::super::PyramidPlanU16::new(image.size(), 3).unwrap();
            pyramid.run(&image).unwrap();
            let mut expected: Image<u16, 1> = image.clone();
            for level in 1..pyramid.levels().len() {
                expected = subsample_naive(&expected);
                prop_assert_eq!(pyramid.levels().get(level).unwrap().as_slice(), expected.as_slice());
            }
        }

        /// A subsampled level never exceeds the source's range: the kernel is a
        /// normalized average, so it cannot overshoot and cannot wrap the cast.
        #[test]
        fn subsample_stays_inside_the_source_range(
            width in 3usize..40,
            height in 3usize..40,
            seed in any::<u64>(),
        ) {
            let image: Image<u16, 1> = random_image(width, height, seed);
            let mut pyramid = super::super::PyramidPlanU16::new(image.size(), 1).unwrap();
            pyramid.run(&image).unwrap();
            let low: u16 = *image.as_slice().iter().min().unwrap();
            let high: u16 = *image.as_slice().iter().max().unwrap();
            for pixel in pyramid.levels().get(1).unwrap().as_slice() {
                prop_assert!(*pixel >= low && *pixel <= high);
            }
        }
    }
}
