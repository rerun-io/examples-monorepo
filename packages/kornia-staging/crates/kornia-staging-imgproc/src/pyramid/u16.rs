//! Exact integer Gaussian downsampling with reusable row scratch.
use kornia_image::{Image, ImageError};

const KERNEL: [i32; 5] = [1, 4, 6, 4, 1];

/// Reusable vertical-filter row for [`pyrdown_u16`].
/// A larger row can serve subsequent, smaller levels without reallocating.
#[derive(Debug, Clone, Default)]
pub struct PyrDownU16Scratch {
    row: Vec<i32>,
}

impl PyrDownU16Scratch {
    /// Allocate scratch for source rows up to `width` pixels.
    ///
    /// # Arguments
    /// * `width` - Largest source row this scratch will filter.
    ///
    /// # Errors
    /// Rejects widths that exceed the single-allocation byte limit.
    pub fn new(width: usize) -> Result<Self, ImageError> {
        let mut scratch = Self::default();
        scratch.prepare(width)?;
        Ok(scratch)
    }

    /// Grow once for a new largest source width; reuse existing storage otherwise.
    ///
    /// # Arguments
    /// * `width` - Required source row length.
    ///
    /// # Errors
    /// Rejects widths that exceed the single-allocation byte limit.
    pub fn prepare(&mut self, width: usize) -> Result<(), ImageError> {
        if width <= self.row.len() {
            return Ok(());
        }
        let max = isize::MAX as usize / size_of::<i32>();
        if width > max {
            return Err(ImageError::InvalidChannelShape(width, max));
        }
        self.row.resize(width, 0);
        Ok(())
    }
}

/// Filter with `[1,4,6,4,1]` in both axes and floor-half the dimensions.
/// Borders use reflect-101; i32 intermediates retain all precision until the
/// final `(sum + 128) >> 8`. Scratch is caller-owned and is not resized here.
///
/// # Arguments
/// * `src` - Dense, single-channel u16 image, at least 3 by 3.
/// * `dst` - Destination with dimensions `(src.width()/2, src.height()/2)`.
/// * `scratch` - Row storage prepared for at least `src.width()` pixels.
///
/// # Errors
/// Rejects incompatible geometry or insufficient scratch before writing output.
///
/// # Examples
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_staging_imgproc::pyramid::{pyrdown_u16, PyrDownU16Scratch};
/// let src = Image::from_size_val(ImageSize { width: 9, height: 7 }, 4242u16)?;
/// let mut dst = Image::from_size_val(ImageSize { width: 4, height: 3 }, 0u16)?;
/// let mut scratch = PyrDownU16Scratch::new(9)?;
/// pyrdown_u16(&src, &mut dst, &mut scratch)?;
/// assert!(dst.as_slice().iter().all(|&pixel| pixel == 4242));
/// # Ok::<(), kornia_image::ImageError>(())
/// ```
pub fn pyrdown_u16(
    src: &Image<u16, 1>,
    dst: &mut Image<u16, 1>,
    scratch: &mut PyrDownU16Scratch,
) -> Result<(), ImageError> {
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
    if scratch.row.len() < src.width() {
        return Err(ImageError::InvalidChannelShape(
            scratch.row.len(),
            src.width(),
        ));
    }
    pyrdown_u16_unchecked(src, dst, scratch);
    Ok(())
}

/// Filter a level whose geometry and scratch have already been checked.
///
/// # Arguments
/// As [`pyrdown_u16`]. Source sides must be at least three, destination sides
/// must be floor-halved, and scratch must cover the source width.
///
/// # Panics
/// Indexing may panic if these preconditions are not met.
#[inline]
pub fn pyrdown_u16_unchecked(
    src: &Image<u16, 1>,
    dst: &mut Image<u16, 1>,
    scratch: &mut PyrDownU16Scratch,
) {
    let scratch = &mut scratch.row;
    let src_width: usize = src.width();
    let src_height: usize = src.height();
    let dst_width: usize = dst.width();
    let dst_height: usize = dst.height();
    debug_assert_eq!(dst_width, src_width >> 1);
    debug_assert_eq!(dst_height, src_height >> 1);

    // Vertical convolution, one accumulator row per destination row.
    for r in 0..dst_height {
        let row2: i64 = 2 * r as i64;
        // `std::abs(2 * r - 2)` and `std::abs(2 * r - 1)`, not `border101`.
        let rows: [usize; 5] = [
            (row2 - 2).unsigned_abs() as usize,
            (row2 - 1).unsigned_abs() as usize,
            row2 as usize,
            border101(row2 + 1, src_height as i64) as usize,
            border101(row2 + 2, src_height as i64) as usize,
        ];
        let [row_m2, row_m1, row_0, row_p1, row_p2]: [&[u16]; 5] = [
            &src.as_slice()[(rows[0]) * src.width()..((rows[0]) + 1) * src.width()],
            &src.as_slice()[(rows[1]) * src.width()..((rows[1]) + 1) * src.width()],
            &src.as_slice()[(rows[2]) * src.width()..((rows[2]) + 1) * src.width()],
            &src.as_slice()[(rows[3]) * src.width()..((rows[3]) + 1) * src.width()],
            &src.as_slice()[(rows[4]) * src.width()..((rows[4]) + 1) * src.width()],
        ];
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
        for (c, pixel) in {
            let width = dst.width();
            &mut dst.as_slice_mut()[(r) * width..((r) + 1) * width]
        }
        .iter_mut()
        .enumerate()
        {
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
/// interchangeable outside their domains (trap 3).
#[inline]
fn border101(x: i64, h: i64) -> i64 {
    h - 1 - (h - 1 - x).abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn zeros(width: usize, height: usize) -> Result<Image<u16, 1>, ImageError> {
        Image::from_size_val(kornia_image::ImageSize { width, height }, 0)
    }
    fn build(image: &Image<u16, 1>, reductions: usize) -> Vec<Image<u16, 1>> {
        let mut levels = vec![image.clone()];
        let mut scratch = PyrDownU16Scratch::new(image.width()).unwrap();
        for level in 0..reductions {
            let source = &levels[level];
            let mut next = zeros(source.width() / 2, source.height() / 2).unwrap();
            pyrdown_u16(source, &mut next, &mut scratch).unwrap();
            levels.push(next);
        }
        levels
    }
    #[test]
    fn scratch_reuses_the_largest_row_and_rejects_unallocatable_widths() {
        let mut scratch = PyrDownU16Scratch::new(96).unwrap();
        let pointer = scratch.row.as_ptr();
        let capacity = scratch.row.capacity();
        for width in [96, 48, 24, 96] {
            scratch.prepare(width).unwrap();
        }
        assert_eq!(scratch.row.as_ptr(), pointer);
        assert_eq!(scratch.row.capacity(), capacity);
        assert!(PyrDownU16Scratch::new(usize::MAX).is_err());
        assert!(PyrDownU16Scratch::new(isize::MAX as usize / size_of::<i32>() + 1).is_err());
    }
    #[test]
    fn checked_boundary_rejects_invalid_geometry_before_writing() {
        let source = zeros(9, 7).unwrap();
        let mut destination = Image::from_size_val(
            kornia_image::ImageSize {
                width: 4,
                height: 3,
            },
            123,
        )
        .unwrap();
        let mut short = PyrDownU16Scratch::new(8).unwrap();
        assert!(pyrdown_u16(&source, &mut destination, &mut short).is_err());
        assert!(destination.as_slice().iter().all(|&v| v == 123));
        let mut scratch = PyrDownU16Scratch::new(9).unwrap();
        assert!(pyrdown_u16(&source, &mut zeros(5, 3).unwrap(), &mut scratch).is_err());
        assert!(pyrdown_u16(
            &zeros(2, 7).unwrap(),
            &mut zeros(1, 3).unwrap(),
            &mut scratch
        )
        .is_err());
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
        let mut dst: Image<u16, 1> = zeros(width, height).unwrap();
        for r in 0..height {
            for c in 0..width {
                let mut sum: i64 = 0;
                for (dy, ky) in KERNEL.iter().enumerate() {
                    let y: i64 =
                        reflect101_naive(2 * r as i64 + dy as i64 - 2, src.height() as i64);
                    for (dx, kx) in KERNEL.iter().enumerate() {
                        let x: i64 =
                            reflect101_naive(2 * c as i64 + dx as i64 - 2, src.width() as i64);
                        let pixel: i64 = i64::from(
                            src.get_pixel(x as usize, y as usize, 0)
                                .copied()
                                .ok()
                                .unwrap(),
                        );
                        sum += i64::from(*ky) * i64::from(*kx) * pixel;
                    }
                }
                dst.set_pixel(c, r, 0, ((sum + 128) >> 8) as u16).unwrap();
            }
        }
        dst
    }

    fn random_image(width: usize, height: usize, seed: u64) -> Image<u16, 1> {
        let mut image: Image<u16, 1> = zeros(width, height).unwrap();
        let mut state: u64 = seed | 1;
        for y in 0..height {
            for pixel in {
                let width = image.width();
                &mut image.as_slice_mut()[(y) * width..((y) + 1) * width]
            } {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                *pixel = (state >> 32) as u16;
            }
        }
        image
    }

    #[test]
    fn border101_matches_the_naive_reflection_over_its_whole_domain() {
        for n in 2i64..24 {
            // Check high-end reflection on `[0, 2*(n-1)]` and absolute-value reflection below
            // zero against the independent implementation (trap 3).
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
            let mut ours: Image<u16, 1> = zeros(width, height).unwrap();
            for (y, row) in bytes.chunks_exact(width).enumerate() {
                for (pixel, byte) in {
                    let width = ours.width();
                    &mut ours.as_slice_mut()[(y) * width..((y) + 1) * width]
                }
                .iter_mut()
                .zip(row)
                {
                    *pixel = u16::from(*byte);
                }
            }
            let mut got: Image<u16, 1> = zeros(width / 2, height / 2).unwrap();
            let mut scratch = PyrDownU16Scratch::new(width).unwrap();
            pyrdown_u16(&ours, &mut got, &mut scratch).unwrap();

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
            let mut got: Image<u16, 1> = zeros(width >> 1, height >> 1).unwrap();
            let mut scratch = PyrDownU16Scratch::new(width).unwrap();
            pyrdown_u16(&image, &mut got, &mut scratch).unwrap();
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
            let pyramid = build(&image, 3);
            let mut expected: Image<u16, 1> = image.clone();
            for level in 1..pyramid.len() {
                expected = subsample_naive(&expected);
                prop_assert_eq!(pyramid.get(level).unwrap().as_slice(), expected.as_slice());
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
            let pyramid = build(&image, 1);
            let low: u16 = *image.as_slice().iter().min().unwrap();
            let high: u16 = *image.as_slice().iter().max().unwrap();
            for pixel in pyramid.get(1).unwrap().as_slice() {
                prop_assert!(*pixel >= low && *pixel <= high);
            }
        }
    }
}
