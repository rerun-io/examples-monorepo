//! Integer-factor 4x4 box pooling of a row-major u8 image: rounded to u8 (half to even, as torch rounds), as unrounded f32
//! means, or as any value of the block sums. The 4x4 case of the box (area) downscale kornia-imgproc `resize` lacks (see also
//! `area::resize_area_u8`).
//! Target upstream: kornia-imgproc `resize`.

use kornia_image::ImageError;

/// The 4x4 block sums of output row `out_row` into `sums` (`width / 4` values).
fn block_sums(src: &[u8], width: usize, out_row: usize, sums: &mut [u16]) {
    sums.fill(0);
    for row in src[out_row * 4 * width..(out_row * 4 + 4) * width].chunks_exact(width) {
        for (sum, block) in sums.iter_mut().zip(row.as_chunks::<4>().0.iter()) {
            *sum += u16::from(block[0])
                + u16::from(block[1])
                + u16::from(block[2])
                + u16::from(block[3]);
        }
    }
}

fn check(src: usize, width: usize, height: usize, dst: usize) -> Result<(), ImageError> {
    let invalid = || ImageError::InvalidImageSize(width, height, width / 4, height / 4);
    if width == 0 || height == 0 || !width.is_multiple_of(4) || !height.is_multiple_of(4) {
        return Err(invalid());
    }
    let src_len = width.checked_mul(height).ok_or_else(invalid)?;
    let dst_len = (width / 4).checked_mul(height / 4).ok_or_else(invalid)?;
    if src != src_len {
        return Err(ImageError::InvalidChannelShape(src, src_len));
    }
    if dst != dst_len {
        return Err(ImageError::InvalidChannelShape(dst, dst_len));
    }
    Ok(())
}

/// 4x4 average pooling, rounding half to even as `torch.round(avg_pool2d(x, 4))` does.
///
/// # Arguments
///
/// * `src` - `width * height` bytes, row-major, no padding.
/// * `width`, `height` - The source size; both nonzero multiples of 4.
/// * `dst` - `(width / 4) * (height / 4)` bytes.
///
/// # Returns
///
/// Nothing; `dst` holds the pooled image.
///
/// # Errors
///
/// `ImageError::InvalidImageSize` for invalid or overflowing dimensions;
/// `ImageError::InvalidChannelShape` for a buffer length mismatch.
///
/// # Example
///
/// ```
/// use kornia_staging_imgproc::resize::pool4_u8;
/// let src = vec![10u8; 8 * 4];
/// let mut dst = vec![0u8; 2];
/// pool4_u8(&src, 8, 4, &mut dst)?;
/// assert_eq!(dst, vec![10, 10]);
/// # Ok::<(), kornia_image::ImageError>(())
/// ```
pub fn pool4_u8(src: &[u8], width: usize, height: usize, dst: &mut [u8]) -> Result<(), ImageError> {
    pool4_from_sums(src, width, height, dst, |sum| {
        let quotient: u16 = sum >> 4;
        let remainder: u16 = sum & 15;
        (quotient + u16::from(remainder > 8 || (remainder == 8 && quotient & 1 == 1))) as u8
    })
}

/// 4x4 average pooling into f32 means on the u8 scale (no rounding).
///
/// # Arguments
///
/// * `src` - `width * height` bytes, row-major, no padding.
/// * `width`, `height` - The source size; both nonzero multiples of 4.
/// * `dst` - `(width / 4) * (height / 4)` values.
///
/// # Returns
///
/// Nothing; `dst` holds the block means.
///
/// # Errors
///
/// `ImageError::InvalidImageSize` for invalid or overflowing dimensions;
/// `ImageError::InvalidChannelShape` for a buffer length mismatch.
///
/// # Example
///
/// ```
/// use kornia_staging_imgproc::resize::pool4_mean_f32;
/// let src: Vec<u8> = (0..16).map(|i| if i < 8 { 1 } else { 0 }).collect();
/// let mut dst = [0.0f32; 1];
/// pool4_mean_f32(&src, 4, 4, &mut dst)?;
/// assert_eq!(dst, [0.5]);
/// # Ok::<(), kornia_image::ImageError>(())
/// ```
pub fn pool4_mean_f32(
    src: &[u8],
    width: usize,
    height: usize,
    dst: &mut [f32],
) -> Result<(), ImageError> {
    pool4_from_sums(src, width, height, dst, |sum| f32::from(sum) / 16.0)
}

/// 4x4 box pooling with the output value chosen by the caller: each output is `value(sum)`, `sum` being its block's 16 source
/// bytes added up (0 to 4080). [`pool4_u8`] and [`pool4_mean_f32`] are this with a rounded and an exact mean; a network input
/// of another type (fp16 bits, say) is written the same way, with no f32 image in between.
///
/// # Arguments
///
/// * `src` - `width * height` bytes, row-major, no padding.
/// * `width`, `height` - The source size; both nonzero multiples of 4.
/// * `dst` - `(width / 4) * (height / 4)` values.
/// * `value` - The output of a block from its sum.
///
/// # Returns
///
/// Nothing; `dst` holds the pooled image.
///
/// # Errors
///
/// `ImageError::InvalidImageSize` for invalid or overflowing dimensions;
/// `ImageError::InvalidChannelShape` for a buffer length mismatch.
///
/// # Example
///
/// ```
/// use kornia_staging_imgproc::resize::pool4_from_sums;
/// let src = vec![3u8; 8 * 4];
/// let mut dst = [0u16; 2];
/// pool4_from_sums(&src, 8, 4, &mut dst, |sum| sum)?;
/// assert_eq!(dst, [48, 48]);
/// # Ok::<(), kornia_image::ImageError>(())
/// ```
pub fn pool4_from_sums<T>(
    src: &[u8],
    width: usize,
    height: usize,
    dst: &mut [T],
    value: impl Fn(u16) -> T,
) -> Result<(), ImageError> {
    check(src.len(), width, height, dst.len())?;
    let mut sums: Vec<u16> = vec![0; width / 4];
    for (out_row, dst_row) in dst.chunks_exact_mut(width / 4).enumerate() {
        block_sums(src, width, out_row, &mut sums);
        for (out, &sum) in dst_row.iter_mut().zip(&sums) {
            *out = value(sum);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool4_rejects_overflow_without_panicking() {
        assert!(matches!(
            pool4_u8(&[], usize::MAX - 3, 4, &mut []),
            Err(ImageError::InvalidImageSize(..))
        ));
    }

    #[test]
    fn pool4_reports_buffer_lengths() {
        assert!(matches!(
            pool4_u8(&[0; 15], 4, 4, &mut [0]),
            Err(ImageError::InvalidChannelShape(15, 16))
        ));
        assert!(matches!(
            pool4_u8(&[0; 16], 4, 4, &mut []),
            Err(ImageError::InvalidChannelShape(0, 1))
        ));
    }

    #[test]
    fn pool4_rounds_half_to_even_like_torch() {
        // One 4x4 block per case: sums 8*16=128 -> 8 (exact); 8 = 0.5 -> 0 (even); 24 = 1.5 -> 2; 9 -> 0.5625 -> 1.
        let cases: [(u16, u8); 5] = [(128, 8), (8, 0), (24, 2), (9, 1), (16 * 255, 255)];
        for (sum, expected) in cases {
            let mut block: Vec<u8> = vec![0; 16];
            let mut left: u16 = sum;
            for value in block.iter_mut() {
                let take: u16 = left.min(255);
                *value = take as u8;
                left -= take;
            }
            let mut dst: [u8; 1] = [0];
            assert!(pool4_u8(&block, 4, 4, &mut dst).is_ok());
            assert_eq!(dst[0], expected, "sum {sum}");
        }
    }

    #[test]
    fn pool4_keeps_layout() {
        let width: usize = 8;
        let height: usize = 8;
        let src: Vec<u8> = (0..width * height)
            .map(|i| {
                if (i % width) < 4 {
                    40
                } else if i / width < 4 {
                    80
                } else {
                    120
                }
            })
            .collect();
        let mut dst: Vec<u8> = vec![0; 4];
        assert!(pool4_u8(&src, width, height, &mut dst).is_ok());
        assert_eq!(dst, vec![40, 80, 40, 120]);
    }

    #[test]
    fn pool4_rejects_bad_sizes() {
        let mut dst: Vec<u8> = vec![0; 2];
        assert!(pool4_u8(&[0; 30], 6, 5, &mut dst).is_err());
        assert!(pool4_mean_f32(&[0; 30], 6, 5, &mut [0.0; 2]).is_err());
        // An empty image is an error, not a zero-width row loop.
        assert!(
            pool4_u8(&[], 0, 4, &mut []).is_err() && pool4_mean_f32(&[], 4, 0, &mut []).is_err()
        );
    }
}
