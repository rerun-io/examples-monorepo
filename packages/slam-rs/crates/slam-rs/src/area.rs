//! Exact integer area downsampling for capture and catalog decoders.

use kornia_image::{Image, ImageError};
use rayon::prelude::*;

mod kernels;

/// Output rows per rayon task, shared with the live downsample stage.
const ROWS_PER_TASK: usize = 16;

/// Downsample an image by integer factors using the rounded (half up) area mean.
/// Returns an image-size error when either factor is not a positive integer.
pub fn resize_area_u8<const C: usize>(
    src: &Image<u8, C>,
    dst: &mut Image<u8, C>,
) -> Result<(), ImageError> {
    let dst_size = (dst.width(), dst.height());
    resize_area_u8_into::<C>(
        src.as_slice(),
        (src.width(), src.height()),
        src.width() * C,
        dst.as_slice_mut(),
        dst_size,
    )
}

/// Write an integer area mean from a strided plane into a tight caller buffer.
/// Padding bytes, including padding after the last visible row, are not read.
/// Returns an image-size or buffer-length error for invalid geometry.
pub fn resize_area_u8_into<const C: usize>(
    src: &[u8],
    (sw, sh): (usize, usize),
    stride: usize,
    dst: &mut [u8],
    (dw, dh): (usize, usize),
) -> Result<(), ImageError> {
    if C == 0
        || dw == 0
        || dh == 0
        || sw % dw != 0
        || sh % dh != 0
        || sw < dw
        || sh < dh
        || stride < sw * C
    {
        return Err(ImageError::InvalidImageSize(sw, sh, dw, dh));
    }
    let needed = (sh - 1) * stride + sw * C;
    if src.len() < needed {
        return Err(ImageError::InvalidChannelShape(src.len(), needed));
    }
    if dst.len() != dw * dh * C {
        return Err(ImageError::InvalidChannelShape(dst.len(), dw * dh * C));
    }
    let (kx, ky) = (sw / dw, sh / dh);
    let dst_stride = dw * C;
    dst.par_chunks_mut(ROWS_PER_TASK * dst_stride)
        .enumerate()
        .for_each(|(chunk, rows)| {
            for (r, out) in rows.chunks_exact_mut(dst_stride).enumerate() {
                let y = chunk * ROWS_PER_TASK + r;
                let block = &src[y * ky * stride..];
                if C == 1 && kx == 3 && ky == 3 {
                    kernels::area3_row(
                        &block[..sw],
                        &block[stride..stride + sw],
                        &block[2 * stride..2 * stride + sw],
                        out,
                    );
                } else {
                    kernels::area_row_generic::<C>(block, stride, kx, ky, out);
                }
            }
        });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{Image, ImageError, ImageSize};
    #[test]
    fn strided_input_ignores_padding_and_needs_only_the_last_visible_pixel()
    -> Result<(), ImageError> {
        let src = [
            0, 3, 6, 9, 9, 9, 255, 255, 0, 3, 6, 9, 9, 9, 255, 255, 0, 3, 6, 9, 9, 9,
        ];
        let mut dst = [0; 2];
        resize_area_u8_into::<1>(&src, (6, 3), 8, &mut dst, (2, 1))?;
        assert_eq!(dst, [3, 9]);
        Ok(())
    }

    const SMALL_SIZE: ImageSize = ImageSize {
        width: 640,
        height: 360,
    };
    fn pseudo_random(len: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 56) as u8
            })
            .collect()
    }

    #[test]
    fn the_3x3_kernel_matches_the_generic_box_mean_on_full_size_frames() -> Result<(), ImageError> {
        let full = ImageSize {
            width: 1920,
            height: 1080,
        };
        for seed in [1, 2] {
            let src = Image::<u8, 1>::new(full, pseudo_random(1920 * 1080, seed))?;
            let mut fast = Image::<u8, 1>::from_size_val(SMALL_SIZE, 0)?;
            resize_area_u8(&src, &mut fast)?;
            let mut reference = vec![0u8; 640 * 360];
            for (y, row) in reference.as_chunks_mut::<640>().0.iter_mut().enumerate() {
                kernels::area_row_generic::<1>(
                    &src.as_slice()[3 * y * 1920..(3 * y + 3) * 1920],
                    1920,
                    3,
                    3,
                    row,
                );
            }
            assert_eq!(fast.as_slice(), reference.as_slice());
        }
        // The largest sum (9 x 255) stays 255.
        let white = Image::<u8, 1>::from_size_val(full, 255)?;
        let mut out = Image::<u8, 1>::from_size_val(SMALL_SIZE, 0)?;
        resize_area_u8(&white, &mut out)?;
        assert!(out.as_slice().iter().all(|&v| v == 255));
        Ok(())
    }

    #[test]
    fn it_reproduces_pr270_slam_luma_on_a_structured_plane() -> Result<(), ImageError> {
        // PR #270's slam_luma test: blocks of constant k plus 4 on one pixel of each block round back to k.
        let mut plane = vec![0u8; 1920 * 1080];
        for y in 0..1080 {
            for x in 0..1920 {
                plane[y * 1920 + x] =
                    (x / 3 % 200) as u8 + if y % 3 == 0 && x % 3 == 0 { 4 } else { 0 };
            }
        }
        plane[0] = 5;
        let src = Image::<u8, 1>::new(
            ImageSize {
                width: 1920,
                height: 1080,
            },
            plane,
        )?;
        let mut out = Image::<u8, 1>::from_size_val(SMALL_SIZE, 0)?;
        resize_area_u8(&src, &mut out)?;
        assert_eq!(out.as_slice()[0], 1, "block sum 5: (5 + 4) / 9 = 1");
        for (y, row) in out.as_slice().as_chunks::<640>().0.iter().enumerate() {
            for (x, &value) in row.iter().enumerate() {
                if (x, y) != (0, 0) {
                    assert_eq!(value, (x % 200) as u8);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn generic_factors_and_channels_and_bad_sizes() -> Result<(), ImageError> {
        let src = Image::<u8, 2>::new(
            ImageSize {
                width: 4,
                height: 2,
            },
            vec![0, 10, 2, 20, 4, 30, 6, 40, 1, 11, 3, 21, 5, 31, 7, 41],
        )?;
        let mut dst = Image::<u8, 2>::from_size_val(
            ImageSize {
                width: 2,
                height: 1,
            },
            0,
        )?;
        resize_area_u8(&src, &mut dst)?;
        assert_eq!(dst.as_slice(), &[2, 16, 6, 36]);
        let mut bad = Image::<u8, 2>::from_size_val(
            ImageSize {
                width: 3,
                height: 1,
            },
            0,
        )?;
        assert!(resize_area_u8(&src, &mut bad).is_err());
        Ok(())
    }
}
