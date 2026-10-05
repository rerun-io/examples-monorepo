//! Bilinear remap of an 8-bit image into an `f32` image, scaled on the way (e.g. by 1/255), with zero padding.
//!
//! kornia-imgproc has `remap` (f32 to f32) and `remap_u8` (u8 to u8, Q10); neither samples a u8 frame into a normalised f32
//! network input without a full-frame conversion or a quantisation. This one does both in one pass and matches PyTorch's
//! `grid_sample(mode="bilinear", padding_mode="zeros", align_corners=False)` when the maps hold pixel-centre coordinates:
//! each of the four taps outside the image contributes zero. Target upstream: kornia-imgproc `interpolation::remap`.

use kornia_image::{Image, ImageError};

/// Sample `src` at `(map_x, map_y)` (pixel centres at integers) with bilinear weights, multiply by `scale` and write `dst`.
///
/// Taps outside the image contribute zero, so pixels within one pixel outside the border fade to zero and non-finite map
/// entries give zero.
///
/// # Arguments
///
/// * `src` - The source image.
/// * `dst` - The destination image, the size of the maps.
/// * `map_x` - Source x per destination pixel.
/// * `map_y` - Source y per destination pixel.
/// * `scale` - Factor applied to every sample (1/255 for a [0, 1] network input).
///
/// # Returns
///
/// `Ok(())` when `dst` was written.
///
/// # Errors
///
/// `ImageError::InvalidImageSize` when the maps and `dst` differ in size.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use robocap_live::kornia_ext::remap::remap_f32_from_u8;
/// let size = ImageSize { width: 2, height: 1 };
/// let src = Image::<u8, 1>::new(size, vec![0, 255]).unwrap();
/// let map_x = Image::<f32, 1>::new(ImageSize { width: 1, height: 1 }, vec![0.5]).unwrap();
/// let map_y = Image::<f32, 1>::new(ImageSize { width: 1, height: 1 }, vec![0.0]).unwrap();
/// let mut dst = Image::<f32, 1>::from_size_val(ImageSize { width: 1, height: 1 }, 0.0).unwrap();
/// remap_f32_from_u8(&src, &mut dst, &map_x, &map_y, 1.0 / 255.0).unwrap();
/// assert!((dst.as_slice()[0] - 0.5).abs() < 1e-6);
/// ```
pub fn remap_f32_from_u8<const C: usize>(
    src: &Image<u8, C>,
    dst: &mut Image<f32, C>,
    map_x: &Image<f32, 1>,
    map_y: &Image<f32, 1>,
    scale: f32,
) -> Result<(), ImageError> {
    if map_x.size() != map_y.size() {
        return Err(ImageError::InvalidImageSize(map_x.cols(), map_x.rows(), map_y.cols(), map_y.rows()));
    }
    if dst.size() != map_x.size() {
        return Err(ImageError::InvalidImageSize(dst.cols(), dst.rows(), map_x.cols(), map_x.rows()));
    }
    let (width, height) = (src.cols(), src.rows());
    let data = src.as_slice();
    let (w, h) = (width as f32, height as f32);
    // A tap's value, zero outside the image (the border path only).
    let tap = |x: isize, y: isize, channel: usize| -> f32 {
        if x < 0 || y < 0 || x >= width as isize || y >= height as isize {
            0.0
        } else {
            f32::from(data[(y as usize * width + x as usize) * C + channel])
        }
    };
    for ((out, &x), &y) in dst.as_slice_mut().chunks_exact_mut(C).zip(map_x.as_slice()).zip(map_y.as_slice()) {
        // Far outside (or NaN, which fails every comparison): all four taps are zero.
        if !(x > -1.0 && y > -1.0 && x < w && y < h) {
            out.fill(0.0);
            continue;
        }
        let (x0, y0) = (x.floor(), y.floor());
        let (wx, wy) = (x - x0, y - y0);
        // grid_sample's weights: nw = (1-wx)(1-wy), ne = wx(1-wy), sw = (1-wx)wy, se = wx wy; the scale folds into them.
        let (nw, ne, sw, se) = ((1.0 - wx) * (1.0 - wy) * scale, wx * (1.0 - wy) * scale, (1.0 - wx) * wy * scale, wx * wy * scale);
        let (ix, iy) = (x0 as isize, y0 as isize);
        if ix >= 0 && iy >= 0 && (ix as usize) + 1 < width && (iy as usize) + 1 < height {
            let top = (iy as usize * width + ix as usize) * C;
            let bottom = top + width * C;
            if let (Some(upper), Some(lower)) = (data.get(top..top + 2 * C), data.get(bottom..bottom + 2 * C)) {
                for (channel, value) in out.iter_mut().enumerate() {
                    *value = nw * f32::from(upper[channel]) + ne * f32::from(upper[C + channel]) + sw * f32::from(lower[channel])
                        + se * f32::from(lower[C + channel]);
                }
                continue;
            }
        }
        for (channel, value) in out.iter_mut().enumerate() {
            *value = nw * tap(ix, iy, channel) + ne * tap(ix + 1, iy, channel) + sw * tap(ix, iy + 1, channel) + se * tap(ix + 1, iy + 1, channel);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;

    fn image<T: Clone>(width: usize, height: usize, data: Vec<T>) -> Result<Image<T, 1>, ImageError> {
        Image::new(ImageSize { width, height }, data)
    }

    #[test]
    fn interior_samples_are_bilinear_and_scaled() -> Result<(), ImageError> {
        let src = image(3, 2, vec![0u8, 10, 20, 30, 40, 50])?;
        let map_x = image(3, 1, vec![0.0f32, 1.5, 0.25])?;
        let map_y = image(3, 1, vec![0.0f32, 0.5, 1.0])?;
        let mut dst = image(3, 1, vec![0f32; 3])?;
        remap_f32_from_u8(&src, &mut dst, &map_x, &map_y, 0.5)?;
        assert_eq!(dst.as_slice(), &[0.0, 0.5 * 30.0, 0.5 * 32.5]);
        Ok(())
    }

    #[test]
    fn taps_outside_the_image_count_as_zero() -> Result<(), ImageError> {
        let src = image(2, 2, vec![100u8; 4])?;
        // Half a pixel left of the first column: half of the weight falls outside.
        let map_x = image(4, 1, vec![-0.5f32, 1.5, -1.0, f32::NAN])?;
        let map_y = image(4, 1, vec![0.0f32, 0.0, 0.0, 0.0])?;
        let mut dst = image(4, 1, vec![1f32; 4])?;
        remap_f32_from_u8(&src, &mut dst, &map_x, &map_y, 1.0)?;
        assert_eq!(dst.as_slice(), &[50.0, 50.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn mismatched_sizes_are_errors() -> Result<(), ImageError> {
        let src = image(2, 2, vec![0u8; 4])?;
        let map = image(2, 1, vec![0f32; 2])?;
        let mut dst = image(1, 1, vec![0f32])?;
        assert!(remap_f32_from_u8(&src, &mut dst, &map, &map, 1.0).is_err());
        Ok(())
    }
}
