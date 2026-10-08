//! Sparse u16 samples with Basalt arithmetic order.
use kornia_image::Image;

/// Floating-point bounds: `border <= x < w - border - 1`, and likewise for y.
/// The extra pixel leaves room for interpolation's unconditional neighbor read.
/// Use border zero for values and at least one for gradients.
///
/// # Arguments
/// * `image` - Dense source image.
/// * `x`, `y` - Pixel coordinates; non-finite coordinates are rejected.
/// * `border` - Non-negative excluded border width.
#[inline]
pub fn in_bounds_u16(image: &Image<u16, 1>, x: f32, y: f32, border: f32) -> bool {
    border <= x
        && x < (image.width() as f32 - border - 1.0)
        && border <= y
        && y < (image.height() as f32 - border - 1.0)
}

/// Bilinear sampling with fixed multiplication grouping and summation order.
/// Interpolate the four surrounding pixels with weights formed from x/y fractions.
/// Truncation towards zero equals floor only for non-negative coordinates;
/// callers must establish `in_bounds(x, y, 0)` first.
///
/// # Arguments
/// * `image` - Dense source image.
/// * `x`, `y` - Coordinates accepted by [`in_bounds_u16`] with border zero.
///
/// # Examples
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_staging_imgproc::interpolation::sample_bilinear_u16;
/// let image = Image::new(ImageSize { width: 2, height: 2 }, vec![100, 200, 300, 400])?;
/// assert_eq!(sample_bilinear_u16(&image, 0.5, 0.5), 250.0);
/// # Ok::<(), kornia_image::ImageError>(())
/// ```
///
/// # Panics
/// If sampling indexes outside image storage.
#[inline]
pub fn sample_bilinear_u16(image: &Image<u16, 1>, x: f32, y: f32) -> f32 {
    U16View::new(image).sample_bilinear(x, y)
}

/// Bilinear value and unit-spacing central differences of the bilinear surface.
/// The gradient is not its analytic derivative. The stencil reaches from `ix-1`
/// to `ix+2` and `iy-1` to `iy+2`, requiring `in_bounds(x, y, 1)`.
/// Sparse 52-tap KLT sampling avoids building a dense gradient image.
///
/// # Arguments
/// * `image` - Dense source image.
/// * `x`, `y` - Coordinates accepted by [`in_bounds_u16`] with border one.
///
/// # Panics
/// If the gradient stencil indexes outside image storage.
#[inline]
pub fn sample_bilinear_with_gradient_u16(image: &Image<u16, 1>, x: f32, y: f32) -> (f32, [f32; 2]) {
    U16View::new(image).sample_bilinear_with_gradient(x, y)
}

/// Dense host pixels borrowed once before a sampling pass.
#[derive(Clone, Copy)]
pub(crate) struct U16View<'a> {
    pub(crate) pixels: &'a [u16],
    pub(crate) width: usize,
    pub(crate) height: usize,
}

impl<'a> U16View<'a> {
    #[inline]
    pub(crate) fn new(image: &'a Image<u16, 1>) -> Self {
        Self {
            pixels: image.as_slice(),
            width: image.width(),
            height: image.height(),
        }
    }

    #[inline]
    pub(crate) fn in_bounds(self, x: f32, y: f32, border: f32) -> bool {
        border <= x
            && x < (self.width as f32 - border - 1.0)
            && border <= y
            && y < (self.height as f32 - border - 1.0)
    }

    #[inline]
    pub(crate) fn sample_bilinear(self, x: f32, y: f32) -> f32 {
        debug_assert!(self.in_bounds(x, y, 0.0), "interp needs InBounds(x, y, 0)");

        let ix: usize = x as usize;
        let iy: usize = y as usize;
        let pixels = self.pixels;
        let at = |px: usize, py: usize| f32::from(pixels[py * self.width + px]);

        let dx: f32 = x - ix as f32;
        let dy: f32 = y - iy as f32;

        let ddx: f32 = 1.0 - dx;
        let ddy: f32 = 1.0 - dy;

        ddx * ddy * at(ix, iy)
            + ddx * dy * at(ix, iy + 1)
            + dx * ddy * at(ix + 1, iy)
            + dx * dy * at(ix + 1, iy + 1)
    }

    #[inline]
    pub(crate) fn sample_bilinear_with_gradient(self, x: f32, y: f32) -> (f32, [f32; 2]) {
        debug_assert!(
            self.in_bounds(x, y, 1.0),
            "interp_grad needs InBounds(x, y, 1)"
        );

        let ix: usize = x as usize;
        let iy: usize = y as usize;
        let pixels = self.pixels;
        let at = |px: usize, py: usize| f32::from(pixels[py * self.width + px]);

        let dx: f32 = x - ix as f32;
        let dy: f32 = y - iy as f32;

        let ddx: f32 = 1.0 - dx;
        let ddy: f32 = 1.0 - dy;

        let px0y0: f32 = at(ix, iy);
        let px1y0: f32 = at(ix + 1, iy);
        let px0y1: f32 = at(ix, iy + 1);
        let px1y1: f32 = at(ix + 1, iy + 1);

        let value: f32 = ddx * ddy * px0y0 + ddx * dy * px0y1 + dx * ddy * px1y0 + dx * dy * px1y1;

        let pxm1y0: f32 = at(ix - 1, iy);
        let pxm1y1: f32 = at(ix - 1, iy + 1);

        let res_mx: f32 =
            ddx * ddy * pxm1y0 + ddx * dy * pxm1y1 + dx * ddy * px0y0 + dx * dy * px0y1;

        let px2y0: f32 = at(ix + 2, iy);
        let px2y1: f32 = at(ix + 2, iy + 1);

        let res_px: f32 = ddx * ddy * px1y0 + ddx * dy * px1y1 + dx * ddy * px2y0 + dx * dy * px2y1;

        let grad_x: f32 = 0.5 * (res_px - res_mx);

        let px0ym1: f32 = at(ix, iy - 1);
        let px1ym1: f32 = at(ix + 1, iy - 1);

        let res_my: f32 =
            ddx * ddy * px0ym1 + ddx * dy * px0y0 + dx * ddy * px1ym1 + dx * dy * px1y0;

        let px0y2: f32 = at(ix, iy + 2);
        let px1y2: f32 = at(ix + 1, iy + 2);

        let res_py: f32 = ddx * ddy * px0y1 + ddx * dy * px0y2 + dx * ddy * px1y1 + dx * dy * px1y2;

        let grad_y: f32 = 0.5 * (res_py - res_my);

        (value, [grad_x, grad_y])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_images::random_image;
    use approx::assert_abs_diff_eq;
    use kornia_image::ImageSize;
    use proptest::prelude::*;
    /// Amplitude of the smooth analytic sampling function.
    const SMOOTH_AMPLITUDE: f64 = 20_000.0;
    const SMOOTH_OFFSET: f64 = 32_768.0;

    fn smooth_value(x: f64, y: f64) -> f64 {
        (x / 100.0 + y / 20.0).sin() * SMOOTH_AMPLITUDE + SMOOTH_OFFSET
    }

    fn smooth_image(width: usize, height: usize) -> Image<u16, 1> {
        let mut image: Image<u16, 1> =
            Image::from_size_val(ImageSize { width, height }, 0).unwrap();
        for y in 0..height {
            for x in 0..width {
                image
                    .set_pixel(x, y, 0, smooth_value(x as f64, y as f64).round() as u16)
                    .unwrap();
            }
        }
        image
    }

    #[test]
    fn in_bounds_excludes_the_last_row_and_column() {
        let image: Image<u16, 1> = Image::from_size_val(
            ImageSize {
                width: 10,
                height: 8,
            },
            0,
        )
        .unwrap();
        assert!(in_bounds_u16(&image, 0.0, 0.0, 0.0));
        assert!(in_bounds_u16(&image, 8.99, 6.99, 0.0));
        // `w - 1` is out of bounds at border 0: `interp` reads `ix + 1`.
        assert!(!in_bounds_u16(&image, 9.0, 3.0, 0.0));
        assert!(!in_bounds_u16(&image, 3.0, 7.0, 0.0));
        // border 1 loses one more pixel on each side.
        assert!(!in_bounds_u16(&image, 0.5, 3.0, 1.0));
        assert!(in_bounds_u16(&image, 1.0, 1.0, 1.0));
        assert!(!in_bounds_u16(&image, 8.0, 3.0, 1.0));
        assert!(!in_bounds_u16(&image, 3.0, 6.0, 1.0));
    }

    #[test]
    fn interp_halfway_is_the_average_of_the_four_neighbours() {
        let mut image: Image<u16, 1> = Image::from_size_val(
            ImageSize {
                width: 4,
                height: 4,
            },
            0,
        )
        .unwrap();
        image.set_pixel(1, 1, 0, 100).unwrap();
        image.set_pixel(2, 1, 0, 200).unwrap();
        image.set_pixel(1, 2, 0, 300).unwrap();
        image.set_pixel(2, 2, 0, 400).unwrap();
        assert_abs_diff_eq!(sample_bilinear_u16(&image, 1.5, 1.5), 250.0, epsilon = 1e-4);
        assert_abs_diff_eq!(sample_bilinear_u16(&image, 1.5, 1.0), 150.0, epsilon = 1e-4);
        assert_abs_diff_eq!(sample_bilinear_u16(&image, 1.0, 1.5), 200.0, epsilon = 1e-4);
    }

    /// Sample a smooth sine into an image and compare `interp_grad` with unit-step
    /// central differences of `interp`. A looser check against the sine's analytic
    /// derivative also catches swapped or sign-flipped axes.
    #[test]
    fn image_interpolate_grad() {
        let image: Image<u16, 1> = smooth_image(512, 256);
        let (x, y): (f32, f32) = (231.4, 123.34345);

        let (value, grad) = sample_bilinear_with_gradient_u16(&image, x, y);
        assert_abs_diff_eq!(value, sample_bilinear_u16(&image, x, y), epsilon = 0.0);

        let numeric: [f32; 2] = [
            0.5 * (sample_bilinear_u16(&image, x + 1.0, y)
                - sample_bilinear_u16(&image, x - 1.0, y)),
            0.5 * (sample_bilinear_u16(&image, x, y + 1.0)
                - sample_bilinear_u16(&image, x, y - 1.0)),
        ];
        assert_abs_diff_eq!(grad[0], numeric[0], epsilon = 1e-2);
        assert_abs_diff_eq!(grad[1], numeric[1], epsilon = 1e-2);

        let phase: f64 = f64::from(x) / 100.0 + f64::from(y) / 20.0;
        let analytic: [f64; 2] = [
            phase.cos() * SMOOTH_AMPLITUDE / 100.0,
            phase.cos() * SMOOTH_AMPLITUDE / 20.0,
        ];
        assert_abs_diff_eq!(
            f64::from(grad[0]),
            analytic[0],
            epsilon = 0.02 * analytic[0].abs()
        );
        assert_abs_diff_eq!(
            f64::from(grad[1]),
            analytic[1],
            epsilon = 0.02 * analytic[1].abs()
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Every integer coordinate inside the image reads back its own pixel:
        /// `dx = dy = 0` makes the first product 1 and the other three 0.
        #[test]
        fn interp_reproduces_the_pixel_at_integer_coordinates(
            seed in any::<u64>(),
            x in 0usize..30,
            y in 0usize..20,
        ) {
            let image: Image<u16, 1> = random_image(31, 21, seed);
            prop_assert_eq!(
                sample_bilinear_u16(&image, x as f32, y as f32),
                f32::from(image.get_pixel(x, y, 0).copied().ok().unwrap())
            );
        }

        /// `interp_grad` is exactly the value and the unit-spacing central
        /// differences of `interp`. Not bit-exact only
        /// because `interp(x + 1, y)` recomputes `dx` from a different float.
        #[test]
        fn interp_grad_is_the_central_difference_of_interp(
            seed in any::<u64>(),
            x in 1.0f32..28.0,
            y in 1.0f32..18.0,
        ) {
            let image: Image<u16, 1> = random_image(31, 21, seed);
            let (value, grad) = sample_bilinear_with_gradient_u16(&image, x, y);
            prop_assert_eq!(value, sample_bilinear_u16(&image, x, y));
            let dx: f32 = 0.5 * (sample_bilinear_u16(&image, x + 1.0, y) - sample_bilinear_u16(&image, x - 1.0, y));
            let dy: f32 = 0.5 * (sample_bilinear_u16(&image, x, y + 1.0) - sample_bilinear_u16(&image, x, y - 1.0));
            prop_assert!((grad[0] - dx).abs() <= 1e-2 * (1.0 + dx.abs()), "{} vs {}", grad[0], dx);
            prop_assert!((grad[1] - dy).abs() <= 1e-2 * (1.0 + dy.abs()), "{} vs {}", grad[1], dy);
        }

        /// On a smooth image the gradient agrees with a central finite
        /// difference of `interp` taken at half-pixel steps.
        #[test]
        fn gradients_match_finite_differences_on_a_smooth_image(
            x in 30.0f32..480.0,
            y in 30.0f32..220.0,
        ) {
            let image: Image<u16, 1> = smooth_image(512, 256);
            let (_, grad) = sample_bilinear_with_gradient_u16(&image, x, y);
            let step: f32 = 0.5;
            let dx: f32 = (sample_bilinear_u16(&image, x + step, y) - sample_bilinear_u16(&image, x - step, y)) / (2.0 * step);
            let dy: f32 = (sample_bilinear_u16(&image, x, y + step) - sample_bilinear_u16(&image, x, y - step)) / (2.0 * step);
            // The two differ by the third derivative of the sine over the step,
            // plus the bilinear interpolation error: both are below 1% of the
            // y gradient, which is the larger of the two.
            let scale: f32 = 0.01 * (SMOOTH_AMPLITUDE as f32 / 20.0);
            prop_assert!((grad[0] - dx).abs() <= scale, "dx {} vs {}", grad[0], dx);
            prop_assert!((grad[1] - dy).abs() <= scale, "dy {} vs {}", grad[1], dy);
        }

    }
}
