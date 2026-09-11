//! Owned u16 grayscale images and bilinear sampling for the frontend.
//! Eight-bit input is widened by `u8 << 8`, preserving interpolation precision
//! (D08). Row strides count u16 elements internally and bytes on input.
//! Honor padded source rows, such as a 1024-byte decoder stride for a 960-pixel
//! image (D28). Flat owned buffers support upload without repacking.

use thiserror::Error;

/// Everything that can go wrong building or filling an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ImageError {
    /// A row pitch cannot be shorter than the row it carries.
    #[error("stride {stride} is shorter than width {width}")]
    StrideTooSmall {
        /// Row length.
        width: usize,
        /// Row pitch that was offered.
        stride: usize,
    },
    /// `stride * height` does not fit in a `usize`, so no buffer can satisfy it.
    #[error("{height} rows of stride {stride} overflow the address space")]
    SizeOverflow {
        /// Number of rows.
        height: usize,
        /// Row pitch.
        stride: usize,
    },
    /// The buffer would be bigger than a single Rust allocation may be.
    ///
    /// `Vec` requires the size in *bytes* to be at most `isize::MAX`, so a
    /// pixel count that fits a `usize` can still abort the allocator. This is
    /// the typed refusal instead (decision D32).
    #[error("{len} pixels is more than the {max} a single allocation may hold")]
    LayoutTooLarge {
        /// Pixels the geometry asks for.
        len: usize,
        /// Most pixels one allocation can hold.
        max: usize,
    },
    /// The source buffer does not hold `stride * height` elements.
    #[error("{height} rows of stride {stride} do not fit in {len} elements")]
    ShortBuffer {
        /// Number of rows.
        height: usize,
        /// Row pitch.
        stride: usize,
        /// Elements actually supplied.
        len: usize,
    },
}

/// An owned flat u16 image with explicit width, height and row stride.
/// This layout can be uploaded as one buffer without per-frame repacking.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ImageU16 {
    data: Vec<u16>,
    width: usize,
    height: usize,
    stride: usize,
}

impl ImageU16 {
    /// A zero-filled image with a dense row stride.
    ///
    /// # Errors
    ///
    /// As [`ImageU16::zeros_with_stride`], which this is `stride == width` of.
    pub fn zeros(width: usize, height: usize) -> Result<Self, ImageError> {
        Self::zeros_with_stride(width, height, width)
    }

    /// A zero-filled image with an explicit row stride in pixels.
    ///
    /// # Errors
    ///
    /// [`ImageError::StrideTooSmall`] if `stride < width`,
    /// [`ImageError::SizeOverflow`] if `stride * height` does not fit in a
    /// `usize`, [`ImageError::LayoutTooLarge`] if the buffer it describes is
    /// bigger than one allocation may be.
    pub fn zeros_with_stride(
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<Self, ImageError> {
        let len: usize = checked_pixel_len(width, height, stride)?;
        Ok(Self {
            data: vec![0; len],
            width,
            height,
            stride,
        })
    }

    /// Widen an 8-bit image with `u8 << 8`, honoring `stride_bytes` between rows.
    /// The buffer must cover `stride_bytes * height` bytes.
    ///
    /// # Errors
    /// Returns stride, overflow or short-buffer errors when geometry is invalid.
    pub fn from_u8_strided(
        bytes: &[u8],
        width: usize,
        height: usize,
        stride_bytes: usize,
    ) -> Result<Self, ImageError> {
        let mut image: Self = Self::default();
        image.fill_from_u8_strided(bytes, width, height, stride_bytes)?;
        Ok(image)
    }

    /// Refill this image from an 8-bit source, reusing the buffer.
    ///
    /// This is the per-frame path: when the capacity already covers
    /// `width * height`, no allocation happens, whatever the previous geometry
    /// was. The resulting image is dense (`stride == width`).
    ///
    /// # Errors
    ///
    /// As [`ImageU16::from_u8_strided`].
    pub fn fill_from_u8_strided(
        &mut self,
        bytes: &[u8],
        width: usize,
        height: usize,
        stride_bytes: usize,
    ) -> Result<(), ImageError> {
        let source_len: usize = checked_len(width, height, stride_bytes)?;
        if bytes.len() < source_len {
            return Err(ImageError::ShortBuffer {
                height,
                stride: stride_bytes,
                len: bytes.len(),
            });
        }
        let len: usize = checked_pixel_len(width, height, width)?;
        // `resize` only allocates when `len` exceeds the current capacity, so a
        // steady stream of same-sized frames never touches the allocator.
        self.data.resize(len, 0);
        self.width = width;
        self.height = height;
        self.stride = width;
        for row in 0..height {
            // Both slices are in range: the source by the `ShortBuffer` check
            // above, the destination because `data.len() == width * height`.
            let source: &[u8] = &bytes[row * stride_bytes..row * stride_bytes + width];
            let target: &mut [u16] = &mut self.data[row * width..row * width + width];
            for (pixel, byte) in target.iter_mut().zip(source.iter()) {
                *pixel = u16::from(*byte) << 8;
            }
        }
        Ok(())
    }

    /// Copy `source` into this image, reusing the buffer.
    ///
    /// The result is dense (`stride == width`) whatever `source`'s stride is,
    /// and no allocation happens when the capacity already covers it. This is
    /// how a caller reads pixels out of a pyramid without borrowing its
    /// storage — see [`crate::pyramid::Pyramid::copy_level_into`].
    ///
    /// # Errors
    ///
    /// [`ImageError::LayoutTooLarge`] or [`ImageError::SizeOverflow`] when
    /// `source`'s geometry cannot be allocated densely. Neither is reachable
    /// from a `source` that already exists.
    pub fn copy_from(&mut self, source: &ImageU16) -> Result<(), ImageError> {
        let width: usize = source.width;
        let height: usize = source.height;
        let len: usize = checked_pixel_len(width, height, width)?;
        self.data.resize(len, 0);
        self.width = width;
        self.height = height;
        self.stride = width;
        for row in 0..height {
            // Both slices are in range: `source` upholds `stride * height`, and
            // `data.len() == width * height`.
            let from: &[u16] = &source.data[row * source.stride..row * source.stride + width];
            self.data[row * width..row * width + width].copy_from_slice(from);
        }
        Ok(())
    }

    /// Row length in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Number of rows.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Distance between the starts of two rows, in pixels.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The whole buffer, `stride * height` pixels long.
    pub fn data(&self) -> &[u16] {
        &self.data
    }

    /// One row, `width` pixels long.
    ///
    /// # Panics
    ///
    /// If `y >= height`. Callers on the frontend path index rows they have
    /// already bounds-checked; this is the checked accessor for everyone else.
    pub fn row(&self, y: usize) -> &[u16] {
        &self.data[y * self.stride..y * self.stride + self.width]
    }

    /// One row, `width` pixels long, mutably.
    ///
    /// # Panics
    ///
    /// If `y >= height`. Callers on the frontend path index rows they have
    /// already bounds-checked; this is the checked accessor for everyone else.
    pub fn row_mut(&mut self, y: usize) -> &mut [u16] {
        &mut self.data[y * self.stride..y * self.stride + self.width]
    }

    /// The pixel at `(x, y)`, or `None` outside the image.
    pub fn get(&self, x: usize, y: usize) -> Option<u16> {
        if x < self.width && y < self.height {
            Some(self.data[y * self.stride + x])
        } else {
            None
        }
    }

    /// Write the pixel at `(x, y)`; a coordinate outside the image is ignored.
    pub fn set(&mut self, x: usize, y: usize, value: u16) {
        if x < self.width && y < self.height {
            self.data[y * self.stride + x] = value;
        }
    }

    /// Read a pixel after validating `x < width && y < height`.
    ///
    /// # Panics
    /// If the resulting slice index is outside storage.
    #[inline]
    fn at(&self, x: usize, y: usize) -> f32 {
        f32::from(self.data[y * self.stride + x])
    }

    /// Floating-point bounds: `border <= x < w - border - 1`, and likewise for y.
    /// The extra pixel leaves room for interpolation's unconditional neighbor read.
    /// Use border zero for values and at least one for gradients.
    pub fn in_bounds(&self, x: f32, y: f32, border: f32) -> bool {
        border <= x
            && x < (self.width as f32 - border - 1.0)
            && border <= y
            && y < (self.height as f32 - border - 1.0)
    }

    /// Bilinear sampling with fixed multiplication grouping and summation order.
    /// Interpolate the four surrounding pixels with weights formed from x/y fractions.
    /// Truncation towards zero equals floor only for non-negative coordinates;
    /// callers must establish `in_bounds(x, y, 0)` first.
    ///
    /// # Panics
    /// If sampling indexes outside image storage.
    #[inline]
    pub fn interp(&self, x: f32, y: f32) -> f32 {
        debug_assert!(self.in_bounds(x, y, 0.0), "interp needs InBounds(x, y, 0)");

        let ix: usize = x as usize;
        let iy: usize = y as usize;

        let dx: f32 = x - ix as f32;
        let dy: f32 = y - iy as f32;

        let ddx: f32 = 1.0 - dx;
        let ddy: f32 = 1.0 - dy;

        ddx * ddy * self.at(ix, iy)
            + ddx * dy * self.at(ix, iy + 1)
            + dx * ddy * self.at(ix + 1, iy)
            + dx * dy * self.at(ix + 1, iy + 1)
    }

    /// Bilinear value and unit-spacing central differences of the bilinear surface.
    /// The gradient is not its analytic derivative. The stencil reaches from `ix-1`
    /// to `ix+2` and `iy-1` to `iy+2`, requiring `in_bounds(x, y, 1)`.
    /// Sparse 52-tap KLT sampling avoids building a dense gradient image.
    ///
    /// # Panics
    /// If the gradient stencil indexes outside image storage.
    #[inline]
    pub fn interp_grad(&self, x: f32, y: f32) -> (f32, [f32; 2]) {
        debug_assert!(
            self.in_bounds(x, y, 1.0),
            "interp_grad needs InBounds(x, y, 1)"
        );

        let ix: usize = x as usize;
        let iy: usize = y as usize;

        let dx: f32 = x - ix as f32;
        let dy: f32 = y - iy as f32;

        let ddx: f32 = 1.0 - dx;
        let ddy: f32 = 1.0 - dy;

        let px0y0: f32 = self.at(ix, iy);
        let px1y0: f32 = self.at(ix + 1, iy);
        let px0y1: f32 = self.at(ix, iy + 1);
        let px1y1: f32 = self.at(ix + 1, iy + 1);

        let value: f32 = ddx * ddy * px0y0 + ddx * dy * px0y1 + dx * ddy * px1y0 + dx * dy * px1y1;

        let pxm1y0: f32 = self.at(ix - 1, iy);
        let pxm1y1: f32 = self.at(ix - 1, iy + 1);

        let res_mx: f32 =
            ddx * ddy * pxm1y0 + ddx * dy * pxm1y1 + dx * ddy * px0y0 + dx * dy * px0y1;

        let px2y0: f32 = self.at(ix + 2, iy);
        let px2y1: f32 = self.at(ix + 2, iy + 1);

        let res_px: f32 = ddx * ddy * px1y0 + ddx * dy * px1y1 + dx * ddy * px2y0 + dx * dy * px2y1;

        let grad_x: f32 = 0.5 * (res_px - res_mx);

        let px0ym1: f32 = self.at(ix, iy - 1);
        let px1ym1: f32 = self.at(ix + 1, iy - 1);

        let res_my: f32 =
            ddx * ddy * px0ym1 + ddx * dy * px0y0 + dx * ddy * px1ym1 + dx * dy * px1y0;

        let px0y2: f32 = self.at(ix, iy + 2);
        let px1y2: f32 = self.at(ix + 1, iy + 2);

        let res_py: f32 = ddx * ddy * px0y1 + ddx * dy * px0y2 + dx * ddy * px1y1 + dx * dy * px1y2;

        let grad_y: f32 = 0.5 * (res_py - res_my);

        (value, [grad_x, grad_y])
    }
}

/// Most elements of type `T` one Rust allocation may hold.
///
/// `Vec` and the global allocator cap an allocation at `isize::MAX` *bytes*
/// (`std::alloc::Layout::from_size_align`), so a count that fits a `usize` can
/// still abort the process. Every allocation in this crate is sized through
/// this bound and refused with a typed error instead (decision D32).
pub(crate) const fn max_elements<T>() -> usize {
    (isize::MAX as usize) / size_of::<T>()
}

/// `stride * height`, refusing a stride shorter than the row or a product that wraps.
///
/// This is an element *count*, not a layout: use it for a source buffer that
/// already exists. Sizing an allocation goes through [`checked_pixel_len`].
fn checked_len(width: usize, height: usize, stride: usize) -> Result<usize, ImageError> {
    if stride < width {
        return Err(ImageError::StrideTooSmall { width, stride });
    }
    stride
        .checked_mul(height)
        .ok_or(ImageError::SizeOverflow { height, stride })
}

/// [`checked_len`], and the `u16` buffer it describes must be allocatable.
fn checked_pixel_len(width: usize, height: usize, stride: usize) -> Result<usize, ImageError> {
    let len: usize = checked_len(width, height, stride)?;
    let max: usize = max_elements::<u16>();
    if len > max {
        return Err(ImageError::LayoutTooLarge { len, max });
    }
    Ok(len)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    /// `sin(x / 100 + y / 20)` sampled into `u16`, the image form of
    /// Smooth analytic sampling function.
    const SMOOTH_AMPLITUDE: f64 = 20_000.0;
    const SMOOTH_OFFSET: f64 = 32_768.0;

    fn smooth_value(x: f64, y: f64) -> f64 {
        (x / 100.0 + y / 20.0).sin() * SMOOTH_AMPLITUDE + SMOOTH_OFFSET
    }

    fn smooth_image(width: usize, height: usize) -> ImageU16 {
        let mut image: ImageU16 = ImageU16::zeros(width, height).unwrap();
        for y in 0..height {
            for x in 0..width {
                image.set(x, y, smooth_value(x as f64, y as f64).round() as u16);
            }
        }
        image
    }

    fn random_image(width: usize, height: usize, seed: u64) -> ImageU16 {
        let mut image: ImageU16 = ImageU16::zeros(width, height).unwrap();
        let mut state: u64 = seed | 1;
        for y in 0..height {
            for pixel in image.row_mut(y) {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                *pixel = (state >> 32) as u16;
            }
        }
        image
    }

    #[test]
    fn widening_shifts_left_by_eight() {
        let bytes: [u8; 4] = [0, 1, 128, 255];
        let image: ImageU16 = ImageU16::from_u8_strided(&bytes, 4, 1, 4).unwrap();
        assert_eq!(image.data(), &[0, 256, 32_768, 65_280]);
    }

    #[test]
    fn widening_honours_the_source_byte_stride() {
        // Two rows of three pixels inside a padded five-byte stride, dav1d's shape.
        let bytes: [u8; 10] = [1, 2, 3, 99, 99, 4, 5, 6, 99, 99];
        let image: ImageU16 = ImageU16::from_u8_strided(&bytes, 3, 2, 5).unwrap();
        assert_eq!(image.width(), 3);
        assert_eq!(image.height(), 2);
        assert_eq!(image.stride(), 3);
        assert_eq!(image.data(), &[256, 512, 768, 1024, 1280, 1536]);
    }

    #[test]
    fn a_bad_geometry_is_an_error_not_a_panic() {
        assert_eq!(
            ImageU16::from_u8_strided(&[0; 4], 4, 1, 2),
            Err(ImageError::StrideTooSmall {
                width: 4,
                stride: 2
            })
        );
        assert_eq!(
            ImageU16::from_u8_strided(&[0; 4], 4, 2, 4),
            Err(ImageError::ShortBuffer {
                height: 2,
                stride: 4,
                len: 4
            })
        );
        assert_eq!(
            ImageU16::from_u8_strided(&[], 1, 2, 1 << 63),
            Err(ImageError::SizeOverflow {
                height: 2,
                stride: 1 << 63
            })
        );
    }

    /// A geometry whose pixel count fits a `usize` but whose *bytes* exceed
    /// `isize::MAX` must be refused, not handed to the allocator: `vec![0; n]`
    /// aborts the process there, and an abort inside the released-GIL region
    /// takes the whole interpreter with it (decision D32).
    ///
    /// Nothing here allocates: every call returns before the `vec!`.
    #[test]
    fn an_unallocatable_layout_is_an_error_not_an_abort() {
        let max: usize = max_elements::<u16>();
        assert_eq!(max, isize::MAX as usize / 2);
        assert_eq!(
            ImageU16::zeros(max + 1, 1),
            Err(ImageError::LayoutTooLarge { len: max + 1, max })
        );
        assert_eq!(
            ImageU16::zeros_with_stride(1, 2, max),
            Err(ImageError::LayoutTooLarge { len: 2 * max, max })
        );
        // A frame that big cannot be filled either. `max + 1` pixels are more
        // than any slice can hold, so the source check fires first; either way
        // it is an error and no allocation is attempted.
        let mut image: ImageU16 = ImageU16::zeros(2, 2).unwrap();
        assert_eq!(
            image.fill_from_u8_strided(&[0; 4], max + 1, 1, max + 1),
            Err(ImageError::ShortBuffer {
                height: 1,
                stride: max + 1,
                len: 4
            })
        );
    }

    #[test]
    fn copy_from_densifies_and_reuses_the_buffer() {
        let source: ImageU16 = {
            let mut padded: ImageU16 = ImageU16::zeros_with_stride(3, 2, 5).unwrap();
            for y in 0..2 {
                for x in 0..3 {
                    padded.set(x, y, (10 * y + x) as u16);
                }
            }
            padded
        };
        let mut target: ImageU16 = ImageU16::zeros(3, 2).unwrap();
        let pointer: *const u16 = target.data().as_ptr();
        target.copy_from(&source).unwrap();
        assert_eq!(target.stride(), 3);
        assert_eq!(target.data(), &[0, 1, 2, 10, 11, 12]);
        assert_eq!(target.data().as_ptr(), pointer, "the buffer moved");
    }

    #[test]
    fn refilling_a_same_sized_frame_never_reallocates() {
        let bytes: Vec<u8> = (0..64 * 32).map(|i| (i % 251) as u8).collect();
        let mut image: ImageU16 = ImageU16::from_u8_strided(&bytes, 64, 32, 64).unwrap();
        let pointer: *const u16 = image.data().as_ptr();
        let capacity: usize = image.data.capacity();
        for _ in 0..8 {
            image.fill_from_u8_strided(&bytes, 64, 32, 64).unwrap();
        }
        assert_eq!(image.data().as_ptr(), pointer, "the buffer moved");
        assert_eq!(image.data.capacity(), capacity, "the buffer was regrown");
        // A smaller frame reuses the same allocation too.
        image.fill_from_u8_strided(&bytes, 32, 16, 64).unwrap();
        assert_eq!(image.data().as_ptr(), pointer);
        assert_eq!(image.data.capacity(), capacity);
        assert_eq!(image.data().len(), 32 * 16);
    }

    #[test]
    fn in_bounds_excludes_the_last_row_and_column() {
        let image: ImageU16 = ImageU16::zeros(10, 8).unwrap();
        assert!(image.in_bounds(0.0, 0.0, 0.0));
        assert!(image.in_bounds(8.99, 6.99, 0.0));
        // `w - 1` is out of bounds at border 0: `interp` reads `ix + 1`.
        assert!(!image.in_bounds(9.0, 3.0, 0.0));
        assert!(!image.in_bounds(3.0, 7.0, 0.0));
        // border 1 loses one more pixel on each side.
        assert!(!image.in_bounds(0.5, 3.0, 1.0));
        assert!(image.in_bounds(1.0, 1.0, 1.0));
        assert!(!image.in_bounds(8.0, 3.0, 1.0));
        assert!(!image.in_bounds(3.0, 6.0, 1.0));
    }

    #[test]
    fn interp_at_an_integer_coordinate_is_the_pixel() {
        let image: ImageU16 = random_image(17, 13, 7);
        for y in 0..12 {
            for x in 0..16 {
                assert_abs_diff_eq!(
                    image.interp(x as f32, y as f32),
                    f32::from(image.get(x, y).unwrap()),
                    epsilon = 0.0
                );
            }
        }
    }

    #[test]
    fn interp_halfway_is_the_average_of_the_four_neighbours() {
        let mut image: ImageU16 = ImageU16::zeros(4, 4).unwrap();
        image.set(1, 1, 100);
        image.set(2, 1, 200);
        image.set(1, 2, 300);
        image.set(2, 2, 400);
        assert_abs_diff_eq!(image.interp(1.5, 1.5), 250.0, epsilon = 1e-4);
        assert_abs_diff_eq!(image.interp(1.5, 1.0), 150.0, epsilon = 1e-4);
        assert_abs_diff_eq!(image.interp(1.0, 1.5), 200.0, epsilon = 1e-4);
    }

    /// Sample a smooth sine into an image and compare `interp_grad` with unit-step
    /// central differences of `interp`. A looser check against the sine's analytic
    /// derivative also catches swapped or sign-flipped axes.
    #[test]
    fn image_interpolate_grad() {
        let image: ImageU16 = smooth_image(512, 256);
        let (x, y): (f32, f32) = (231.4, 123.34345);

        let (value, grad) = image.interp_grad(x, y);
        assert_abs_diff_eq!(value, image.interp(x, y), epsilon = 0.0);

        let numeric: [f32; 2] = [
            0.5 * (image.interp(x + 1.0, y) - image.interp(x - 1.0, y)),
            0.5 * (image.interp(x, y + 1.0) - image.interp(x, y - 1.0)),
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
            let image: ImageU16 = random_image(31, 21, seed);
            prop_assert_eq!(
                image.interp(x as f32, y as f32),
                f32::from(image.get(x, y).unwrap())
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
            let image: ImageU16 = random_image(31, 21, seed);
            let (value, grad) = image.interp_grad(x, y);
            prop_assert_eq!(value, image.interp(x, y));
            let dx: f32 = 0.5 * (image.interp(x + 1.0, y) - image.interp(x - 1.0, y));
            let dy: f32 = 0.5 * (image.interp(x, y + 1.0) - image.interp(x, y - 1.0));
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
            let image: ImageU16 = smooth_image(512, 256);
            let (_, grad) = image.interp_grad(x, y);
            let step: f32 = 0.5;
            let dx: f32 = (image.interp(x + step, y) - image.interp(x - step, y)) / (2.0 * step);
            let dy: f32 = (image.interp(x, y + step) - image.interp(x, y - step)) / (2.0 * step);
            // The two differ by the third derivative of the sine over the step,
            // plus the bilinear interpolation error: both are below 1% of the
            // y gradient, which is the larger of the two.
            let scale: f32 = 0.01 * (SMOOTH_AMPLITUDE as f32 / 20.0);
            prop_assert!((grad[0] - dx).abs() <= scale, "dx {} vs {}", grad[0], dx);
            prop_assert!((grad[1] - dy).abs() <= scale, "dy {} vs {}", grad[1], dy);
        }

        /// Whatever the geometry, the widened image is dense, the right length
        /// and every pixel is its source byte shifted left by eight.
        #[test]
        fn widening_is_stride_correct_for_any_padding(
            width in 1usize..17,
            height in 1usize..13,
            padding in 0usize..7,
        ) {
            let stride: usize = width + padding;
            let bytes: Vec<u8> = (0..stride * height).map(|i| (i % 256) as u8).collect();
            let image: ImageU16 = ImageU16::from_u8_strided(&bytes, width, height, stride)?;
            prop_assert_eq!(image.stride(), width);
            prop_assert_eq!(image.data().len(), width * height);
            for y in 0..height {
                for x in 0..width {
                    prop_assert_eq!(
                        image.get(x, y),
                        Some(u16::from(bytes[y * stride + x]) << 8)
                    );
                }
            }
        }
    }
}
