//! Validated frame ingestion into dense Kornia images.
use kornia_image::{Image, ImageSize};
use kornia_staging_imgproc::color::widen_u8_shift8_strided;
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

/// Allocate a dense zero-filled frame after checking its allocation bounds.
///
/// # Errors
/// Returns a geometry or allocation-layout error before allocating invalid sizes.
pub fn zeros(width: usize, height: usize) -> Result<Image<u16, 1>, ImageError> {
    let len = checked_pixel_len(width, height, width)?;
    Image::new(ImageSize { width, height }, vec![0; len]).map_err(|_| ImageError::LayoutTooLarge {
        len,
        max: max_elements::<u16>(),
    })
}

/// Empty reusable ingestion buffer.
#[allow(
    clippy::unwrap_used,
    reason = "zero geometry always constructs an empty tensor"
)]
pub fn empty() -> Image<u16, 1> {
    // No allocation; zero dimensions are valid for Kornia's empty tensor.
    Image::new(
        ImageSize {
            width: 0,
            height: 0,
        },
        Vec::new(),
    )
    .unwrap()
}

/// Widen padded source rows directly into a dense image.
///
/// # Errors
/// Rejects short rows, truncated input, overflowing geometry or allocation bounds.
pub fn from_u8_strided(
    bytes: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<Image<u16, 1>, ImageError> {
    let mut image = empty();
    fill_from_u8_strided(&mut image, bytes, width, height, stride)?;
    Ok(image)
}

/// Refill a frame, retaining the allocation for unchanged geometry.
///
/// # Errors
/// Rejects invalid source geometry before changing the destination.
pub fn fill_from_u8_strided(
    image: &mut Image<u16, 1>,
    bytes: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(), ImageError> {
    let source_len = checked_len(width, height, stride)?;
    if bytes.len() < source_len {
        return Err(ImageError::ShortBuffer {
            height,
            stride,
            len: bytes.len(),
        });
    }
    checked_pixel_len(width, height, width)?;
    if image.width() != width || image.height() != height {
        *image = zeros(width, height)?;
    }
    widen_u8_shift8_strided(bytes, stride, image).map_err(|_| ImageError::ShortBuffer {
        height,
        stride,
        len: bytes.len(),
    })
}

/// Copy a dense image, reusing storage when its geometry matches.
///
/// # Errors
/// Returns an allocation-layout error for unrepresentable geometry.
pub fn copy_image(source: &Image<u16, 1>, target: &mut Image<u16, 1>) -> Result<(), ImageError> {
    if source.size() != target.size() {
        *target = zeros(source.width(), source.height())?;
    }
    target.as_slice_mut().copy_from_slice(source.as_slice());
    Ok(())
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
    use kornia_staging_imgproc::interpolation::{
        sample_bilinear_u16, sample_bilinear_with_gradient_u16,
    };
    use proptest::prelude::*;

    #[test]
    fn widening_shifts_left_by_eight() {
        let image = from_u8_strided(&[0, 1, 128, 255], 4, 1, 4).unwrap();
        assert_eq!(image.as_slice(), &[0, 256, 32768, 65280]);
    }

    #[test]
    fn a_bad_geometry_is_an_error_not_a_panic() {
        assert_eq!(
            from_u8_strided(&[0; 4], 4, 1, 2).err(),
            Some(ImageError::StrideTooSmall {
                width: 4,
                stride: 2
            })
        );
        assert_eq!(
            from_u8_strided(&[0; 4], 4, 2, 4).err(),
            Some(ImageError::ShortBuffer {
                height: 2,
                stride: 4,
                len: 4
            })
        );
        assert_eq!(
            from_u8_strided(&[], 1, 2, 1 << 63).err(),
            Some(ImageError::SizeOverflow {
                height: 2,
                stride: 1 << 63
            })
        );
    }

    #[test]
    fn an_unallocatable_layout_is_an_error_not_an_abort() {
        let max = max_elements::<u16>();
        assert_eq!(
            zeros(max + 1, 1).err(),
            Some(ImageError::LayoutTooLarge { len: max + 1, max })
        );
        let mut image = zeros(2, 2).unwrap();
        assert_eq!(
            fill_from_u8_strided(&mut image, &[0; 4], max + 1, 1, max + 1),
            Err(ImageError::ShortBuffer {
                height: 1,
                stride: max + 1,
                len: 4
            })
        );
    }

    #[test]
    fn densified_images_support_sampling_after_each_refill() {
        let mut image = from_u8_strided(&[99; 25], 5, 5, 5).unwrap();
        let first: Vec<u8> = (0..5)
            .flat_map(|y| (0..6).map(move |x| x + 2 * y))
            .collect();
        fill_from_u8_strided(&mut image, &first, 5, 5, 6).unwrap();
        assert_eq!(sample_bilinear_u16(&image, 1.25, 1.5), 1088.0);
        assert_eq!(
            sample_bilinear_with_gradient_u16(&image, 1.25, 1.5),
            (1088.0, [256.0, 512.0])
        );
        let second: Vec<u8> = (0..5)
            .flat_map(|y| (0..6).map(move |x| 20 + 3 * x + 4 * y))
            .collect();
        fill_from_u8_strided(&mut image, &second, 5, 5, 6).unwrap();
        assert_eq!(sample_bilinear_u16(&image, 1.25, 1.5), 7616.0);
        assert_eq!(
            sample_bilinear_with_gradient_u16(&image, 1.25, 1.5),
            (7616.0, [768.0, 1024.0])
        );
    }

    #[test]
    fn refilling_and_copying_same_sized_frames_never_reallocates() {
        let bytes: Vec<u8> = (0..70 * 32).map(|i| (i % 251) as u8).collect();
        let mut image = from_u8_strided(&bytes, 64, 32, 70).unwrap();
        let pointer = image.as_slice().as_ptr();
        let mut copy = zeros(64, 32).unwrap();
        let copy_pointer = copy.as_slice().as_ptr();
        for _ in 0..8 {
            fill_from_u8_strided(&mut image, &bytes, 64, 32, 70).unwrap();
            copy_image(&image, &mut copy).unwrap();
        }
        assert_eq!(image.as_slice().as_ptr(), pointer);
        assert_eq!(copy.as_slice().as_ptr(), copy_pointer);
        assert_eq!(copy.as_slice(), image.as_slice());
    }

    proptest! {
        #[test]
        fn widening_is_stride_correct_for_any_padding(width in 1usize..17, height in 1usize..13, padding in 0usize..7) {
            let stride = width + padding;
            let bytes: Vec<u8> = (0..stride * height).map(|i| (i % 256) as u8).collect();
            let image = from_u8_strided(&bytes, width, height, stride)?;
            prop_assert_eq!(image.as_slice().len(), width * height);
            for y in 0..height { for x in 0..width {
                prop_assert_eq!(image.as_slice()[y * width + x], u16::from(bytes[y * stride + x]) << 8);
            }}
        }
    }
}
