//! Integer grayscale depth conversion.
use kornia_image::{Image, ImageError};

/// Densify padded byte rows during widening, without an intermediate copy.
///
/// # Arguments
/// * `src` - Byte storage covering `stride * dst.height()` elements.
/// * `stride` - Source row pitch in bytes.
/// * `dst` - Dense destination; its size defines the visible source rectangle.
///
/// # Errors
/// Returns [`ImageError::InvalidImageSize`] if the source row is too short or
/// the storage does not contain the requested number of full rows.
pub fn widen_u8_shift8_strided(
    src: &[u8],
    stride: usize,
    dst: &mut Image<u16, 1>,
) -> Result<(), ImageError> {
    let width = dst.width();
    let height = dst.height();
    if stride < width {
        return Err(ImageError::InvalidImageSize(stride, height, width, height));
    }
    if let Some(source_rows) = src.len().checked_div(stride) {
        if source_rows < height {
            return Err(ImageError::InvalidImageSize(
                stride,
                source_rows,
                stride,
                height,
            ));
        }
    }
    widen_u8_shift8_strided_unchecked(src, stride, dst);
    Ok(())
}

/// Widen previously validated padded byte rows into a dense image.
///
/// Each visible byte becomes `u16::from(byte) << 8`. Storage and size checks
/// belong at the caller's ingest boundary; this operation allocates no memory.
///
/// # Arguments
/// * `src` - Byte storage covering `stride * dst.height()` elements.
/// * `stride` - Source row pitch, at least `dst.width()`.
/// * `dst` - Dense destination defining the visible source rectangle.
///
/// # Panics
/// May panic if the caller has not met the source storage and stride requirements.
#[inline]
pub fn widen_u8_shift8_strided_unchecked(src: &[u8], stride: usize, dst: &mut Image<u16, 1>) {
    let width = dst.width();
    if width == 0 {
        return;
    }
    let length = stride * dst.height();
    for (source, target) in src[..length]
        .chunks_exact(stride)
        .zip(dst.as_slice_mut().chunks_exact_mut(width))
    {
        for (pixel, byte) in target.iter_mut().zip(source) {
            *pixel = u16::from(*byte) << 8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_images::zeros;

    #[test]
    fn rejects_short_rows_and_unrepresentable_source_geometry() {
        let mut image = zeros(2, 2);
        for (bytes, stride) in [(&[0; 4][..], 1), (&[0; 3][..], 2), (&[][..], usize::MAX)] {
            assert!(matches!(
                widen_u8_shift8_strided(bytes, stride, &mut image),
                Err(ImageError::InvalidImageSize(..))
            ));
            assert_eq!(image.as_slice(), &[0; 4]);
        }
    }

    #[test]
    fn widens_padded_rows_without_changing_allocation() {
        let mut image = zeros(2, 2);
        let pointer = image.as_slice().as_ptr();
        widen_u8_shift8_strided(&[0, 255, 99, 1, 128, 99], 3, &mut image).unwrap();
        assert_eq!(image.as_slice(), &[0, 65280, 256, 32768]);
        assert_eq!(image.as_slice().as_ptr(), pointer);
    }

    #[test]
    fn accepts_empty_images() {
        widen_u8_shift8_strided(&[], 0, &mut zeros(0, 4)).unwrap();
        widen_u8_shift8_strided(&[], 4, &mut zeros(4, 0)).unwrap();
    }
}
