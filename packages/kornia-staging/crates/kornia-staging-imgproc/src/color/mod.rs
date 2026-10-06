//! Pixel depth and color conversions.
//!
//! For dense images, use [`kornia_image::ops::cast_and_scale`]. The strided
//! operation here widens padded byte rows directly into caller-owned storage.
//!
//! ```
//! use kornia_image::{ops::cast_and_scale, Image, ImageSize};
//! let size = ImageSize { width: 2, height: 1 };
//! let source: Image<u8, 1> = Image::new(size, vec![1u8, 255])?;
//! let mut target = Image::from_size_val(size, 0u16)?;
//! cast_and_scale(&source, &mut target, 256u16)?;
//! assert_eq!(target.as_slice(), &[256, 65280]);
//! # Ok::<(), kornia_image::ImageError>(())
//! ```
mod depth;
pub use depth::{widen_u8_shift8_strided, widen_u8_shift8_strided_unchecked};
