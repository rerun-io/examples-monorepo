//! Image processing extensions prepared for kornia-imgproc.
#![deny(missing_docs)]

#[cfg(test)]
mod test_images;

/// Image resizing and pooling.
pub mod resize;

/// Image interpolation.
pub mod interpolation;

/// Feature extraction and decoding.
pub mod features;

/// Contour geometry.
pub mod contours;

/// Pixel depth and color conversions.
pub mod color;

/// Gaussian image pyramids.
pub mod pyramid;
