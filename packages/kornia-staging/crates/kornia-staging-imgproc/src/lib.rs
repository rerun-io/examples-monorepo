//! Image processing extensions prepared for kornia-imgproc.
#![deny(missing_docs)]

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

/// Sparse optical flow.
pub mod optical_flow;

#[cfg(any(test, feature = "test-fixtures"))]
pub mod test_fixtures;
