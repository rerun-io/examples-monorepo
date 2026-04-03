//! Shared constants for the Gaussian splatting pipeline.
//!
//! These values are used by the GPU compute pipeline and the CPU pre-pass
//! (frustum culling + depth sorting).  They match the constants used by the
//! [Brush](https://github.com/ArthurBrussee/brush) renderer so that our
//! output is numerically comparable.

/// Hard cap on the number of splats sent to the GPU per frame.  Dense scenes
/// with millions of splats are truncated to this count (keeping the closest
/// ones) to maintain interactive frame rates.
pub const MAX_SPLATS_RENDERED: usize = 200_000;

/// Splats whose projected radius is smaller than this (in pixels) are discarded.
/// Prevents wasting GPU time on sub-pixel Gaussians.
pub const MIN_RADIUS_PX: f32 = 0.35;

/// Global opacity multiplier.  1.0 = no change.
pub const OPACITY_SCALE: f32 = 1.0;

/// Number of standard deviations used to compute the Gaussian's screen-space
/// bounding box.  3σ covers ~99.7% of the Gaussian's energy.
pub const SIGMA_COVERAGE: f32 = 3.0;

/// When the splat count exceeds this threshold, visibility culling and sorting
/// switch from single-threaded to parallel (using `rayon`).
pub const PARALLEL_SPLAT_THRESHOLD: usize = 16_384;

/// Zeroth spherical-harmonic coefficient: `1 / (2 * sqrt(π))`.
/// Used to convert the DC (degree-0) SH coefficient to a base color.
pub const SH_C0: f32 = 0.282_094_8;

/// Small blur added to the 2D covariance after projection.  Prevents
/// infinitely sharp splats from creating aliasing artifacts.  Matches the
/// value used by the Brush renderer.
pub const BRUSH_COVARIANCE_BLUR_PX: f32 = 0.3;

/// Minimum alpha contribution a splat must have to be considered visible.
/// Splats below this threshold (1/255) would not affect any pixel.
pub const BRUSH_VISIBILITY_ALPHA_THRESHOLD: f32 = 1.0 / 255.0;
