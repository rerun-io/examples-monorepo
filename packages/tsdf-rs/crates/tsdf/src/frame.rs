use crate::error::TsdfError;
use crate::geometry::validate_non_negative;

/// A pinhole camera at one pose.
///
/// Both matrices are row-major. `intrinsics` is the 3-by-3 matrix `K` that
/// maps camera coordinates to homogeneous pixels (OpenCV convention: +Z
/// forward, pixel centers at integer coordinates). `camera_from_world` is a
/// rigid 4-by-4 transform from world to camera coordinates.
///
/// # Example
///
/// ```
/// use tsdf::Camera;
///
/// let camera = Camera::pinhole([100.0, 100.0, 31.5, 23.5], Camera::IDENTITY);
/// assert_eq!(camera.intrinsics[2], 31.5);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    /// Row-major 3-by-3 intrinsic matrix `K`.
    pub intrinsics: [f32; 9],
    /// Row-major 4-by-4 rigid world-to-camera transform.
    pub camera_from_world: [f32; 16],
}

impl Camera {
    /// The identity pose: camera and world frames coincide.
    pub const IDENTITY: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ];

    /// A camera with `K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]` from
    /// `[fx, fy, cx, cy]` in pixels.
    pub fn pinhole(focal_principal: [f32; 4], camera_from_world: [f32; 16]) -> Self {
        let [fx, fy, cx, cy] = focal_principal;
        Self {
            intrinsics: [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
            camera_from_world,
        }
    }
}

/// Depths a frame contributes: `min <= depth <= max`, in world units.
///
/// Non-finite and non-positive depths never contribute. The default range
/// accepts every other depth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthRange {
    /// Nearest depth integrated; finite and non-negative.
    pub min: f32,
    /// Farthest depth integrated; at least `min`, may be infinite.
    pub max: f32,
}

impl Default for DepthRange {
    fn default() -> Self {
        Self {
            min: 0.0,
            max: f32::INFINITY,
        }
    }
}

impl DepthRange {
    pub(crate) fn validate(&self) -> Result<(), TsdfError> {
        validate_non_negative("min_depth", self.min)?;
        if self.max.is_nan() || self.max < self.min {
            return Err(TsdfError::InvalidParameter {
                name: "max_depth",
                reason: "must be at least min_depth",
                value: self.max,
            });
        }
        Ok(())
    }

    /// Whether `depth` is a valid sample inside the range.
    #[inline]
    pub(crate) fn contains(&self, depth: f32) -> bool {
        depth > 0.0 && depth.is_finite() && depth >= self.min && depth <= self.max
    }
}

/// One depth image to integrate, with optional color.
///
/// Images are row-major: pixel `(u, v)` is element `v * width + u`.
///
/// # Example
///
/// ```
/// use tsdf::{DepthFrame, DepthRange};
///
/// let depth = [1.0_f32; 6];
/// let color = [128_u8; 18];
/// let frame = DepthFrame::new(&depth, [2, 3])
///     .with_color(&color)
///     .with_depth_range(DepthRange { min: 0.2, max: 8.0 });
/// assert_eq!(frame.image_size, [2, 3]);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct DepthFrame<'a> {
    /// Camera Z depth (not ray length) per pixel, in world units.
    pub depth: &'a [f32],
    /// `[height, width]` in pixels.
    pub image_size: [usize; 2],
    /// RGB bytes per pixel, `height * width * 3` values.
    pub color: Option<&'a [u8]>,
    /// Depths outside this range are not integrated.
    pub depth_range: DepthRange,
}

impl<'a> DepthFrame<'a> {
    /// A frame without color that integrates every valid depth.
    pub fn new(depth: &'a [f32], image_size: [usize; 2]) -> Self {
        Self {
            depth,
            image_size,
            color: None,
            depth_range: DepthRange::default(),
        }
    }

    /// The same frame with RGB bytes per pixel.
    pub fn with_color(self, color: &'a [u8]) -> Self {
        Self {
            color: Some(color),
            ..self
        }
    }

    /// The same frame integrating only depths inside `depth_range`.
    pub fn with_depth_range(self, depth_range: DepthRange) -> Self {
        Self {
            depth_range,
            ..self
        }
    }
}
