//! Core data structures for the Gaussian splatting pipeline.
//!
//! These types carry Gaussian splat data through the rendering pipeline —
//! from PLY loading through projection, culling, and rasterization.  None of
//! them depend on Rerun; the viewer module wraps them with Rerun-specific
//! component descriptors and query logic.

use std::sync::Arc;

use glam::{Affine3A, Mat4, Quat, Vec2, Vec3};

/// Spherical harmonic coefficients in a flat, GPU-uploadable layout.
#[derive(Clone, Debug)]
pub struct RenderShCoefficients {
    /// Number of SH coefficients per channel, including DC.
    /// Valid values: 1 (degree 0), 4 (degree 1), 9 (degree 2), 16 (degree 3),
    /// 25 (degree 4).
    pub coeffs_per_channel: usize,
    /// Flat storage: `coefficients[splat * coeffs_per_channel * 3 + coeff * 3 + channel]`.
    pub coefficients: Arc<[f32]>,
}

/// Packed, renderer-ready representation of a Gaussian splat cloud.
/// All positions are in world space (with the entity transform already applied).
#[derive(Clone, Debug)]
pub struct RenderGaussianCloud {
    /// World-space center of each Gaussian.
    pub means_world: Arc<[Vec3]>,
    /// Rotation quaternion for each Gaussian (unit-length).
    pub quats: Arc<[Quat]>,
    /// Per-axis scale factors for each Gaussian.
    pub scales: Arc<[Vec3]>,
    /// Per-splat opacity in `[0, 1]`.
    pub opacities: Arc<[f32]>,
    /// Base RGB color (from SH DC coefficient or vertex colors).
    pub colors_dc: Arc<[[f32; 3]]>,
    /// Optional higher-order SH coefficients for view-dependent color.
    pub sh_coeffs: Option<RenderShCoefficients>,
    /// Axis-aligned bounding box of all splat centers in world space.
    /// Used by the fallback camera when the user hasn't interacted yet.
    pub bounds_world: Option<(Vec3, Vec3)>,
}

impl RenderGaussianCloud {
    /// Number of Gaussian splats in this cloud.
    pub fn len(&self) -> usize {
        self.means_world.len()
    }

    /// Whether this cloud contains zero splats.
    pub fn is_empty(&self) -> bool {
        self.means_world.is_empty()
    }

    /// Construct a cloud from raw arrays (no Rerun dependency).
    ///
    /// This is the primary constructor used by the PLY loader and tests.
    /// The Rerun viewer has its own constructor that reads from the data store.
    pub fn from_raw(
        means_world: Vec<Vec3>,
        quats: Vec<Quat>,
        scales: Vec<Vec3>,
        opacities: Vec<f32>,
        colors_dc: Vec<[f32; 3]>,
        sh_coeffs: Option<RenderShCoefficients>,
    ) -> Self {
        let bounds_world = approximate_bounds_from_points(&means_world);
        Self {
            means_world: Arc::from(means_world),
            quats: Arc::from(quats),
            scales: Arc::from(scales),
            opacities: Arc::from(opacities),
            colors_dc: Arc::from(colors_dc),
            sh_coeffs,
            bounds_world,
        }
    }
}

/// Simplified camera parameters for rendering.
///
/// Used for frustum culling on the CPU and as uniforms for the GPU shaders.
/// Can be constructed from a Rerun 3D view state, a NeRF transforms JSON,
/// or manually via [`CameraApproximation::from_look_at`].
#[derive(Clone, Debug)]
pub struct CameraApproximation {
    /// View matrix: transforms world-space points into view (camera) space.
    /// Right-handed coordinate system with -Z pointing into the screen.
    pub view_from_world: Affine3A,
    /// Perspective projection matrix for the current viewport.
    pub projection_from_view: Mat4,
    /// Camera position in world space (used for SH view-direction computation).
    pub world_position: Vec3,
    /// Viewport dimensions in physical pixels (not logical/UI points).
    pub viewport_size_px: Vec2,
    /// Distance to the near clipping plane.
    pub near_plane: f32,
}

/// A splat that passed the CPU visibility test, carrying its index back into
/// the cloud arrays and its depth from the camera (for sorting).
#[derive(Clone, Copy, Debug)]
pub struct SortedSplatIndex {
    /// Index back into the cached render cloud arrays.
    pub splat_index: u32,
    /// Positive distance from camera.  Larger = farther away.
    pub camera_depth: f32,
}

/// Result of projecting one Gaussian onto the screen (CPU fallback path).
/// Contains everything the instanced-quad shader needs to render a splat.
#[derive(Clone, Copy, Debug)]
pub struct ProjectedGaussian {
    /// Screen-space center in normalized device coordinates (NDC), range `[-1, 1]`.
    pub center_ndc: Vec2,
    /// Depth in NDC for depth testing.
    pub ndc_depth: f32,
    /// Inverse of the 2D covariance matrix in NDC (upper triangle: xx, xy, yy).
    /// Used by the fragment shader to evaluate the Gaussian falloff.
    pub inv_cov_ndc_xx_xy_yy: [f32; 3],
    /// Conservative bounding radius in NDC for the instanced quad.
    pub radius_ndc: f32,
}

/// A fully prepared splat ready for GPU upload (CPU fallback path).
/// Combines the projected geometry with the final color and opacity.
#[derive(Clone, Copy, Debug)]
pub struct PreparedSplat {
    pub center_ndc: Vec2,
    pub ndc_depth: f32,
    pub inv_cov_ndc_xx_xy_yy: [f32; 3],
    pub radius_ndc: f32,
    pub color_rgb: [f32; 3],
    pub opacity: f32,
}

/// Compute the axis-aligned bounding box of a set of 3D points.
/// Returns `None` if the point set is empty or contains non-finite values.
pub fn approximate_bounds_from_points(points: &[Vec3]) -> Option<(Vec3, Vec3)> {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for point in points {
        min = min.min(*point);
        max = max.max(*point);
    }
    min.is_finite().then_some((min, max))
}
