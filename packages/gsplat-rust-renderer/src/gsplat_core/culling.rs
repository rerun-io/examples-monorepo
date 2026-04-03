//! CPU-side visibility culling and depth sorting.
//!
//! Quickly rejects splats that are behind the camera, have zero opacity, or
//! project outside the screen.  The GPU compute shader does the exact
//! projection later — this is an approximate first pass to reduce the number
//! of splats uploaded to the GPU.

use rayon::prelude::*;

use super::constants::{MAX_SPLATS_RENDERED, OPACITY_SCALE, PARALLEL_SPLAT_THRESHOLD};
use super::types::{CameraApproximation, RenderGaussianCloud, SortedSplatIndex};

/// Build a sorted list of visible splat indices from the cloud.
///
/// This is an **approximate** first pass on the CPU.  It quickly rejects splats
/// that are behind the camera, have zero opacity, or project outside the
/// screen.  The GPU compute shader does the exact projection later.
///
/// The output is sorted **back-to-front** (farthest first) because alpha
/// blending requires this ordering.  If the cloud exceeds [`MAX_SPLATS_RENDERED`],
/// the closest splats are kept.
pub fn rebuild_visible_indices(
    visible_indices: &mut Vec<SortedSplatIndex>,
    cloud: &RenderGaussianCloud,
    camera: &CameraApproximation,
) {
    visible_indices.clear();

    // Use rayon for parallel filtering when there are enough splats to
    // amortize the threading overhead.
    if cloud.len() >= PARALLEL_SPLAT_THRESHOLD {
        *visible_indices = cloud
            .means_world
            .par_iter()
            .enumerate()
            .filter_map(|(index, mean_world)| {
                build_compute_candidate(index, *mean_world, cloud, camera)
            })
            .collect();
    } else {
        visible_indices.extend(cloud.means_world.iter().enumerate().filter_map(
            |(index, mean_world)| build_compute_candidate(index, *mean_world, cloud, camera),
        ));
    }

    // If we have more visible splats than the GPU cap, keep only the closest
    // ones.  `select_nth_unstable_by` is O(n) partial sort.
    if visible_indices.len() > MAX_SPLATS_RENDERED {
        let nth = MAX_SPLATS_RENDERED.saturating_sub(1);
        visible_indices.select_nth_unstable_by(nth, |left, right| {
            left.camera_depth
                .partial_cmp(&right.camera_depth)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        visible_indices.truncate(MAX_SPLATS_RENDERED);
    }

    // Sort back-to-front (descending depth) for correct alpha blending.
    if visible_indices.len() >= PARALLEL_SPLAT_THRESHOLD {
        visible_indices.par_sort_unstable_by(|left, right| {
            right
                .camera_depth
                .partial_cmp(&left.camera_depth)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        visible_indices.sort_by(|left, right| {
            right
                .camera_depth
                .partial_cmp(&left.camera_depth)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}

/// Test whether a single splat should be included in the visible set.
///
/// Rejection criteria (any one causes `None`):
/// - Zero or negative opacity
/// - Behind the near plane
/// - Non-finite position after projection
/// - Projects outside a generous screen margin (1.5× NDC bounds)
pub fn build_compute_candidate(
    index: usize,
    mean_world: glam::Vec3,
    cloud: &RenderGaussianCloud,
    camera: &CameraApproximation,
) -> Option<SortedSplatIndex> {
    let opacity = cloud.opacities[index] * OPACITY_SCALE;
    if opacity <= 0.0 {
        return None;
    }

    // Transform to view space.  In our right-handed convention, -Z points into
    // the screen, so `camera_depth = -mean_view.z`.
    let mean_view = camera.view_from_world.transform_point3(mean_world);
    let camera_depth = -mean_view.z;
    if !camera_depth.is_finite() || camera_depth <= camera.near_plane {
        return None;
    }

    // Project to clip space and check for degenerate w values.
    let clip = camera.projection_from_view * mean_view.extend(1.0);
    if !clip.is_finite() || clip.w.abs() <= 1e-6 {
        return None;
    }

    // Perspective divide to get NDC.  Reject splats well outside the screen.
    let center_ndc = clip.truncate() / clip.w;
    if center_ndc.x.abs() > 1.5 || center_ndc.y.abs() > 1.5 {
        return None;
    }

    Some(SortedSplatIndex {
        splat_index: index as u32,
        camera_depth,
    })
}
