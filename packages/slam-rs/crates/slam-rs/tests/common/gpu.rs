//! Shared fixtures for GPU integration binaries.

use slam_rs::frontend::detect::BandRequest;
use slam_rs::frontend::se2::AffineCompact2f;
use slam_rs::frontend::tracker::{FlowTransforms, PointsSoA};

/// One band of a 50-pixel cell grid, keyed the way
/// `detect_keypoints_with_cells` keys it: `row` is the grid row and `rung` the
/// place on the threshold ladder, and the cache is indexed by the pair.
#[allow(
    dead_code,
    reason = "used by gpu_kernels and gpu_pool; other binaries compile a subset"
)]
pub fn band_at(row: usize, rung: usize, y: usize, rows: usize, threshold: i32) -> BandRequest {
    BandRequest {
        row,
        rung,
        y,
        rows,
        threshold,
    }
}

/// The keypoint budget both lanes are sized for, well over what the grid needs.
#[allow(
    dead_code,
    reason = "used by gpu_kernels and gpu_pool; other binaries compile a subset"
)]
pub const MAX_KEYPOINTS: usize = 1024;
/// `optical_flow_max_iterations` in every shipped config.
#[allow(
    dead_code,
    reason = "used by gpu_kernels and gpu_pool; other binaries compile a subset"
)]
pub const MAX_ITERATIONS: usize = 5;
/// `optical_flow_max_recovered_dist2` in every shipped config.
#[allow(
    dead_code,
    reason = "used by gpu_kernels and gpu_pool; other binaries compile a subset"
)]
pub const MAX_RECOVERED_DIST2: f32 = 0.09;

/// The pyramid geometry the shipped msd configs run: `optical_flow_levels = 3`
/// on a 960x960 frame, so four levels.
#[allow(
    dead_code,
    reason = "used by gpu_kernels and gpu_pool; other binaries compile a subset"
)]
pub const LEVELS: usize = 3;

/// Guesses that say "the point has not moved", which is what the frontend hands
/// the tracker when it has no pose prediction.
#[allow(
    dead_code,
    reason = "used by gpu_kernels and gpu_pool; other binaries compile a subset"
)]
pub fn guesses_at(positions: &PointsSoA) -> FlowTransforms {
    let mut guesses: FlowTransforms = FlowTransforms::with_capacity(positions.len());
    guesses.resize(positions.len());
    for index in 0..positions.len() {
        guesses.set(index, &AffineCompact2f::at(positions.get(index)));
    }
    guesses
}
