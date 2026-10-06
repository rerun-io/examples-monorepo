//! The KLT tracker's own tests: the CPU patch set, the warp arrays and
//! `track_point`'s sub-pixel recovery.
//!
//! The independent analytic texture below defines the 0.01 pixel recovery oracle.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::fixtures::{pool, pyramid_of};
use super::{CpuPatchTracker, PatchTracker, TrackBatch, TrackInput, TrackPhase};
use kornia_image::Image;
use kornia_staging_imgproc::optical_flow::patch_se2::AffineCompact2f;
use kornia_staging_imgproc::optical_flow::patch_se2::{Pattern, Pattern51};
use kornia_staging_imgproc::optical_flow::patch_tracker::limits::{
    validate_exit_step, MAX_CAPACITY, MAX_LEVELS,
};
use kornia_staging_imgproc::optical_flow::patch_tracker::*;
use kornia_staging_imgproc::pyramid::PyramidPlanU16;
use nalgebra::{Matrix2, Vector2};
use proptest::prelude::*;

/// A band-limited texture: twelve plane waves with wavelengths between 16
/// and 56 pixels, in fixed pseudo-random directions and phases.
///
/// Band-limited matters twice over. Below the Nyquist of the finest pyramid
/// level the `[1,4,6,4,1]` subsample does not alias, so the coarse levels
/// really do carry the same shift; and away from the sampling limit bilinear
/// interpolation reconstructs the field closely, so the residual's fixed
/// point sits near the true shift rather than a fraction of a pixel off it.
/// Twelve components in different directions also keep every patch's `H_se2`
/// well conditioned: a single wave, or a field that is locally almost affine,
/// is the aperture problem and no tracker recovers a shift from it.
fn texture(x: f64, y: f64) -> f64 {
    // (wavelength, direction in turns, phase in turns)
    const WAVES: [(f64, f64, f64); 16] = [
        (22.0000, 0.000000, 0.000000),
        (23.5218, 0.381966, 0.618034),
        (25.1489, 0.763932, 0.236068),
        (26.8886, 0.145898, 0.854102),
        (28.7486, 0.527864, 0.472136),
        (30.7373, 0.909830, 0.090170),
        (32.8635, 0.291796, 0.708204),
        (35.1368, 0.673762, 0.326238),
        (37.5674, 0.055728, 0.944272),
        (40.1661, 0.437694, 0.562306),
        (42.9446, 0.819660, 0.180340),
        (45.9153, 0.201626, 0.798374),
        (49.0914, 0.583592, 0.416408),
        (52.4873, 0.965558, 0.034442),
        (56.1181, 0.347524, 0.652476),
        (60.0000, 0.729490, 0.270510),
    ];
    let tau: f64 = std::f64::consts::TAU;
    let mut sum: f64 = 0.0;
    for (wavelength, direction, phase) in WAVES {
        let angle: f64 = tau * direction;
        let projection: f64 = x * angle.cos() + y * angle.sin();
        sum += (tau * (projection / wavelength + phase)).sin();
    }
    sum / WAVES.len() as f64
}

/// A textured frame, shifted by `(dx, dy)`: the same continuous field
/// resampled at `(x - dx, y - dy)`, so the shift is exact by construction.
fn shifted_image(width: usize, height: usize, dx: f32, dy: f32) -> Image<u16, 1> {
    let mut image: Image<u16, 1> =
        Image::from_size_val(kornia_image::ImageSize { width, height }, 0u16).unwrap();
    for y in 0..height {
        for x in 0..width {
            let fx: f64 = f64::from(x as f32 - dx);
            let fy: f64 = f64::from(y as f32 - dy);
            let value: f64 = 32_000.0 + 28_000.0 * texture(fx, fy);
            image.set_pixel(x, y, 0, value as u16).unwrap();
        }
    }
    image
}

struct Fixture {
    prev: PyramidPlanU16,
    next: PyramidPlanU16,
    patches: PatchSoA<Pattern51>,
    transforms: FlowTransforms,
    positions: PointsSoA,
}

fn fixture(dx: f32, dy: f32, levels: usize) -> Fixture {
    let base: Image<u16, 1> = shifted_image(160, 160, 0.0, 0.0);
    let moved: Image<u16, 1> = shifted_image(160, 160, dx, dy);
    let prev: PyramidPlanU16 = pyramid_of(&base, levels);
    let next: PyramidPlanU16 = pyramid_of(&moved, levels);

    let mut positions: PointsSoA = PointsSoA::default();
    for y in (40..120).step_by(16) {
        for x in (40..120).step_by(16) {
            positions.push(Vector2::new(x as f32, y as f32));
        }
    }
    let mut transforms: FlowTransforms = FlowTransforms::default();
    for index in 0..positions.len() {
        transforms.push(&AffineCompact2f::at(positions.get(index)));
    }

    let mut patches: PatchSoA<Pattern51> = PatchSoA::new(positions.len(), levels + 1).unwrap();
    patches.build(&prev, &positions, None).unwrap();

    Fixture {
        prev,
        next,
        patches,
        transforms,
        positions,
    }
}

fn tracker(capacity: usize, levels: usize, threads: usize) -> CpuPatchTracker<Pattern51> {
    CpuPatchTracker::new(capacity, levels + 1, 5, 0.04, pool(threads)).unwrap()
}

fn batch_input(ids: &[u64], positions: &PointsSoA) -> TrackInput {
    let mut input = TrackInput {
        ids: ids.to_vec(),
        positions: positions.clone(),
        ..TrackInput::default()
    };
    for index in 0..positions.len() {
        input
            .guesses
            .push(&AffineCompact2f::at(positions.get(index)));
    }
    input
}

mod batch;
mod cache;
mod lifecycle;
mod numerics;

fn plan(capacity: usize, levels: usize, threads: usize) -> PatchTrackerPlan<Pattern51> {
    PatchTrackerPlan::new(capacity, levels + 1, 5, 0.04, pool(threads)).unwrap()
}
