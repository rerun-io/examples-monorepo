use crate::optical_flow::{
    patch_se2::AffineCompact2f,
    patch_tracker::{FlowTransforms, PointsSoA},
};
/// The keypoint budget both lanes are sized for, well over what the grid needs.
pub const MAX_KEYPOINTS: usize = 1024;
/// Iterations per pyramid level in the shared fixtures.
pub const MAX_ITERATIONS: usize = 5;
/// Maximum squared forward/backward distance in the shared fixtures.
pub const MAX_RECOVERED_DIST2: f32 = 0.09;

/// Three pyramid reductions, giving four levels.
pub const LEVELS: usize = 3;

/// Initial transforms with zero pixel displacement.
pub fn guesses_at(positions: &PointsSoA) -> FlowTransforms {
    let mut guesses: FlowTransforms = FlowTransforms::with_capacity(positions.len());
    guesses.resize(positions.len());
    for index in 0..positions.len() {
        guesses.set(index, &AffineCompact2f::at(positions.get(index)));
    }
    guesses
}

/// Interior positions separated by 71 pixels.
pub fn grid_positions(size: usize) -> PointsSoA {
    let mut positions: PointsSoA = PointsSoA::with_capacity(256);
    let mut y: usize = 96;
    while y + 96 < size {
        let mut x: usize = 96;
        while x + 96 < size {
            positions.push([x as f32 + 0.37, y as f32 - 0.21]);
            x += 71;
        }
        y += 71;
    }
    positions
}
