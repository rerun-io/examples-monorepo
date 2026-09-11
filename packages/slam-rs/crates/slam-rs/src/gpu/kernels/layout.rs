//! layout kernels and their launchers.
use cubecl::prelude::*;

// Per-frame launchers use launch_unchecked. Stage shape checks establish
// the binding lengths; each raw binding keeps its handle and element count
// together. The storage probe alone retains checked launch mode.
/// Units per cube on the per-patch kernels.
///
/// One unit per pattern tap, rounded up to two warps: the largest pattern is 52
/// taps ([`crate::frontend::patterns::MAX_PATTERN_SIZE`]), so 64 covers every
/// pattern and the reductions stay inside one cube.
pub const TAP_UNITS: u32 = 64;

/// [`TAP_UNITS`] as the `#[comptime]` length every per-patch shared array is
/// declared with.
///
/// The nine arrays below are indexed by unit, so raising `TAP_UNITS` for a
/// pattern larger than 52 taps while they stayed at a literal 64 would have
/// units 64.. writing past every one of them, under `launch_unchecked` and with
/// no test that would say so. Off one constant, the coupling is the compiler's.
pub(crate) const TAP_SLOTS: usize = TAP_UNITS as usize;

/// Units per cube on the per-element bookkeeping kernels.
pub(crate) const LINEAR_UNITS: u32 = 256;

/// Cube width on the pyramid kernel, the 32x8 tile the kernel comparison test
/// settled on.
pub const TILE_W: u32 = 32;
/// Cube height on the pyramid kernel.
pub const TILE_H: u32 = 8;

// The four numbers below are the CPU lane's own, aliased rather than
// re-declared: a kernel that drifted from its reference by a constant would
// still compile, and `FILTER_LANES` below already shows the shape.
/// `border` on every patch tap, `PATCH_BORDER`.
pub(crate) const PATCH_BORDER: f32 = crate::frontend::patch::PATCH_BORDER;
/// `const int filter_margin = 2`.
pub(crate) const FILTER_MARGIN: f32 = crate::frontend::tracker::FILTER_MARGIN;
/// Upper bound for a valid increment, aliased from the CPU tracker so both
/// lanes share one number.
pub(crate) const MAX_INCREMENT_INFINITY_NORM: f32 =
    crate::frontend::tracker::MAX_INCREMENT_INFINITY_NORM;
/// `Sophus::Constants<float>::epsilon()`.
pub(crate) const SOPHUS_EPSILON: f32 = <f32 as crate::lie::LieScalar>::SOPHUS_EPSILON;

/// A device buffer and the element count the kernel will see in it.
///
/// Spelled once rather than at each of the launcher parameters below, all of
/// which take exactly this: the count is a promise `BufferArg::from_raw_parts`
/// cannot check, so keeping it beside the handle is what makes the promise
/// visible at the call site.
pub(crate) type Buffer<'a> = (&'a cubecl::server::Handle, usize);

/// The dispatch every image-shaped kernel uses: one unit per pixel, over
/// [`TILE_W`] x [`TILE_H`] tiles.
///
/// Two-dimensional cube dims with `ABSOLUTE_POS_X`/`_Y` rather than a linear
/// index and a `div`/`mod`, which is the layout the kernel comparison
/// measurement settled on.
pub(crate) fn tile_2d(width: usize, height: usize) -> (CubeCount, CubeDim) {
    (
        CubeCount::Static(
            (width as u32).div_ceil(TILE_W),
            (height as u32).div_ceil(TILE_H),
            1,
        ),
        CubeDim::new_3d(TILE_W, TILE_H, 1),
    )
}

/// Cubes per dispatch dimension a WebGPU implementation must allow, and the
/// number every one of them stops at.
///
/// wgpu reports its adapter's own `max_compute_workgroups_per_dimension` and on
/// every adapter measured that is exactly this floor, so the portable lane
/// treats it as the limit rather than as a minimum.
pub(crate) const MAX_CUBES_PER_DIM: u32 = 65_535;

/// The dispatch every per-element bookkeeping kernel uses: one unit per element,
/// found through [`ABSOLUTE_POS`].
///
/// Two dimensions rather than one, because a single row of cubes reaches only
/// `65,535 * 256 = 2^24` elements and the frame copy is one unit per **pixel**:
/// a 4097x4097 frame is 16,785,409 of them, and the portable lane failed that
/// dispatch at validation — a wgpu error on its own thread, so the caller saw a
/// panicking read rather than a refusal. Both compilers flatten `ABSOLUTE_POS`
/// as `z * cubes_x * units * cubes_y + y * cubes_x * units + x`, so the second
/// row is a continuation of the first and the mapping is the same dense
/// enumeration either way. Anything under the ceiling — every shipped frame and
/// every keypoint buffer — still dispatches exactly one row, unchanged.
pub(crate) fn linear_1d(count: usize) -> (CubeCount, CubeDim) {
    let cubes: u32 = (count as u32).div_ceil(LINEAR_UNITS);
    (
        CubeCount::Static(
            cubes.min(MAX_CUBES_PER_DIM),
            cubes.div_ceil(MAX_CUBES_PER_DIM),
            1,
        ),
        CubeDim::new_3d(LINEAR_UNITS, 1, 1),
    )
}

/// The shape every per-patch launcher needs: capacity, taps, levels, count.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PatchShape {
    /// Patch slots the store was allocated for.
    pub capacity: usize,
    /// Pattern taps.
    pub taps: usize,
    /// Pyramid levels, which is `optical_flow_levels + 1`.
    pub num_levels: usize,
    /// Patches actually filled.
    pub count: usize,
}

/// Where a launcher reads positions and the selection flag inside one buffer.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PositionBases {
    /// Index of patch 0's `x`.
    pub x: usize,
    /// Index of patch 0's `y`.
    pub y: usize,
    /// Index of patch 0's selection flag, non-zero meaning selected.
    pub selected: usize,
}
