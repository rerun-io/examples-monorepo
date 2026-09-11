//! cell select kernels and their launchers.
use super::layout::Buffer;
use cubecl::prelude::*;

// Per-frame launchers use launch_unchecked. Stage shape checks establish
// the binding lengths; each raw binding keeps its handle and element count
// together. The storage probe alone retains checked launch mode.
// ── per-cell corner selection ────────────────────────────────────────────────

/// Cube width on the per-cell selection kernel.
///
/// 32x8 units, the same 256-unit tile the image kernels use: a cell's candidate
/// window is at most `cell - 6` wide, so a 32-wide row of units walks it in one
/// or two coalesced steps and the shared reduction stays at WebGPU's 256-unit
/// floor.
const SELECT_DIM_X: usize = 32;
/// Cube height on the per-cell selection kernel.
const SELECT_DIM_Y: usize = 8;
/// Units per cube, and the length of the shared reduction array.
const SELECT_SLOTS: usize = SELECT_DIM_X * SELECT_DIM_Y;
/// Halvings a 256-slot tree reduction takes.
const SELECT_STEPS: usize = 8;

/// The FAST ring radius, off the CPU detector's own constant.
const SELECT_MARGIN: usize = crate::frontend::detect::FAST_BORDER;
/// `EDGE_THRESHOLD`, likewise: the kernel applies the same gate the host loop
/// applied to the corner it chose.
const SELECT_EDGE: f32 = crate::frontend::detect::EDGE_THRESHOLD;

// The packed cell key's layout is the host detector's, aliased rather than
// re-declared for the same reason `FILTER_LANES` is: the kernel writes the keys
// and `detect_keypoints_with_cells` takes them apart, and a shift that drifted
// by one would move every corner without failing to compile.
/// Where the packed key keeps `255 - score`.
const KEY_SCORE_SHIFT: u32 = crate::frontend::cell::KEY_SCORE_SHIFT;
/// Where the packed key keeps the row.
const KEY_ROW_SHIFT: u32 = crate::frontend::cell::KEY_ROW_SHIFT;
/// `NO_CELL_WINNER`, which the kernel spells as a literal below because the
/// `#[cube]` macro keeps a *named* integer constant comptime and the local it
/// initialises is then assigned from the runtime reduction. This is what pins
/// the literal to the constant.
const _: () = assert!(crate::frontend::detect::NO_CELL_WINNER == 4_294_967_295);

/// One in-window candidate's score, or zero.
///
/// The zero is load-bearing rather than a convenience: the host's
/// `suppress_non_maxima` compares a candidate against a `side x side` scratch
/// grid that is **zero wherever this cell wrote nothing**, so a neighbour
/// outside the cell's own column and row window scores `0.0` there even where
/// `kept` holds a real corner. Reproducing that means testing the window before
/// the load, which is also what keeps the load in range: the caller only ever
/// asks about a pixel one step outside the window.
#[cube]
#[allow(clippy::too_many_arguments)]
fn cell_candidate(
    kept: &Array<u8>,
    width: usize,
    x: usize,
    y: usize,
    first_x: usize,
    last_x: usize,
    first_y: usize,
    last_y: usize,
    threshold: u32,
) -> u32 {
    let mut value: u32 = 0u32;
    if x >= first_x && x < last_x && y >= first_y && y < last_y {
        let score = u32::cast_from(kept[y * width + x]);
        if score > threshold {
            value = score;
        }
    }
    value
}

/// `detectKeypointsWithCells`' inner loop, one cube per grid cell.
///
/// The host used to download the candidate image and its bitmask — 2.07 MB per
/// two-camera frameset on MIO10 — walk one row band per cell row, suppress
/// non-maxima per cell and sort the survivors. All of that is this kernel, and
/// what comes back is one packed key per cell.
///
/// **Why one key is the whole answer.** Every shipped config sets
/// `optical_flow_detection_num_points_cell = 1`, and the threshold ladder cannot
/// change which corner that one point is: a candidate at rung `t` is
/// `kept > t`, suppression kills `p` only through a neighbour `q` with
/// `score_q >= score_p`, and such a `q` is itself a candidate at every rung `p`
/// is. So suppression survival does not depend on the rung, and the ladder
/// simply admits survivors in descending score. The cell's outcome is the first
/// survivor in the host's own total order — score descending, then row, then
/// column ascending, which is the order the band walk produces and
/// `sort_by` is stable in — that clears `safe_radius`, the masks and
/// `EDGE_THRESHOLD`.
///
/// That total order is what the packed key
/// `((255 - score) << 24) | (y << 12) | x` reduces under integer **minimum**:
/// exact, associative and commutative, so the shared tree below agrees with the
/// host walk bit for bit and a subgroup fast path would agree with both. It
/// holds while the frame is under 4,096 pixels on a side, which the caller
/// checks.
///
/// The masks are not here: they arrive as whole cells (`flow.rs`'s overlap
/// rectangles are the cell rectangles), so a masked cell is wholly masked and
/// the host drops its key instead of the kernel testing 361 rectangles per
/// pixel.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn fast_cell_select_kernel(
    kept: &Array<u8>,
    best: &mut Array<u32>,
    width: usize,
    height: usize,
    cell: usize,
    x_start: usize,
    y_start: usize,
    cells_x: usize,
    cells_y: usize,
    threshold: u32,
    safe_radius: f32,
    centre_x: f32,
    centre_y: f32,
) {
    let column = usize::cast_from(CUBE_POS_X);
    let row = usize::cast_from(CUBE_POS_Y);
    let unit = usize::cast_from(UNIT_POS_Y) * SELECT_DIM_X + usize::cast_from(UNIT_POS_X);

    // Cube-uniform, so every unit leaves together and no barrier below is
    // reached by only part of the cube. The launch sizes the grid exactly, so
    // this is the guard against a future launcher rather than a live branch.
    if column >= cells_x || row >= cells_y {
        terminate!();
    }

    // The band `detect_keypoints_with_cells` asks for is `[y + 3, y + cell - 3)`
    // clamped to `height - 3`, and the columns a cell keeps out of it are
    // `[x + 3, min(x + cell - 3, width - 3))`.
    let first_x = x_start + column * cell + SELECT_MARGIN;
    let first_y = y_start + row * cell + SELECT_MARGIN;
    let mut last_x = x_start + column * cell + cell - SELECT_MARGIN;
    if last_x > width - SELECT_MARGIN {
        last_x = width - SELECT_MARGIN;
    }
    let mut last_y = y_start + row * cell + cell - SELECT_MARGIN;
    if last_y > height - SELECT_MARGIN {
        last_y = height - SELECT_MARGIN;
    }

    // `border <= x && x < w - border - 1`, hoisted.
    let edge_x = f32::cast_from(width) - SELECT_EDGE - 1.0f32;
    let edge_y = f32::cast_from(height) - SELECT_EDGE - 1.0f32;

    // `NO_CELL_WINNER`; see the assertion beside `KEY_ROW_SHIFT`.
    let mut key: u32 = 4_294_967_295u32;
    let mut y = first_y + usize::cast_from(UNIT_POS_Y);
    while y < last_y {
        let mut x = first_x + usize::cast_from(UNIT_POS_X);
        while x < last_x {
            let score = cell_candidate(
                kept, width, x, y, first_x, last_x, first_y, last_y, threshold,
            );
            if score > 0u32 {
                // OpenCV's rule: strictly greater than all eight neighbours, a
                // neighbour that is not a candidate scoring zero. Both sides
                // strict, so a plateau of equal scores yields nothing.
                let mut rival: u32 = 0u32;
                rival = max(
                    rival,
                    cell_candidate(
                        kept,
                        width,
                        x - 1usize,
                        y - 1usize,
                        first_x,
                        last_x,
                        first_y,
                        last_y,
                        threshold,
                    ),
                );
                rival = max(
                    rival,
                    cell_candidate(
                        kept,
                        width,
                        x,
                        y - 1usize,
                        first_x,
                        last_x,
                        first_y,
                        last_y,
                        threshold,
                    ),
                );
                rival = max(
                    rival,
                    cell_candidate(
                        kept,
                        width,
                        x + 1usize,
                        y - 1usize,
                        first_x,
                        last_x,
                        first_y,
                        last_y,
                        threshold,
                    ),
                );
                rival = max(
                    rival,
                    cell_candidate(
                        kept,
                        width,
                        x - 1usize,
                        y,
                        first_x,
                        last_x,
                        first_y,
                        last_y,
                        threshold,
                    ),
                );
                rival = max(
                    rival,
                    cell_candidate(
                        kept,
                        width,
                        x + 1usize,
                        y,
                        first_x,
                        last_x,
                        first_y,
                        last_y,
                        threshold,
                    ),
                );
                rival = max(
                    rival,
                    cell_candidate(
                        kept,
                        width,
                        x - 1usize,
                        y + 1usize,
                        first_x,
                        last_x,
                        first_y,
                        last_y,
                        threshold,
                    ),
                );
                rival = max(
                    rival,
                    cell_candidate(
                        kept,
                        width,
                        x,
                        y + 1usize,
                        first_x,
                        last_x,
                        first_y,
                        last_y,
                        threshold,
                    ),
                );
                rival = max(
                    rival,
                    cell_candidate(
                        kept,
                        width,
                        x + 1usize,
                        y + 1usize,
                        first_x,
                        last_x,
                        first_y,
                        last_y,
                        threshold,
                    ),
                );
                if score > rival {
                    let fx = f32::cast_from(x);
                    let fy = f32::cast_from(y);
                    let dx = fx - centre_x;
                    let dy = fy - centre_y;
                    // Use sqrt of the sum of squares, not hypot.
                    let distance = f32::sqrt(dx * dx + dy * dy);
                    let mut inside = true;
                    if safe_radius != 0.0f32 && distance >= safe_radius {
                        inside = false;
                    }
                    if fx < SELECT_EDGE || fx >= edge_x || fy < SELECT_EDGE || fy >= edge_y {
                        inside = false;
                    }
                    if inside {
                        let candidate = ((255u32 - score) << KEY_SCORE_SHIFT)
                            | (u32::cast_from(y) << KEY_ROW_SHIFT)
                            | u32::cast_from(x);
                        key = min(key, candidate);
                    }
                }
            }
            x += SELECT_DIM_X;
        }
        y += SELECT_DIM_Y;
    }

    // Integer minimum is associative and commutative, so this tree is the same
    // answer as the ascending serial reduction the float kernels are obliged to
    // use — and every barrier is cube-uniform.
    let mut reduce = SharedMemory::<u32>::new(SELECT_SLOTS);
    reduce[unit] = key;
    sync_cube();
    #[unroll]
    for step in 0..SELECT_STEPS {
        // Comptime, and it has to be: a `let mut` seeded from a constant is a
        // const variable inside `#[cube]` and `stride /= 2` on one panics the
        // expansion on cubecl's own thread — which reaches the caller as a
        // buffer of zeros, not as an error.
        let stride = SELECT_SLOTS >> (step + 1usize);
        if unit < stride {
            reduce[unit] = min(reduce[unit], reduce[unit + stride]);
        }
        sync_cube();
    }
    if unit == 0usize {
        best[row * cells_x + column] = reduce[0usize];
    }
}

/// The detection grid one [`launch_fast_cell_select`] covers.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CellSelectGeometry {
    /// Frame width in pixels.
    pub width: usize,
    /// Frame height in pixels.
    pub height: usize,
    /// `PATCH_SIZE`.
    pub cell: usize,
    /// The grid's first cell's left edge.
    pub x_start: usize,
    /// The grid's first cell's top edge.
    pub y_start: usize,
    /// Cells the grid walks across, which is `w / cell`.
    pub cells_x: usize,
    /// Cells the grid walks down.
    pub cells_y: usize,
    /// The last rung the host ladder visits, already clamped into the score's
    /// own `u8` range.
    pub threshold: u32,
    /// `optical_flow_image_safe_radius`; `0` switches the gate off.
    pub safe_radius: f32,
    /// `(width / 2) as f32`, an integer halving.
    pub centre_x: f32,
    /// `(height / 2) as f32`.
    pub centre_y: f32,
}

/// One packed winner key per grid cell, from the candidate image.
pub(crate) fn launch_fast_cell_select<R: Runtime>(
    client: &ComputeClient<R>,
    kept: Buffer<'_>,
    best: Buffer<'_>,
    geometry: CellSelectGeometry,
) {
    super::super::submission::launch(client);
    unsafe {
        fast_cell_select_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(geometry.cells_x as u32, geometry.cells_y as u32, 1),
            CubeDim {
                x: SELECT_DIM_X as u32,
                y: SELECT_DIM_Y as u32,
                z: 1,
            },
            ArrayArg::from_raw_parts(kept.0.clone(), kept.1),
            ArrayArg::from_raw_parts(best.0.clone(), best.1),
            geometry.width,
            geometry.height,
            geometry.cell,
            geometry.x_start,
            geometry.y_start,
            geometry.cells_x,
            geometry.cells_y,
            geometry.threshold,
            geometry.safe_radius,
            geometry.centre_x,
            geometry.centre_y,
        );
    }
}
