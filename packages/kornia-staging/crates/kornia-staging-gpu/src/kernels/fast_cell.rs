//! Cell-local FAST-9 scoring, suppression and exact winner selection.
use super::{
    cell_select::{cell_key, min_tree, CellSelectGeometry, NO_WINNER},
    layout::Buffer,
};
use cubecl::prelude::*;

const UNITS: usize = 128;
const BORDER: usize = kornia_staging_imgproc::features::FAST_BORDER;
const EDGE: f32 = kornia_staging_imgproc::features::EDGE_THRESHOLD;

#[cube]
fn tile_pixel(tile: &Shared<[u32]>, slot: usize) -> u32 {
    (tile[slot / 4usize] >> u32::cast_from(8usize * (slot % 4usize))) & 255u32
}

/// Scores below the final threshold cannot suppress an admitted corner. Every
/// nine-point arc contains a point from each antipodal cardinal pair, so the
/// four-point test rejects only pixels whose score cannot clear the threshold.
#[cube]
fn score_tile(
    tile: &Shared<[u32]>,
    x: usize,
    y: usize,
    threshold: u32,
    #[comptime] cell: usize,
) -> u32 {
    let center = i32::cast_from(tile_pixel(tile, y * cell + x));
    let t = i32::cast_from(threshold);
    let right = i32::cast_from(tile_pixel(tile, y * cell + x + 3usize)) - center;
    let down = i32::cast_from(tile_pixel(tile, (y + 3usize) * cell + x)) - center;
    let left = i32::cast_from(tile_pixel(tile, y * cell + x - 3usize)) - center;
    let up = i32::cast_from(tile_pixel(tile, (y - 3usize) * cell + x)) - center;
    let bright = max(right, left) > t && max(down, up) > t;
    let dark = min(right, left) < -t && min(down, up) < -t;
    let mut score = 0u32;
    if bright || dark {
        let mut diff = Array::<i32>::new(16usize);
        #[unroll]
        for k in 0..16usize {
            let dy =
                comptime!(kornia_staging_imgproc::features::backend::FAST_RING_ROW[k] + 3) as usize;
            let dx = comptime!(kornia_staging_imgproc::features::backend::FAST_RING_COLUMN[k] + 3)
                as usize;
            diff[k] = i32::cast_from(tile_pixel(tile, (y + dy - 3usize) * cell + x + dx - 3usize))
                - center;
        }
        let mut low4 = Array::<i32>::new(16usize);
        let mut high4 = Array::<i32>::new(16usize);
        #[unroll]
        for k in 0..16usize {
            low4[k] = min(
                min(diff[k], diff[(k + 1usize) % 16usize]),
                min(diff[(k + 2usize) % 16usize], diff[(k + 3usize) % 16usize]),
            );
            high4[k] = max(
                max(diff[k], diff[(k + 1usize) % 16usize]),
                max(diff[(k + 2usize) % 16usize], diff[(k + 3usize) % 16usize]),
            );
        }
        let mut best = 0i32;
        #[unroll]
        for k in 0..16usize {
            let low9 = min(
                min(low4[k], low4[(k + 4usize) % 16usize]),
                diff[(k + 8usize) % 16usize],
            );
            let high9 = max(
                max(high4[k], high4[(k + 4usize) % 16usize]),
                diff[(k + 8usize) % 16usize],
            );
            best = max(best, max(low9, -high9));
        }
        if best > t {
            score = u32::cast_from(best);
        }
    }
    score
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn fast_cell_kernel(
    frame: &[u16],
    best: &mut [u32],
    width: usize,
    height: usize,
    x_start: usize,
    y_start: usize,
    cells_x: usize,
    threshold: u32,
    safe_radius: f32,
    centre_x: f32,
    centre_y: f32,
    frame_base: usize,
    frame_stride: usize,
    key_stride: usize,
    #[comptime] cell: usize,
) {
    let unit = usize::cast_from(UNIT_POS_X);
    let camera = usize::cast_from(CUBE_POS_Z);
    let frame_base = frame_base + camera * frame_stride;
    let column = usize::cast_from(CUBE_POS_X);
    let row = usize::cast_from(CUBE_POS_Y);
    let origin_x = x_start + column * cell;
    let origin_y = y_start + row * cell;
    let side = cell - 2usize * BORDER;
    // Pack four gray8 pixels per word without requiring 8-bit workgroup storage.
    // At cell=50 this cuts shared storage from 17.3 KiB to 10.0 KiB. The
    // benchmark checks whether lower storage pressure pays for unpacking.
    let mut tile = Shared::<[u32]>::new_slice((cell * cell).div_ceil(4).max(UNITS));
    let mut scores = Shared::<[u32]>::new_slice((cell - 2 * BORDER) * (cell - 2 * BORDER));
    for word in range_stepped(unit, (cell * cell).div_ceil(4), UNITS) {
        let mut packed = 0u32;
        #[unroll]
        for byte in 0..4usize {
            let i = word * 4usize + byte;
            let x = origin_x + i % cell;
            let y = origin_y + i / cell;
            if i < cell * cell && x < width && y < height {
                let value = u32::cast_from(frame[frame_base + y * width + x]) >> 8u32;
                packed |= value << u32::cast_from(8usize * byte);
            }
        }
        tile[word] = packed;
    }
    sync_cube();
    for i in range_stepped(unit, side * side, UNITS) {
        let x = i % side + BORDER;
        let y = i / side + BORDER;
        let mut value = 0u32;
        if origin_x + x + BORDER < width && origin_y + y + BORDER < height {
            value = score_tile(&tile, x, y, threshold, cell);
        }
        scores[i] = value;
    }
    sync_cube();
    let mut key = NO_WINNER.runtime();
    for i in range_stepped(unit, side * side, UNITS) {
        let value = scores[i];
        if value > 0u32 {
            let x = i % side;
            let y = i / side;
            let mut rival = 0u32;
            #[unroll]
            for dy in 0..3usize {
                #[unroll]
                for dx in 0..3usize {
                    if comptime!(dx != 1 || dy != 1) {
                        let nx = x + dx;
                        let ny = y + dy;
                        if nx > 0usize && nx <= side && ny > 0usize && ny <= side {
                            rival = max(rival, scores[(ny - 1usize) * side + nx - 1usize]);
                        }
                    }
                }
            }
            if value > rival {
                let px = origin_x + x + BORDER;
                let py = origin_y + y + BORDER;
                key = min(
                    key,
                    cell_key(
                        px,
                        py,
                        value,
                        f32::cast_from(width) - EDGE - 1.0f32,
                        f32::cast_from(height) - EDGE - 1.0f32,
                        safe_radius,
                        centre_x,
                        centre_y,
                    ),
                );
            }
        }
    }
    // The tile is dead now. Reuse it for the workgroup's exact integer minimum.
    sync_cube();
    tile[unit] = key;
    min_tree(&mut tile, unit, UNITS);
    if unit == 0usize {
        best[camera * key_stride + row * cells_x + column] = tile[0usize];
    }
}

/// Shared storage required by the fused detector. The launcher is selected only
/// for cells of 12..=64 pixels, with no CPU SIMD block filter.
pub(crate) fn cell_shared_bytes(cell: usize) -> usize {
    ((cell * cell).div_ceil(4).max(UNITS) + (cell - 2 * BORDER).pow(2)) * size_of::<u32>()
}

/// One dispatch replaces dense score, candidate copy, and cell selection.
pub(crate) fn launch_fast_cell<R: Runtime>(
    client: &ComputeClient<R>,
    frame: Buffer<'_>,
    best: Buffer<'_>,
    g: CellSelectGeometry,
) {
    launch_fast_cell_batch(client, frame, best, g, 0, 0, 0, 1);
}

/// Packed cell selection across contiguous cameras in a shared pyramid arena.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_fast_cell_batch<R: Runtime>(
    client: &ComputeClient<R>,
    frame: Buffer<'_>,
    best: Buffer<'_>,
    g: CellSelectGeometry,
    frame_base: usize,
    frame_stride: usize,
    key_stride: usize,
    cameras: usize,
) {
    // SAFETY: The scanner allocates one key per grid cell and camera, with these camera
    // strides. Each group owns one cell; pixel reads are clipped to the supplied image
    // geometry.
    unsafe {
        fast_cell_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(g.cells_x as u32, g.cells_y as u32, cameras as u32),
            CubeDim::new_1d(UNITS as u32),
            BufferArg::from_raw_parts(frame.0.clone(), frame.1),
            BufferArg::from_raw_parts(best.0.clone(), best.1),
            g.width,
            g.height,
            g.x_start,
            g.y_start,
            g.cells_x,
            g.threshold,
            g.safe_radius,
            g.centre_x,
            g.centre_y,
            frame_base,
            frame_stride,
            key_stride,
            g.cell,
        );
    }
}
