//! Exact cell order and bounded stereo input preparation on the device.
//!
//! Parameters: depth, survivor ratio, last detection count, point capacity,
//! reprojection flag, camera-zero radtan8 intrinsics (12 values), then each
//! destination's quaternion xyzw, translation xyz and 12 intrinsics.
//! Selected points: live count followed by interleaved x,y in column-major
//! cell order. Stereo slots retain the fused kernel's ten-float point layout.
use super::klt_fused::FUSED_RUNS;
use kornia_staging_imgproc::features::backend::NO_CELL_WINNER;

const KEY_ROW_SHIFT: u32 = kornia_staging_imgproc::features::backend::KEY_ROW_SHIFT;
const KEY_FIELD_MASK: u32 = kornia_staging_imgproc::features::backend::KEY_FIELD_MASK;

pub(crate) const PARAM_HEADER: usize = 5;
pub(crate) const RADTAN8_PARAMS: usize = 12;
pub(crate) const PER_CAMERA_PARAMS: usize = 7 + RADTAN8_PARAMS;
pub(crate) const CAMERA_PARAMS_START: usize = PARAM_HEADER + RADTAN8_PARAMS;

use crate::gpu::finite::is_finite;
use cubecl::prelude::*;

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn occupancy(
    temporal: &[f32],
    occupied: &mut [u32],
    count: usize,
    columns: usize,
    rows: usize,
    x_start: usize,
    y_start: usize,
    cell: usize,
) {
    let index = usize::cast_from(ABSOLUTE_POS);
    if index >= columns * rows {
        terminate!();
    }
    let column = index % columns;
    let row = index / columns;
    let mut survivors = 0u32;
    let mut present = 0u32;
    for point in 0..count {
        let base = point * FUSED_RUNS;
        if temporal[base + 6usize] != 0.0f32 {
            survivors += 1u32;
            let x = temporal[base + 4usize];
            let y = temporal[base + 5usize];
            if x >= f32::cast_from(x_start)
                && y >= f32::cast_from(y_start)
                && x < f32::cast_from(x_start + columns * cell)
                && y < f32::cast_from(y_start + rows * cell)
            {
                let px = usize::cast_from((x - f32::cast_from(x_start)) / f32::cast_from(cell));
                let py = usize::cast_from((y - f32::cast_from(y_start)) / f32::cast_from(cell));
                if px == column && py == row {
                    present = 1u32;
                }
            }
        }
    }
    occupied[index + 1usize] = present;
    if index == 0usize {
        occupied[0usize] = survivors;
    }
}

#[cube(launch_unchecked)]
pub(crate) fn select(
    occupied: &[u32],
    keys: &[u32],
    params: &[f32],
    selected: &mut [f32],
    columns: usize,
    rows: usize,
) {
    let survivors = occupied[0usize];
    selected[0usize] = 0.0f32;
    let ratio = params[1usize];
    if is_finite(ratio)
        && ratio > 0.0f32
        && params[2usize] != 0.0f32
        && f32::cast_from(survivors) >= ratio * params[2usize]
    {
        terminate!();
    }
    let budget = usize::cast_from(params[3usize]) - usize::cast_from(survivors);
    let mut added = 0usize;
    for column in 0..columns {
        for row in 0..rows {
            if added < budget && occupied[1usize + row * columns + column] == 0u32 {
                let key = keys[row * columns + column];
                if key != NO_CELL_WINNER {
                    selected[1usize + added * 2usize] = f32::cast_from(key & KEY_FIELD_MASK);
                    selected[2usize + added * 2usize] =
                        f32::cast_from((key >> KEY_ROW_SHIFT) & KEY_FIELD_MASK);
                    added += 1usize;
                }
            }
        }
    }
    selected[0usize] = f32::cast_from(added);
}

#[cube]
fn distort(
    params: &[f32],
    base: usize,
    x: f32,
    y: f32,
    result: &mut Array<f32>,
    #[comptime] jacobian: bool,
) {
    let k1 = params[base + 4usize];
    let k2 = params[base + 5usize];
    let p1 = params[base + 6usize];
    let p2 = params[base + 7usize];
    let k3 = params[base + 8usize];
    let k4 = params[base + 9usize];
    let k5 = params[base + 10usize];
    let k6 = params[base + 11usize];
    let rp2 = x * x + y * y;
    let cdist = (1.0f32 + rp2 * (k1 + rp2 * (k2 + rp2 * k3)))
        / (1.0f32 + rp2 * (k4 + rp2 * (k5 + rp2 * k6)));
    let dx = 2.0f32 * p1 * x * y + p2 * (rp2 + 2.0f32 * x * x);
    let dy = 2.0f32 * p2 * x * y + p1 * (rp2 + 2.0f32 * y * y);
    result[0usize] = x * cdist + dx;
    result[1usize] = y * cdist + dy;
    if jacobian {
        let v0 = x * x;
        let v1 = y * y;
        let v2 = v0 + v1;
        let v3 = k6 * v2;
        let v4 = k4 + v2 * (k5 + v3);
        let v5 = v2 * v4 + 1.0f32;
        let v6 = v5 * v5;
        let v7 = 1.0f32 / v6;
        let v8 = p1 * y;
        let v9 = p2 * x;
        let v10 = 2.0f32 * v6;
        let v11 = k3 * v2;
        let v12 = k1 + v2 * (k2 + v11);
        let v13 = v12 * v2 + 1.0f32;
        let v14 = v13 * (v2 * (k5 + 2.0f32 * v3) + v4);
        let v15 = 2.0f32 * v14;
        let v16 = v12 + v2 * (k2 + 2.0f32 * v11);
        let v17 = 2.0f32 * v16;
        let v18 = x * y;
        let v19 = 2.0f32 * v7 * (-v14 * v18 + v16 * v18 * v5 + v6 * (p1 * x + p2 * y));
        result[2usize] = v7 * (-v0 * v15 + v10 * (v8 + 3.0f32 * v9) + v5 * (v0 * v17 + v13));
        result[3usize] = v19;
        result[4usize] = v7 * (-v1 * v15 + v10 * (3.0f32 * v8 + v9) + v5 * (v1 * v17 + v13));
    }
}

#[cube(launch_unchecked)]
pub(crate) fn stereo_inputs(
    selected: &[f32],
    params: &[f32],
    io: &mut [f32],
    cells: usize,
    capacity: usize,
) {
    let point = usize::cast_from(ABSOLUTE_POS);
    if point >= capacity {
        terminate!();
    }
    let camera = point / cells;
    let index = point % cells;
    let base = point * FUSED_RUNS;
    io[base + 6usize] = 0.0f32;
    io[base + 9usize] = f32::cast_from(camera);
    if index >= usize::cast_from(selected[0usize]) {
        terminate!();
    }
    let sx = selected[1usize + 2usize * index];
    let sy = selected[2usize + 2usize * index];
    let mut guess_x = sx;
    let mut guess_y = sy;
    if params[4usize] != 0.0f32 {
        let mx = (sx - params[PARAM_HEADER + 2usize]) / params[PARAM_HEADER];
        let my = (sy - params[PARAM_HEADER + 3usize]) / params[PARAM_HEADER + 1usize];
        let mut x = mx;
        let mut y = my;
        let mut d = Array::<f32>::new(5usize);
        for _iteration in 0..5usize {
            distort(params, PARAM_HEADER, x, y, &mut d, true);
            let rx = d[0usize] - mx;
            let ry = d[1usize] - my;
            let det = d[2usize] * d[4usize] - d[3usize] * d[3usize];
            let inv = 1.0f32 / det;
            let ix = d[4usize] * inv * rx + -d[3usize] * inv * ry;
            let iy = -d[3usize] * inv * rx + d[2usize] * inv * ry;
            x -= ix;
            y -= iy;
            if f32::sqrt(rx * rx + ry * ry) < 0.0031622776f32 {
                break;
            }
        }
        let norm_inv = 1.0f32 / f32::sqrt(x * x + y * y + 1.0f32);
        let px = x * norm_inv * params[0usize];
        let py = y * norm_inv * params[0usize];
        let pz = norm_inv * params[0usize];
        let c = CAMERA_PARAMS_START + camera * PER_CAMERA_PARAMS;
        let qx = params[c];
        let qy = params[c + 1usize];
        let qz = params[c + 2usize];
        let qw = params[c + 3usize];
        let b2 = (qx * qx + qy * qy) + qz * qz;
        let dot = (px * qx + py * qy) + pz * qz;
        let a = qw * qw - b2;
        let b = dot * 2.0f32;
        let w = qw * 2.0f32;
        let vx = (px * a + qx * b) + (qy * pz - qz * py) * w + params[c + 4usize];
        let vy = (py * a + qy * b) + (qz * px - qx * pz) * w + params[c + 5usize];
        let vz = (pz * a + qz * b) + (qx * py - qy * px) * w + params[c + 6usize];
        distort(params, c + 7usize, vx / vz, vy / vz, &mut d, false);
        guess_x = params[c + 7usize] * d[0usize] + params[c + 9usize];
        guess_y = params[c + 8usize] * d[1usize] + params[c + 10usize];
    }
    io[base] = 1.0f32;
    io[base + 1usize] = 0.0f32;
    io[base + 2usize] = 0.0f32;
    io[base + 3usize] = 1.0f32;
    io[base + 4usize] = guess_x;
    io[base + 5usize] = guess_y;
    io[base + 6usize] = 1.0f32;
    io[base + 7usize] = sx;
    io[base + 8usize] = sy;
}
