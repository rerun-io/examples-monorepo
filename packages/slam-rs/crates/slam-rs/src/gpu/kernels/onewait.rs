//! Exact cell order and bounded stereo input preparation on the device.
//!
//! Parameters: depth, survivor ratio, last detection count, point capacity,
//! reprojection flag, camera-zero Brown8 parameters (13 values), then each
//! destination's quaternion xyzw, translation xyz and 13 Brown8 parameters.
//! Selected points: live count followed by interleaved x,y in column-major
//! cell order. Stereo slots retain the fused kernel's ten-float point layout.
use kornia_staging_gpu::{
    camera::{BROWN8_PARAMETERS, brown8},
    optical_flow::{
        FUSED_RUNS, RUN_CAMERA, RUN_SOURCE_X, RUN_SOURCE_Y, RUN_TARGET_X, RUN_TARGET_Y, RUN_VALID,
        RUN_WARP,
    },
};
use kornia_staging_imgproc::features::backend::NO_CELL_WINNER;

const KEY_ROW_SHIFT: u32 = kornia_staging_imgproc::features::backend::KEY_ROW_SHIFT;
const KEY_FIELD_MASK: u32 = kornia_staging_imgproc::features::backend::KEY_FIELD_MASK;

pub(crate) const PARAM_HEADER: usize = 5;
pub(crate) const PER_CAMERA_PARAMS: usize = 7 + BROWN8_PARAMETERS;
pub(crate) const CAMERA_PARAMS_START: usize = PARAM_HEADER + BROWN8_PARAMETERS;

use cubecl::prelude::*;
use kornia_staging_gpu::kernels::finite::is_finite;

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
        if temporal[base + RUN_VALID] != 0.0f32 {
            survivors += 1u32;
            let x = temporal[base + RUN_TARGET_X];
            let y = temporal[base + RUN_TARGET_Y];
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
    io[base + RUN_VALID] = 0.0f32;
    io[base + RUN_CAMERA] = f32::cast_from(camera);
    if index >= usize::cast_from(selected[0usize]) {
        terminate!();
    }
    let sx = selected[1usize + 2usize * index];
    let sy = selected[2usize + 2usize * index];
    // Source coordinates identify every record, including rejected candidates.
    io[base + RUN_SOURCE_X] = sx;
    io[base + RUN_SOURCE_Y] = sy;
    let mut guess_x = sx;
    let mut guess_y = sy;
    if params[4usize] != 0.0f32 {
        let mut bearing = Array::<f32>::new(3usize);
        if brown8::unproject(params, PARAM_HEADER, sx, sy, &mut bearing) != brown8::VALID {
            terminate!();
        }
        let px = bearing[0usize] * params[0usize];
        let py = bearing[1usize] * params[0usize];
        let pz = bearing[2usize] * params[0usize];
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
        let mut pixel = Array::<f32>::new(2usize);
        if brown8::project(params, c + 7usize, vx, vy, vz, &mut pixel) != brown8::VALID {
            terminate!();
        }
        guess_x = pixel[0usize];
        guess_y = pixel[1usize];
    }
    io[base + RUN_WARP] = 1.0f32;
    io[base + RUN_WARP + 1usize] = 0.0f32;
    io[base + RUN_WARP + 2usize] = 0.0f32;
    io[base + RUN_WARP + 3usize] = 1.0f32;
    io[base + RUN_TARGET_X] = guess_x;
    io[base + RUN_TARGET_Y] = guess_y;
    io[base + RUN_VALID] = 1.0f32;
}

#[cfg(all(test, feature = "gpu-wgpu"))]
mod tests {
    use super::*;
    use kornia_staging_gpu::{
        runtime::{GpuError, gpu_client},
        transfer::{read_buffers, upload},
    };

    #[test]
    fn stereo_inputs_reject_invalid_camera_rays_before_tracking() -> Result<(), GpuError> {
        let client = gpu_client()?;
        for (depth, k1, pixel, source_radius, target_radius, valid) in [
            (0.001, 0.0, 0.0, 0.0, 0.0, false),
            (1.0, -1.0, 1.0, 0.0, 0.0, false),
            (1.0, 0.0, f32::NAN, 0.0, 0.0, false),
            (1.0, 0.0, 2.0, 1.0, 0.0, false),
            (1.0, 0.0, 2.0, 0.0, 1.0, false),
            (1.0, 0.0, 0.5, 1.0, 1.0, true),
        ] {
            let mut params = vec![0.0f32; CAMERA_PARAMS_START + PER_CAMERA_PARAMS];
            params[0] = depth;
            params[4] = 1.0;
            params[PARAM_HEADER] = 1.0;
            params[PARAM_HEADER + 1] = 1.0;
            params[PARAM_HEADER + 4] = k1;
            params[PARAM_HEADER + 12] = source_radius;
            params[CAMERA_PARAMS_START + 3] = 1.0;
            params[CAMERA_PARAMS_START + 7] = 1.0;
            params[CAMERA_PARAMS_START + 8] = 1.0;
            params[CAMERA_PARAMS_START + 19] = target_radius;
            let selected = upload(&client, f32::as_bytes(&[1.0, pixel, 0.0]))?;
            let params_handle = upload(&client, f32::as_bytes(&params))?;
            let io = upload(&client, f32::as_bytes(&[0.0; FUSED_RUNS]))?;
            // SAFETY: One selected point, one camera and complete parameter/output records.
            unsafe {
                stereo_inputs::launch_unchecked::<kornia_staging_gpu::GpuRuntime>(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(1),
                    BufferArg::from_raw_parts(selected, 3),
                    BufferArg::from_raw_parts(params_handle, params.len()),
                    BufferArg::from_raw_parts(io.clone(), FUSED_RUNS),
                    1,
                    1,
                );
            }
            let read = read_buffers(&client, vec![io], "stereo refusal")?;
            let values = f32::from_bytes(&read[0]);
            assert_eq!(
                values[6],
                if valid { 1.0 } else { 0.0 },
                "depth={depth}, k1={k1}, pixel={pixel}"
            );
            if valid {
                approx::assert_abs_diff_eq!(values[4], pixel, epsilon = 1e-3);
            }
            assert!(values[..7].iter().all(|value| value.is_finite()));
            assert_eq!(
                values[7].to_bits(),
                pixel.to_bits(),
                "refused records must preserve source order"
            );
            assert_eq!(values[8], 0.0);
        }
        Ok(())
    }
}
