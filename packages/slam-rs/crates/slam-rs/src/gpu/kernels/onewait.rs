//! Exact cell order and bounded stereo input preparation on the device.
//!
//! Parameters: depth, survivor ratio, last detection count, point capacity,
//! reprojection flag, camera-zero device-camera parameters, then each
//! destination's quaternion xyzw, translation xyz and its device-camera parameters.
//! Every camera of a rig has the same model, a comptime tag (`camera::MODEL_*`),
//! so each model compiles its own kernel and the Brown8 one keeps its source.
//! Selected points: live count followed by interleaved x,y in column-major
//! cell order. Stereo slots retain the fused kernel's ten-float point layout.
use kornia_staging_gpu::{
    camera::{brown8, dispatch, parameter_count},
    optical_flow::{
        FUSED_RUNS, RUN_CAMERA, RUN_SOURCE_X, RUN_SOURCE_Y, RUN_TARGET_X, RUN_TARGET_Y, RUN_VALID,
        RUN_WARP,
    },
};
use kornia_staging_imgproc::features::backend::NO_CELL_WINNER;

const KEY_ROW_SHIFT: u32 = kornia_staging_imgproc::features::backend::KEY_ROW_SHIFT;
const KEY_FIELD_MASK: u32 = kornia_staging_imgproc::features::backend::KEY_FIELD_MASK;

pub(crate) const PARAM_HEADER: usize = 5;

/// Values per destination camera: pose then the model's parameters.
pub(crate) const fn per_camera_params(model: u32) -> usize {
    7 + parameter_count(model)
}

/// Offset of the first destination camera's record.
pub(crate) const fn camera_params_start(model: u32) -> usize {
    PARAM_HEADER + parameter_count(model)
}

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
    #[comptime] model: u32,
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
        if dispatch::unproject(params, PARAM_HEADER, sx, sy, &mut bearing, model) != brown8::VALID {
            terminate!();
        }
        let px = bearing[0usize] * params[0usize];
        let py = bearing[1usize] * params[0usize];
        let pz = bearing[2usize] * params[0usize];
        let c =
            comptime!(camera_params_start(model)) + camera * comptime!(per_camera_params(model));
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
        if dispatch::project(params, c + 7usize, vx, vy, vz, &mut pixel, model) != brown8::VALID {
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
    #![allow(clippy::unwrap_used)]

    use super::*;
    use kornia_staging_3d::camera::{
        CameraModelKind, DoubleSphere, Equirectangular, Eucm, KannalaBrandt4,
    };
    use kornia_staging_gpu::{
        camera::{DeviceCamera, MODEL_BROWN8},
        runtime::{GpuError, gpu_client},
        transfer::{read_buffers, upload},
    };

    /// Run `stereo_inputs` for `pixels` into `destinations` cameras; one record per pixel and camera.
    fn launch(
        model: u32,
        params: &[f32],
        pixels: &[[f32; 2]],
        destinations: usize,
    ) -> Result<Vec<f32>, GpuError> {
        let client = gpu_client()?;
        let cells = pixels.len();
        let count = cells * destinations;
        let mut selected = vec![cells as f32];
        selected.extend(pixels.as_flattened());
        let selected = upload(&client, f32::as_bytes(&selected))?;
        let params_handle = upload(&client, f32::as_bytes(params))?;
        let io = upload(&client, f32::as_bytes(&vec![0.0; FUSED_RUNS * count]))?;
        // SAFETY: `cells` selected points per destination and complete parameter/output records.
        unsafe {
            stereo_inputs::launch_unchecked::<kornia_staging_gpu::GpuRuntime>(
                &client,
                CubeCount::Static(count.div_ceil(64) as u32, 1, 1),
                CubeDim::new_1d(64),
                BufferArg::from_raw_parts(selected, 1 + 2 * cells),
                BufferArg::from_raw_parts(params_handle, params.len()),
                BufferArg::from_raw_parts(io.clone(), FUSED_RUNS * count),
                cells,
                count,
                model,
            );
        }
        let read = read_buffers(&client, vec![io], "stereo inputs")?;
        Ok(f32::from_bytes(&read[0]).to_vec())
    }

    /// Header, camera zero and one destination per `(quaternion xyzw, translation)`.
    fn pack(camera: &DeviceCamera, depth: f32, poses: &[([f32; 4], [f32; 3])]) -> Vec<f32> {
        let mut params = vec![depth, 0.0, 0.0, 0.0, 1.0];
        params.extend(camera.device_parameters());
        for (rotation, translation) in poses {
            params.extend(rotation);
            params.extend(translation);
            params.extend(camera.device_parameters());
        }
        assert_eq!(
            params.len(),
            camera_params_start(camera.model_id())
                + poses.len() * per_camera_params(camera.model_id())
        );
        params
    }

    #[test]
    fn stereo_inputs_reject_invalid_camera_rays_before_tracking() -> Result<(), GpuError> {
        let client = gpu_client()?;
        let start = camera_params_start(MODEL_BROWN8);
        for (depth, k1, pixel, source_radius, target_radius, valid) in [
            (0.001, 0.0, 0.0, 0.0, 0.0, false),
            (1.0, -1.0, 1.0, 0.0, 0.0, false),
            (1.0, 0.0, f32::NAN, 0.0, 0.0, false),
            (1.0, 0.0, 2.0, 1.0, 0.0, false),
            (1.0, 0.0, 2.0, 0.0, 1.0, false),
            (1.0, 0.0, 0.5, 1.0, 1.0, true),
        ] {
            let mut params = vec![0.0f32; start + per_camera_params(MODEL_BROWN8)];
            params[0] = depth;
            params[4] = 1.0;
            params[PARAM_HEADER] = 1.0;
            params[PARAM_HEADER + 1] = 1.0;
            params[PARAM_HEADER + 4] = k1;
            params[PARAM_HEADER + 12] = source_radius;
            params[start + 3] = 1.0;
            params[start + 7] = 1.0;
            params[start + 8] = 1.0;
            params[start + 19] = target_radius;
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
                    MODEL_BROWN8,
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

    /// Per model: a centre pixel maps to itself through the identity pose; a non-finite
    /// pixel, a pixel outside the source domain and a point pushed behind the target
    /// (outside its domain) are refused with complete, ordered records.
    #[test]
    fn stereo_inputs_accept_and_refuse_per_fisheye_model() -> Result<(), GpuError> {
        let models = [
            (
                CameraModelKind::Kb4(
                    KannalaBrandt4::new([
                        212.1, 211.6, 318.4, 174.8, 0.0617, -0.0211, 0.0372, -0.0135,
                    ])
                    .unwrap(),
                ),
                [900.0, 174.8],
            ),
            (
                CameraModelKind::DoubleSphere(
                    DoubleSphere::new([158.3, 158.3, 255.0, 256.9, -0.172, 0.593]).unwrap(),
                ),
                [900.0, 256.9],
            ),
            (
                CameraModelKind::Eucm(
                    Eucm::new([191.1, 191.1, 255.0, 256.9, 0.629, 1.042]).unwrap(),
                ),
                [900.0, 256.9],
            ),
            (
                CameraModelKind::Equirectangular(
                    Equirectangular::hemisphere_tile(1088, 960, 305.6, 102f32.to_radians())
                        .unwrap(),
                ),
                [1088.0, 479.5],
            ),
        ];
        for (cpu, outside) in models {
            let camera = DeviceCamera::from_model(&cpu).unwrap();
            let p = camera.device_parameters();
            let centre = [p[2] + 10.0, p[3] - 7.0];
            let pixels = [centre, [f32::NAN, p[3]], outside];
            let identity = ([0.0, 0.0, 0.0, 1.0], [0.0; 3]);
            let behind = ([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -5.0]);
            let values = launch(
                camera.model_id(),
                &pack(&camera, 1.0, &[identity, behind]),
                &pixels,
                2,
            )?;
            let valid: Vec<f32> = values
                .chunks(FUSED_RUNS)
                .map(|record| record[RUN_VALID])
                .collect();
            assert_eq!(valid, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "{cpu:?}");
            approx::assert_abs_diff_eq!(values[RUN_TARGET_X], centre[0], epsilon = 1e-3);
            approx::assert_abs_diff_eq!(values[RUN_TARGET_Y], centre[1], epsilon = 1e-3);
            for (index, record) in values.chunks(FUSED_RUNS).enumerate() {
                assert_eq!(
                    record[RUN_SOURCE_X].to_bits(),
                    pixels[index % 3][0].to_bits(),
                    "source order"
                );
                assert_eq!(record[RUN_CAMERA], (index / 3) as f32);
            }
        }
        Ok(())
    }

    /// RoboCap (4 x KB4 640x360): device stereo guesses equal the host's
    /// `project_between_cams` with the same refusals: 95 % within 1e-3 px, all within 1e-2 px.
    /// The excess sits at the cam-0 image corners near KB4's `r_max`, where the radial slope
    /// tends to zero and f32 bearings differ by up to 5e-3 px (kornia-staging camera parity).
    #[test]
    fn kb4_rig_stereo_guesses_match_the_host() -> Result<(), GpuError> {
        use crate::calib::Calibration;
        use crate::camera::RigCamera;
        use crate::frontend::flow::project_between_cams;
        let calib = Calibration::<f64>::from_json_str(include_str!(
            "../../../../../configs/robocap_calib_downscale3.json"
        ))
        .unwrap()
        .cast::<f32>();
        let cameras = RigCamera::from_calibration(&calib).unwrap();
        let devices: Vec<_> = cameras
            .iter()
            .map(|camera| DeviceCamera::from_model(&camera.model.inner).unwrap())
            .collect();
        let mut pixels = Vec::new();
        for row in 0..12 {
            for column in 0..20 {
                pixels.push([column as f32 * 33.3 + 3.0, row as f32 * 31.1 + 2.0]);
            }
        }
        for depth in [0.5f32, 2.0] {
            let mut params = vec![depth, 0.0, 0.0, 0.0, 1.0];
            params.extend(devices[0].device_parameters());
            for (t_i_c, device) in calib.t_i_c.iter().zip(&devices).skip(1) {
                let transform = (calib.t_i_c[0].inverse() * *t_i_c).inverse();
                params.extend(transform.rotation.quaternion_xyzw());
                params.extend(transform.translation.iter());
                params.extend(device.device_parameters());
            }
            let values = launch(devices[0].model_id(), &params, &pixels, cameras.len() - 1)?;
            let (mut worst, mut within, mut accepted) = (0f32, 0, 0);
            for camera in 1..cameras.len() {
                let t_c0_ci = calib.t_i_c[0].inverse() * calib.t_i_c[camera];
                for (index, pixel) in pixels.iter().enumerate() {
                    let record = &values[FUSED_RUNS * ((camera - 1) * pixels.len() + index)..];
                    let (valid, guess) = project_between_cams(
                        &cameras,
                        &nalgebra::Vector2::from(*pixel),
                        depth,
                        &t_c0_ci,
                        0,
                        camera,
                    );
                    assert_eq!(
                        record[RUN_VALID] == 1.0,
                        valid,
                        "camera {camera}, pixel {pixel:?}"
                    );
                    if valid {
                        accepted += 1;
                        let error = (record[RUN_TARGET_X] - guess.x)
                            .abs()
                            .max((record[RUN_TARGET_Y] - guess.y).abs());
                        worst = worst.max(error);
                        within += usize::from(error <= 1e-3);
                        assert!(
                            error <= 1e-2,
                            "camera {camera}, pixel {pixel:?}: {error} px"
                        );
                    }
                }
            }
            println!(
                "RoboCap KB4 depth {depth} m: {accepted} guesses, {within} within 1e-3 px, worst {worst:e} px"
            );
            assert!(
                accepted > 0 && within * 100 >= accepted * 95,
                "{within} of {accepted} within 1e-3 px"
            );
        }
        Ok(())
    }
}
