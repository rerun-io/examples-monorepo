//! Device selection connects temporal KLT to stereo without an intermediate read.

use crate::camera::SlamCamera;
use crate::gpu::kernels::onewait::{camera_params_start, per_camera_params};
use cubecl::prelude::*;

use super::GpuStages;
use crate::config::MatchingGuessType;
use crate::frontend::flow::FrontendError;
use crate::frontend::stages::StereoContext;
use crate::gpu::{kernels, pyramid::GpuPyramid, submission};
use kornia_staging_gpu::camera::{DeviceCamera, MODEL_BROWN8};
use kornia_staging_gpu::optical_flow::{
    FUSED_RUNS, FusedKltPlan, FusedLaunch, RUN_SOURCE_X, RUN_SOURCE_Y,
};
use kornia_staging_gpu::runtime::GpuError;
use kornia_staging_gpu::runtime::guarded;
use kornia_staging_imgproc::features::{CellGrid, CellSelect};
use kornia_staging_imgproc::optical_flow::patch_se2::Pattern;
use kornia_staging_slam::tracking::optical_flow::{PatchTracker, TrackPhase};

pub(super) enum Phase {
    Off,
    Prepared(Box<FusedLaunch>),
    Submitted,
    Ready(cubecl::bytes::Bytes),
}

pub(super) struct OneWait<P: Pattern, R: Runtime> {
    params: cubecl::server::Handle,
    selected: cubecl::server::Handle,
    occupied: cubecl::server::Handle,
    pub(super) io: cubecl::server::Handle,
    plan: FusedKltPlan<P, R>,
    grid: CellGrid,
    cameras: usize,
    camera_models: Vec<SlamCamera<f32>>,
    device_cameras: Vec<DeviceCamera>,
    /// Comptime model tag shared by every camera of the rig.
    model: u32,
    cells: usize,
    pub(super) phase: Phase,
}

/// The rig's shared device-camera model tag, or `None` when a camera needs the
/// general CPU path or the rig mixes models (each model is its own kernel).
pub(crate) fn rig_device_model(cameras: &[crate::camera::RigCamera<f32>]) -> Option<u32> {
    let mut models = cameras.iter().map(|camera| {
        DeviceCamera::from_model(&camera.model.inner).map(|device| device.model_id())
    });
    let first = models.next()??;
    models.all(|model| model == Some(first)).then_some(first)
}

impl<P: Pattern, R: Runtime> GpuStages<P, R> {
    /// Enable the single-read path only for the supported, unmasked shared rig.
    pub(super) fn prepare_one_wait(
        &mut self,
        context: StereoContext<'_>,
        selects: &[Option<CellSelect>],
    ) -> Result<bool, FrontendError> {
        let StereoContext {
            cameras,
            calib,
            config,
            depth,
            last_detect_count,
            eligible,
        } = context;
        let pyramids = &self.current.pyramids;

        if let Some(state) = &mut self.one_wait {
            state.phase = Phase::Off;
        }
        let Some(select) = selects.first().copied().flatten() else {
            return Ok(false);
        };
        if !eligible
            || cameras.len() < 2
            || !config.optical_flow_detection_nonoverlap
            || !selects.iter().all(|other| *other == Some(select))
        {
            return Ok(false);
        }
        let Some(model) = rig_device_model(cameras) else {
            return Ok(false);
        };
        if model != MODEL_BROWN8 && !config.optical_flow_gpu_one_wait_any_model {
            return Ok(false);
        }
        let Some(arena) = pyramids.first().and_then(GpuPyramid::arena) else {
            return Ok(false);
        };
        if !pyramids.iter().all(|pyramid| {
            pyramid
                .arena()
                .is_some_and(|other| std::sync::Arc::ptr_eq(arena, other))
        }) {
            return Ok(false);
        }
        let (columns, rows) = select.grid.dimensions();
        let cells = columns * rows;
        let lanes = cameras.len() - 1;
        let fits = self.one_wait.as_ref().is_some_and(|state| {
            state.grid == select.grid
                && state.cameras == cameras.len()
                && state
                    .camera_models
                    .iter()
                    .zip(cameras)
                    .all(|(a, b)| *a == b.model)
        });
        let state = match &mut self.one_wait {
            Some(existing) if fits => existing,
            slot => {
                let plan = self.tracker.plan_like(cells, lanes)?;
                let io = plan.result_handle(cells * lanes)?;
                let device_cameras = cameras
                    .iter()
                    .map(|camera| {
                        DeviceCamera::from_model(&camera.model.inner)
                            .ok_or(crate::camera::CameraError::InvalidCalibration)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                slot.insert(guarded(
                    GpuError::DeviceLost {
                        what: "stereo allocation",
                    },
                    || {
                        Ok::<_, FrontendError>(OneWait {
                            params: self.client.empty(
                                (camera_params_start(model) + lanes * per_camera_params(model))
                                    * size_of::<f32>(),
                            ),
                            selected: self.client.empty((1 + cells * 2) * size_of::<f32>()),
                            occupied: self.client.empty((1 + cells) * size_of::<u32>()),
                            io,
                            plan,
                            grid: select.grid,
                            cameras: cameras.len(),
                            camera_models: cameras.iter().map(|camera| camera.model).collect(),
                            device_cameras,
                            model,
                            cells,
                            phase: Phase::Off,
                        })
                    },
                )?)
            }
        };
        let mut params = vec![
            depth,
            config.port_redetect_survivor_ratio,
            last_detect_count as f32,
            self.tracker.capacity() as f32,
            if config.optical_flow_matching_guess_type != MatchingGuessType::SamePixel {
                1.0
            } else {
                0.0
            },
        ];
        params.extend(state.device_cameras[0].device_parameters());
        let mut pairs = Vec::with_capacity(lanes);
        for camera in 1..cameras.len() {
            // Match the host's two inversions and quaternion action exactly.
            let transform = (calib.t_i_c[0].inverse() * calib.t_i_c[camera]).inverse();
            params.extend(transform.rotation.quaternion_xyzw());
            params.extend(transform.translation.iter());
            params.extend(state.device_cameras[camera].device_parameters());
            pairs.push((&pyramids[0], &pyramids[camera]));
        }

        guarded(
            GpuError::DeviceLost {
                what: "stereo parameters",
            },
            || {
                self.client
                    .write(&state.params, cubecl::bytes::Bytes::from_elems(params));
                Ok::<_, FrontendError>(())
            },
        )?;
        state.plan.set_exit_step_px(self.tracker.exit_step_px())?;
        // SAFETY: stereo_inputs writes complete records before this launch on the same stream.
        let launch = unsafe {
            state
                .plan
                .prepare_device(&pairs, (arena, arena), cells * lanes)?
        };
        state.io = state.plan.result_handle(cells * lanes)?;
        state.phase = Phase::Prepared(Box::new(launch));
        Ok(true)
    }

    pub(super) fn submit_stereo(&mut self) -> Result<(), FrontendError> {
        let Some(state) = self
            .one_wait
            .as_mut()
            .filter(|state| matches!(state.phase, Phase::Prepared(_)))
        else {
            return Ok(());
        };
        let (temporal_io, temporal_len, temporal_count) = self.tracker.packed_result();
        let Some(keys) = self
            .current
            .detector
            .scanner_mut()
            .inner
            .staged_handles()
            .filter(|keys| keys.len() == 1)
        else {
            // An unbatchable detector uses the normal host selection and stereo pass.
            state.phase = Phase::Off;
            return Ok(());
        };
        let klt = match std::mem::replace(&mut state.phase, Phase::Submitted) {
            Phase::Prepared(klt) => klt,
            phase => {
                state.phase = phase;
                return Ok(());
            }
        };
        self.launches.dispatch(
            &self.client,
            submission::Launch::Stereo(StereoLaunch {
                stereo_params: state.params.clone(),
                selected: state.selected.clone(),
                occupied: state.occupied.clone(),
                io: state.io.clone(),
                klt,
                grid: state.grid,
                cameras: state.cameras,
                model: state.model,
                cells: state.cells,
                temporal_io: temporal_io.clone(),
                temporal_len: FUSED_RUNS * temporal_len.max(1),
                temporal_count,
                keys: keys[0].clone(),
            }),
        )?;
        Ok(())
    }

    pub(super) fn take_stereo(
        &mut self,
        phase: TrackPhase<'_>,
        slots: &mut [usize],
    ) -> Result<bool, FrontendError> {
        use crate::pyramid::Pyramid;
        let inputs = phase.validate(
            self.tracker.capacity(),
            self.tracker.num_levels(),
            slots.len(),
            None,
            |i| self.current.pyramids.get(i).map(Pyramid::num_levels),
            |i| self.current.pyramids.get(i).map(Pyramid::num_levels),
        )?;
        let Some(state) = &mut self.one_wait else {
            return Ok(false);
        };
        let bytes = match std::mem::replace(&mut state.phase, Phase::Off) {
            Phase::Ready(bytes) => bytes,
            phase => {
                state.phase = phase;
                return Ok(false);
            }
        };
        if bytes.len() != FUSED_RUNS * state.cells * (state.cameras - 1) * size_of::<f32>() {
            return Err(GpuError::DeviceReadFailed {
                what: "one-wait stereo result length",
            }
            .into());
        }
        let values = f32::from_bytes(&bytes);
        for (lane, input) in inputs.iter().enumerate() {
            let count = input.positions.len();
            if count > state.cells {
                return Err(GpuError::DeviceReadFailed {
                    what: "one-wait corner count",
                }
                .into());
            }
            slots[lane] = lane;
            for index in 0..count {
                let base = FUSED_RUNS * (lane * state.cells + index);
                let source = input.positions.get(index);
                if values[base + RUN_SOURCE_X] != source[0]
                    || values[base + RUN_SOURCE_Y] != source[1]
                {
                    return Err(GpuError::DeviceReadFailed {
                        what: "one-wait corner order",
                    }
                    .into());
                }
            }
            let start = FUSED_RUNS * lane * state.cells * size_of::<f32>();
            let end = start + FUSED_RUNS * count * size_of::<f32>();
            self.tracker.publish_lane(lane, count, &bytes[start..end])?;
        }
        Ok(true)
    }
}

pub(in crate::gpu) struct StereoLaunch {
    stereo_params: cubecl::server::Handle,
    selected: cubecl::server::Handle,
    occupied: cubecl::server::Handle,
    io: cubecl::server::Handle,
    klt: Box<FusedLaunch>,
    grid: CellGrid,
    cameras: usize,
    model: u32,
    cells: usize,
    temporal_io: cubecl::server::Handle,
    temporal_len: usize,
    temporal_count: usize,
    keys: cubecl::server::Handle,
}
impl StereoLaunch {
    pub(in crate::gpu) fn run<R: Runtime>(self, client: &ComputeClient<R>) {
        let Self {
            stereo_params,
            selected,
            occupied,
            io,
            klt,
            grid,
            cameras,
            model,
            cells,
            temporal_io,
            temporal_len,
            temporal_count,
            keys,
        } = self;
        let (columns, rows) = grid.dimensions();

        // SAFETY: Packed temporal storage has FUSED_RUNS coefficients per point;
        // occupancy has a count plus one entry per cell. The kernel bounds the
        // rounded-up launch.
        unsafe {
            kernels::onewait::occupancy::launch_unchecked::<R>(
                client,
                CubeCount::Static(cells.div_ceil(64) as u32, 1, 1),
                CubeDim::new_1d(64),
                BufferArg::from_raw_parts(temporal_io, temporal_len),
                BufferArg::from_raw_parts(occupied.clone(), 1 + cells),
                temporal_count,
                columns,
                rows,
                grid.x_start,
                grid.y_start,
                grid.cell,
            );
        }

        // SAFETY: The scanner supplied one packed key buffer. Params and selected
        // storage were allocated for this grid and camera count; selection runs as
        // one unit.
        unsafe {
            kernels::onewait::select::launch_unchecked::<R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(occupied.clone(), 1 + cells),
                BufferArg::from_raw_parts(
                    keys.clone(),
                    keys.size_in_used() as usize / size_of::<u32>(),
                ),
                BufferArg::from_raw_parts(
                    stereo_params.clone(),
                    camera_params_start(model) + per_camera_params(model) * (cameras - 1),
                ),
                BufferArg::from_raw_parts(selected.clone(), 1 + cells * 2),
                columns,
                rows,
            );
        }
        let count = cells * (cameras - 1);

        // SAFETY: There is one stereo candidate per cell and non-host camera.
        // Selected/params use the same grid, and io holds FUSED_RUNS values per
        // candidate. Extra launch units check count.
        unsafe {
            kernels::onewait::stereo_inputs::launch_unchecked::<R>(
                client,
                CubeCount::Static(count.div_ceil(64) as u32, 1, 1),
                CubeDim::new_1d(64),
                BufferArg::from_raw_parts(selected.clone(), 1 + cells * 2),
                BufferArg::from_raw_parts(
                    stereo_params.clone(),
                    camera_params_start(model) + per_camera_params(model) * (cameras - 1),
                ),
                BufferArg::from_raw_parts(io.clone(), FUSED_RUNS * count),
                cells,
                count,
                model,
            );
        }
        // SAFETY: The prepared pass follows stereo_inputs on the same client stream.
        unsafe {
            (*klt).run(client);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::rig_device_model;
    use crate::calib::Calibration;
    use crate::camera::RigCamera;
    use kornia_staging_gpu::camera::{MODEL_BROWN8, MODEL_KB4};

    fn rig(text: &str) -> Vec<RigCamera<f32>> {
        RigCamera::from_calibration(
            &Calibration::<f64>::from_json_str(text)
                .unwrap()
                .cast::<f32>(),
        )
        .unwrap()
    }

    #[test]
    fn one_wait_takes_any_single_model_device_rig() {
        let msd = rig(include_str!("../../../tests/fixtures/msdmg_calib.json"));
        let robocap = rig(include_str!(
            "../../../../../configs/robocap_calib_downscale3.json"
        ));
        assert_eq!(rig_device_model(&msd), Some(MODEL_BROWN8));
        assert_eq!(rig_device_model(&robocap), Some(MODEL_KB4));
        // Mixed models compile to different kernels: the rig stays on the host path.
        let mixed = [msd[0], robocap[1]];
        assert_eq!(rig_device_model(&mixed), None);
        assert_eq!(rig_device_model(&[]), None);
        // A camera without a device model (Brown with prism terms) keeps the rig off.
        let mut parameters = [0.0f32; 18];
        parameters[..5].copy_from_slice(&[400.0, 400.0, 320.0, 240.0, 0.01]);
        parameters[12] = 1e-4;
        let prism = kornia_staging_3d::camera::BrownConrady::new(parameters, None).unwrap();
        let mut general = msd.clone();
        general[1].model.inner = kornia_staging_3d::camera::CameraModelKind::BrownConrady(prism);
        assert_eq!(rig_device_model(&general), None);
    }
}
