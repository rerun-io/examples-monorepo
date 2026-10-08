//! Device selection connects temporal KLT to stereo without an intermediate read.

use crate::camera::SlamCamera;
use crate::gpu::kernels::onewait::{CAMERA_PARAMS_START, PER_CAMERA_PARAMS};
use cubecl::prelude::*;

use super::GpuStages;
use crate::config::MatchingGuessType;
use crate::frontend::flow::FrontendError;
use crate::frontend::stages::StereoContext;
use crate::gpu::{kernels, pyramid::GpuPyramid, submission};
use kornia_staging_3d::camera::CameraModelKind;
use kornia_staging_gpu::camera::Brown8;
use kornia_staging_gpu::optical_flow::{
    FUSED_RUNS, FusedKltPlan, FusedLaunch, RUN_SOURCE_X, RUN_SOURCE_Y,
};
use kornia_staging_gpu::runtime::GpuError;
use kornia_staging_gpu::runtime::guarded;
use kornia_staging_imgproc::features::{CellGrid, CellSelect};
use kornia_staging_imgproc::optical_flow::patch_se2::Pattern;
use kornia_staging_slam::tracking::optical_flow::{TrackPhase, PatchTracker};

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
    device_cameras: Vec<Brown8>,
    cells: usize,
    pub(super) phase: Phase,
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
            || !cameras
                .iter()
                .all(|camera| matches!(&camera.model.inner, CameraModelKind::BrownConrady(model) if Brown8::from_brown_conrady(model).is_some()))
        {
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
                        let CameraModelKind::BrownConrady(model) = &camera.model.inner else { unreachable!("Brown8 eligibility checked"); };
                        Brown8::from_brown_conrady(model).ok_or(crate::camera::CameraError::InvalidCalibration)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                slot.insert(guarded(
                    GpuError::DeviceLost {
                        what: "stereo allocation",
                    },
                    || {
                        Ok::<_, FrontendError>(OneWait {
                            params: self.client.empty(
                                (CAMERA_PARAMS_START + lanes * PER_CAMERA_PARAMS)
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
                    CAMERA_PARAMS_START + PER_CAMERA_PARAMS * (cameras - 1),
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
                    CAMERA_PARAMS_START + PER_CAMERA_PARAMS * (cameras - 1),
                ),
                BufferArg::from_raw_parts(io.clone(), FUSED_RUNS * count),
                cells,
                count,
            );
        }
        // SAFETY: The prepared pass follows stereo_inputs on the same client stream.
        unsafe {
            (*klt).run(client);
        }
    }
}

