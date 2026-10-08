//! Device selection connects temporal KLT to stereo without an intermediate read.

use crate::gpu::kernels::onewait::{CAMERA_PARAMS_START, PER_CAMERA_PARAMS};
use cubecl::prelude::*;

use super::GpuStages;
use kornia_staging_3d::camera::CameraModelKind;
use crate::config::MatchingGuessType;
use crate::frontend::detect::{CellGrid, CellSelect};
use crate::frontend::patterns::Pattern;
use crate::frontend::stages::StereoContext;
use crate::frontend::tracker::{TrackInput, TrackerError};
use crate::gpu::kernels::klt_fused::{CachedU32Upload, FUSED_RUNS, decode_point, launch_fused};
use crate::gpu::{GpuError, guarded, kernels, pyramid::GpuPyramid, submission};

pub(super) enum Phase {
    Off,
    Prepared,
    Submitted,
    Ready(cubecl::bytes::Bytes),
}

pub(super) struct OneWait {
    params: cubecl::server::Handle,
    selected: cubecl::server::Handle,
    occupied: cubecl::server::Handle,
    pub(super) io: cubecl::server::Handle,
    meta: CachedU32Upload,
    grid: CellGrid,
    cameras: usize,
    cells: usize,
    pub(super) phase: Phase,
}

impl<P: Pattern, R: Runtime> GpuStages<P, R> {
    /// Enable the single-read path only for the supported, unmasked shared rig.
    pub(super) fn prepare_one_wait(
        &mut self,
        context: StereoContext<'_>,
        selects: &[Option<CellSelect>],
    ) -> Result<bool, TrackerError> {
        let StereoContext {
            cameras,
            calib,
            config,
            depth,
            last_detect_count,
            eligible,
        } = context;
        let pyramids = &self.current.pyramids;
        guarded(
            GpuError::DeviceLost {
                what: "stereo preparation",
            },
            || {
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
                        .all(|camera| matches!(camera.model.inner, CameraModelKind::BrownConrady(_)))
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
                    state.grid == select.grid && state.cameras == cameras.len()
                });
                let state = match &mut self.one_wait {
                    Some(existing) if fits => existing,
                    slot => slot.insert(OneWait {
                        params: self.client.empty(
                            (CAMERA_PARAMS_START + lanes * PER_CAMERA_PARAMS) * size_of::<f32>(),
                        ),
                        selected: self.client.empty((1 + cells * 2) * size_of::<f32>()),
                        occupied: self.client.empty((1 + cells) * size_of::<u32>()),
                        io: self
                            .client
                            .empty(FUSED_RUNS * cells * lanes * size_of::<f32>()),
                        meta: CachedU32Upload::new(
                            &self.client,
                            crate::gpu::kernels::klt_fused::meta_len(
                                self.tracker.num_levels,
                                lanes,
                                P::OFFSETS.len(),
                            ),
                        ),
                        grid: select.grid,
                        cameras: cameras.len(),
                        cells,
                        phase: Phase::Off,
                    }),
                };
                let mut params = vec![
                    depth,
                    config.port_redetect_survivor_ratio,
                    last_detect_count as f32,
                    self.tracker.capacity as f32,
                    if config.optical_flow_matching_guess_type != MatchingGuessType::SamePixel {
                        1.0
                    } else {
                        0.0
                    },
                ];
                if let CameraModelKind::BrownConrady(camera) = cameras[0].model.inner {
                    params.extend(camera.params()[..12].iter());
                }
                self.geometry.clear();
                for camera in 1..cameras.len() {
                    // Match the host's two inversions and quaternion action exactly.
                    let transform = (calib.t_i_c[0].inverse() * calib.t_i_c[camera]).inverse();
                    params.extend(transform.rotation.quaternion_xyzw());
                    params.extend(transform.translation.iter());
                    if let CameraModelKind::BrownConrady(model) = cameras[camera].model.inner {
                        params.extend(model.params()[..12].iter());
                    }
                    pyramids[0].append_geometry(&mut self.geometry, Some(arena));
                    pyramids[camera].append_geometry(&mut self.geometry, Some(arena));
                }
                self.geometry
                    .extend(P::OFFSETS.iter().flat_map(|tap| tap.map(f32::to_bits)));
                self.client
                    .write(&state.params, cubecl::bytes::Bytes::from_elems(params));
                state.meta.update(&self.client, &self.geometry);
                state.phase = Phase::Prepared;
                Ok(true)
            },
        )
    }

    pub(super) fn submit_stereo(&mut self) -> Result<(), TrackerError> {
        let pyramids = &self.current.pyramids;
        let params = self.tracker.fused_params();
        let Some(state) = self
            .one_wait
            .as_mut()
            .filter(|state| matches!(state.phase, Phase::Prepared))
        else {
            return Ok(());
        };
        let packed = &self.tracker.packed_fused;
        let Some(arena) = pyramids[0].arena() else {
            return Ok(());
        };
        let Some(keys) = self
            .current
            .detector
            .scanner
            .staged_handles()
            .filter(|keys| keys.len() == 1)
        else {
            // An unbatchable detector uses the normal host selection and stereo pass.
            state.phase = Phase::Off;
            return Ok(());
        };
        state.phase = Phase::Submitted;
        self.launches.dispatch(
            &self.client,
            submission::Launch::Stereo(StereoLaunch {
                stereo_params: state.params.clone(),
                selected: state.selected.clone(),
                occupied: state.occupied.clone(),
                io: state.io.clone(),
                meta: (state.meta.handle.clone(), state.meta.len()),
                grid: state.grid,
                cameras: state.cameras,
                cells: state.cells,
                temporal_io: packed.io.clone(),
                temporal_len: FUSED_RUNS * packed.count.max(1),
                temporal_count: self.tracker.pending.first().copied().unwrap_or(0),
                keys: keys[0].clone(),
                arena: arena.clone(),
                params,
            }),
        );
        Ok(())
    }

    pub(super) fn take_stereo(&mut self, inputs: &mut [TrackInput]) -> Result<bool, TrackerError> {
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
        for (lane, input) in inputs.iter_mut().enumerate() {
            let count = input.positions.len();
            if count > state.cells {
                return Err(GpuError::DeviceReadFailed {
                    what: "one-wait corner count",
                }
                .into());
            }
            self.tracker.batch.slot_mut(lane, self.tracker.capacity);
            input.result = lane;
            self.tracker.batch.slots[lane].reset(count);
            for index in 0..count {
                let base = FUSED_RUNS * (lane * state.cells + index);
                let source = input.positions.get(index);
                if values[base + 7] != source.x || values[base + 8] != source.y {
                    return Err(GpuError::DeviceReadFailed {
                        what: "one-wait corner order",
                    }
                    .into());
                }
                decode_point(
                    &values[base..base + FUSED_RUNS],
                    &mut self.tracker.batch.slots[lane],
                    index,
                );
            }
            self.tracker.batch.slots[lane].finish(count);
        }
        Ok(true)
    }
}

pub(in crate::gpu) struct StereoLaunch {
    stereo_params: cubecl::server::Handle,
    selected: cubecl::server::Handle,
    occupied: cubecl::server::Handle,
    io: cubecl::server::Handle,
    meta: (cubecl::server::Handle, usize),
    grid: CellGrid,
    cameras: usize,
    cells: usize,
    temporal_io: cubecl::server::Handle,
    temporal_len: usize,
    temporal_count: usize,
    keys: cubecl::server::Handle,
    arena: std::sync::Arc<crate::gpu::pyramid::FrameArena>,
    params: crate::gpu::kernels::klt_fused::FusedParams,
}
impl StereoLaunch {
    pub(in crate::gpu) fn run<R: Runtime>(self, client: &ComputeClient<R>) {
        let Self {
            stereo_params,
            selected,
            occupied,
            io,
            meta,
            grid,
            cameras,
            cells,
            temporal_io,
            temporal_len,
            temporal_count,
            keys,
            arena,
            params,
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
        let [even, odd] = arena.bindings();
        launch_fused(
            client,
            [even, odd, even, odd],
            (&meta.0, meta.1),
            &io,
            count,
            cameras - 1,
            params,
        );
    }
}
