//! Fused forward/backward SE(2) patch tracking with reusable device storage.
mod buffers;
use crate::kernels::klt_fused::{decode_point, encode_point, meta_len, FusedParams};
pub use crate::kernels::klt_fused::{
    FUSED_RUNS, RUN_CAMERA, RUN_SOURCE_X, RUN_SOURCE_Y, RUN_TARGET_X, RUN_TARGET_Y, RUN_VALID,
    RUN_WARP,
};
use crate::{
    pyramid::{FrameArena, GpuPyramid},
    runtime::{guarded, GpuError},
};
pub use buffers::FusedLaunch;
use buffers::KltBuffers;
use cubecl::prelude::*;
use kornia_staging_imgproc::optical_flow::{
    patch_se2::Pattern,
    patch_tracker::{self, FlowResult, FlowTransforms, PointsSoA, TrackerError},
};

/// A rejected tracking input or device operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TrackingError {
    /// Invalid tracking parameters or input shape.
    #[error(transparent)]
    Input(#[from] TrackerError),
    /// Device failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),
}

/// One camera's host point records for a fused pass.
pub struct TrackPoints<'a> {
    /// Source pixel coordinates.
    pub positions: &'a PointsSoA,
    /// Initial destination transforms in the same order.
    pub guesses: &'a FlowTransforms,
    /// Optional source validity mask.
    pub selected: Option<&'a [bool]>,
}

/// Reusable storage for one forward/backward pass, independent of frame scheduling.
///
/// A plan samples templates in kernel registers; it stores no keypoint identities
/// or temporal/stereo state. Inputs use the CPU patch tracker's types.
pub struct FusedKltPlan<P: Pattern, R: Runtime> {
    client: ComputeClient<R>,
    buffers: KltBuffers,
    params: FusedParams,
    capacity: usize,
    cameras: usize,
    geometry: Vec<u32>,
    points: Vec<f32>,
    pattern: std::marker::PhantomData<P>,
}
impl<P: Pattern, R: Runtime> FusedKltPlan<P, R> {
    /// Allocate one reusable pass on `client`.
    ///
    /// # Arguments
    /// * `capacity` - Maximum input points.
    /// * `levels` - Pyramid levels including full resolution.
    /// * `iterations` - Maximum iterations per pyramid level.
    /// * `max_dist2` - Maximum squared forward/backward recovery error.
    ///
    /// # Errors
    /// Rejects invalid parameters and device allocation/subgroup failures.
    ///
    /// ```no_run
    /// # #[cfg(feature = "wgpu")] {
    /// use kornia_staging_gpu::{optical_flow::FusedKltPlan, runtime::gpu_client};
    /// use kornia_staging_imgproc::optical_flow::patch_se2::Pattern51;
    /// let plan = FusedKltPlan::<Pattern51, _>::new(gpu_client()?, 128, 4, 5, 0.04)?;
    /// # }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(
        client: ComputeClient<R>,
        capacity: usize,
        levels: usize,
        iterations: usize,
        max_dist2: f32,
    ) -> Result<Self, TrackingError> {
        Self::new_batched(client, capacity, levels, iterations, max_dist2, 1)
    }

    /// Reserve a fused pass for up to `cameras` source/target pairs.
    ///
    /// # Errors
    /// Rejects invalid tracking parameters, unsupported capacities and device failures.
    pub fn new_batched(
        client: ComputeClient<R>,
        capacity: usize,
        levels: usize,
        iterations: usize,
        max_dist2: f32,
        cameras: usize,
    ) -> Result<Self, TrackingError> {
        patch_tracker::limits::validate_tracking_parameters(iterations, max_dist2)?;
        patch_tracker::limits::checked_patch_shape(capacity, levels, P::SIZE)?;
        crate::runtime::probe_subgroups(&client)?;
        guarded(
            GpuError::DeviceLost {
                what: "tracker allocation",
            },
            || {
                Ok(Self {
                    buffers: KltBuffers::new(&client, capacity, levels, cameras, P::OFFSETS.len())?,
                    params: FusedParams {
                        levels,
                        taps: P::OFFSETS.len(),
                        iterations,
                        max_dist2,
                        exit_step_px: None,
                    },
                    capacity,
                    cameras,
                    geometry: Vec::with_capacity(meta_len(levels, cameras, P::OFFSETS.len())),
                    points: Vec::with_capacity(capacity * cameras * FUSED_RUNS),
                    client,
                    pattern: std::marker::PhantomData,
                })
            },
        )
    }
    /// Set a positive finite convergence threshold, or disable early exit.
    ///
    /// # Errors
    /// Rejects nonpositive or nonfinite thresholds.
    pub fn set_exit_step_px(&mut self, threshold: Option<f32>) -> Result<(), TrackingError> {
        patch_tracker::limits::validate_exit_step(threshold)?;
        self.params.exit_step_px = threshold;
        Ok(())
    }
    /// Track the supplied source positions and initial transforms, then collect results.
    ///
    /// # Arguments
    /// * `prev`, `next` - Pyramids built on this plan's client and stream.
    /// * `positions`, `guesses` - Matching source positions and destination guesses.
    /// * `selected` - Optional mask; false entries remain invalid.
    /// * `out` - Reused result storage, cleared on any failure.
    ///
    /// # Errors
    /// Rejects mismatched shapes, short readbacks and device failures.
    pub fn track(
        &mut self,
        prev: &GpuPyramid<R>,
        next: &GpuPyramid<R>,
        positions: &PointsSoA,
        guesses: &FlowTransforms,
        selected: Option<&[bool]>,
        out: &mut FlowResult,
    ) -> Result<(), TrackingError> {
        out.clear();
        let count = guesses.len();
        let launch = self.prepare(
            &[(prev, next)],
            None,
            &[TrackPoints {
                positions,
                guesses,
                selected,
            }],
        )?;
        if count == 0 {
            return Ok(());
        }
        guarded(
            GpuError::DeviceLost {
                what: "tracker launch",
            },
            || {
                // SAFETY: Validated records and retained pyramids use this plan's stream.
                unsafe {
                    launch.run(&self.client);
                }
                Ok::<_, TrackingError>(())
            },
        )?;
        let reads = crate::transfer::read_buffers(
            &self.client,
            vec![self.result_handle(count)?],
            "the tracker result",
        )?;
        let [bytes] = reads.as_slice() else {
            return Err(GpuError::DeviceReadFailed {
                what: "the tracker result",
            }
            .into());
        };
        Self::decode(bytes, count, out)
    }

    /// Prepare host point records and geometry without launching or waiting.
    ///
    /// Each pair uses this plan's device and stream. A multi-camera pass requires
    /// matching shared arenas. Run the returned work before preparing this plan again.
    ///
    /// # Errors
    /// Rejects inconsistent shapes, arena ownership, capacity overflow and device failures.
    pub fn prepare(
        &mut self,
        pairs: &[(&GpuPyramid<R>, &GpuPyramid<R>)],
        arena: Option<(&FrameArena, &FrameArena)>,
        inputs: &[TrackPoints<'_>],
    ) -> Result<FusedLaunch, TrackingError> {
        if pairs.len() != inputs.len() {
            return Err(TrackerError::InvalidParameter("KLT camera inputs").into());
        }
        for ((prev, next), input) in pairs.iter().zip(inputs) {
            patch_tracker::limits::check_track_inputs(
                input.guesses.len(),
                input.positions.len(),
                self.params.levels,
                prev.num_levels(),
                next.num_levels(),
                self.capacity,
                self.params.levels,
            )?;
            patch_tracker::limits::check_patch_inputs(
                input.positions.len(),
                self.capacity,
                input.selected,
                prev.num_levels(),
                self.params.levels,
            )?;
        }
        self.points.clear();
        for (camera, input) in inputs.iter().enumerate() {
            for index in 0..input.guesses.len() {
                self.points.extend(encode_point(
                    input.guesses.coefficients(index),
                    f32::from(u8::from(input.selected.is_none_or(|mask| mask[index]))),
                    input.positions.get(index),
                    camera,
                ));
            }
        }
        let count = self.points.len() / FUSED_RUNS;
        let launch = self.prepare_geometry(pairs, arena, count)?;
        guarded(
            GpuError::DeviceLost {
                what: "tracker inputs",
            },
            || {
                if count != 0 {
                    self.client.write(
                        self.buffers.io(),
                        cubecl::bytes::Bytes::from_elems(self.points.clone()),
                    );
                }
                Ok::<_, TrackingError>(launch)
            },
        )
    }

    /// Prepare a pass whose point records will be filled on the device.
    ///
    /// # Errors
    /// Rejects inconsistent geometry, arena ownership, capacity overflow and device failures.
    ///
    /// # Safety
    /// Before running the returned launch, write `count` complete records through
    /// `result_handle(count)` on the same stream. Every camera index must address
    /// `pairs`, every validity flag must be zero or one, and selected coordinates
    /// and transforms must be finite. Run before preparing this plan again.
    pub unsafe fn prepare_device(
        &mut self,
        pairs: &[(&GpuPyramid<R>, &GpuPyramid<R>)],
        arena: (&FrameArena, &FrameArena),
        count: usize,
    ) -> Result<FusedLaunch, TrackingError> {
        self.prepare_geometry(pairs, Some(arena), count)
    }

    fn prepare_geometry(
        &mut self,
        pairs: &[(&GpuPyramid<R>, &GpuPyramid<R>)],
        arena: Option<(&FrameArena, &FrameArena)>,
        count: usize,
    ) -> Result<FusedLaunch, TrackingError> {
        if pairs.is_empty()
            || pairs.len() > self.cameras
            || count > self.capacity * self.cameras
            || (pairs.len() > 1 && arena.is_none())
        {
            return Err(TrackerError::InvalidParameter("KLT dispatch shape").into());
        }
        self.geometry.clear();
        for &(prev, next) in pairs {
            if prev.num_levels() < self.params.levels || next.num_levels() < self.params.levels {
                return Err(TrackerError::InvalidParameter("KLT pyramid levels").into());
            }
            if let Some((a, b)) = arena {
                if !prev
                    .arena()
                    .is_some_and(|owner| std::ptr::eq(owner.as_ref(), a))
                    || !next
                        .arena()
                        .is_some_and(|owner| std::ptr::eq(owner.as_ref(), b))
                {
                    return Err(TrackerError::InvalidParameter("KLT pyramid arena").into());
                }
            }
            prev.append_geometry_levels(
                &mut self.geometry,
                arena.map(|pair| pair.0),
                self.params.levels,
            );
            next.append_geometry_levels(
                &mut self.geometry,
                arena.map(|pair| pair.1),
                self.params.levels,
            );
        }
        self.geometry
            .extend(P::OFFSETS.iter().flat_map(|tap| tap.map(f32::to_bits)));
        let (a, b) = arena.map_or_else(
            || (pairs[0].0.buffers(), pairs[0].1.buffers()),
            |(a, b)| (a.bindings(), b.bindings()),
        );
        guarded(
            GpuError::DeviceLost {
                what: "tracker preparation",
            },
            || {
                Ok(self.buffers.prepare(
                    &self.client,
                    [a[0], a[1], b[0], b[1]],
                    &self.geometry,
                    count,
                    pairs.len(),
                    self.params,
                ))
            },
        )
    }

    /// The used record prefix for a readback or a device input producer.
    ///
    /// # Errors
    /// Rejects a count beyond the reserved record capacity.
    pub fn result_handle(&self, count: usize) -> Result<cubecl::server::Handle, TrackingError> {
        if count > self.capacity * self.cameras {
            return Err(TrackerError::InvalidParameter("KLT result count").into());
        }
        let handle = self.buffers.io();
        Ok(handle
            .clone()
            .offset_end(handle.size_in_used() - (count * FUSED_RUNS * size_of::<f32>()) as u64))
    }

    /// Decode one lane's exact record slice, clearing stale output on failure.
    ///
    /// # Errors
    /// Rejects incomplete or extra result bytes.
    pub fn decode(bytes: &[u8], count: usize, out: &mut FlowResult) -> Result<(), TrackingError> {
        out.clear();
        let expected = count
            .checked_mul(FUSED_RUNS * size_of::<f32>())
            .ok_or(TrackerError::InvalidParameter("KLT result count"))?;
        if bytes.len() != expected {
            return Err(GpuError::ShortRead {
                what: "the tracker result",
                actual: bytes.len(),
                expected,
            }
            .into());
        }
        out.reset(count);
        for (index, point) in f32::from_bytes(bytes)
            .as_chunks::<FUSED_RUNS>()
            .0
            .iter()
            .enumerate()
        {
            decode_point(point, out, index);
        }
        out.finish();
        Ok(())
    }
}
