//! Persistent KLT records and owned deferred launches.
use super::{FusedParams, TrackingError, FUSED_RUNS};
use crate::kernels::klt_fused::launch_fused;
use crate::kernels::klt_fused::CachedU32Upload;
use cubecl::prelude::*;
use kornia_staging_imgproc::optical_flow::patch_tracker::limits::checked_patch_shape;
use kornia_staging_imgproc::optical_flow::patch_tracker::TrackerError;

/// Persistent point records and cached pyramid/pattern metadata for one dispatch.
/// The consumer may retain one allocation per independent in-flight pass.
pub(super) struct KltBuffers {
    io: cubecl::server::Handle,
    meta: CachedU32Upload,
}
impl KltBuffers {
    /// Reserve device storage for `capacity` points per camera.
    ///
    /// # Arguments
    /// * `client` - Device used for all later preparation and execution.
    /// * `capacity`, `levels`, `cameras`, `taps` - Bounds of prepared dispatches.
    ///   Total `capacity * cameras` may not exceed 262140 (four points per cube).
    ///
    /// # Errors
    /// Rejects overflowing or unsupported shapes and allocation failures.
    pub fn new<R: Runtime>(
        client: &ComputeClient<R>,
        capacity: usize,
        levels: usize,
        cameras: usize,
        taps: usize,
    ) -> Result<Self, TrackingError> {
        checked_patch_shape(capacity, levels, taps)?;
        let invalid = || TrackerError::InvalidParameter("KLT dispatch shape");
        let points = capacity.checked_mul(cameras).ok_or_else(invalid)?;
        let records = points.checked_mul(FUSED_RUNS).ok_or_else(invalid)?;
        let metadata = levels
            .checked_mul(8)
            .and_then(|v| v.checked_mul(cameras))
            .and_then(|v| taps.checked_mul(2).and_then(|t| v.checked_add(t)))
            .ok_or_else(invalid)?;
        if cameras == 0
            || points > crate::kernels::MAX_CUBES_PER_DIM as usize * 4
            || records > u32::MAX as usize
            || metadata > u32::MAX as usize
        {
            return Err(invalid().into());
        }
        let bytes = records.checked_mul(size_of::<f32>()).ok_or_else(invalid)?;
        metadata.checked_mul(size_of::<u32>()).ok_or_else(invalid)?;
        Ok(Self {
            io: client.empty(bytes),
            meta: CachedU32Upload::new(client, metadata),
        })
    }
    /// Point input/output allocation, for a caller that batches compatible readbacks.
    pub fn io(&self) -> &cubecl::server::Handle {
        &self.io
    }

    pub(super) fn prepare<R: Runtime>(
        &mut self,
        client: &ComputeClient<R>,
        images: [crate::kernels::Buffer<'_>; 4],
        geometry: &[u32],
        count: usize,
        cameras: usize,
        params: FusedParams,
    ) -> FusedLaunch {
        self.meta.update(client, geometry);
        FusedLaunch {
            buffers: images.map(|(handle, len)| (handle.clone(), len)),
            meta: (self.meta.handle.clone(), self.meta.len()),
            io: self.io.clone(),
            count,
            cameras,
            params,
        }
    }
}

/// Owned bindings for one prepared fused forward/backward dispatch.
pub struct FusedLaunch {
    buffers: [(cubecl::server::Handle, usize); 4],
    meta: (cubecl::server::Handle, usize),
    io: cubecl::server::Handle,
    count: usize,
    cameras: usize,
    params: FusedParams,
}
impl FusedLaunch {
    /// Enqueue this dispatch on the preparation stream.
    ///
    /// # Safety
    /// Use the same client and stream as preparation. Keep all inputs unchanged
    /// since preparation, and exclude concurrent access to the retained buffers.
    pub unsafe fn run<R: Runtime>(self, client: &ComputeClient<R>) {
        // SAFETY: The caller upholds the prepared launch's validated binding contract.
        unsafe {
            launch_fused(
                client,
                self.buffers.each_ref().map(|(handle, len)| (handle, *len)),
                (&self.meta.0, self.meta.1),
                &self.io,
                self.count,
                self.cameras,
                self.params,
            );
        }
    }
}
