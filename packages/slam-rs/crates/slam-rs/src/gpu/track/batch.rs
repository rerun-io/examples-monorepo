//! One fused KLT dispatch over a phase's cameras in shared pyramid arenas.

use crate::frontend::flow::FrontendError;
use std::sync::Arc;

use cubecl::prelude::*;

use super::GpuPatchTracker;
use crate::gpu::{pyramid::GpuPyramid, submission};

use kornia_staging_gpu::optical_flow::TrackPoints;
use kornia_staging_imgproc::optical_flow::patch_se2::Pattern;

impl<P: Pattern, R: Runtime> GpuPatchTracker<P, R> {
    pub(super) fn submit_packed(
        &mut self,
        prev: &[GpuPyramid<R>],
        next: &[GpuPyramid<R>],
        inputs: ValidatedPhase<'_>,
        slots: &mut [usize],
    ) -> Result<bool, FrontendError> {
        let Some(first) = inputs.iter().next() else {
            return Ok(true);
        };
        let (Some(a), Some(b)) = (prev[first.source].arena(), next[first.destination].arena())
        else {
            return Ok(false);
        };
        if !self.pending.is_empty()
            || !inputs.iter().all(|input| {
                prev[input.source]
                    .arena()
                    .is_some_and(|other| Arc::ptr_eq(a, other))
                    && next[input.destination]
                        .arena()
                        .is_some_and(|other| Arc::ptr_eq(b, other))
            })
        {
            return Ok(false);
        }
        let packed = &mut self.packed_fused;
        let pairs: Vec<_> = inputs
            .iter()
            .map(|input| (&prev[input.source], &next[input.destination]))
            .collect();
        let points: Vec<_> = inputs
            .iter()
            .map(|input| TrackPoints {
                positions: input.positions,
                guesses: input.guesses,
                selected: None,
            })
            .collect();
        packed.count = inputs.iter().map(|input| input.guesses.len()).sum();
        let launch = packed.plan.prepare(&pairs, Some((a, b)), &points)?;
        for (lane, input) in inputs.iter().enumerate() {
            slots[lane] = lane;
            self.batch.slot_mut(lane, self.capacity);
            self.pending.push(input.guesses.len());
        }
        if packed.count != 0 {
            self.launches
                .dispatch(&self.client, submission::Launch::Klt(launch))?;
        }
        Ok(true)
    }
}
use kornia_staging_slam::tracking::optical_flow::ValidatedPhase;
