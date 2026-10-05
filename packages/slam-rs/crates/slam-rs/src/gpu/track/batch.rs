//! One fused KLT dispatch over a phase's cameras in shared pyramid arenas.

use std::sync::Arc;

use cubecl::prelude::*;

use super::{FUSED_RUNS, FusedLaunch, GpuPatchTracker, stage_points};
use crate::frontend::patterns::Pattern;
use crate::frontend::tracker::{TrackInput, TrackerError, check_track_inputs};
use crate::gpu::{GpuError, guarded, pyramid::GpuPyramid, submission};
use crate::pyramid::Pyramid;

impl<P: Pattern, R: Runtime> GpuPatchTracker<P, R> {
    pub(super) fn submit_packed(
        &mut self,
        prev: &[GpuPyramid<R>],
        next: &[GpuPyramid<R>],
        inputs: &mut [TrackInput],
    ) -> Result<bool, TrackerError> {
        let params = self.fused_params();
        let Some(first) = inputs.first() else {
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
        if inputs.len() > self.results.len() {
            return Err(TrackerError::TooManyPasses {
                submitted: inputs.len(),
                lanes: self.results.len(),
            });
        }
        for input in inputs.iter() {
            check_track_inputs(
                input.guesses.len(),
                input.positions.len(),
                self.num_levels,
                prev[input.source].num_levels(),
                next[input.destination].num_levels(),
                self.capacity,
                self.num_levels,
            )?;
        }
        guarded(
            GpuError::DeviceLost {
                what: "packed tracker",
            },
            || {
                self.geometry.clear();
                self.staging.clear();
                for (lane, input) in inputs.iter_mut().enumerate() {
                    prev[input.source].append_geometry(&mut self.geometry, Some(a));
                    next[input.destination].append_geometry(&mut self.geometry, Some(b));
                    let count = input.guesses.len();
                    stage_points(&mut self.staging, &input.guesses, lane, |index| {
                        (1.0, input.positions.get(index))
                    });
                    input.result = lane;
                    self.batch.slot_mut(lane, self.capacity);
                    self.pending.push(count);
                }
                self.geometry
                    .extend(P::OFFSETS.iter().flat_map(|tap| tap.map(f32::to_bits)));
                packed.meta.update(&self.client, &self.geometry);
                packed.count = self.staging.len() / FUSED_RUNS;
                if packed.count == 0 {
                    return Ok(true);
                }
                self.client.write(
                    &packed.io,
                    cubecl::bytes::Bytes::from_elems(self.staging.clone()),
                );
                self.launches.dispatch(
                    &self.client,
                    submission::Launch::Klt(FusedLaunch::new(
                        [
                            a.bindings()[0],
                            a.bindings()[1],
                            b.bindings()[0],
                            b.bindings()[1],
                        ],
                        &packed.meta,
                        &packed.io,
                        packed.count,
                        inputs.len(),
                        params,
                    )),
                );
                Ok(true)
            },
        )
    }
}
