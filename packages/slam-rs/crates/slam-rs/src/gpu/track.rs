//! The GPU [`PatchTracker`]: both passes and the recovered-distance test on the
//! device, one wait per call.

use cubecl::prelude::*;

use super::kernels::{self, PositionBases};
use super::patches::GpuPatches;
use super::pyramid::GpuPyramid;
use super::{GpuError, guarded};
use crate::frontend::patterns::Pattern;
use crate::frontend::tracker::{
    FlowResult, FlowTransforms, PatchTracker, SourcePatches, TrackerError, check_track_inputs,
};
use crate::pyramid::Pyramid;

/// Runs of `count` in a transform buffer: the six warp coefficients and the
/// validity flag.
const TRANSFORM_RUNS: usize = 7;

/// The CPU tracker's counterpart, with the two passes and the check between them
/// on the device.
///
/// One [`PatchTracker::submit_prepared`] call is five launches and no wait: the
/// forward sweep, the backward pass's inputs, the backward patch build, the
/// backward sweep, and the recovered-distance test that combines them. Nothing
/// crosses back to the host in between — the backward pass reads the forward
/// result where it lies — so the only download is the packed result, seven `f32`
/// per keypoint, and [`PatchTracker::collect`] takes every submitted pass's
/// result in **one** download.
///
/// That split is the whole point of the type. A synchronising read on this lane
/// costs about 0.12 ms of host time before it has moved a byte — a staging
/// reservation, a submit, a `map_async`, and two thread handoffs to the runtime's
/// polling thread — so a frameset's host cost is set by how many reads it makes.
/// A two-camera frameset made five and spent 1.51 ms of its 1.64 in them.
///
/// The passes of one batch share every intermediate buffer, and may: the device
/// stream is ordered, so pass *k*'s `finish` has read the patch stores and the
/// backward transforms before pass *k+1*'s kernels write them. The **result**
/// is the one buffer that must survive until the download, so there is one per
/// lane and a lane is a camera.
///
/// The buffers a call needs are allocated once at construction and reused; the
/// one exception is the forward pass's inputs, which arrive from the host and so
/// must be a fresh `create_from_slice` every call.
pub struct GpuPatchTracker<P: Pattern, R: Runtime> {
    client: ComputeClient<R>,
    capacity: usize,
    num_levels: usize,
    max_iterations: usize,
    max_recovered_dist2: f32,
    /// Backward source patches, built from `next` at the forward result.
    backward_patches: GpuPatches<P, R>,
    /// The backward pass's inputs, then its results in place.
    backward: cubecl::server::Handle,
    /// One combined result per lane; a batch writes `results[lane]` and
    /// [`PatchTracker::collect`] downloads every one of them at once.
    results: Vec<cubecl::server::Handle>,
    /// The keypoint counts of the passes submitted since the last collect, in
    /// submission order, which is also their lane order.
    pending: Vec<usize>,
    /// The staging buffer the transform inputs are uploaded from.
    staging: Vec<f32>,
    /// `source position - guess` per patch, the offset the backward guess adds.
    offset_x: Vec<f32>,
    /// The `y` half of the same offset.
    offset_y: Vec<f32>,
    /// Buffers another stage on this client launched and left for whichever
    /// download comes next; [`PatchTracker::collect`] appends them to its own,
    /// which is the corner scanner's cell keys arriving for free (D78).
    reads: super::ReadRelay,
}

impl<P: Pattern, R: Runtime> GpuPatchTracker<P, R> {
    /// A tracker sized for `capacity` keypoints over `num_levels` levels, able
    /// to hold `lanes` passes in flight — the rig's camera count.
    ///
    /// # Errors
    ///
    /// As [`GpuPatches::new`]: every buffer product is checked against the
    /// `usize` range and the two ceilings before anything is allocated, and the
    /// three allocations run inside the module's guard, so a device that dies
    /// under them is [`super::GpuError::DeviceLost`] and not an unwind through
    /// the caller (decision D32).
    pub fn new(
        client: ComputeClient<R>,
        capacity: usize,
        num_levels: usize,
        max_iterations: usize,
        max_recovered_dist2: f32,
        lanes: usize,
    ) -> Result<Self, TrackerError> {
        guarded(
            GpuError::DeviceLost {
                what: "tracker allocation",
            },
            || {
                let backward_patches: GpuPatches<P, R> =
                    GpuPatches::new(client.clone(), capacity, num_levels)?;
                let transform_bytes: usize = TRANSFORM_RUNS * capacity * size_of::<f32>();
                // Allocated here and never again: the pool has to be flat from
                // the first frameset, which a lane allocated on first use would
                // break (`the_whole_gpu_path_holds_the_pool_flat`).
                let results: Vec<cubecl::server::Handle> = (0..lanes.max(1))
                    .map(|_| super::empty(&client, transform_bytes))
                    .collect();
                Ok(Self {
                    capacity,
                    num_levels,
                    max_iterations,
                    max_recovered_dist2,
                    backward_patches,
                    backward: super::empty(&client, transform_bytes),
                    results,
                    pending: Vec::with_capacity(lanes.max(1)),
                    staging: vec![0.0; TRANSFORM_RUNS * capacity],
                    offset_x: vec![0.0; capacity],
                    offset_y: vec![0.0; capacity],
                    reads: super::ReadRelay::default(),
                    client,
                })
            },
        )
    }

    /// Drain `relay` inside this tracker's own download.
    ///
    /// Wired by [`super::gpu_backends`]: the corner scanner stages its cell-key
    /// buffers there, and a read on this lane costs 0.12 ms of host time before
    /// it moves a byte, so carrying them is free where a second read is not
    /// (D78).
    pub fn share_reads(&mut self, relay: super::ReadRelay) {
        self.reads = relay;
    }
}

impl<P: Pattern, R: Runtime> PatchTracker for GpuPatchTracker<P, R> {
    type Pattern = P;
    type Pyramid = GpuPyramid<R>;
    type Patches = GpuPatches<P, R>;

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn num_levels(&self) -> usize {
        self.num_levels
    }

    fn make_patches(&self) -> Result<GpuPatches<P, R>, TrackerError> {
        guarded(
            GpuError::DeviceLost {
                what: "patch allocation",
            },
            || GpuPatches::new(self.client.clone(), self.capacity, self.num_levels),
        )
    }

    fn track(
        &mut self,
        prev: &GpuPyramid<R>,
        next: &GpuPyramid<R>,
        patches: &GpuPatches<P, R>,
        transforms_in: &FlowTransforms,
        out: &mut FlowResult,
    ) -> Result<(), TrackerError> {
        self.submit_inner(prev, next, patches, transforms_in, false)?;
        self.collect(std::slice::from_mut(out))
    }

    fn track_prepared(
        &mut self,
        prev: &GpuPyramid<R>,
        next: &GpuPyramid<R>,
        patches: &GpuPatches<P, R>,
        transforms_in: &FlowTransforms,
        out: &mut FlowResult,
    ) -> Result<(), TrackerError> {
        self.submit_prepared(prev, next, patches, transforms_in, out)?;
        self.collect(std::slice::from_mut(out))
    }

    fn submit_prepared(
        &mut self,
        prev: &GpuPyramid<R>,
        next: &GpuPyramid<R>,
        patches: &GpuPatches<P, R>,
        transforms_in: &FlowTransforms,
        _out: &mut FlowResult,
    ) -> Result<(), TrackerError> {
        self.submit_inner(prev, next, patches, transforms_in, true)
    }

    /// Every submitted pass's result in one download, decoded in order.
    fn collect(&mut self, outs: &mut [FlowResult]) -> Result<(), TrackerError> {
        // The one call per frameset that waits on the device, and the one the
        // frontend makes with the GIL released: a lost device panics inside
        // CubeCL's own client, and the guard is what turns that into this
        // method's error rather than a `PanicException` in Python (decision D32).
        let outcome: Result<(), TrackerError> =
            guarded(GpuError::DeviceLost { what: "tracker" }, || {
                self.collect_inner(outs)
            });
        self.pending.clear();
        outcome
    }

    fn discard(&mut self) {
        self.pending.clear();
    }
}

impl<P: Pattern, R: Runtime> GpuPatchTracker<P, R> {
    /// `trackPoints` (`frame_to_frame_optical_flow.h:294-375`) on the device,
    /// launched into the next free lane and left there.
    fn submit_inner(
        &mut self,
        prev: &GpuPyramid<R>,
        next: &GpuPyramid<R>,
        patches: &GpuPatches<P, R>,
        transforms_in: &FlowTransforms,
        build_source: bool,
    ) -> Result<(), TrackerError> {
        // Launches only, but they still reach the device, and a lost one panics
        // inside CubeCL's own client: the guard is what turns that into this
        // method's error rather than a `PanicException` through the released GIL
        // (decision D32).
        guarded(GpuError::DeviceLost { what: "tracker" }, || {
            let count: usize = transforms_in.len();
            check_track_inputs(
                count,
                patches.len(),
                patches.num_levels(),
                prev.num_levels(),
                next.num_levels(),
                self.capacity,
                self.num_levels,
            )?;
            let lane: usize = self.pending.len();
            if lane >= self.results.len() {
                return Err(TrackerError::TooManyPasses {
                    submitted: lane + 1,
                    lanes: self.results.len(),
                });
            }
            self.pending.push(count);
            if count == 0 {
                return Ok(());
            }

            // ── the forward pass's inputs: the source warps and the guesses.
            for index in 0..count {
                let coefficients: [f32; 6] = transforms_in.coefficients(index);
                for (run, value) in coefficients.into_iter().enumerate() {
                    self.staging[run * count + index] = value;
                }
            }
            self.staging[6 * count..TRANSFORM_RUNS * count].fill(0.0);
            // A local, not a field beside `backward` and `result`. Those two are
            // `client.empty` once at construction and reused; this one is a
            // host-to-device write, and `create_from_slice` is the only one CubeCL
            // 0.10 has, so a fresh buffer is allocated on every call whatever holds
            // it — and as a field it also meant one allocation at construction that
            // no path ever read (the first call replaced it, and a `count == 0` call
            // returns before touching it). Measured: this form costs about 0.05 ms
            // of the lane's 5.9 and an `Option` field about 0.14, both inside this
            // host's drift and above its 0.02 ms pair-to-pair floor.
            let forward: cubecl::server::Handle = super::submission::upload(
                &self.client,
                f32::as_bytes(&self.staging[..TRANSFORM_RUNS * count]),
            );

            // `off = source position - guess` (`:339`), which the backward guess
            // adds back (`:357`). Both terms are on the host already, so the offset
            // rides along in the backward patch set's positions buffer instead of
            // costing a kernel.
            let guess_x: &[f32] = transforms_in.translations_x();
            let guess_y: &[f32] = transforms_in.translations_y();
            for index in 0..count {
                let source = patches.position(index);
                self.offset_x[index] = source.x - guess_x[index];
                self.offset_y[index] = source.y - guess_y[index];
            }

            // `shape.count` is `patches.len()`, which the guard above proved equal
            // to `count`.
            let shape = patches.shape();
            let forward_view = (&forward, TRANSFORM_RUNS * count);

            self.backward_patches
                .accept(count, None, next.num_levels())?;
            self.backward_patches
                .upload_offsets(&self.offset_x[..count], &self.offset_y[..count]);
            if build_source {
                patches.launch_build(prev, patches.bases());
            }

            // ── forward: `trackPoint(pyr_1, pyr_2, transform_1, transform_2)` (`:349`).
            kernels::launch_klt::<R>(
                &self.client,
                next.buffers(),
                next.meta(),
                patches.store(),
                forward_view,
                shape,
                self.max_iterations,
                true,
            );

            // ── the backward source patches, from `next` at the forward result.
            kernels::launch_prepare_backward::<R>(
                &self.client,
                forward_view,
                self.backward_patches.position_buffer(),
                (&self.backward, TRANSFORM_RUNS * count),
                count,
                self.backward_patches.offset_bases(),
            );
            // The backward patches sit at the forward translations, which live in
            // runs 4 and 5 of the forward buffer, with its validity flag as the
            // selection mask — `patches.build(next, forward, Some(forward_valid))`
            // in the CPU tracker, without the round trip.
            self.backward_patches.launch_build_from(
                next,
                forward_view,
                PositionBases {
                    x: 4 * count,
                    y: 5 * count,
                    selected: 6 * count,
                },
            );

            // ── backward: `trackPoint(pyr_2, pyr_1, transform_2, recovered)` (`:359`).
            let backward_view = (&self.backward, TRANSFORM_RUNS * count);
            kernels::launch_klt::<R>(
                &self.client,
                prev.buffers(),
                prev.meta(),
                self.backward_patches.store(),
                backward_view,
                shape,
                self.max_iterations,
                false,
            );

            // ── `dist2 = (t1 - t1_recovered).squaredNorm() < max` (`:362`).
            kernels::launch_finish::<R>(
                &self.client,
                forward_view,
                backward_view,
                patches.position_buffer(),
                (&self.results[lane], TRANSFORM_RUNS * count),
                count,
                patches.bases(),
                self.max_recovered_dist2,
            );
            Ok(())
        })
    }

    /// The batch's one wait: every submitted lane's packed result downloaded
    /// together, then decoded into `outs` in submission order.
    fn collect_inner(&mut self, outs: &mut [FlowResult]) -> Result<(), TrackerError> {
        if self.pending.len() > outs.len() {
            return Err(TrackerError::LengthMismatch {
                first_name: "submitted passes",
                first: self.pending.len(),
                second_name: "results offered",
                second: outs.len(),
            });
        }
        for (lane, count) in self.pending.iter().enumerate() {
            outs[lane].reset(*count);
        }
        // A pass that offered nothing launched nothing, so it has no buffer to
        // read; the download is over the lanes that do.
        let mut reads: Vec<cubecl::server::Handle> = self
            .pending
            .iter()
            .enumerate()
            .filter(|(_, count)| **count != 0)
            .map(|(lane, count)| {
                let expected: u64 = (TRANSFORM_RUNS * count * size_of::<f32>()) as u64;
                let handle: &cubecl::server::Handle = &self.results[lane];
                handle.clone().offset_end(handle.size_in_used() - expected)
            })
            .collect();
        // Whatever another stage staged rides along on the tail, and the tail is
        // handed back to it: one synchronisation for the frameset's two answers
        // instead of one each (D78). Taken **before** the empty-read shortcut,
        // so a frameset whose every lane offered nothing still carries them.
        let lanes: usize = reads.len();
        // The tag comes back with the tail, so the bytes are handed to the
        // stage that staged them and to no other (see [`super::RelayTag`]).
        let staged: Option<super::RelayTag> = self.reads.take_staged().map(|(tag, handles)| {
            reads.extend(handles);
            tag
        });
        let mut bytes: Vec<cubecl::bytes::Bytes> = if reads.is_empty() {
            Vec::new()
        } else {
            super::read_blocking(
                &self.client,
                reads,
                "the tracker result",
                &super::seam::READ_TRACK,
            )?
        };
        if let Some(tag) = staged.filter(|_| bytes.len() >= lanes) {
            let tail: Vec<cubecl::bytes::Bytes> = bytes.split_off(lanes);
            self.reads.deliver(tag, tail);
        }

        let mut read: usize = 0;
        for (lane, count) in self.pending.iter().enumerate() {
            let count: usize = *count;
            if count == 0 {
                outs[lane].finish(0);
                continue;
            }
            let expected: usize = TRANSFORM_RUNS * count * size_of::<f32>();
            // One buffer per non-empty lane, in the order they were asked for;
            // anything else is the runtime breaking its own contract.
            let Some(buffer) = bytes.get(read) else {
                return Err(super::GpuError::DeviceReadFailed {
                    what: "a tracker lane's result",
                }
                .into());
            };
            read += 1;
            // The kernel packs by count; the unused capacity tail stays on the
            // device. Keep the short-read refusal before decoding any values.
            if buffer.len() < expected {
                return Err(TrackerError::LengthMismatch {
                    first_name: "result bytes expected",
                    first: expected,
                    second_name: "returned",
                    second: buffer.len(),
                });
            }
            let values: &[f32] = f32::from_bytes(buffer);
            let (valid, transforms) = outs[lane].parts_mut();
            let [m00, m01, m10, m11, tx, ty] = transforms.coefficients_mut();
            for index in 0..count {
                m00[index] = values[index];
                m01[index] = values[count + index];
                m10[index] = values[2 * count + index];
                m11[index] = values[3 * count + index];
                tx[index] = values[4 * count + index];
                ty[index] = values[5 * count + index];
                valid[index] = values[6 * count + index] != 0.0;
            }
            outs[lane].finish(count);
        }
        Ok(())
    }
}

impl<P: Pattern, R: Runtime> std::fmt::Debug for GpuPatchTracker<P, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuPatchTracker")
            .field("capacity", &self.capacity)
            .field("num_levels", &self.num_levels)
            .field("max_iterations", &self.max_iterations)
            .field("max_recovered_dist2", &self.max_recovered_dist2)
            .field("lanes", &self.results.len())
            .finish()
    }
}
