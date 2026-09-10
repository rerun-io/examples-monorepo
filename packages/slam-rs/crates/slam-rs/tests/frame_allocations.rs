//! What the per-frame paths ask of the allocator, counted.
//!
//! The frontend's first, then the estimator's: same counting allocator, same
//! thread-scoped gate, and in both cases a claim that had to be measured
//! rather than asserted in a comment.
//!
//! `cubecl-portability.md` §12.2 asks for no allocation on the per-frame path,
//! because a GPU kernel cannot grow a buffer and because an allocator call
//! inside the released-GIL region is a latency spike nobody sees. The claim was
//! made once and turned out to be false — `#[derive(Clone)]` writes only
//! `clone`, so `clone_from` fell back to `*self = source.clone()` and the
//! keypoint snapshot replaced sixteen buffers per stereo frame — so it is
//! measured here rather than asserted in a comment.
//!
//! ## The counter is thread-scoped, and has to be
//!
//! The measurement is a counting global allocator over this test binary alone;
//! the shipped library never sees it. Both the gate **and the counters** are
//! thread-locals, so an allocation is counted only when it happens on the thread
//! that opened the gate.
//!
//! A global flag under a mutex is not enough, and was observed failing: the
//! mutex serialises the *measurements*, but libtest runs each test on its own
//! thread and those threads keep allocating — printing, panicking, unwinding,
//! setting up the next test — while one of them holds the gate open. Under build
//! load one run of `copying_the_warp_arrays_costs_nothing_once_warm` counted two
//! reallocations that were another thread's. Thread-locals remove the shared
//! state entirely: no mutex, no cross-talk, and the numbers do not depend on how
//! many tests the harness decides to run at once.
//!
//! A `thread_local!` with a `const` initialiser allocates nothing itself and
//! registers no destructor for a `Cell`, so it cannot recurse into the allocator
//! it is counting; `try_with` is still used, so a call during thread teardown
//! degrades to "not counting" instead of panicking inside `alloc`.
//!
//! One consequence to know about: work the frontend does on *other* threads is
//! not counted. Every measurement here runs at `FrontendOptions::threads == 1`,
//! where `WorkPool` takes its sequential path and never starts a rayon worker, so
//! there is no such work. A future test at a wider thread budget would have to
//! account for it.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use slam_rs::config::VioConfig;
use slam_rs::frontend::detect::{CellGrid, CpuCornerScan};
use slam_rs::frontend::flow::{
    FlowFrame, FrameToFrameOpticalFlow, FrontendOptions, Keypoints, PosePrediction,
};
use slam_rs::frontend::patterns::Pattern51;
use slam_rs::frontend::se2::AffineCompact2f;
use slam_rs::frontend::tracker::FlowTransforms;
use slam_rs::image::ImageU16;

mod common;

use common::{dotted_image, flow_config, flow_rig};

// ── the counting allocator ────────────────────────────────────────────────

thread_local! {
    /// Whether this thread is inside a [`measure`] call.
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    /// This thread's counts since the gate opened.
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static REALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static DEALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

/// Add one to `counter`, but only on a thread that is measuring.
///
/// `try_with` rather than `with`: a thread tearing down its locals may no longer
/// have them, and a panic inside `alloc` would be an abort.
#[inline]
fn count(counter: &'static std::thread::LocalKey<Cell<usize>>) {
    let counting: bool = COUNTING.try_with(Cell::get).unwrap_or(false);
    if counting {
        let _ = counter.try_with(|slot| slot.set(slot.get() + 1));
    }
}

struct Counting;

// SAFETY: every method forwards to `System` with the layout it was given and
// changes nothing about the returned pointer. The only addition is a
// thread-local increment, which cannot allocate: the locals are `Cell`s with
// `const` initialisers, so they need no lazy setup and register no destructor.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(&ALLOCATIONS);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        count(&DEALLOCATIONS);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(&REALLOCATIONS);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// What one counted region cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Allocations {
    allocations: usize,
    reallocations: usize,
    deallocations: usize,
}

impl Allocations {
    /// Every call that reaches the allocator, however it is spelled.
    fn total(&self) -> usize {
        self.allocations + self.reallocations + self.deallocations
    }
}

/// Run `body` with this thread's allocator counting, and report what it cost.
///
/// Re-entrant by accident is impossible — `body` is `FnOnce` and the gate is
/// this thread's — and concurrent measurements on other threads are independent,
/// because there is no shared counter to share.
fn measure<T>(body: impl FnOnce() -> T) -> (T, Allocations) {
    ALLOCATIONS.set(0);
    REALLOCATIONS.set(0);
    DEALLOCATIONS.set(0);
    COUNTING.set(true);
    let value: T = body();
    COUNTING.set(false);
    let counted: Allocations = Allocations {
        allocations: ALLOCATIONS.get(),
        reallocations: REALLOCATIONS.get(),
        deallocations: DEALLOCATIONS.get(),
    };
    (value, counted)
}

// ── the frontend under test ───────────────────────────────────────────────

/// The snapshot's exact shape: one `Keypoints` per camera, copied in place.
///
/// This is the operation that once cost sixteen allocations and sixteen frees
/// per call, on this thread only. Every type in the chain — `Vec`, `Keypoints` and
/// `FlowTransforms` — now implements `clone_from` by hand, so a copy into
/// buffers that are already big enough reaches the allocator zero times.
#[test]
fn snapshotting_a_camera_costs_nothing_once_the_buffers_are_warm() {
    let mut source: Vec<Keypoints> = vec![Keypoints::default(); 2];
    for (camera, keypoints) in source.iter_mut().enumerate() {
        for index in 0..400u64 {
            keypoints.ids.push(slam_rs::types::KeypointId(index));
            keypoints
                .transforms
                .push(&AffineCompact2f::at(nalgebra::Vector2::new(
                    index as f32,
                    camera as f32,
                )));
            keypoints.responses.push(index as f32);
        }
    }

    // Warm-up: the first copy is allowed to allocate, and does.
    let mut destination: Vec<Keypoints> = source.clone();
    let (_, warm_up) = measure(|| destination.clone_from(&source));
    assert_eq!(destination, source);

    // Steady state, both directions and a shorter source, which must reuse the
    // buffer rather than shrink and regrow it.
    let (_, again) = measure(|| destination.clone_from(&source));
    assert_eq!(
        again.total(),
        0,
        "a warm snapshot cost {again:?} (the first copy cost {warm_up:?})"
    );

    let mut shorter: Vec<Keypoints> = source.clone();
    for keypoints in &mut shorter {
        keypoints.ids.truncate(100);
        keypoints.responses.truncate(100);
    }
    let (_, shrunk) = measure(|| destination.clone_from(&shorter));
    assert_eq!(
        shrunk.total(),
        0,
        "copying a shorter camera cost {shrunk:?}"
    );

    let (_, regrown) = measure(|| destination.clone_from(&source));
    assert_eq!(regrown.total(), 0, "copying back cost {regrown:?}");

    // The occupancy counts travel with them, through `Vec<Vec<i32>>`.
    let cells: Vec<Vec<i32>> = vec![vec![3; 441]; 2];
    let mut cells_copy: Vec<Vec<i32>> = cells.clone();
    let (_, counts) = measure(|| cells_copy.clone_from(&cells));
    assert_eq!(counts.total(), 0, "copying the cell counts cost {counts:?}");
}

/// `FlowTransforms` on its own, since it is the one nested a level down and the
/// one a derived `clone_from` would have replaced silently.
#[test]
fn copying_the_warp_arrays_costs_nothing_once_warm() {
    let mut source: FlowTransforms = FlowTransforms::default();
    for index in 0..500 {
        source.push(&AffineCompact2f::at(nalgebra::Vector2::new(
            index as f32,
            -(index as f32),
        )));
    }
    let mut destination: FlowTransforms = source.clone();
    let (_, _warm) = measure(|| destination.clone_from(&source));
    let (_, steady) = measure(|| destination.clone_from(&source));
    assert_eq!(steady.total(), 0, "a warm copy cost {steady:?}");
    assert_eq!(destination, source);
}

/// The structural bound on one frame's allocator calls, from the band scans.
///
/// The detector scans one whole-width band per **cell row** per rung of the
/// threshold ladder — every cell of a grid row filters its own columns out of
/// the same band, see `Band` in `frontend::detect` — over one detection pass per
/// camera. Each scan costs one `Vec` per row of the band, allocated and freed
/// (`features/fast.rs:465`, the `row_cap` buffer), plus the one
/// `fast_detect_rect_u8` returns (`features/cells.rs:141`); 512 is the slack for
/// everything else a frame touches.
///
/// The count is `cameras x cell rows x rungs x rows-per-band`, which is the
/// **worst** frame: the one where no cell ever fills, so every row is scanned at
/// every rung. A textured frame scans far fewer bands, which is why the flat
/// frame below is the one this bounds tightly. A detector that scanned per
/// **cell** instead — the shape this replaced — would multiply the band term by
/// `grid.columns` and fail it.
fn band_scan_bound(grid: &CellGrid, cameras: usize) -> usize {
    let rows: usize = (grid.y_stop - grid.y_start) / grid.cell + 1;
    // 40, 20, 10, 5.
    let ladder: usize = 4;
    let rows_per_band: usize = grid.cell - 2 * 3;
    cameras * rows * ladder * (rows_per_band + 1) * 2 + 512
}

/// What a whole steady-state frame costs, and where that goes.
///
/// A frame is **not** allocation-free and is not claimed to be. Every one of
/// those allocations is kornia's, not the port's, and [`band_scan_bound`] is
/// where they come from. A *textured* frame here costs a few hundred, because
/// most cells are full and never ask for a band; the flat frame in the next test
/// asks for all of them and costs a few thousand.
///
/// The next test is the attribution: a frame that finds **no keypoints at all**
/// costs more than a textured one, so nothing on the port's own per-frame path —
/// the snapshot, the keypoint arrays, the patch storage — is the source.
///
/// Fixing it means an upstream change (a `fast_detect_rect_u8_into` taking a
/// caller buffer) or forking ~200 lines of kornia; it is recorded rather than
/// papered over. What this test asserts is the **structural bound**, so a new
/// per-keypoint or per-patch allocation on the frame path shows up against it.
#[test]
fn a_steady_state_frame_reports_its_allocation_count() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> =
        FrameToFrameOpticalFlow::new(flow_config(), &flow_rig(2), FrontendOptions::default())
            .unwrap();
    let frames: Vec<[ImageU16; 2]> = (0..6)
        .map(|step| [dotted_image(step), dotted_image(step)])
        .collect();

    // Warm-up: three frames is enough for every buffer to reach its high-water
    // mark on this scene.
    for (step, images) in frames.iter().enumerate().take(3) {
        flow.process_frame(step as i64, images, &PosePrediction::default(), &[])
            .unwrap();
    }

    let grid: CellGrid = flow.occupancy_grid();
    let bound: usize = band_scan_bound(&grid, 2);

    let mut counts: Vec<usize> = Vec::new();
    for (step, images) in frames.iter().enumerate().skip(3) {
        let (_, counted) = measure(|| {
            flow.process_frame(step as i64, images, &PosePrediction::default(), &[])
                .unwrap();
        });
        println!("steady-state frame {step}: {counted:?}");
        counts.push(counted.total());
    }

    assert!(
        counts.iter().all(|count| *count <= bound),
        "a steady-state frame reached the allocator {counts:?} times, over the \
         structural bound of {bound} for {} cell rows",
        (grid.y_stop - grid.y_start) / grid.cell + 1
    );
}

/// The same frame over a flat image, which yields no keypoints at all.
///
/// If the per-frame allocations came from the keypoints — growing the id, warp
/// and response arrays, or the snapshot copying them — this would be near zero.
/// It is not: it is the same order as the textured frame, which is what pins the
/// cost on the detector's band scans rather than on anything the port owns.
///
/// This frame is also the one [`band_scan_bound`] is tight on: no cell ever
/// fills, so every cell row is scanned at every rung of the ladder and the count
/// is the whole band term. That is what makes the upper bound here a regression
/// test and the textured one only a ceiling — a per-cell scan multiplies this
/// frame's count by the grid's column count and fails.
#[test]
fn a_frame_that_finds_nothing_costs_the_same_order() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> =
        FrameToFrameOpticalFlow::new(flow_config(), &flow_rig(2), FrontendOptions::default())
            .unwrap();
    let flat: ImageU16 = {
        let mut image: ImageU16 = ImageU16::zeros(common::FLOW_WIDTH, common::FLOW_HEIGHT).unwrap();
        for y in 0..common::FLOW_HEIGHT {
            for x in 0..common::FLOW_WIDTH {
                image.set(x, y, 100u16 << 8);
            }
        }
        image
    };
    let images: [ImageU16; 2] = [flat.clone(), flat];

    for step in 0..3 {
        flow.process_frame(step, &images, &PosePrediction::default(), &[])
            .unwrap();
    }
    assert!(
        flow.frame().cameras.iter().all(|camera| camera.is_empty()),
        "the flat image was supposed to yield no keypoints"
    );

    let (_, counted) = measure(|| {
        flow.process_frame(3, &images, &PosePrediction::default(), &[])
            .unwrap();
    });
    println!("flat frame, zero keypoints: {counted:?}");
    assert!(
        counted.total() > 1_000,
        "a frame with no keypoints at all cost only {counted:?}, so the \
         detector's band scans are not the dominant source after all"
    );
    let bound: usize = band_scan_bound(&flow.occupancy_grid(), 2);
    assert!(
        counted.total() <= bound,
        "the frame that scans every band reached the allocator {} times, over \
         the structural bound of {bound}: the detector is scanning more than one \
         band per cell row and rung",
        counted.total()
    );
}

/// The restore path is free: a frame that fails half way costs no more than one
/// that succeeds.
///
/// The failure is injected through a `PatchTracker` that refuses its n-th call,
/// written against the public seam, so the frontend takes the snapshot, runs
/// part of the tracking, and then puts everything back.
#[test]
fn a_restored_frame_costs_no_more_than_a_successful_one() {
    use slam_rs::frontend::parallel::WorkPool;
    use slam_rs::frontend::tracker::{CpuPatchTracker, PatchSoA, PatchTracker, TrackerError};
    use slam_rs::pyramid::{CpuPyramidBuilder, PyramidU16};

    #[derive(Debug)]
    struct FailingTracker {
        inner: CpuPatchTracker<Pattern51>,
        calls: usize,
        fail_from: usize,
    }

    impl PatchTracker for FailingTracker {
        fn batch(&self) -> &slam_rs::frontend::tracker::TrackBatch {
            self.inner.batch()
        }
        fn batch_mut(&mut self) -> &mut slam_rs::frontend::tracker::TrackBatch {
            self.inner.batch_mut()
        }

        type Pattern = Pattern51;
        type Pyramid = PyramidU16;
        type Patches = PatchSoA<Pattern51>;

        fn capacity(&self) -> usize {
            self.inner.capacity()
        }

        fn num_levels(&self) -> usize {
            self.inner.num_levels()
        }

        fn make_patches(&self) -> Result<PatchSoA<Pattern51>, TrackerError> {
            self.inner.make_patches()
        }

        fn submit_prepared(
            &mut self,
            prev: &PyramidU16,
            next: &PyramidU16,
            patches: &PatchSoA<Pattern51>,
            transforms_in: &FlowTransforms,
        ) -> Result<usize, TrackerError> {
            self.calls += 1;
            if self.calls >= self.fail_from {
                return Err(TrackerError::CapacityExceeded {
                    offered: usize::MAX,
                    capacity: 0,
                });
            }
            self.inner
                .submit_prepared(prev, next, patches, transforms_in)
        }
    }

    let options: FrontendOptions = FrontendOptions::default();
    let configuration: VioConfig = flow_config();
    let inner: CpuPatchTracker<Pattern51> = CpuPatchTracker::new(
        options.max_keypoints,
        configuration.optical_flow_levels as usize + 1,
        configuration.optical_flow_max_iterations as usize,
        configuration.optical_flow_max_recovered_dist2,
        WorkPool::new(options.threads).unwrap(),
    )
    .unwrap();
    // Frame 1 makes one call and each later frame three, so failing from call 8
    // lets four frames through and then refuses every frame after.
    let mut flow = FrameToFrameOpticalFlow::with_backends(
        configuration,
        &flow_rig(2),
        options,
        CpuPyramidBuilder::new(),
        FailingTracker {
            inner,
            calls: 0,
            fail_from: 8,
        },
        Box::new(CpuCornerScan::default()),
    )
    .unwrap();

    let frames: Vec<[ImageU16; 2]> = (0..8)
        .map(|step| [dotted_image(step), dotted_image(step)])
        .collect();
    for (step, images) in frames.iter().enumerate().take(3) {
        flow.process_frame(step as i64, images, &PosePrediction::default(), &[])
            .unwrap();
    }
    let before: FlowFrame = flow.frame().clone();

    // The first frame that fails, and then two more, all restoring.
    let mut counts: Vec<usize> = Vec::new();
    for (step, images) in frames.iter().enumerate().skip(3).take(3) {
        let (outcome, counted) = measure(|| {
            flow.process_frame(step as i64, images, &PosePrediction::default(), &[])
                .map(|_| ())
        });
        assert!(outcome.is_err(), "frame {step} was supposed to fail");
        println!("restored frame {step}: {counted:?}");
        counts.push(counted.total());
    }
    assert_eq!(flow.frame(), &before, "a restored frame changed the state");

    // The later restores, with every buffer warm, must not reach the allocator
    // at all: the snapshot and the restore are the only per-frame copies left on
    // that path once tracking has been refused.
    assert_eq!(
        counts[2], 0,
        "a warm restore cost {counts:?} allocator calls"
    );
}

// ── the estimator's per-frame path ────────────────────────────────────────

/// Framesets the oracle fixture covers; the estimator walks all of them.
const ESTIMATOR_FRAMESETS: usize = 60;

/// Framesets before the first measurement.
///
/// `opt_started` flips at frameset 4, the second keyframe arrives at 7 and the
/// prior reaches its 22x27 shape at 9, so by 20 the window is at its steady
/// shape and every scratch buffer has reached its high-water mark.
const ESTIMATOR_WARMUP: usize = 20;

/// One estimator over the oracle fixture's calibration and config, with the
/// whole inertial window pushed in advance — `vio_oracle::window`, which the
/// integration tests cannot share because each has its own binary.
fn estimator_window() -> slam_rs::estimator::SqrtKeypointVio<f32> {
    let mut estimator: slam_rs::estimator::SqrtKeypointVio<f32> =
        slam_rs::estimator::SqrtKeypointVio::with_default_gravity(
            common::calibration().cast(),
            common::config(),
        )
        .unwrap();
    for row in common::IMU.iter() {
        estimator.push_imu(slam_rs::imu::ImuSample {
            t_ns: row.t_ns,
            gyro: nalgebra::Vector3::from(row.gyro),
            accel: nalgebra::Vector3::from(row.accel),
        });
    }
    estimator
}

/// The dense reduction over a workspace that persists: zero, after the first.
///
/// This is the estimator's hottest allocation site made visible on its own.
/// `get_dense_h_b` runs **once per inner Levenberg-Marquardt step** — two to
/// eight times per frameset on this window, seven on the median MIO10 frame —
/// and it used to take a fresh accumulator, a fresh subtree partial per
/// recursion depth and a fresh leaf transpose every time: on a 55-landmark
/// window seven `opt_size`-square matrices, or 38 allocations and their frees,
/// per step. They are one `DenseHbWorkspace` now, and the assertion is that
/// the second call over the same workspace reaches the allocator zero times.
///
/// The problem below is synthetic and small — three frames, twelve landmarks,
/// each seen in every frame — because the count being asserted is zero and
/// zero does not depend on the window's size. What it does need is a window
/// whose reduction actually recurses, so the subtree partials at every depth
/// are allocated on the first call and reused on the second.
#[test]
fn the_dense_reduction_allocates_nothing_after_its_first_call() {
    let mut calibration: slam_rs::calib::Calibration<f32> = common::calibration().cast::<f32>();
    calibration.t_i_c.truncate(2);
    calibration.intrinsics.truncate(2);
    let mut estimator: slam_rs::ba_base::BundleAdjustmentBase<f32> =
        slam_rs::ba_base::BundleAdjustmentBase::new(calibration, 1.0, 1.0).unwrap();

    let frames: [i64; 3] = [0, 1, 2];
    let mut aom: slam_rs::types::AbsOrderMap = slam_rs::types::AbsOrderMap::new();
    for (index, &t_ns) in frames.iter().enumerate() {
        let pose: slam_rs::lie::Se3<f32> = slam_rs::lie::Se3::new(
            slam_rs::lie::So3::identity(),
            nalgebra::Vector3::new(0.01 * index as f32, 0.0, 0.0),
        );
        estimator.frame_poses.insert(
            t_ns,
            slam_rs::types::PoseStateWithLin::new(t_ns, pose, false),
        );
        aom.push(t_ns, slam_rs::types::POSE_SIZE).unwrap();
    }

    let host: slam_rs::types::TimeCamId = slam_rs::types::TimeCamId::new(frames[0], 0);
    for index in 0..12u64 {
        let id: slam_rs::types::LandmarkId = slam_rs::types::LandmarkId(index);
        let direction: nalgebra::Vector2<f32> =
            nalgebra::Vector2::new(0.01 * index as f32 - 0.05, -0.02);
        let landmark: slam_rs::landmark::Landmark<f32> =
            slam_rs::landmark::Landmark::new(id, host, direction, 0.25);
        estimator.lmdb.add_landmark(id, &landmark);
        for &t_ns in &frames {
            estimator
                .lmdb
                .add_observation(
                    slam_rs::types::TimeCamId::new(t_ns, 0),
                    id,
                    nalgebra::Vector2::new(500.0 + index as f32, 510.0),
                )
                .unwrap();
        }
    }

    let inputs: slam_rs::linearize::LinearizationInputs<'_, f32> = Default::default();
    let mut lqr: slam_rs::linearize::LinearizationAbsQR<f32> =
        slam_rs::linearize::LinearizationAbsQR::new(
            &estimator,
            &aom,
            slam_rs::linearize::LinearizationOptions::default(),
            &inputs,
        )
        .unwrap();
    lqr.linearize_problem(&estimator, &inputs).unwrap();
    lqr.perform_qr().unwrap();

    let mut workspace: slam_rs::linearize::DenseHbWorkspace<f32> = Default::default();
    // The first call is allowed to allocate, and does: this is where every
    // buffer reaches its size.
    let (_, first) = measure(|| {
        let _ = lqr
            .get_dense_h_b_into(&estimator, &inputs, &mut workspace)
            .unwrap();
    });
    for repeat in 0..3 {
        let (_, again) = measure(|| {
            let _ = lqr
                .get_dense_h_b_into(&estimator, &inputs, &mut workspace)
                .unwrap();
        });
        assert_eq!(
            again.total(),
            0,
            "call {} over a warm workspace cost {again:?} (the first cost {first:?})",
            repeat + 2,
        );
    }
    println!("dense reduction: first call {first:?}, then zero");
}

/// What one steady-state frameset costs the estimator, and how that cost grows
/// with the number of Levenberg-Marquardt steps it takes.
///
/// A frameset is **not** allocation-free and is not claimed to be: the landmark
/// database is `BTreeMap`-shaped and every new observation can split a node,
/// the LM trail is a `Vec<LmIteration>` inside a boxed `FrameStats`, and the
/// marginalization builds its own square-root system — which on this window is
/// every steady-state frameset, all forty of them marginalize. Measured here:
/// 805 to 1,389 allocator calls, and the bound below is that with room.
///
/// The second assertion is the one that measures **this** change. The inner LM
/// loop runs two to eight times per frameset (seven on the median MIO10 frame)
/// and each pass used to take a fresh set of buffers:
///
/// | per inner step | allocations |
/// |---|---:|
/// | the dense reduction's accumulator (`DMatrix`, `DVector`, two `Vec`s) | 4 |
/// | one subtree partial per recursion depth, `ceil(log2 55) = 6` | 24 |
/// | the depth vector itself, grown to six | ~3 |
/// | the leaf transpose buffers | 2 |
/// | `h.clone()` per damping attempt | 1 |
/// | `EigenLdlt`'s transpositions, `temp` and accumulator | 3 |
/// | the right-hand side clone inside the solve | 1 |
///
/// All of them are now buffers the estimator owns and resets. Measured on this
/// fixture, with the layout commits in and the pooling out (`9a248147`) and
/// then with the pooling in: **126.8 allocator calls per LM step and 1,531 per
/// frameset, down to 59.5 and 1,133**. What is left per step is
/// `compute_delta` and the prior's `H · delta`, once each, and the IMU blocks.
///
/// The gate is a slope as well as a total, because the total is dominated by
/// the per-frame database work: a per-step buffer coming back moves the slope
/// long before it is visible in a mean.
#[test]
fn the_estimators_per_frame_cost_does_not_grow_with_the_lm_step_count() {
    /// Measured on this fixture: 805 to 1,389 allocator calls per frameset.
    const BOUND: usize = 1_600;
    /// Measured slope: 59.5 allocator calls per LM step, against 126.8 before
    /// the buffers were pooled, so this sits well inside the gap.
    const CALLS_PER_STEP: f64 = 85.0;

    let flow: Vec<std::sync::Arc<slam_rs::estimator::FlowObservations>> = common::ORACLE
        .flow
        .iter()
        .take(ESTIMATOR_FRAMESETS)
        .map(common::observations)
        .collect();
    let mut estimator = estimator_window();
    for frame in flow.iter().take(ESTIMATOR_WARMUP) {
        estimator
            .process_frame(std::sync::Arc::clone(frame))
            .unwrap();
    }

    let mut measured: Vec<(f64, f64)> = Vec::new();
    for frame in flow.iter().skip(ESTIMATOR_WARMUP) {
        let (outcome, counted) = measure(|| {
            estimator
                .process_frame(std::sync::Arc::clone(frame))
                .unwrap()
        });
        let slam_rs::estimator::FrameOutcome::Measured(stats) = outcome else {
            panic!("the fixture's inertial window is complete, so every frameset measures");
        };
        println!(
            "frameset {}: {} LM steps, marginalized {}, {counted:?}",
            stats.t_ns,
            stats.lm.len(),
            stats.marginalization.is_some(),
        );
        assert!(
            counted.total() <= BOUND,
            "a steady-state frameset with {} LM steps reached the allocator {counted:?} times, \
             over the structural bound of {BOUND}",
            stats.lm.len(),
        );
        measured.push((stats.lm.len() as f64, counted.total() as f64));
    }

    // Least squares on (LM steps, allocator calls). The fixture has to spread
    // the step count or the slope is not identified: without that spread the
    // total bound above would hold for a per-step allocation too.
    let steps: Vec<f64> = measured.iter().map(|(x, _)| *x).collect();
    let low: f64 = steps.iter().copied().fold(f64::INFINITY, f64::min);
    let high: f64 = steps.iter().copied().fold(0.0, f64::max);
    assert!(
        high >= low + 4.0,
        "the measured framesets took {low} to {high} LM steps, too narrow to \
         tell a per-step allocation from a per-frame one"
    );
    let count: f64 = measured.len() as f64;
    let mean_x: f64 = steps.iter().sum::<f64>() / count;
    let mean_y: f64 = measured.iter().map(|(_, y)| *y).sum::<f64>() / count;
    let covariance: f64 = measured
        .iter()
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>();
    let variance: f64 = steps
        .iter()
        .map(|x| (x - mean_x) * (x - mean_x))
        .sum::<f64>();
    let slope: f64 = covariance / variance;
    println!(
        "{count} framesets, {low}-{high} LM steps: {slope:.1} allocator calls per step, \
         {mean_y:.0} mean per frameset"
    );
    assert!(
        slope <= CALLS_PER_STEP,
        "the estimator reaches the allocator {slope:.1} times per LM step, over the \
         {CALLS_PER_STEP} the per-step path is allowed: the inner LM step allocates \
         noticeably more than it did (this is a bulk gate; one small buffer can hide under it)"
    );
}
