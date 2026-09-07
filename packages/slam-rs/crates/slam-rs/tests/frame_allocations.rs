//! What the frontend's per-frame path asks of the allocator, counted.
//!
//! `cubecl-portability.md` §12.2 asks for no allocation on the per-frame path,
//! because a GPU kernel cannot grow a buffer and because an allocator call
//! inside the released-GIL region is a latency spike nobody sees. The claim was
//! made once and turned out to be false — `#[derive(Clone)]` writes only
//! `clone`, so `clone_from` fell back to `*self = source.clone()` and the
//! keypoint snapshot replaced sixteen buffers per stereo frame — so it is
//! measured here rather than asserted in a comment.
//!
//! The measurement is a counting global allocator over this test binary alone.
//! It is `#[cfg(test)]` only in the sense that this is a test target: the shipped
//! library never sees it. Counting is gated by a flag held under a mutex, so two
//! tests in this binary cannot pollute each other's numbers.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use nalgebra::Vector3;

use slam_rs::calib::{CalibAccelBias, CalibGyroBias, Calibration, CameraModel, PinholeParams};
use slam_rs::config::{MatchingGuessType, VioConfig};
use slam_rs::frontend::flow::{
    FlowFrame, FrameToFrameOpticalFlow, FrontendOptions, Keypoints, PosePrediction,
};
use slam_rs::frontend::patterns::Pattern51;
use slam_rs::frontend::se2::AffineCompact2f;
use slam_rs::frontend::tracker::FlowTransforms;
use slam_rs::image::ImageU16;
use slam_rs::lie::{Se3, So3};

// ── the counting allocator ────────────────────────────────────────────────

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static REALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static COUNTING: AtomicBool = AtomicBool::new(false);
/// Only one measurement at a time, so the harness's other test threads cannot
/// land inside a counted region.
static MEASURING: Mutex<()> = Mutex::new(());

struct Counting;

// SAFETY: every method forwards to `System` with the layout it was given and
// changes nothing about the returned pointer; the counters are the only
// addition, and they are plain relaxed atomics.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if COUNTING.load(Ordering::Relaxed) {
            DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            REALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
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

/// Run `body` with the allocator counting, and report what it cost.
fn measure<T>(body: impl FnOnce() -> T) -> (T, Allocations) {
    let guard = MEASURING.lock().unwrap_or_else(|error| error.into_inner());
    ALLOCATIONS.store(0, Ordering::Relaxed);
    REALLOCATIONS.store(0, Ordering::Relaxed);
    DEALLOCATIONS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    let value: T = body();
    COUNTING.store(false, Ordering::Relaxed);
    let counted: Allocations = Allocations {
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        reallocations: REALLOCATIONS.load(Ordering::Relaxed),
        deallocations: DEALLOCATIONS.load(Ordering::Relaxed),
    };
    drop(guard);
    (value, counted)
}

// ── the frontend under test ───────────────────────────────────────────────

const WIDTH: usize = 200;
const HEIGHT: usize = 200;

fn rig(count: usize) -> Calibration<f64> {
    let intrinsics: CameraModel<f64> = CameraModel::Pinhole(PinholeParams {
        fx: 180.0,
        fy: 180.0,
        cx: WIDTH as f64 / 2.0,
        cy: HEIGHT as f64 / 2.0,
    });
    Calibration {
        t_i_c: (0..count)
            .map(|index| Se3::new(So3::identity(), Vector3::new(0.05 * index as f64, 0.0, 0.0)))
            .collect(),
        intrinsics: vec![intrinsics; count],
        resolution: vec![[WIDTH as u32, HEIGHT as u32]; count],
        vignette: Vec::new(),
        cam_time_offset_ns: 0,
        calib_accel_bias: CalibAccelBias::default(),
        calib_gyro_bias: CalibGyroBias::default(),
        imu_update_rate: 200.0,
        gyro_noise_std: Vector3::repeat(1e-4),
        accel_noise_std: Vector3::repeat(1e-3),
        gyro_bias_std: Vector3::repeat(1e-5),
        accel_bias_std: Vector3::repeat(1e-4),
        unknown: BTreeMap::new(),
    }
}

fn config() -> VioConfig {
    VioConfig {
        optical_flow_matching_guess_type: MatchingGuessType::SamePixel,
        ..VioConfig::default()
    }
}

/// Bright squares on a gently varying background, shifted by `shift` pixels.
fn dotted_image(shift: i32) -> ImageU16 {
    let mut image: ImageU16 = ImageU16::zeros(WIDTH, HEIGHT).unwrap();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let fx: f64 = f64::from(x as i32 - shift);
            let background: f64 = 60.0 + 25.0 * (fx * 0.09).sin() * (y as f64 * 0.07).cos();
            image.set(x, y, (background as u16) << 8);
        }
    }
    let mut cy: usize = 14;
    while cy + 5 < HEIGHT {
        let mut cx: usize = 14;
        while cx + 5 < WIDTH {
            for dy in 0..5 {
                for dx in 0..5 {
                    let x: i32 = (cx + dx) as i32 + shift;
                    if x >= 0 && (x as usize) < WIDTH {
                        image.set(x as usize, cy + dy, 200u16 << 8);
                    }
                }
            }
            cx += 17;
        }
        cy += 17;
    }
    image
}

/// The snapshot's exact shape: one `Keypoints` per camera, copied in place.
///
/// This is the operation the review measured at sixteen allocations and sixteen
/// frees per call. Every type in the chain — `Vec`, `Keypoints` and
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

/// What a whole steady-state frame costs, and where that goes.
///
/// A frame is **not** allocation-free and is not claimed to be. Every one of
/// those allocations is kornia's, not the port's: `fast_detect_rect_u8` returns a
/// fresh `Vec<FastCorner>` (`features/cells.rs:141`) and the row kernel it calls
/// allocates one `Vec` **per image row** of the region
/// (`features/fast.rs:465`, the `row_cap` buffer). The detector runs that once
/// per grid cell per rung of the threshold ladder, over two detection passes, so
/// the count is `cells x rungs x rows-per-cell x passes` and depends on how many
/// cells fall all the way down the ladder — which is why a *textured* frame here
/// costs a few hundred and the flat frame in the next test costs a few thousand.
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
        FrameToFrameOpticalFlow::new(config(), &rig(2), FrontendOptions::default()).unwrap();
    let frames: Vec<[ImageU16; 2]> = (0..6)
        .map(|step| [dotted_image(step), dotted_image(step)])
        .collect();

    // Warm-up: three frames is enough for every buffer to reach its high-water
    // mark on this scene.
    for (step, images) in frames.iter().enumerate().take(3) {
        flow.process_frame(step as i64, images, &PosePrediction::default(), &[])
            .unwrap();
    }

    let grid = flow.occupancy_grid();
    let cells: usize = ((grid.x_stop - grid.x_start) / grid.cell + 1)
        * ((grid.y_stop - grid.y_start) / grid.cell + 1);
    // 40, 20, 10, 5.
    let ladder: usize = 4;
    // One `Vec` per row of the cell's inner region, plus the returned one, each
    // allocated and freed, over two detection passes.
    let rows_per_cell: usize = grid.cell - 6;
    let bound: usize = cells * ladder * (rows_per_cell + 1) * 2 * 2 + 512;

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
         structural bound of {bound} for {cells} cells"
    );
}

/// The same frame over a flat image, which yields no keypoints at all.
///
/// If the per-frame allocations came from the keypoints — growing the id, warp
/// and response arrays, or the snapshot copying them — this would be near zero.
/// It is not: it is the same order as the textured frame, which is what pins the
/// cost on the detector's per-cell `Vec` rather than on anything the port owns.
#[test]
fn a_frame_that_finds_nothing_costs_the_same_order() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> =
        FrameToFrameOpticalFlow::new(config(), &rig(2), FrontendOptions::default()).unwrap();
    let flat: ImageU16 = {
        let mut image: ImageU16 = ImageU16::zeros(WIDTH, HEIGHT).unwrap();
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
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
         detector's per-cell allocation is not the dominant source after all"
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
    use slam_rs::frontend::tracker::{
        CpuPatchTracker, FlowResult, PatchSoA, PatchTracker, TrackerError,
    };
    use slam_rs::pyramid::{CpuPyramidBuilder, PyramidU16};

    #[derive(Debug)]
    struct FailingTracker {
        inner: CpuPatchTracker<Pattern51>,
        calls: usize,
        fail_from: usize,
    }

    impl PatchTracker for FailingTracker {
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

        fn track(
            &mut self,
            prev: &PyramidU16,
            next: &PyramidU16,
            patches: &PatchSoA<Pattern51>,
            transforms_in: &FlowTransforms,
            out: &mut FlowResult,
        ) -> Result<(), TrackerError> {
            self.calls += 1;
            if self.calls >= self.fail_from {
                return Err(TrackerError::CapacityExceeded {
                    offered: usize::MAX,
                    capacity: 0,
                });
            }
            self.inner.track(prev, next, patches, transforms_in, out)
        }
    }

    let options: FrontendOptions = FrontendOptions::default();
    let configuration: VioConfig = config();
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
        &rig(2),
        options,
        CpuPyramidBuilder::new(),
        FailingTracker {
            inner,
            calls: 0,
            fail_from: 8,
        },
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
