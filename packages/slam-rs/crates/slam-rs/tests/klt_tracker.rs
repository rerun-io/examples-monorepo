//! The KLT tracker's own tests: the CPU patch set, the warp arrays and
//! `track_point`'s sub-pixel recovery.
//!
//! Moved out of `src/frontend/tracker.rs` (S25). The wave table below is the
//! one these tests were measured against — the 0.01 px recovery bound is
//! stated against *this* field, not `tests/common`'s twelve-wave one — so it
//! moved with them rather than being swapped for the shared generator.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use nalgebra::{Matrix2, Vector2};
use proptest::prelude::*;
use slam_rs::frontend::parallel::WorkPool;
use slam_rs::frontend::patterns::{Pattern, Pattern51};
use slam_rs::frontend::se2::AffineCompact2f;
use slam_rs::frontend::tracker::*;
use slam_rs::image::ImageU16;
use slam_rs::pyramid::PyramidU16;

mod common;

use common::pyramid_of;

/// A band-limited texture: twelve plane waves with wavelengths between 16
/// and 56 pixels, in fixed pseudo-random directions and phases.
///
/// Band-limited matters twice over. Below the Nyquist of the finest pyramid
/// level the `[1,4,6,4,1]` subsample does not alias, so the coarse levels
/// really do carry the same shift; and away from the sampling limit bilinear
/// interpolation reconstructs the field closely, so the residual's fixed
/// point sits near the true shift rather than a fraction of a pixel off it.
/// Twelve components in different directions also keep every patch's `H_se2`
/// well conditioned: a single wave, or a field that is locally almost affine,
/// is the aperture problem and no tracker recovers a shift from it.
fn texture(x: f64, y: f64) -> f64 {
    // (wavelength, direction in turns, phase in turns)
    const WAVES: [(f64, f64, f64); 16] = [
        (22.0000, 0.000000, 0.000000),
        (23.5218, 0.381966, 0.618034),
        (25.1489, 0.763932, 0.236068),
        (26.8886, 0.145898, 0.854102),
        (28.7486, 0.527864, 0.472136),
        (30.7373, 0.909830, 0.090170),
        (32.8635, 0.291796, 0.708204),
        (35.1368, 0.673762, 0.326238),
        (37.5674, 0.055728, 0.944272),
        (40.1661, 0.437694, 0.562306),
        (42.9446, 0.819660, 0.180340),
        (45.9153, 0.201626, 0.798374),
        (49.0914, 0.583592, 0.416408),
        (52.4873, 0.965558, 0.034442),
        (56.1181, 0.347524, 0.652476),
        (60.0000, 0.729490, 0.270510),
    ];
    let tau: f64 = std::f64::consts::TAU;
    let mut sum: f64 = 0.0;
    for (wavelength, direction, phase) in WAVES {
        let angle: f64 = tau * direction;
        let projection: f64 = x * angle.cos() + y * angle.sin();
        sum += (tau * (projection / wavelength + phase)).sin();
    }
    sum / WAVES.len() as f64
}

/// A textured frame, shifted by `(dx, dy)`: the same continuous field
/// resampled at `(x - dx, y - dy)`, so the shift is exact by construction.
fn shifted_image(width: usize, height: usize, dx: f32, dy: f32) -> ImageU16 {
    let mut image: ImageU16 = ImageU16::zeros(width, height).unwrap();
    for y in 0..height {
        for x in 0..width {
            let fx: f64 = f64::from(x as f32 - dx);
            let fy: f64 = f64::from(y as f32 - dy);
            let value: f64 = 32_000.0 + 28_000.0 * texture(fx, fy);
            image.set(x, y, value as u16);
        }
    }
    image
}

struct Fixture {
    prev: PyramidU16,
    next: PyramidU16,
    patches: PatchSoA<Pattern51>,
    transforms: FlowTransforms,
    positions: PointsSoA,
}

fn fixture(dx: f32, dy: f32, levels: usize) -> Fixture {
    let base: ImageU16 = shifted_image(160, 160, 0.0, 0.0);
    let moved: ImageU16 = shifted_image(160, 160, dx, dy);
    let prev: PyramidU16 = pyramid_of(&base, levels);
    let next: PyramidU16 = pyramid_of(&moved, levels);

    let mut positions: PointsSoA = PointsSoA::default();
    for y in (40..120).step_by(16) {
        for x in (40..120).step_by(16) {
            positions.push(Vector2::new(x as f32, y as f32));
        }
    }
    let mut transforms: FlowTransforms = FlowTransforms::default();
    for index in 0..positions.len() {
        transforms.push(&AffineCompact2f::at(positions.get(index)));
    }

    let mut patches: PatchSoA<Pattern51> = PatchSoA::new(positions.len(), levels + 1).unwrap();
    patches.build(&prev, &positions, None).unwrap();

    Fixture {
        prev,
        next,
        patches,
        transforms,
        positions,
    }
}

fn tracker(capacity: usize, levels: usize, threads: usize) -> CpuPatchTracker<Pattern51> {
    CpuPatchTracker::new(
        capacity,
        levels + 1,
        5,
        0.04,
        WorkPool::new(threads).unwrap(),
    )
    .unwrap()
}

#[test]
fn an_integer_shift_is_recovered() {
    let levels: usize = 3;
    let scene: Fixture = fixture(2.0, -1.0, levels);
    let mut tracker: CpuPatchTracker<Pattern51> = tracker(scene.positions.len(), levels, 1);
    let mut out: FlowResult = FlowResult::with_capacity(scene.positions.len());
    tracker
        .track(
            &scene.prev,
            &scene.next,
            &scene.patches,
            &scene.transforms,
            &mut out,
        )
        .unwrap();

    assert!(
        out.len() >= scene.positions.len() / 2,
        "tracked {}",
        out.len()
    );
    for index in out.tracked() {
        let index: usize = *index as usize;
        let moved: Vector2<f32> = out.transform(index).translation - scene.positions.get(index);
        // An integer shift moves the samples themselves, so the two
        // bilinear reconstructions are exact translates of each other and
        // the residual's fixed point is the true shift.
        assert!(
            (moved.x - 2.0).abs() < 0.01 && (moved.y + 1.0).abs() < 0.01,
            "patch {index} moved by {moved:?}, expected (2, -1)"
        );
    }
}

/// A sub-pixel shift, up to the pattern's own radius.
///
/// The tolerance is not the tracker's convergence — it converges to five
/// decimal places in three iterations — but the **bias of the fixed point
/// itself**. `interp` reconstructs the image bilinearly and `interpGrad`
/// differentiates that reconstruction by central differences
/// (`image.h:396-469`), so for a shift that is not a whole number of samples
/// the residual vanishes not at the true shift but a little beside it.
///
/// The size of that displacement depends only on the **fractional** part of
/// the shift, not on its magnitude: on this texture an exactly integer shift
/// is recovered to `0.0000` px, a shift of 0.02 px to 0.0004, and a shift of
/// half a pixel to 0.035 on the median patch and 0.13 on the worst — the same
/// numbers whether the shift is 0.5 or 3.5 pixels. Shortening the texture's
/// wavelengths raises the floor and lengthening them makes the patches
/// ill-conditioned instead; basalt's C++ has the same property, because this
/// is its arithmetic. The gate is therefore the median, with a cap on the tail.
fn sub_pixel_shift_error(dx: f32, dy: f32) -> (f32, f32, usize, usize) {
    let levels: usize = 3;
    let scene: Fixture = fixture(dx, dy, levels);
    let mut tracker: CpuPatchTracker<Pattern51> = tracker(scene.positions.len(), levels, 1);
    let mut out: FlowResult = FlowResult::with_capacity(scene.positions.len());
    tracker
        .track(
            &scene.prev,
            &scene.next,
            &scene.patches,
            &scene.transforms,
            &mut out,
        )
        .unwrap();

    let mut errors: Vec<f32> = Vec::new();
    for index in out.tracked() {
        let index: usize = *index as usize;
        let moved: Vector2<f32> = out.transform(index).translation - scene.positions.get(index);
        errors.push((moved.x - dx).abs().max((moved.y - dy).abs()));
    }
    errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median: f32 = errors
        .get(errors.len() / 2)
        .copied()
        .unwrap_or(f32::INFINITY);
    let worst: f32 = errors.last().copied().unwrap_or(f32::INFINITY);
    (median, worst, out.len(), scene.positions.len())
}

#[test]
fn a_sub_pixel_shift_is_recovered() {
    let (median, worst, tracked, total) = sub_pixel_shift_error(0.6, 1.4);
    assert_eq!(tracked, total);
    assert!(median < 0.05, "median error {median}");
    assert!(worst < 0.2, "worst error {worst}");
}

/// The forward-backward gate (`frame_to_frame_optical_flow.h:362-364`) is
/// what rejects a track onto an unrelated image.
#[test]
fn a_mismatched_pair_is_rejected() {
    let levels: usize = 3;
    let scene: Fixture = fixture(0.0, 0.0, levels);
    // A different texture entirely, not a shift of the first.
    let mut other: ImageU16 = ImageU16::zeros(160, 160).unwrap();
    for y in 0..160 {
        for x in 0..160 {
            let value: f64 = 25_000.0
                + 9_000.0 * ((x as f64) * 0.61).cos()
                + 6_000.0 * ((y as f64) * 0.47).sin();
            other.set(x, y, value as u16);
        }
    }
    let unrelated: PyramidU16 = pyramid_of(&other, levels);

    let mut tracker: CpuPatchTracker<Pattern51> = tracker(scene.positions.len(), levels, 1);
    let mut out: FlowResult = FlowResult::with_capacity(scene.positions.len());
    tracker
        .track(
            &scene.prev,
            &unrelated,
            &scene.patches,
            &scene.transforms,
            &mut out,
        )
        .unwrap();
    assert!(
        out.len() * 4 < scene.positions.len(),
        "{} of {} tracks survived an unrelated image",
        out.len(),
        scene.positions.len()
    );
}

#[test]
fn one_thread_and_four_threads_agree_exactly() {
    let levels: usize = 3;
    let scene: Fixture = fixture(1.3, -0.7, levels);

    let mut single: FlowResult = FlowResult::with_capacity(scene.positions.len());
    tracker(scene.positions.len(), levels, 1)
        .track(
            &scene.prev,
            &scene.next,
            &scene.patches,
            &scene.transforms,
            &mut single,
        )
        .unwrap();

    let mut wide: FlowResult = FlowResult::with_capacity(scene.positions.len());
    tracker(scene.positions.len(), levels, 4)
        .track(
            &scene.prev,
            &scene.next,
            &scene.patches,
            &scene.transforms,
            &mut wide,
        )
        .unwrap();

    assert_eq!(single.tracked(), wide.tracked());
    assert!(!single.is_empty());
    for index in single.tracked() {
        let index: usize = *index as usize;
        assert_eq!(single.transform(index), wide.transform(index));
    }
}

#[test]
fn two_runs_of_the_same_tracker_agree_exactly() {
    let levels: usize = 3;
    let scene: Fixture = fixture(0.9, 0.4, levels);
    let mut tracker: CpuPatchTracker<Pattern51> = tracker(scene.positions.len(), levels, 4);

    let mut first: FlowResult = FlowResult::with_capacity(scene.positions.len());
    let mut second: FlowResult = FlowResult::with_capacity(scene.positions.len());
    for out in [&mut first, &mut second] {
        tracker
            .track(
                &scene.prev,
                &scene.next,
                &scene.patches,
                &scene.transforms,
                out,
            )
            .unwrap();
    }
    assert_eq!(first.tracked(), second.tracked());
    for index in first.tracked() {
        let index: usize = *index as usize;
        assert_eq!(first.transform(index), second.transform(index));
    }
}

#[test]
fn more_keypoints_than_capacity_is_refused() {
    let levels: usize = 1;
    let scene: Fixture = fixture(0.0, 0.0, levels);
    let mut tracker: CpuPatchTracker<Pattern51> = tracker(2, levels, 1);
    let mut out: FlowResult = FlowResult::with_capacity(scene.positions.len());
    let error = tracker
        .track(
            &scene.prev,
            &scene.next,
            &scene.patches,
            &scene.transforms,
            &mut out,
        )
        .unwrap_err();
    assert_eq!(
        error,
        TrackerError::CapacityExceeded {
            offered: scene.positions.len(),
            capacity: 2
        }
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(12))]

    /// Any shift up to the pattern's radius (3.5 px for `Pattern51`) is
    /// recovered, with the interpolation bias documented on
    /// [`sub_pixel_shift_error`] as the tolerance.
    #[test]
    fn any_shift_within_the_pattern_radius_is_recovered(
        dx in -3.5f32..3.5,
        dy in -3.5f32..3.5,
    ) {
        let (median, worst, tracked, total) = sub_pixel_shift_error(dx, dy);
        prop_assert!(tracked * 4 >= total * 3, "tracked {tracked} of {total}");
        prop_assert!(median < 0.05, "median error {median} for ({dx}, {dy})");
        prop_assert!(worst < 0.2, "worst error {worst} for ({dx}, {dy})");
    }
}
/// A patch set shallower than the tracker used to index past the end of its
/// validity array; it is a typed error now (decision D32).
#[test]
fn a_shallow_patch_set_is_refused() {
    for threads in [1, 4] {
        let levels: usize = 3;
        let scene: Fixture = fixture(0.0, 0.0, levels);
        let mut shallow: PatchSoA<Pattern51> = PatchSoA::new(scene.positions.len(), 1).unwrap();
        shallow.build(&scene.prev, &scene.positions, None).unwrap();

        let mut tracker: CpuPatchTracker<Pattern51> =
            tracker(scene.positions.len(), levels, threads);
        let mut out: FlowResult = FlowResult::with_capacity(scene.positions.len());
        let error = tracker
            .track(
                &scene.prev,
                &scene.next,
                &shallow,
                &scene.transforms,
                &mut out,
            )
            .unwrap_err();
        assert_eq!(
            error,
            TrackerError::LevelMismatch {
                what: "the patch set",
                expected: levels + 1,
                actual: 1
            }
        );
    }
}

/// A pyramid shallower than the tracker is refused too, on either side.
#[test]
fn a_shallow_pyramid_is_refused() {
    let levels: usize = 3;
    let scene: Fixture = fixture(0.0, 0.0, levels);
    let shallow: PyramidU16 = pyramid_of(&shifted_image(160, 160, 0.0, 0.0), 1);
    let mut tracker: CpuPatchTracker<Pattern51> = tracker(scene.positions.len(), levels, 1);
    let mut out: FlowResult = FlowResult::with_capacity(scene.positions.len());
    let error = tracker
        .track(
            &shallow,
            &scene.next,
            &scene.patches,
            &scene.transforms,
            &mut out,
        )
        .unwrap_err();
    assert_eq!(
        error,
        TrackerError::LevelMismatch {
            what: "the previous pyramid",
            expected: levels + 1,
            actual: 2
        }
    );
}

/// A selection mask shorter than the positions used to index past its end.
#[test]
fn a_short_selection_mask_is_refused() {
    let levels: usize = 1;
    let scene: Fixture = fixture(0.0, 0.0, levels);
    let mut patches: PatchSoA<Pattern51> =
        PatchSoA::new(scene.positions.len(), levels + 1).unwrap();
    let short: Vec<bool> = vec![true; 2];
    let error = patches
        .build(&scene.prev, &scene.positions, Some(&short))
        .unwrap_err();
    assert_eq!(
        error,
        TrackerError::LengthMismatch {
            first_name: "positions",
            first: scene.positions.len(),
            second_name: "selection flags",
            second: 2,
        }
    );
}

/// More positions than the patch storage holds is a typed error too.
#[test]
fn more_positions_than_patch_capacity_is_refused() {
    let levels: usize = 1;
    let scene: Fixture = fixture(0.0, 0.0, levels);
    let mut patches: PatchSoA<Pattern51> = PatchSoA::new(2, levels + 1).unwrap();
    let error = patches
        .build(&scene.prev, &scene.positions, None)
        .unwrap_err();
    assert_eq!(
        error,
        TrackerError::CapacityExceeded {
            offered: scene.positions.len(),
            capacity: 2
        }
    );
}

/// The warp storage is six flat arrays; the round trip through them is exact.
/// A capacity past the ceiling is refused before a byte is allocated.
///
/// `Vec::with_capacity(2^63)` panics with `capacity overflow`, and a panic in
/// here reaches Python as a `PanicException` that ordinary `except Exception`
/// handlers do not catch (decision D32). The ceiling is what makes the
/// request answerable instead.
#[test]
fn a_capacity_over_the_ceiling_is_refused() {
    for capacity in [MAX_CAPACITY + 1, 1 << 40, usize::MAX / 2, usize::MAX] {
        assert_eq!(
            PatchSoA::<Pattern51>::new(capacity, 4).unwrap_err(),
            TrackerError::CapacityTooLarge {
                capacity,
                ceiling: MAX_CAPACITY,
            }
        );
        assert_eq!(
            CpuPatchTracker::<Pattern51>::new(capacity, 4, 5, 0.04, WorkPool::new(1).unwrap())
                .unwrap_err(),
            TrackerError::CapacityTooLarge {
                capacity,
                ceiling: MAX_CAPACITY,
            }
        );
    }
}

/// A pyramid deeper than the ceiling is refused before the allocation.
///
/// `optical_flow_levels = 10^12` sized a `Vec` of 6e17 floats. A `Vec` whose
/// length fits in a `usize` but whose bytes do not exist does not panic — the
/// allocator handler **aborts** the process, taking the Python interpreter
/// with it, so this is checked rather than attempted.
#[test]
fn more_levels_than_the_ceiling_is_refused() {
    for num_levels in [MAX_LEVELS + 1, 1_000_000_000_001, usize::MAX] {
        assert_eq!(
            PatchSoA::<Pattern51>::new(3000, num_levels).unwrap_err(),
            TrackerError::TooManyLevels {
                num_levels,
                ceiling: MAX_LEVELS,
            }
        );
    }
}

/// The two ceilings bound every buffer product, in `usize` and in `u32`.
///
/// This is what makes [`TrackerError::BufferShapeOverflow`] unreachable
/// today: it is the guard that fires if either ceiling is ever raised past
/// the point where a product wraps, and this test is the proof that it does
/// not have to fire now.
#[test]
fn the_ceilings_bound_every_buffer_product() {
    let flags: usize = MAX_LEVELS.checked_mul(MAX_CAPACITY).unwrap();
    let taps: usize = flags.checked_mul(Pattern51::SIZE).unwrap();
    let jacobians: usize = taps.checked_mul(3).unwrap();
    assert!(
        jacobians <= u32::MAX as usize,
        "{jacobians} elements would wrap a 32-bit usize"
    );
}

/// The ceiling is well clear of anything the port runs.
#[test]
fn the_default_budget_is_far_under_the_ceiling() {
    // `FrontendOptions::default().max_keypoints` is 3000.
    const { assert!(MAX_CAPACITY > 300 * 3000) };
    let patches: PatchSoA<Pattern51> = PatchSoA::new(3000, 4).unwrap();
    assert_eq!(patches.capacity(), 3000);
    assert_eq!(patches.num_levels(), 4);
}

#[test]
fn flow_transforms_round_trip_through_the_soa_arrays() {
    let mut transforms: FlowTransforms = FlowTransforms::default();
    let warps: [AffineCompact2f; 3] = [
        AffineCompact2f::at(Vector2::new(1.0, 2.0)),
        AffineCompact2f {
            linear: Matrix2::new(0.5, -0.25, 0.25, 0.5),
            translation: Vector2::new(-3.0, 4.5),
        },
        AffineCompact2f::identity(),
    ];
    for warp in &warps {
        transforms.push(warp);
    }
    assert_eq!(transforms.len(), 3);
    for (index, warp) in warps.iter().enumerate() {
        assert_eq!(transforms.get(index), *warp);
        assert_eq!(transforms.translation(index), warp.translation);
        assert_eq!(transforms.coefficients(index), warp.coefficients());
    }
    // One coefficient of every warp is contiguous, which is the point.
    assert_eq!(transforms.translations_x(), &[1.0, -3.0, 0.0]);
    assert_eq!(transforms.translations_y(), &[2.0, 4.5, 0.0]);

    transforms.remove(1);
    assert_eq!(transforms.len(), 2);
    assert_eq!(transforms.get(1), warps[2]);
    transforms.insert(1, &warps[1]);
    assert_eq!(transforms.get(1), warps[1]);
    transforms.set(0, &warps[2]);
    assert_eq!(transforms.get(0), warps[2]);
}
/// A second [`PatchTracker`] implementation, written only against the public
/// API, proving a backend outside this module can publish results.
///
/// It reports every input as tracked, at the guess it was given, through
/// [`FlowResult::reset`], [`FlowResult::set_track`] and
/// [`FlowResult::finish`]; the bulk path through [`FlowResult::parts_mut`]
/// and [`FlowTransforms::coefficients_mut`] is what [`CpuPatchTracker`]
/// itself uses, so both halves of the writing surface are exercised.
#[derive(Debug, Default)]
struct EchoTracker {
    batch: slam_rs::frontend::tracker::TrackBatch,
    capacity: usize,
    num_levels: usize,
}

impl PatchTracker for EchoTracker {
    fn batch(&self) -> &slam_rs::frontend::tracker::TrackBatch {
        &self.batch
    }
    fn batch_mut(&mut self) -> &mut slam_rs::frontend::tracker::TrackBatch {
        &mut self.batch
    }

    type Pattern = Pattern51;
    type Pyramid = PyramidU16;
    type Patches = PatchSoA<Pattern51>;

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn num_levels(&self) -> usize {
        self.num_levels
    }

    fn make_patches(&self) -> Result<PatchSoA<Pattern51>, TrackerError> {
        PatchSoA::new(self.capacity, self.num_levels)
    }

    fn submit_prepared(
        &mut self,
        _prev: &PyramidU16,
        _next: &PyramidU16,
        patches: &PatchSoA<Pattern51>,
        transforms_in: &FlowTransforms,
    ) -> Result<usize, TrackerError> {
        let count: usize = transforms_in.len();
        if count != patches.len() {
            return Err(TrackerError::LengthMismatch {
                first_name: "patches",
                first: patches.len(),
                second_name: "transforms",
                second: count,
            });
        }
        let (pass, out) = self.batch.submit_slot(self.capacity);
        out.reset(count);
        for index in 0..count {
            out.set_track(index, true, &transforms_in.get(index));
        }
        // The bulk path over the same buffers, to prove it is reachable.
        let (valid, transforms) = out.parts_mut();
        let [m00, ..] = transforms.coefficients_mut();
        assert_eq!(m00.len(), valid.len().max(m00.len()));
        out.finish(count);
        Ok(pass)
    }
}

#[test]
fn a_second_backend_can_publish_results_through_the_public_api() {
    let levels: usize = 3;
    let scene: Fixture = fixture(0.7, -1.2, levels);
    let mut echo: EchoTracker = EchoTracker {
        batch: Default::default(),
        capacity: scene.positions.len(),
        num_levels: levels + 1,
    };
    let mut out: FlowResult = FlowResult::default();
    echo.track(
        &scene.prev,
        &scene.next,
        &scene.patches,
        &scene.transforms,
        &mut out,
    )
    .unwrap();

    assert_eq!(out.len(), scene.positions.len());
    for index in out.tracked() {
        let index: usize = *index as usize;
        assert!(out.is_valid(index));
        assert_eq!(out.transform(index), scene.transforms.get(index));
    }
}

/// The write sequence is usable on its own: reset, set, finish.
#[test]
fn the_flow_result_writing_surface_compacts_what_it_is_given() {
    let mut out: FlowResult = FlowResult::default();
    out.reset(5);
    assert!(out.is_empty());
    out.set_track(1, true, &AffineCompact2f::at(Vector2::new(3.0, 4.0)));
    out.set_track(4, true, &AffineCompact2f::at(Vector2::new(-1.0, 0.5)));
    out.set_track(2, false, &AffineCompact2f::identity());
    out.finish(5);

    assert_eq!(out.tracked(), &[1, 4]);
    assert_eq!(out.transform(1).translation, Vector2::new(3.0, 4.0));
    assert_eq!(out.transform(4).translation, Vector2::new(-1.0, 0.5));
    assert!(!out.is_valid(0));
    assert!(!out.is_valid(2));

    // A reset clears the survivors without dropping the allocation.
    out.reset(3);
    assert!(out.tracked().is_empty());
    assert!(!out.is_valid(1));
}
