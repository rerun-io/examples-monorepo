//! Numerical forward/backward KLT contracts against the staged CPU operation.
#![cfg(feature = "wgpu")]
use flow::{
    grid_positions, guesses_at, LEVELS, MAX_ITERATIONS, MAX_KEYPOINTS, MAX_RECOVERED_DIST2,
};
use images::{cornered_image, texture, textured_image};
use kornia_image::Image;
use kornia_staging_gpu::{
    optical_flow::FusedKltPlan, pyramid::GpuPyramidBuilder, runtime::gpu_client,
};
use kornia_staging_imgproc::test_fixtures as flow;
use kornia_staging_imgproc::test_fixtures as images;
use kornia_staging_imgproc::{
    optical_flow::{
        patch_se2::{AffineCompact2f, Pattern51},
        patch_tracker::{
            FlowResult, FlowTransforms, PatchSoA, PatchTrackerPlan as CpuTrackerPlan, PointsSoA,
        },
    },
    pyramid::PyramidPlanU16,
};
use nalgebra::Vector2;
fn separable_image(width: usize, height: usize, dx: f32, dy: f32) -> Image<u16, 1> {
    let column: Vec<f64> = (0..width)
        .map(|x| 0.5 + 0.4 * texture(x as f64 - f64::from(dx), 0.0))
        .collect();
    let row: Vec<f64> = (0..height)
        .map(|y| 0.5 + 0.4 * texture(0.0, y as f64 - f64::from(dy)))
        .collect();
    let mut image: Image<u16, 1> =
        Image::from_size_val(kornia_image::ImageSize { width, height }, 0u16).unwrap();
    for (y, vertical) in row.iter().enumerate() {
        let scaled: f64 = vertical * 65535.0;
        for (x, horizontal) in column.iter().enumerate() {
            image
                .set_pixel(x, y, 0, (horizontal * scaled).clamp(0.0, 65535.0) as u16)
                .unwrap();
        }
    }
    image
}

/// A level base past `2^24` reaches the kernels as the integer it is.
///
/// The level bases go to the device in the metadata array both per-patch
/// kernels read. They were `f32` there, and `f32` holds only every second
/// integer above `2^24`: a 4097x4097 frame puts level 2 — the second level of
/// the even buffer — at base `4097 * 4097 = 16,785,409`, which `f32` stores as
/// 16,785,408, so every level-2 sample read one pixel early on both lanes with
/// nothing anywhere reporting it.
///
/// `the_gpu_pyramid_is_bit_exact_with_the_cpu` cannot see it and no bigger
/// version of it could: `copy_level_into` downloads with the **host** side's
/// exact base and never reads the device's metadata at all. So this is the
/// fixture that reads it, the only way a caller can — through the two kernels
/// that index with it — at the smallest geometry whose alternating-buffer base
/// is past the bound, and at the same tolerances every other fixture here uses.
#[test]
fn a_level_base_past_f32_precision_reaches_the_kernels_exactly() {
    // Odd times odd is the point, not the size: `f32`'s spacing at this
    // magnitude is 2, so an even product this big is still exact and only an
    // odd one rounds. 4097 x 4097 is the smallest square that is both.
    const SIDE: usize = 4097;
    const SHIFT: f32 = 2.75;
    assert_eq!(SIDE * SIDE, 16_785_409);
    assert_ne!(
        (SIDE * SIDE) as f32 as usize,
        SIDE * SIDE,
        "an f32 base of {} would have been exact, so this fixture measures nothing",
        SIDE * SIDE
    );

    let started: std::time::Instant = std::time::Instant::now();
    let first: Image<u16, 1> = separable_image(SIDE, SIDE, 0.0, 0.0);
    let second: Image<u16, 1> = separable_image(SIDE, SIDE, SHIFT, -1.5);
    // Patches well inside the frame at every level: level 3 is 512x512, so a
    // level-0 position must stay under 4064 to keep its taps in bounds there.
    let mut positions: PointsSoA = PointsSoA::with_capacity(32);
    for row in 0..5 {
        for column in 0..5 {
            positions.push(Vector2::new(
                200.0 + 900.0 * column as f32 + 0.37,
                200.0 + 900.0 * row as f32 - 0.21,
            ));
        }
    }
    let count: usize = positions.len();

    // ── one KLT step, which reads it once per level per iteration
    let guesses: FlowTransforms = guesses_at(&positions);
    let (cpu_result, gpu_result) = track_both_lanes(SIDE, &first, &second, &positions, &guesses);
    let tracked: usize = (0..count)
        .filter(|&index| gpu_result.is_valid(index))
        .count();
    assert_eq!(tracked, count, "{tracked} of {count} patches survived");
    let worst: f32 = assert_lanes_agree(&cpu_result, &gpu_result, count, "4097x4097");
    println!(
        "4097x4097 (level-2 base {}): {tracked} patches tracked, worst lane gap \
         {worst:.3e} px, whole fixture {:.1} s",
        SIDE * SIDE,
        started.elapsed().as_secs_f64()
    );
}

/// How far apart the two lanes' tracked positions may sit, in pixels.
///
/// 1e-3, ten times the worst any fixture in this file measures and ten times
/// below the 1e-2 it used to allow. What is measured on wgpu, in pixels:
/// the grid shift 3.146e-5, the border margin 7.780e-5,
/// a bad guess 2.158e-5. The factor of ten is not slack for the
/// backend: it is the room a *different* fixture — another texture, another
/// shift, another adapter — may need for the same cause, which is that the shader compiler
/// contracts `a * b + c` into a fused multiply-add and the host does not. A
/// change that needs more than this is a change in the arithmetic, not in the
/// rounding.
const LANE_POSITION_BOUND: f32 = 1e-3;

/// One frame pair through both lanes, from the same patches and the same guesses.
///
/// The five seam tests below differ only in the frames, the positions and the
/// guesses; every one of them needs both lanes driven identically from the same
/// inputs, and a copy each would be five places for the two lanes to drift
/// apart in.
fn track_both_lanes(
    size: usize,
    first: &Image<u16, 1>,
    second: &Image<u16, 1>,
    positions: &PointsSoA,
    guesses: &FlowTransforms,
) -> (FlowResult, FlowResult) {
    // ── the CPU lane
    let mut cpu_prev = PyramidPlanU16::new(first.size(), LEVELS).unwrap();
    let mut cpu_next = PyramidPlanU16::new(second.size(), LEVELS).unwrap();
    cpu_prev.run(first).unwrap();
    cpu_next.run(second).unwrap();
    let mut cpu_tracker = CpuTrackerPlan::<Pattern51>::new(
        MAX_KEYPOINTS,
        LEVELS + 1,
        MAX_ITERATIONS,
        MAX_RECOVERED_DIST2,
        None,
    )
    .unwrap();
    let mut cpu_patches: PatchSoA<Pattern51> = PatchSoA::new(MAX_KEYPOINTS, LEVELS + 1).unwrap();
    cpu_patches.build(&cpu_prev, positions, None).unwrap();
    let mut cpu_result: FlowResult = FlowResult::with_capacity(MAX_KEYPOINTS);
    cpu_tracker
        .track(&cpu_prev, &cpu_next, &cpu_patches, guesses, &mut cpu_result)
        .unwrap();

    // ── the GPU lane
    let client = gpu_client().unwrap();
    let mut gpu_builder = GpuPyramidBuilder::new(client.clone());
    let mut gpu_prev = gpu_builder.allocate(size, size, LEVELS).unwrap();
    let mut gpu_next = gpu_builder.allocate(size, size, LEVELS).unwrap();
    gpu_builder.build(0, first, &mut gpu_prev).unwrap();
    gpu_builder.build(0, second, &mut gpu_next).unwrap();
    let mut gpu_tracker = FusedKltPlan::<Pattern51, _>::new(
        client,
        MAX_KEYPOINTS,
        LEVELS + 1,
        MAX_ITERATIONS,
        MAX_RECOVERED_DIST2,
    )
    .unwrap();
    let mut gpu_result = FlowResult::with_capacity(MAX_KEYPOINTS);
    gpu_tracker
        .track(
            &gpu_prev,
            &gpu_next,
            positions,
            guesses,
            None,
            &mut gpu_result,
        )
        .unwrap();
    (cpu_result, gpu_result)
}

/// Every patch's outcome flag on the two lanes, and the worst shared position.
///
/// The flag is asserted **exactly**, not as a fraction: these fixtures are built
/// so no patch sits on a threshold, and a lane that starts disagreeing about
/// whether a track survived is the failure these tests exist to catch.
fn assert_lanes_agree(cpu: &FlowResult, gpu: &FlowResult, count: usize, label: &str) -> f32 {
    let mut worst: f32 = 0.0;
    for index in 0..count {
        assert_eq!(
            cpu.is_valid(index),
            gpu.is_valid(index),
            "{label}: patch {index} survived on one lane and not the other \
             (CPU {}, GPU {})",
            cpu.is_valid(index),
            gpu.is_valid(index)
        );
        if cpu.is_valid(index) {
            let difference = Vector2::from(cpu.transform(index).translation)
                - Vector2::from(gpu.transform(index).translation);
            worst = worst.max(difference.norm());
        }
    }
    assert!(
        worst < LANE_POSITION_BOUND,
        "{label}: positions differ by up to {worst} px between the lanes, \
         against a bound of {LANE_POSITION_BOUND}"
    );
    worst
}

#[test]
fn the_gpu_tracker_recovers_the_same_shift_as_the_cpu() {
    const SHIFT: f32 = 2.75;
    let first: Image<u16, 1> = textured_image(512, 512, 0.0, 0.0);
    let second: Image<u16, 1> = textured_image(512, 512, SHIFT, -1.5);
    let positions: PointsSoA = grid_positions(512);
    let count: usize = positions.len();
    let guesses: FlowTransforms = guesses_at(&positions);
    let (cpu_result, gpu_result) = track_both_lanes(512, &first, &second, &positions, &guesses);

    // ── the shift both lanes must find, before they are compared with each other
    //
    // Neither lane recovers the shift exactly: the residual's fixed point sits a
    // fraction of a pixel from the true shift because bilinear interpolation is
    // not the band-limited reconstruction, which is a property of the algorithm
    // and not of the backend. So the bound is the same for both and the *lanes*
    // are what get the tight comparison below.
    let mut tracked: usize = 0;
    let mut worst_shift: f32 = 0.0;
    let mut worst_cpu_shift: f32 = 0.0;
    for index in 0..count {
        if gpu_result.is_valid(index) {
            tracked += 1;
            let moved = Vector2::from(gpu_result.transform(index).translation)
                - Vector2::from(positions.get(index));
            worst_shift = worst_shift
                .max((moved.x - SHIFT).abs())
                .max((moved.y + 1.5).abs());
        }
        if cpu_result.is_valid(index) {
            let moved = Vector2::from(cpu_result.transform(index).translation)
                - Vector2::from(positions.get(index));
            worst_cpu_shift = worst_cpu_shift
                .max((moved.x - SHIFT).abs())
                .max((moved.y + 1.5).abs());
        }
    }
    assert!(
        tracked * 4 >= count * 3,
        "the GPU tracker kept only {tracked} of {count} patches"
    );
    assert!(
        worst_shift < 0.1,
        "the GPU tracker's worst recovered shift is off by {worst_shift} px, \
         against the CPU tracker's {worst_cpu_shift} px on the same frames"
    );
    assert!(
        worst_shift < 1.5 * worst_cpu_shift.max(1e-3),
        "the GPU tracker's worst recovered shift ({worst_shift} px) is more than \
         half again the CPU tracker's ({worst_cpu_shift} px)"
    );

    // ── the two lanes against each other
    //
    // A Gauss-Newton fixed point reached from the same start converges to the
    // same place; fused multiply-add moves the last steps, not the answer. Every
    // patch here is far from the border and every one survives, so the flag is
    // asserted exactly rather than as a fraction — the cases where it may be
    // decided by a threshold have their own tests below.
    let worst_position: f32 = assert_lanes_agree(&cpu_result, &gpu_result, count, "grid shift");
    println!(
        "tracker: {tracked} of {count} kept, worst recovered shift GPU {worst_shift:.4} px \
         / CPU {worst_cpu_shift:.4} px, worst lane-to-lane position {worst_position:.3e} px"
    );
}

/// A track onto an unrelated frame fails on both lanes, and fails the same way.
///
/// The CPU suite's `a_mismatched_pair_is_rejected` at the seam: the
/// forward-backward gate is what
/// rejects a track onto an image the patch is not in, and it is the one exit
/// the grid fixture above never takes. A backend that let a failed track
/// through — or that failed a different set of patches from the CPU's — would
/// put keypoints on nothing and pass every other test in this file. The
/// rejection is not trivial: `the_gpu_tracker_recovers_the_same_shift_as_the_cpu`
/// puts these same patches and these same guesses through both lanes against a
/// *shifted* frame and keeps all 25.
#[test]
fn both_lanes_reject_a_track_onto_an_unrelated_frame() {
    let first: Image<u16, 1> = textured_image(512, 512, 0.0, 0.0);
    // A different field, not a shift of the first: `cornered_image` is one LCG
    // plus a much shorter wave, so nothing in it correlates with the plane-wave
    // texture the patches were built on.
    let unrelated: Image<u16, 1> = cornered_image(512, 512);
    let positions: PointsSoA = grid_positions(512);
    let count: usize = positions.len();
    let guesses: FlowTransforms = guesses_at(&positions);
    let (cpu_result, gpu_result) = track_both_lanes(512, &first, &unrelated, &positions, &guesses);

    let worst: f32 = assert_lanes_agree(&cpu_result, &gpu_result, count, "unrelated frame");
    println!(
        "unrelated frame: {} of {count} survived on each lane, worst lane-to-lane \
         position {worst:.3e} px",
        cpu_result.len()
    );
    // The CPU suite's own bound on this fixture, restated here because a lane
    // pair that agreed on keeping everything would agree and be wrong.
    assert!(
        cpu_result.len() * 4 < count,
        "{} of {count} tracks survived an unrelated image",
        cpu_result.len()
    );
}

/// Patches on both sides of the border margin get the same verdict on both lanes.
///
/// Two thresholds meet near an edge and the grid fixture is built to stay away
/// from both: the patch build needs its 52 taps in range at every level, and
/// each Gauss-Newton step needs the new centre `FILTER_MARGIN = 2` pixels inside
/// *that level's* image. At the coarsest
/// of four levels a 512-pixel frame is 64 wide, so the margin is 16 full-
/// resolution pixels; this walks a column from 4 to 60 pixels from the left
/// edge, which crosses it, and asserts the two lanes take the same branch at
/// every one.
#[test]
fn both_lanes_agree_at_the_border_margin() {
    let first: Image<u16, 1> = textured_image(512, 512, 0.0, 0.0);
    let second: Image<u16, 1> = textured_image(512, 512, 0.75, -0.25);
    let mut positions: PointsSoA = PointsSoA::with_capacity(256);
    let mut x: usize = 4;
    while x <= 60 {
        let mut y: usize = 96;
        while y + 96 < 512 {
            positions.push(Vector2::new(x as f32 + 0.37, y as f32 - 0.21));
            y += 71;
        }
        x += 2;
    }
    let count: usize = positions.len();
    let guesses: FlowTransforms = guesses_at(&positions);
    let (cpu_result, gpu_result) = track_both_lanes(512, &first, &second, &positions, &guesses);

    let worst: f32 = assert_lanes_agree(&cpu_result, &gpu_result, count, "border margin");
    // Both sides of the threshold have to be represented, or the test is only
    // checking one branch under a name that promises two.
    let survivors: usize = cpu_result.len();
    println!(
        "border margin: {survivors} of {count} survived on each lane, worst \
         lane-to-lane position {worst:.3e} px"
    );
    assert!(
        survivors > 0 && survivors < count,
        "the border column put {survivors} of {count} patches through, so one \
         side of the margin is untested"
    );
}

/// A guess far from the truth converges — or fails — the same way on both lanes.
///
/// Two shapes in one fixture, because they take different exits. A guess
/// displaced between 4 and 28 pixels is inside the frame and straddles what
/// four pyramid levels recover from this texture, so it is the tracker's own
/// convergence that decides and it decides both ways; a guess at a negative
/// coordinate is refused before any level runs (the
/// `t2(0) >= 0 && ... < w` gate). Both lanes must take all three exits
/// identically.
#[test]
fn both_lanes_agree_on_a_bad_initial_guess() {
    let first: Image<u16, 1> = textured_image(512, 512, 0.0, 0.0);
    let second: Image<u16, 1> = textured_image(512, 512, 1.25, 0.5);
    let positions: PointsSoA = grid_positions(512);
    let count: usize = positions.len();
    let mut guesses: FlowTransforms = FlowTransforms::with_capacity(count);
    guesses.resize(count);
    for index in 0..count {
        let displaced: Vector2<f32> = if index % 4 == 0 {
            // Outside the frame entirely: the pre-track bound, not the tracker.
            Vector2::new(-8.0, -8.0)
        } else {
            let offset: f32 = 4.0 + 6.0 * (index % 5) as f32;
            Vector2::from(positions.get(index)) + Vector2::new(offset, -offset)
        };
        guesses.set(index, &AffineCompact2f::at(displaced));
    }
    let (cpu_result, gpu_result) = track_both_lanes(512, &first, &second, &positions, &guesses);

    let worst: f32 = assert_lanes_agree(&cpu_result, &gpu_result, count, "bad guess");
    println!(
        "bad guess: {} of {count} survived on each lane, worst lane-to-lane \
         position {worst:.3e} px",
        cpu_result.len()
    );
    // Both sides of convergence have to be represented: every patch here is one
    // the grid fixture tracks from a correct guess, so a fixture where none
    // recovers would only be testing the refusal.
    assert!(
        !cpu_result.is_empty() && cpu_result.len() < count,
        "the bad-guess column put {} of {count} patches through, so one side of \
         convergence is untested",
        cpu_result.len()
    );
    // The out-of-frame quarter must be refused on both lanes; `assert_lanes_agree`
    // has already made the two agree, so asserting the CPU's is asserting both.
    for index in (0..count).step_by(4) {
        assert!(
            !cpu_result.is_valid(index),
            "patch {index} was guessed outside the frame and tracked anyway"
        );
    }
}

#[test]
fn a_shallow_plan_accepts_deeper_unequal_pyramids_and_clears_failed_output() {
    let first = textured_image(512, 512, 0.0, 0.0);
    let second = textured_image(512, 512, 0.25, -0.125);
    let points = grid_positions(512);
    let guesses = guesses_at(&points);
    let client = gpu_client().unwrap();
    let mut builder = GpuPyramidBuilder::new(client.clone());
    let mut prev = builder.allocate(512, 512, 3).unwrap();
    let mut next = builder.allocate(512, 512, 2).unwrap();
    builder.build(0, &first, &mut prev).unwrap();
    builder.build(0, &second, &mut next).unwrap();
    let mut cpu_prev = PyramidPlanU16::new(first.size(), 3).unwrap();
    let mut cpu_next = PyramidPlanU16::new(second.size(), 2).unwrap();
    cpu_prev.run(&first).unwrap();
    cpu_next.run(&second).unwrap();
    for levels in [1, 2] {
        let mut gpu =
            FusedKltPlan::<Pattern51, _>::new(client.clone(), 128, levels, 5, 0.09).unwrap();
        let mut cpu = CpuTrackerPlan::<Pattern51>::new(128, levels, 5, 0.09, None).unwrap();
        let mut patches = PatchSoA::new(128, levels).unwrap();
        patches.build(&cpu_prev, &points, None).unwrap();
        let mut expected = FlowResult::default();
        let mut actual = FlowResult::default();
        cpu.track(&cpu_prev, &cpu_next, &patches, &guesses, &mut expected)
            .unwrap();
        gpu.track(&prev, &next, &points, &guesses, None, &mut actual)
            .unwrap();
        assert_lanes_agree(&expected, &actual, points.len(), "shallow plan");
        assert!(!actual.is_empty());
        assert!(gpu
            .track(
                &prev,
                &next,
                &points,
                &FlowTransforms::default(),
                None,
                &mut actual
            )
            .is_err());
        assert!(actual.is_empty());
        assert!(
            actual.parts_mut().0.is_empty(),
            "failed output must not retain valid slots"
        );
    }
}

#[test]
fn klt_buffers_reject_shapes_beyond_the_portable_dispatch_limit() {
    use kornia_staging_gpu::optical_flow::FusedKltPlan;
    let client = kornia_staging_gpu::runtime::gpu_client().unwrap();
    assert!(
        FusedKltPlan::<Pattern51, _>::new_batched(client.clone(), 262_140, 1, 5, 0.09, 1).is_ok()
    );
    assert!(
        FusedKltPlan::<Pattern51, _>::new_batched(client.clone(), 262_141, 1, 5, 0.09, 1).is_err()
    );
    assert!(FusedKltPlan::<Pattern51, _>::new_batched(client, 131_071, 1, 5, 0.09, 2).is_err());
}
