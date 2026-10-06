//! Per-kernel tolerance tests for the CubeCL frontend backend (decision D21).
//!
//! Each test states its tolerance and why, and prints the measured maximum
//! absolute difference in the assertion message so a regression says how far it
//! moved rather than only that it moved.
//!
//! * the pyramid is **exact**. Its arithmetic is integer: the fused 5x5 pass on
//!   the GPU and the separable two-pass form on the CPU are the same sum of
//!   `u32` terms with one rounding at the end, so anything but equality is a
//!   bug, not a rounding.
//! * the patch build and the tracker are **not** exact, for one reason: the shader compiler
//!   contracts `a * b + c` into a fused multiply-add, which the CPU does not.
//!   Contraction only ever raises the accuracy of a term, but it changes the
//!   bits, and a Gauss-Newton fixed point amplifies the change until the
//!   iteration converges. The bounds below are what that costs.
//!
//! These run only under `--features gpu-wgpu` and need a working CubeCL runtime;
//! `cargo test --features gpu-wgpu` is the gate.
#![cfg(feature = "gpu-core")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use cubecl::frontend::CompilationArg;
use kornia_image::Image;
use kornia_imgproc::features::FastCorner;
use nalgebra::Vector2;
use slam_rs::frontend::detect::{BandRequest, CornerScan, CpuCornerScan, DetectError};
use slam_rs::frontend::parallel::WorkPool;
use slam_rs::frontend::patterns::Pattern51;
use slam_rs::frontend::se2::AffineCompact2f;
use slam_rs::frontend::tracker::{
    CpuPatchTracker, FlowResult, FlowTransforms, PatchSoA, PatchTracker, PointsSoA, SourcePatches,
};
use slam_rs::gpu::{
    GpuCornerScan, GpuPatchSources, GpuPatchTracker, GpuPyramid, GpuPyramidBuilder, GpuRuntime,
    gpu_client,
};
use slam_rs::pyramid::{CpuPyramidBuilder, Pyramid, PyramidBuilder, PyramidError, PyramidU16};

mod common;

#[path = "../src/gpu/finite.rs"]
mod finite;

use common::gpu::{
    LEVELS, MAX_ITERATIONS, MAX_KEYPOINTS, MAX_RECOVERED_DIST2, band_at, guesses_at,
};
use common::{cornered_image, grid_positions, texture, textured_image};

fn packed_texture(width: usize, height: usize, dx: f32, dy: f32) -> Image<u16, 1> {
    let source = textured_image(width, height, dx, dy);
    let bytes: Vec<u8> = source
        .as_slice()
        .iter()
        .map(|pixel| (pixel >> 8) as u8)
        .collect();
    let mut image = slam_rs::image::empty();
    slam_rs::image::fill_from_u8_strided(&mut image, &bytes, width, height, width).unwrap();
    image
}

/// `image`'s pyramid on both lanes, `LEVELS` deep and the same geometry.
fn both_pyramids(image: &Image<u16, 1>) -> (PyramidU16, GpuPyramid<GpuRuntime>) {
    let (width, height): (usize, usize) = (image.width(), image.height());

    let mut cpu_builder: CpuPyramidBuilder = CpuPyramidBuilder::new();
    let mut cpu: PyramidU16 = cpu_builder.allocate(width, height, LEVELS).unwrap();
    cpu_builder.build(0, image, &mut cpu).unwrap();

    let mut gpu_builder: GpuPyramidBuilder<GpuRuntime> =
        GpuPyramidBuilder::new(gpu_client().unwrap(), Default::default());
    let mut gpu: GpuPyramid<GpuRuntime> = gpu_builder.allocate(width, height, LEVELS).unwrap();
    gpu_builder.build(0, image, &mut gpu).unwrap();

    (cpu, gpu)
}

/// Every level of the two pyramids holds the same geometry and the same pixels.
///
/// The message names the worst pixel and where it is, so a regression says how
/// far it moved rather than only that it moved. Equality is the bound: the
/// arithmetic is integer on both lanes (module doc).
fn assert_levels_equal(cpu: &PyramidU16, gpu: &GpuPyramid<GpuRuntime>, label: &str) {
    assert_eq!(gpu.num_levels(), cpu.num_levels(), "{label}");
    let mut expected: Image<u16, 1> = slam_rs::image::empty();
    let mut actual: Image<u16, 1> = slam_rs::image::empty();
    for level in 0..cpu.num_levels() {
        assert_eq!(
            gpu.level_size(level),
            cpu.level_size(level),
            "{label}, level {level}"
        );
        cpu.copy_level_into(level, &mut expected).unwrap();
        gpu.copy_level_into(level, &mut actual).unwrap();
        let mut worst: i64 = 0;
        let mut worst_at: (usize, usize) = (0, 0);
        for y in 0..expected.height() {
            for x in 0..expected.width() {
                let difference: i64 = i64::from(actual.get_pixel(x, y, 0).copied().ok().unwrap())
                    - i64::from(expected.get_pixel(x, y, 0).copied().ok().unwrap());
                if difference.abs() > worst {
                    worst = difference.abs();
                    worst_at = (x, y);
                }
            }
        }
        assert_eq!(
            worst,
            0,
            "{label}, level {level} differs: max-abs-diff {worst} at {worst_at:?} \
             ({}x{}); the fused 5x5 pass is integer arithmetic and must be exact",
            expected.width(),
            expected.height()
        );
    }
}

#[test]
fn the_gpu_pyramid_is_bit_exact_with_the_cpu() {
    for (width, height) in [(960, 960), (640, 480), (517, 193), (64, 64)] {
        let image: Image<u16, 1> = textured_image(width, height, 0.0, 0.0);
        let (cpu, gpu) = both_pyramids(&image);
        assert_levels_equal(&cpu, &gpu, &format!("{width}x{height}"));
    }
}

#[test]
fn batched_gpu_pyramids_keep_every_camera_pixel_exact() {
    let mut builder = GpuPyramidBuilder::new(gpu_client().unwrap(), Default::default());
    let mut cpu_builder = CpuPyramidBuilder::new();
    let pool = WorkPool::new(1).unwrap();
    for (width, height) in [(640, 480), (517, 193), (65, 67)] {
        let mut pyramids: Vec<_> = (0..4)
            .map(|_| builder.allocate(width, height, LEVELS).unwrap())
            .collect();
        for frame in 0..2 {
            let images: Vec<_> = (0..4)
                .map(|camera| {
                    packed_texture(width, height, (camera * 7 + frame) as f32, camera as f32)
                })
                .collect();
            builder.build_frames(&images, &mut pyramids, &pool).unwrap();
            for (camera, image) in images.iter().enumerate() {
                let mut expected = cpu_builder.allocate(width, height, LEVELS).unwrap();
                cpu_builder.build(camera, image, &mut expected).unwrap();
                assert_levels_equal(
                    &expected,
                    &pyramids[camera],
                    &format!("camera {camera}, frame {frame}"),
                );
            }
        }
    }
}

#[test]
fn prepared_gpu_pixels_survive_reusing_the_source_image() {
    let mut image = textured_image(96, 96, 0.0, 0.0);
    let mut builder = GpuPyramidBuilder::new(gpu_client().unwrap(), Default::default());
    let mut pyramid = builder.allocate(96, 96, 1).unwrap();
    let expected = image.clone();
    builder
        .prepare_images(std::slice::from_ref(&image))
        .unwrap();
    slam_rs::image::fill_from_u8_strided(&mut image, &vec![0; 96 * 96], 96, 96, 96).unwrap();
    builder.build(0, &image, &mut pyramid).unwrap();
    let mut actual = slam_rs::image::empty();
    pyramid.copy_level_into(0, &mut actual).unwrap();
    assert_eq!(actual.as_slice(), expected.as_slice());
}

/// A pyramid of level 0 alone is refused rather than allocated empty.
///
/// With `optical_flow_levels = 0` the odd buffer holds no level, and
/// `client.empty(0)` is a zero-sized allocation that wgpu rejects at
/// validation — on cubecl's own worker thread, where a panic reaches the caller
/// as data rather than as an error. That is the failure mode `probe_storage`
/// exists to prevent, reachable through a config value instead, so the geometry
/// is refused the way every other unbuildable one is.
#[test]
fn a_pyramid_of_one_level_is_refused_rather_than_allocated_empty() {
    let builder = GpuPyramidBuilder::new(gpu_client().unwrap(), Default::default());
    let refused = builder.allocate(64, 48, 0);
    assert!(
        matches!(
            refused,
            Err(PyramidError::TooSmall {
                width: 64,
                height: 48,
                num_levels: 0
            })
        ),
        "a single-level pyramid was accepted: {refused:?}"
    );
}

/// A frame whose stride is wider than its width — dav1d's shape — must upload
/// the same pixels, not the padding.
#[test]
fn a_strided_frame_uploads_its_rows_and_not_its_padding() {
    let bytes: Vec<u8> = (0..96 * 48).map(|i| (i % 251) as u8).collect();
    let strided = slam_rs::image::from_u8_strided(&bytes, 64, 48, 96).unwrap();
    assert_eq!(strided.as_slice().len(), 64 * 48);
    for y in 0..48 {
        for x in 0..64 {
            assert_eq!(
                strided.as_slice()[y * 64 + x],
                u16::from(bytes[y * 96 + x]) << 8
            );
        }
    }

    let (cpu, gpu) = both_pyramids(&strided);
    assert_levels_equal(&cpu, &gpu, "a 64x48 frame with stride 96");
}

/// The pyramid the builder allocates is reused frame after frame, so the second
/// frame must not see the first one's pixels anywhere.
#[test]
fn a_reused_pyramid_carries_only_the_newest_frame() {
    let first: Image<u16, 1> = textured_image(128, 96, 0.0, 0.0);
    let second: Image<u16, 1> = textured_image(128, 96, 7.0, -3.0);

    let mut gpu_builder: GpuPyramidBuilder<GpuRuntime> =
        GpuPyramidBuilder::new(gpu_client().unwrap(), Default::default());
    let mut gpu: GpuPyramid<GpuRuntime> = gpu_builder.allocate(128, 96, LEVELS).unwrap();
    gpu_builder.build(0, &first, &mut gpu).unwrap();
    gpu_builder.build(0, &second, &mut gpu).unwrap();

    let mut cpu_builder: CpuPyramidBuilder = CpuPyramidBuilder::new();
    let mut cpu: PyramidU16 = cpu_builder.allocate(128, 96, LEVELS).unwrap();
    cpu_builder.build(0, &second, &mut cpu).unwrap();

    assert_levels_equal(&cpu, &gpu, "the second frame of a reused pyramid");
}

/// [`textured_image`]'s field at 16.8 M pixels, in a fraction of the time.
///
/// `textured_image` costs twelve `sin` per pixel, which is seconds at
/// 4097x4097, and this is the only fixture that needs a frame that big. The
/// same texture is sampled along each axis once and the field is the product of
/// the two, so it costs one multiply per pixel, keeps a gradient in both axes —
/// a patch with a singular `H` would be dropped by both lanes and compare
/// vacuously — and stays an exact translation of itself under `(dx, dy)`, which
/// is what the tracking half of the fixture measures.
fn separable_image(width: usize, height: usize, dx: f32, dy: f32) -> Image<u16, 1> {
    let column: Vec<f64> = (0..width)
        .map(|x| 0.5 + 0.4 * texture(x as f64 - f64::from(dx), 0.0))
        .collect();
    let row: Vec<f64> = (0..height)
        .map(|y| 0.5 + 0.4 * texture(0.0, y as f64 - f64::from(dy)))
        .collect();
    let mut image: Image<u16, 1> = slam_rs::image::zeros(width, height).unwrap();
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
    let mut cpu_builder: CpuPyramidBuilder = CpuPyramidBuilder::new();
    let mut cpu_prev: PyramidU16 = cpu_builder.allocate(size, size, LEVELS).unwrap();
    let mut cpu_next: PyramidU16 = cpu_builder.allocate(size, size, LEVELS).unwrap();
    cpu_builder.build(0, first, &mut cpu_prev).unwrap();
    cpu_builder.build(0, second, &mut cpu_next).unwrap();
    let mut cpu_tracker: CpuPatchTracker<Pattern51> = CpuPatchTracker::new(
        MAX_KEYPOINTS,
        LEVELS + 1,
        MAX_ITERATIONS,
        MAX_RECOVERED_DIST2,
        WorkPool::new(1).unwrap(),
    )
    .unwrap();
    let mut cpu_patches: PatchSoA<Pattern51> = cpu_tracker.make_patches().unwrap();
    cpu_patches.build(&cpu_prev, positions, None).unwrap();
    let mut cpu_result: FlowResult = FlowResult::with_capacity(MAX_KEYPOINTS);
    cpu_tracker
        .track(&cpu_prev, &cpu_next, &cpu_patches, guesses, &mut cpu_result)
        .unwrap();

    // ── the GPU lane
    let client = gpu_client().unwrap();
    let mut gpu_builder = GpuPyramidBuilder::new(client.clone(), Default::default());
    let mut gpu_prev = gpu_builder.allocate(size, size, LEVELS).unwrap();
    let mut gpu_next = gpu_builder.allocate(size, size, LEVELS).unwrap();
    gpu_builder.build(0, first, &mut gpu_prev).unwrap();
    gpu_builder.build(0, second, &mut gpu_next).unwrap();
    let mut gpu_tracker: GpuPatchTracker<Pattern51, _> = GpuPatchTracker::new(
        client.clone(),
        MAX_KEYPOINTS,
        LEVELS + 1,
        MAX_ITERATIONS,
        MAX_RECOVERED_DIST2,
        1,
        Default::default(),
    )
    .unwrap();
    let mut gpu_patches: GpuPatchSources<Pattern51, _> = gpu_tracker.make_patches().unwrap();
    gpu_patches.build(&gpu_prev, positions, None).unwrap();
    let mut gpu_result: FlowResult = FlowResult::with_capacity(MAX_KEYPOINTS);
    gpu_tracker
        .track(&gpu_prev, &gpu_next, &gpu_patches, guesses, &mut gpu_result)
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
            let difference = cpu.transform(index).translation - gpu.transform(index).translation;
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
            let moved = gpu_result.transform(index).translation - positions.get(index);
            worst_shift = worst_shift
                .max((moved.x - SHIFT).abs())
                .max((moved.y + 1.5).abs());
        }
        if cpu_result.is_valid(index) {
            let moved = cpu_result.transform(index).translation - positions.get(index);
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
            positions.get(index) + Vector2::new(offset, -offset)
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

/// Every band the detector asks for, at every rung of the shipped ladder, from
/// two scanners — and the count of corners they agreed on.
///
/// One per grid row of a 50-pixel cell, which is the shape
/// `detect_keypoints_with_cells` drives. Written once because both corner tests
/// need exactly it: the "exact against kornia" one and the "reads the pyramid"
/// one, which would otherwise drift into checking different amounts.
fn bands_agree(
    reference: &mut impl CornerScan,
    actual: &mut impl CornerScan,
    height: usize,
    label: &str,
) -> usize {
    let mut total: usize = 0;
    for (row, band_y) in (3..height - 3).step_by(50).enumerate() {
        for (rung, threshold) in [40i32, 20, 10, 5, 1].into_iter().enumerate() {
            let request: BandRequest = band_at(row, rung, band_y, 44, threshold);
            let want: Vec<FastCorner> = reference.band(request).unwrap().to_vec();
            let got: &[FastCorner] = actual.band(request).unwrap();
            assert_eq!(
                got.len(),
                want.len(),
                "{label} band {band_y} threshold {threshold}: {} corners against {}",
                got.len(),
                want.len()
            );
            for (index, (got, want)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    (got.xy, got.response),
                    (want.xy, want.response),
                    "{label} band {band_y} threshold {threshold}, corner {index}"
                );
            }
            total += want.len();
        }
    }
    total
}

/// The GPU corner scanner is **exact**, not within a tolerance.
///
/// kornia's candidate test at threshold `t` is the same statement as
/// `corner_score_9 > t`, and its in-block local-maximum filter compares raw
/// scores, so one dense score image answers every rung of the ladder. That is
/// what `tests/fast_model.rs` establishes on the CPU against kornia itself;
/// this checks that the two CubeCL kernels implement the same thing, corner for
/// corner, response for response, and in the same row-major order.
#[test]
fn the_gpu_corner_scan_is_exact_against_kornia() {
    for (width, height) in [(960usize, 240usize), (512, 192)] {
        let image: Image<u16, 1> = cornered_image(width, height);
        let mut cpu: CpuCornerScan = CpuCornerScan::default();
        let mut gpu: GpuCornerScan<_> =
            GpuCornerScan::new(gpu_client().unwrap(), Default::default()).unwrap();
        cpu.scan(0, &image).unwrap();
        gpu.scan(0, &image).unwrap();

        let total: usize = bands_agree(&mut cpu, &mut gpu, height, &format!("{width}x{height}"));
        println!("{width}x{height}: {total} corners over every band and rung, identical");
    }
}

/// The GPU lane refuses a band before a scan with the same typed error the CPU
/// lane returns, rather than caching an empty one and reporting success (D32).
#[test]
fn a_gpu_band_before_a_scan_is_refused() {
    let mut gpu: GpuCornerScan<_> =
        GpuCornerScan::new(gpu_client().unwrap(), Default::default()).unwrap();
    assert_eq!(
        gpu.band(band_at(0, 0, 0, 32, 5)).unwrap_err(),
        DetectError::NotScanned
    );
}

/// The scanner is reused frame after frame, so the second frame's bands must be
/// the second frame's.
#[test]
fn a_reused_corner_scan_carries_only_the_newest_frame() {
    let first: Image<u16, 1> = cornered_image(512, 128);
    let second: Image<u16, 1> = slam_rs::image::zeros(512, 128).unwrap();

    let mut gpu: GpuCornerScan<_> =
        GpuCornerScan::new(gpu_client().unwrap(), Default::default()).unwrap();
    gpu.scan(0, &first).unwrap();
    assert!(
        !gpu.band(band_at(0, 0, 3, 44, 5)).unwrap().is_empty(),
        "the textured frame has corners"
    );
    gpu.scan(0, &second).unwrap();
    assert!(
        gpu.band(band_at(0, 0, 3, 44, 5)).unwrap().is_empty(),
        "a black frame has none"
    );
}

/// The detector reads the pyramid's level 0, and uploads nothing.
///
/// The corner scanner and the pyramid builder are handed the same pixels once
/// per camera per frameset — the detector's input *is* level 0
///  — and until the camera index reached
/// both seams the GPU scanner had no way to know that, so it uploaded the frame
/// a second time. This is the test that the sharing is exact rather than merely
/// cheaper: the same corners at every rung, from a scanner that uploaded zero
/// frames, on two cameras of different geometry so the per-camera table is
/// actually indexed.
#[test]
fn the_gpu_corner_scan_reads_the_pyramid_and_uploads_nothing() {
    let frames: [Image<u16, 1>; 2] = [cornered_image(960, 240), cornered_image(512, 192)];
    let client = gpu_client().unwrap();

    // The lane the frontend runs: one builder, one scanner, one client, the
    // level-0 table between them.
    let mut builder = GpuPyramidBuilder::new(client.clone(), Default::default());
    let mut shared: GpuCornerScan<_> =
        GpuCornerScan::new(client.clone(), Default::default()).unwrap();

    // The lane before this change: the scanner uploads its own copy.
    let mut alone: GpuCornerScan<_> = GpuCornerScan::new(client, Default::default()).unwrap();

    let mut pyramids: Vec<_> = frames
        .iter()
        .map(|frame| builder.allocate(frame.width(), frame.height(), 3).unwrap())
        .collect();
    for (camera, frame) in frames.iter().enumerate() {
        builder.build(camera, frame, &mut pyramids[camera]).unwrap();
    }
    shared.use_level0(&mut builder);
    for (camera, frame) in frames.iter().enumerate() {
        shared.scan(camera, frame).unwrap();
        alone.scan(camera, frame).unwrap();
        let total: usize = bands_agree(
            &mut alone,
            &mut shared,
            frame.height(),
            &format!("camera {camera}"),
        );
        println!(
            "camera {camera} ({}x{}): {total} corners identical, uploads shared {} / alone {}",
            frame.width(),
            frame.height(),
            shared.frame_uploads(),
            alone.frame_uploads()
        );
    }
    assert_eq!(
        shared.frame_uploads(),
        0,
        "the shared scanner uploaded a frame the pyramid had already put on the device"
    );
    assert_eq!(alone.frame_uploads(), frames.len());

    // And each camera's three device buffers are allocated once for the life of
    // the scanner, not once per frameset. A one-slot geometry cache holds only
    // for a rig whose cameras are all the same size; on this one — 960x240 next
    // to 512x192, which is why the test drives two — a scanner without the
    // cache misses on every scan and re-allocates 4 MB.
    let allocations: usize = shared.buffer_allocations();
    assert_eq!(allocations, frames.len(), "one geometry, one allocation");
    for _ in 0..3 {
        for (camera, frame) in frames.iter().enumerate() {
            shared.scan(camera, frame).unwrap();
        }
    }
    assert_eq!(
        shared.buffer_allocations(),
        allocations,
        "three more framesets of the same rig re-allocated the scan buffers"
    );
    assert_eq!(shared.frame_uploads(), 0);

    // A frame whose geometry does not match the published entry is refused
    // rather than read: the fallback upload is what keeps a stale table safe.
    let odd: Image<u16, 1> = cornered_image(256, 128);
    shared.scan(0, &odd).unwrap();
    assert_eq!(shared.frame_uploads(), 1);
}

/// This runtime stores every element width the kernels bind.
///
/// The one test that would fire on a fleet machine before any of the others
/// mean anything. The `cubecl-wgpu` WGSL compiler on `u16`/`u8` can
/// panic on cubecl's own worker thread, so the launch reports success and every
/// read comes back as zeros. `probe_storage` copies a known pattern on the
/// device and refuses the runtime if it does not survive; measured on this
/// host, the portable lane without `cubecl-wgpu/spirv` fails exactly here.
#[test]
fn the_runtime_stores_every_element_width_the_kernels_bind() {
    slam_rs::gpu::probe_storage(&gpu_client().unwrap()).unwrap();
}

/// A host with no GPU is a typed error, in a subprocess that really has none.
///
/// A missing adapter must not raise `pyo3_runtime.PanicException`, even though
/// CubeCL unwraps its own bring-up on its worker thread and the process's
/// documented contract is a `ValueError` and never a Rust panic (decision D32).
/// It cannot be tested in-process — a client is a per-process singleton and the
/// environment is read once — so each case re-runs *this test binary* with one
/// variable changed and reads what the child printed.
///
/// The child asserts, so a child that stopped reaching the probe fails rather
/// than passing quietly; each case additionally says whether rustc's own
/// `panicked at` belongs in the child's output — for the case the probe answers
/// it must not appear, and for the one that reaches cubecl's own unwrap it
/// must, because the caught panic's message is the only account of a failure no
/// probe anticipated.
mod absent_gpu {
    use std::process::{Command, Output};

    /// Names the child answers to, so the parent can tell it which case to run.
    const CASE: &str = "SLAM_RS_ABSENT_GPU_CASE";

    /// Run this test binary again as the child of `case`, with `variables` set.
    ///
    /// `--test-threads=1` and `--nocapture` so the child's `println!` reaches
    /// the parent whatever the harness would otherwise do with it.
    fn child(test: &str, case: &str, variables: &[(&str, &str)]) -> String {
        let exe: std::path::PathBuf = std::env::current_exe().unwrap();
        run(Command::new(exe), test, case, variables)
    }

    /// Run `command` as the child of `case` and return everything it printed.
    fn run(mut command: Command, test: &str, case: &str, variables: &[(&str, &str)]) -> String {
        command
            .args(["--exact", test, "--nocapture", "--test-threads=1"])
            .env(CASE, case);
        for (name, value) in variables {
            command.env(name, value);
        }
        let output: Output = command.output().unwrap();
        let text: String = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.status.success(), "the {case} child failed:\n{text}");
        text
    }

    /// `panicked at` is rustc's own header and nothing else prints it, so it is
    /// how a case says whether a panic happened at all — which is not the same
    /// question as whether one escaped. A case that reaches the runtime's own
    /// unwrap panics and is caught; a case the probe answers never panics. Both
    /// return the typed error, and each test below says which it is.
    fn panicked(text: &str) -> bool {
        text.contains("panicked at")
    }

    /// The case this process is the child of, or `None` if it is the parent.
    fn case() -> Option<String> {
        std::env::var(CASE).ok()
    }

    /// Why this host has no client — `unwrap_err` cannot say it, because a
    /// `ComputeClient` is not `Debug`.
    fn client_error() -> slam_rs::gpu::GpuError {
        match slam_rs::gpu::gpu_client() {
            Err(error) => error,
            Ok(_) => panic!("this child was supposed to have no GPU, and it built a client"),
        }
    }

    /// The shape both cases share, run once as the child and once as the
    /// parent.
    ///
    /// In the child: the client's error is `expected`, and it is printed with
    /// the `CHILD` prefix the parent greps for. In the parent: `spawn` starts
    /// the child, its output contains `expected_text`, and whether rustc's own
    /// `panicked at` appears is exactly `expect_panic` — the two questions each
    /// shim's doc answers for its own case. `None` from `spawn` is a skip and
    /// not a pass.
    fn assert_absent_gpu_case(
        expected: slam_rs::gpu::GpuError,
        expected_text: &str,
        expect_panic: bool,
        spawn: impl FnOnce() -> Option<String>,
    ) {
        if case().is_some() {
            let error: slam_rs::gpu::GpuError = client_error();
            assert_eq!(error, expected);
            println!("CHILD {error}");
            return;
        }
        let Some(text) = spawn() else {
            println!("SKIPPED: this case's child could not be started, so the case is not a pass");
            return;
        };
        assert!(text.contains(expected_text), "{text}");
        assert_eq!(
            panicked(&text),
            expect_panic,
            "the child's `panicked at` lines are not this case's (expected {expect_panic}):\n{text}"
        );
    }

    /// No Vulkan ICD: the loader enumerates nothing and wgpu has no adapter.
    #[cfg(feature = "gpu-wgpu")]
    #[cfg_attr(
        target_os = "macos",
        ignore = "requires the Vulkan ICD loader; Metal ignores VK_DRIVER_FILES"
    )]
    #[test]
    fn a_wgpu_host_with_no_adapter_is_a_typed_error() {
        const NAME: &str = "absent_gpu::a_wgpu_host_with_no_adapter_is_a_typed_error";
        // The probe answers this one too.
        assert_absent_gpu_case(
            slam_rs::gpu::GpuError::NoAdapter { backend: "vulkan" },
            "CHILD wgpu found no vulkan adapter",
            false,
            || {
                Some(child(
                    NAME,
                    "no-adapter",
                    &[
                        ("VK_DRIVER_FILES", "/nonexistent/no-such-icd.json"),
                        // Older Vulkan loaders recognize only the legacy name.
                        ("VK_ICD_FILENAMES", "/nonexistent/no-such-icd.json"),
                    ],
                ))
            },
        );
    }

    /// `CUBECL_WGPU_DEFAULT_DEVICE` naming an index the host does not have.
    ///
    /// The case the adapter probe cannot see: cubecl-wgpu selects by
    /// enumeration here rather than by power preference, and panics on its own
    /// thread. It is what the `catch_unwind` in `gpu_client` is for, and it is
    /// the only test that exercises it.
    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn a_wgpu_device_index_past_the_end_is_a_typed_error() {
        const NAME: &str = "absent_gpu::a_wgpu_device_index_past_the_end_is_a_typed_error";
        // This one reaches cubecl's own unwrap, so the runtime's own message
        // must survive to stderr: the child caught the panic and returned the
        // typed error, and without the quiet panic hook the one clue about a
        // case no probe anticipated is still printed rather than swallowed.
        assert_absent_gpu_case(
            slam_rs::gpu::GpuError::ClientPanicked { runtime: "wgpu" },
            "CHILD building the wgpu client panicked",
            true,
            || {
                Some(child(
                    NAME,
                    "bad-index",
                    &[("CUBECL_WGPU_DEFAULT_DEVICE", "DiscreteGpu(99)")],
                ))
            },
        );
    }
}

#[cubecl::prelude::cube(launch_unchecked)]
fn finite_probe(input: &[f32], output: &mut [u32]) {
    use cubecl::prelude::*;
    let i = ABSOLUTE_POS;
    if i < 12 {
        let mut value = input[i];
        if i >= 9 {
            value = input[i] / input[4];
        }
        output[i] = u32::cast_from(finite::is_finite(value));
    }
}

#[test]
// The production helper must classify uploaded values and runtime device division.
fn finite_predicates_match_ieee_classification() {
    use cubecl::prelude::*;
    let values = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        0.0,
        1.0,
        f32::MAX,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        0.0,
        1.0,
        -1.0,
    ];
    let client = gpu_client().unwrap();
    let input = client.create_from_slice(f32::as_bytes(&values));
    let output = client.empty(12 * size_of::<u32>());
    unsafe {
        finite_probe::launch_unchecked::<GpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(32),
            BufferArg::from_raw_parts(input, 12),
            BufferArg::from_raw_parts(output.clone(), 12),
        );
    }
    let bytes = client.read_one(output).unwrap();
    let classifications = u32::from_bytes(&bytes);
    for (i, classification) in classifications.iter().enumerate() {
        let expected = (3..9).contains(&i);
        assert_eq!(
            *classification,
            u32::from(expected),
            "production bit classification at {i}"
        );
    }
}

#[path = "../src/gpu/trig.rs"]
mod trig;

#[cubecl::prelude::cube(launch_unchecked)]
fn small_angle_probe(input: &[f32], output: &mut [f32]) {
    use cubecl::prelude::*;
    let i = ABSOLUTE_POS;
    if i < input.len() {
        output[2 * i] = trig::sin(input[i]);
        output[2 * i + 1] = f32::cos(input[i]); // Native cosine is what ships.
    }
}

/// SE(2) divides sin(theta) by theta: absolute-error-only trig is insufficient.
#[test]
fn small_angle_trig_stays_within_two_ulps_of_the_cpu() {
    use cubecl::prelude::*;
    let mut values = Vec::new();
    for extent in [0.001f32, 0.5] {
        for i in 0..=1024 {
            values.push((i as f32 / 512.0 - 1.0) * extent);
        }
    }
    let client = gpu_client().unwrap();
    let input = client.create_from_slice(f32::as_bytes(&values));
    let output = client.empty(values.len() * 2 * size_of::<f32>());
    unsafe {
        small_angle_probe::launch_unchecked::<GpuRuntime>(
            &client,
            CubeCount::Static(values.len().div_ceil(256) as u32, 1, 1),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(input, values.len()),
            BufferArg::from_raw_parts(output.clone(), values.len() * 2),
        );
    }
    let bytes = client.read_one(output).unwrap();
    let actual = f32::from_bytes(&bytes);
    let mut worst = [0u32; 2];
    for (i, theta) in values.iter().enumerate() {
        for (operation, expected) in [theta.sin(), theta.cos()].iter().enumerate() {
            let measured = actual[2 * i + operation];
            assert!(measured.is_finite());
            // Same-sign floats have monotonic bit patterns. Zero signs are equivalent.
            let ulps = if measured == *expected {
                0
            } else {
                assert_eq!(measured.is_sign_negative(), expected.is_sign_negative());
                measured.to_bits().abs_diff(expected.to_bits())
            };
            worst[operation] = worst[operation].max(ulps);
            assert!(
                ulps <= 2,
                "theta={theta:e}, operation={operation}, GPU={measured:e}, CPU={expected:e}, ulps={ulps}"
            );
        }
    }
    println!("small-angle sin/cos maximum ULP errors: {worst:?}");
}

/// Two passes in flight over **one** patch set answer what two separate calls do.
///
/// This is the claim the batched tracker rests on: a batch's passes share every
/// intermediate buffer — the source and backward patch stores, the backward
/// transforms — and may, because the device stream is ordered, so pass 0's
/// `finish` has read them before pass 1's kernels write them. Only the packed
/// result is per lane. If that ordering did not hold, the second `prepare` would
/// corrupt the first pass and lane 0 would come back wrong; the reference here
/// is the same two passes run one at a time, which is what the frontend did
/// before D77.
#[test]
fn a_batch_of_two_passes_answers_what_two_calls_do() {
    const SIZE: usize = 512;
    let first: Image<u16, 1> = textured_image(SIZE, SIZE, 0.0, 0.0);
    let second: Image<u16, 1> = textured_image(SIZE, SIZE, 2.75, -1.5);
    let lane0: PointsSoA = grid_positions(SIZE);
    // A different pass, so a batch that answered both lanes from one of them
    // would be caught: half the patches, a quarter-pixel off the grid.
    let mut lane1: PointsSoA = PointsSoA::default();
    for index in (0..lane0.len()).step_by(2) {
        let point = lane0.get(index);
        lane1.push(Vector2::new(point.x + 0.25, point.y - 0.25));
    }
    let guesses: [FlowTransforms; 2] = [guesses_at(&lane0), guesses_at(&lane1)];
    let points: [&PointsSoA; 2] = [&lane0, &lane1];

    let client = gpu_client().unwrap();
    let mut builder = GpuPyramidBuilder::new(client.clone(), Default::default());
    let mut prev = builder.allocate(SIZE, SIZE, LEVELS).unwrap();
    let mut next = builder.allocate(SIZE, SIZE, LEVELS).unwrap();
    builder.build(0, &first, &mut prev).unwrap();
    builder.build(0, &second, &mut next).unwrap();

    let mut tracker: GpuPatchTracker<Pattern51, _> = GpuPatchTracker::new(
        client.clone(),
        MAX_KEYPOINTS,
        LEVELS + 1,
        MAX_ITERATIONS,
        MAX_RECOVERED_DIST2,
        2,
        Default::default(),
    )
    .unwrap();
    let mut patches: GpuPatchSources<Pattern51, _> = tracker.make_patches().unwrap();

    // ── one at a time, which is the reference
    let mut alone: Vec<FlowResult> = Vec::new();
    for lane in 0..2 {
        let mut out: FlowResult = FlowResult::with_capacity(MAX_KEYPOINTS);
        patches.build(&prev, points[lane], None).unwrap();
        tracker
            .track(&prev, &next, &patches, &guesses[lane], &mut out)
            .unwrap();
        alone.push(out);
    }

    // ── both launched, then one download
    let mut passes = Vec::new();
    for lane in 0..2 {
        patches.build(&prev, points[lane], None).unwrap();
        passes.push(
            tracker
                .submit(&prev, &next, &patches, &guesses[lane])
                .unwrap(),
        );
    }
    tracker.collect().unwrap();

    for lane in 0..2 {
        assert_eq!(
            tracker.result(passes[lane]).tracked(),
            alone[lane].tracked(),
            "lane {lane} kept a different set out of the batch"
        );
        for index in 0..points[lane].len() {
            assert_eq!(
                tracker.result(passes[lane]).is_valid(index),
                alone[lane].is_valid(index),
                "lane {lane}: patch {index} survived out of the batch and not alone"
            );
            if alone[lane].is_valid(index) {
                assert_eq!(
                    tracker.result(passes[lane]).transform(index).translation,
                    alone[lane].transform(index).translation,
                    "lane {lane}: patch {index} moved"
                );
            }
        }
    }
    assert_ne!(
        alone[0].tracked().len(),
        alone[1].tracked().len(),
        "the two lanes tracked the same set, so crossing them would not show"
    );
}

/// The fused kernel preserves accepted translations and failure flags,
/// including a partial final workgroup and camera boundaries inside subgroups.
#[test]
fn fused_temporal_batch_matches_cpu() {
    use slam_rs::frontend::tracker::TrackInput;
    let client = gpu_client().unwrap();
    let mut builder = CpuPyramidBuilder::new();
    let mut pyramids = [Vec::new(), Vec::new()];
    let mut images = [Vec::new(), Vec::new()];
    let mut positions = Vec::new();
    let mut guesses = Vec::new();
    let mut expected = Vec::new();
    for camera in 0..4 {
        for (frame, frame_pyramids) in pyramids.iter_mut().enumerate() {
            let mut image = packed_texture(512, 512, frame as f32 * 2.75, frame as f32 * -1.5);
            if camera == 3 {
                slam_rs::image::fill_from_u8_strided(
                    &mut image,
                    &vec![0; 512 * 512],
                    512,
                    512,
                    512,
                )
                .unwrap();
            }
            let mut pyramid = builder.allocate(512, 512, LEVELS).unwrap();
            builder.build(camera, &image, &mut pyramid).unwrap();
            frame_pyramids.push(pyramid);
            images[frame].push(image);
        }
        let mut points = PointsSoA::with_capacity(33);
        let mut input = FlowTransforms::with_capacity(33);
        for i in 0..33 {
            let point = Vector2::new(90.0 + (i % 6) as f32 * 55.0, 90.0 + (i / 6) as f32 * 55.0);
            points.push(point);
            let mut guess = point;
            if i % 11 == 0 {
                guess.x = -5.0;
            }
            input.push(&AffineCompact2f::at(guess));
        }
        let mut tracker = CpuPatchTracker::<Pattern51>::new(
            33,
            LEVELS + 1,
            MAX_ITERATIONS,
            MAX_RECOVERED_DIST2,
            WorkPool::new(1).unwrap(),
        )
        .unwrap();
        let mut patches = tracker.make_patches().unwrap();
        patches.build(&pyramids[0][camera], &points, None).unwrap();
        let mut result = FlowResult::with_capacity(33);
        tracker
            .track(
                &pyramids[0][camera],
                &pyramids[1][camera],
                &patches,
                &input,
                &mut result,
            )
            .unwrap();
        expected.push(result);
        positions.push(points);
        guesses.push(input);
    }
    assert!(
        expected
            .iter()
            .take(3)
            .all(|r| (0..33).filter(|&i| r.is_valid(i)).count() >= 25)
    );
    let mut builder = GpuPyramidBuilder::new(client.clone(), Default::default());
    let mut gpu_pyramids: Vec<Vec<_>> = images
        .iter()
        .map(|frame| {
            frame
                .iter()
                .map(|image| {
                    builder
                        .allocate(image.width(), image.height(), LEVELS)
                        .unwrap()
                })
                .collect()
        })
        .collect();
    for frame in 0..2 {
        builder
            .build_frames(
                &images[frame],
                &mut gpu_pyramids[frame],
                &WorkPool::new(1).unwrap(),
            )
            .unwrap();
    }
    let mut tracker = GpuPatchTracker::<Pattern51, _>::new(
        client,
        33,
        LEVELS + 1,
        MAX_ITERATIONS,
        MAX_RECOVERED_DIST2,
        4,
        Default::default(),
    )
    .unwrap();
    let mut patches = tracker.make_patches().unwrap();
    let mut inputs: Vec<_> = positions
        .into_iter()
        .zip(guesses)
        .enumerate()
        .map(|(camera, (positions, guesses))| TrackInput {
            source: camera,
            destination: camera,
            positions,
            guesses,
            ..Default::default()
        })
        .collect();
    tracker
        .submit_batch(
            &gpu_pyramids[0],
            &gpu_pyramids[1],
            &mut inputs,
            &mut patches,
            true,
        )
        .unwrap();
    tracker.collect().unwrap();
    for (camera, input) in inputs.iter().enumerate() {
        let actual = tracker.result(input.result);
        let cpu = &expected[camera];
        for point in 0..33 {
            assert_eq!(
                actual.is_valid(point),
                cpu.is_valid(point),
                "camera {camera}, point {point}"
            );
            if cpu.is_valid(point) {
                let delta = actual.transform(point).translation - cpu.transform(point).translation;
                assert!(
                    delta.norm() < LANE_POSITION_BOUND,
                    "camera {camera}, point {point}, error {}",
                    delta.norm()
                );
            }
        }
    }
}
