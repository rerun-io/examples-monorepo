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
//! * the patch build and the tracker are **not** exact, for one reason: NVRTC
//!   contracts `a * b + c` into a fused multiply-add, which the CPU does not.
//!   Contraction only ever raises the accuracy of a term, but it changes the
//!   bits, and a Gauss-Newton fixed point amplifies the change until the
//!   iteration converges. The bounds below are what that costs.
//!
//! These run only under `--features gpu` and need a working CubeCL runtime;
//! `cargo test --features gpu` is the gate.
#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use kornia_imgproc::features::FastCorner;
use nalgebra::Vector2;
use slam_rs::frontend::detect::{CornerScan, CpuCornerScan};
use slam_rs::frontend::parallel::WorkPool;
use slam_rs::frontend::patch::OpticalFlowPatch;
use slam_rs::frontend::patterns::{Pattern, Pattern51};
use slam_rs::frontend::se2::AffineCompact2f;
use slam_rs::frontend::tracker::{
    CpuPatchTracker, FlowResult, FlowTransforms, PatchSoA, PatchTracker, PointsSoA, SourcePatches,
};
use slam_rs::gpu::{GpuCornerScan, GpuPatchTracker, GpuPatches, GpuPyramidBuilder, cuda_client};
use slam_rs::image::ImageU16;
use slam_rs::pyramid::{CpuPyramidBuilder, Pyramid, PyramidBuilder, PyramidU16};

/// The keypoint budget both lanes are sized for, well over what the grid needs.
const MAX_KEYPOINTS: usize = 1024;
/// `optical_flow_max_iterations` in every shipped config.
const MAX_ITERATIONS: usize = 5;
/// `optical_flow_max_recovered_dist2` in every shipped config.
const MAX_RECOVERED_DIST2: f32 = 0.09;

/// Twelve plane waves between 16 and 56 pixels, in fixed pseudo-random
/// directions and phases: the same band-limited field the CPU tracker's tests
/// use, so a shift really does survive down the pyramid and every patch's
/// `H_se2` is well conditioned.
fn texture(x: f64, y: f64) -> f64 {
    const WAVES: [(f64, f64, f64); 12] = [
        (16.0, 0.031, 0.11),
        (19.0, 0.187, 0.37),
        (23.0, 0.311, 0.63),
        (27.0, 0.451, 0.05),
        (31.0, 0.077, 0.81),
        (35.0, 0.229, 0.29),
        (39.0, 0.383, 0.55),
        (43.0, 0.497, 0.73),
        (47.0, 0.143, 0.19),
        (51.0, 0.271, 0.91),
        (54.0, 0.419, 0.43),
        (56.0, 0.353, 0.67),
    ];
    let mut total: f64 = 0.0;
    for (wavelength, direction, phase) in WAVES {
        let angle: f64 = std::f64::consts::TAU * direction;
        let projection: f64 = x * angle.cos() + y * angle.sin();
        total += (std::f64::consts::TAU * (projection / wavelength + phase)).sin();
    }
    total / WAVES.len() as f64
}

/// The texture rendered into a `u16` image, shifted by `(dx, dy)`.
fn textured_image(width: usize, height: usize, dx: f32, dy: f32) -> ImageU16 {
    let mut image: ImageU16 = ImageU16::zeros(width, height).unwrap();
    for y in 0..height {
        for x in 0..width {
            let value: f64 = texture(x as f64 - f64::from(dx), y as f64 - f64::from(dy));
            let scaled: f64 = (value * 0.4 + 0.5) * 65535.0;
            image.set(x, y, scaled.clamp(0.0, 65535.0) as u16);
        }
    }
    image
}

/// The pyramid geometry the shipped msd configs run: `optical_flow_levels = 3`
/// on a 960x960 frame, so four levels.
const LEVELS: usize = 3;

#[test]
fn the_gpu_pyramid_is_bit_exact_with_the_cpu() {
    let image: ImageU16 = textured_image(960, 960, 0.0, 0.0);

    let mut cpu_builder: CpuPyramidBuilder = CpuPyramidBuilder::new();
    let mut cpu: PyramidU16 = cpu_builder.allocate(960, 960, LEVELS).unwrap();
    cpu_builder.build(0, &image, &mut cpu).unwrap();

    let client = cuda_client();
    let mut gpu_builder = GpuPyramidBuilder::new(client, &[[0.0, 0.0]]);
    let mut gpu = gpu_builder.allocate(960, 960, LEVELS).unwrap();
    gpu_builder.build(0, &image, &mut gpu).unwrap();

    assert_eq!(gpu.num_levels(), cpu.num_levels());
    let mut expected: ImageU16 = ImageU16::default();
    let mut actual: ImageU16 = ImageU16::default();
    for level in 0..cpu.num_levels() {
        assert_eq!(
            gpu.level_size(level),
            cpu.level_size(level),
            "level {level}"
        );
        cpu.copy_level_into(level, &mut expected).unwrap();
        gpu.copy_level_into(level, &mut actual).unwrap();
        let mut worst: i64 = 0;
        let mut worst_at: (usize, usize) = (0, 0);
        for y in 0..expected.height() {
            for x in 0..expected.width() {
                let difference: i64 =
                    i64::from(actual.get(x, y).unwrap()) - i64::from(expected.get(x, y).unwrap());
                if difference.abs() > worst {
                    worst = difference.abs();
                    worst_at = (x, y);
                }
            }
        }
        assert_eq!(
            worst,
            0,
            "level {level} differs: max-abs-diff {worst} at {worst_at:?} \
             ({}x{}); the fused 5x5 pass is integer arithmetic and must be exact",
            expected.width(),
            expected.height()
        );
    }
}

/// A frame whose stride is wider than its width — dav1d's shape — must upload
/// the same pixels, not the padding.
#[test]
fn a_strided_frame_uploads_its_rows_and_not_its_padding() {
    let source: ImageU16 = textured_image(64, 48, 0.0, 0.0);
    let mut strided: ImageU16 = ImageU16::zeros_with_stride(64, 48, 96).unwrap();
    for y in 0..48 {
        strided.row_mut(y).copy_from_slice(source.row(y));
    }

    let mut cpu_builder: CpuPyramidBuilder = CpuPyramidBuilder::new();
    let mut cpu: PyramidU16 = cpu_builder.allocate(64, 48, LEVELS).unwrap();
    cpu_builder.build(0, &strided, &mut cpu).unwrap();

    let mut gpu_builder = GpuPyramidBuilder::new(cuda_client(), &[[0.0, 0.0]]);
    let mut gpu = gpu_builder.allocate(64, 48, LEVELS).unwrap();
    gpu_builder.build(0, &strided, &mut gpu).unwrap();

    let mut expected: ImageU16 = ImageU16::default();
    let mut actual: ImageU16 = ImageU16::default();
    for level in 0..cpu.num_levels() {
        cpu.copy_level_into(level, &mut expected).unwrap();
        gpu.copy_level_into(level, &mut actual).unwrap();
        assert_eq!(actual.data(), expected.data(), "level {level}");
    }
}

/// The pyramid the builder allocates is reused frame after frame, so the second
/// frame must not see the first one's pixels anywhere.
#[test]
fn a_reused_pyramid_carries_only_the_newest_frame() {
    let first: ImageU16 = textured_image(128, 96, 0.0, 0.0);
    let second: ImageU16 = textured_image(128, 96, 7.0, -3.0);

    let mut gpu_builder = GpuPyramidBuilder::new(cuda_client(), &[[0.0, 0.0]]);
    let mut gpu = gpu_builder.allocate(128, 96, LEVELS).unwrap();
    gpu_builder.build(0, &first, &mut gpu).unwrap();
    gpu_builder.build(0, &second, &mut gpu).unwrap();

    let mut cpu_builder: CpuPyramidBuilder = CpuPyramidBuilder::new();
    let mut cpu: PyramidU16 = cpu_builder.allocate(128, 96, LEVELS).unwrap();
    cpu_builder.build(0, &second, &mut cpu).unwrap();

    let mut expected: ImageU16 = ImageU16::default();
    let mut actual: ImageU16 = ImageU16::default();
    for level in 0..cpu.num_levels() {
        cpu.copy_level_into(level, &mut expected).unwrap();
        gpu.copy_level_into(level, &mut actual).unwrap();
        assert_eq!(actual.data(), expected.data(), "level {level}");
    }
}

/// A grid of source positions well inside a `size` x `size` frame, spaced so no
/// two patches overlap and every one is far enough from the border for the
/// coarsest level's 52-tap pattern.
fn grid_positions(size: usize) -> PointsSoA {
    let mut positions: PointsSoA = PointsSoA::with_capacity(256);
    let mut y: usize = 96;
    while y + 96 < size {
        let mut x: usize = 96;
        while x + 96 < size {
            positions.push(Vector2::new(x as f32 + 0.37, y as f32 - 0.21));
            x += 71;
        }
        y += 71;
    }
    positions
}

#[test]
fn the_gpu_patch_build_matches_the_cpu_within_tolerance() {
    let image: ImageU16 = textured_image(512, 512, 0.0, 0.0);
    let positions: PointsSoA = grid_positions(512);
    let count: usize = positions.len();
    assert!(count > 20, "the grid produced only {count} patches");

    let mut cpu_builder: CpuPyramidBuilder = CpuPyramidBuilder::new();
    let mut cpu: PyramidU16 = cpu_builder.allocate(512, 512, LEVELS).unwrap();
    cpu_builder.build(0, &image, &mut cpu).unwrap();

    let client = cuda_client();
    let mut gpu_builder = GpuPyramidBuilder::new(client.clone(), Pattern51::OFFSETS);
    let mut gpu = gpu_builder.allocate(512, 512, LEVELS).unwrap();
    gpu_builder.build(0, &image, &mut gpu).unwrap();

    let mut patches: GpuPatches<Pattern51, _> =
        GpuPatches::new(client, MAX_KEYPOINTS, LEVELS + 1).unwrap();
    patches.build(&gpu, &positions, None).unwrap();
    let store: Vec<f32> = patches.read_store().unwrap();

    let mut worst_data: f32 = 0.0;
    let mut worst_jacobian: f32 = 0.0;
    let mut jacobian_scale: f32 = 0.0;
    let mut level_image: ImageU16 = ImageU16::default();
    for level in 0..=LEVELS {
        cpu.copy_level_into(level, &mut level_image).unwrap();
        let scale: f32 = (1u32 << level) as f32;
        for patch in 0..count {
            let reference: OpticalFlowPatch<Pattern51> =
                OpticalFlowPatch::new(&level_image, positions.get(patch) / scale);
            assert_eq!(
                patches.valid_in(&store, level, patch),
                reference.valid,
                "validity differs at level {level}, patch {patch}"
            );
            for tap in 0..Pattern51::SIZE {
                worst_data = worst_data
                    .max((patches.data_in(&store, level, tap, patch) - reference.data[tap]).abs());
                for row in 0..3 {
                    let expected: f32 = reference.h_se2_inv_j_se2_t[row][tap];
                    let actual: f32 = patches.jacobian_in(&store, level, row, tap, patch);
                    worst_jacobian = worst_jacobian.max((actual - expected).abs());
                    jacobian_scale = jacobian_scale.max(expected.abs());
                }
            }
        }
    }

    println!(
        "patch build: data max-abs-diff {worst_data:.3e}, H^-1 J^T max-abs-diff \
         {worst_jacobian:.3e} on a largest coefficient of {jacobian_scale:.3e} \
         ({:.2e} relative), over {count} patches x {} taps x {} levels",
        worst_jacobian / jacobian_scale,
        Pattern51::SIZE,
        LEVELS + 1
    );
    // The taps are mean-normalised, so `data` sits near 1 and an absolute bound
    // is a relative one. Fused multiply-add is the whole difference.
    assert!(
        worst_data < 1e-5,
        "patch data max-abs-diff {worst_data} over {count} patches x {} taps x {} levels",
        Pattern51::SIZE,
        LEVELS + 1
    );
    // `H^-1 J^T` inherits `H`'s conditioning, so the bound is relative to the
    // largest coefficient the reference produced on this texture.
    assert!(
        worst_jacobian < 1e-3 * jacobian_scale,
        "H^-1 J^T max-abs-diff {worst_jacobian} against a largest coefficient of \
         {jacobian_scale} ({:.2e} relative)",
        worst_jacobian / jacobian_scale
    );
}

#[test]
fn the_gpu_tracker_recovers_the_same_shift_as_the_cpu() {
    const SHIFT: f32 = 2.75;
    let first: ImageU16 = textured_image(512, 512, 0.0, 0.0);
    let second: ImageU16 = textured_image(512, 512, SHIFT, -1.5);
    let positions: PointsSoA = grid_positions(512);
    let count: usize = positions.len();

    // ── the CPU lane
    let mut cpu_builder: CpuPyramidBuilder = CpuPyramidBuilder::new();
    let mut cpu_prev: PyramidU16 = cpu_builder.allocate(512, 512, LEVELS).unwrap();
    let mut cpu_next: PyramidU16 = cpu_builder.allocate(512, 512, LEVELS).unwrap();
    cpu_builder.build(0, &first, &mut cpu_prev).unwrap();
    cpu_builder.build(0, &second, &mut cpu_next).unwrap();
    let mut cpu_tracker: CpuPatchTracker<Pattern51> = CpuPatchTracker::new(
        MAX_KEYPOINTS,
        LEVELS + 1,
        MAX_ITERATIONS,
        MAX_RECOVERED_DIST2,
        WorkPool::new(1).unwrap(),
    )
    .unwrap();
    let mut cpu_patches: PatchSoA<Pattern51> = cpu_tracker.make_patches().unwrap();
    cpu_patches.build(&cpu_prev, &positions, None).unwrap();
    let mut guesses: FlowTransforms = FlowTransforms::with_capacity(count);
    guesses.resize(count);
    for index in 0..count {
        guesses.set(index, &AffineCompact2f::at(positions.get(index)));
    }
    let mut cpu_result: FlowResult = FlowResult::with_capacity(MAX_KEYPOINTS);
    cpu_tracker
        .track(
            &cpu_prev,
            &cpu_next,
            &cpu_patches,
            &guesses,
            &mut cpu_result,
        )
        .unwrap();

    // ── the GPU lane
    let client = cuda_client();
    let mut gpu_builder = GpuPyramidBuilder::new(client.clone(), Pattern51::OFFSETS);
    let mut gpu_prev = gpu_builder.allocate(512, 512, LEVELS).unwrap();
    let mut gpu_next = gpu_builder.allocate(512, 512, LEVELS).unwrap();
    gpu_builder.build(0, &first, &mut gpu_prev).unwrap();
    gpu_builder.build(0, &second, &mut gpu_next).unwrap();
    let mut gpu_tracker: GpuPatchTracker<Pattern51, _> = GpuPatchTracker::new(
        client.clone(),
        MAX_KEYPOINTS,
        LEVELS + 1,
        MAX_ITERATIONS,
        MAX_RECOVERED_DIST2,
    )
    .unwrap();
    let mut gpu_patches: GpuPatches<Pattern51, _> = gpu_tracker.make_patches().unwrap();
    gpu_patches.build(&gpu_prev, &positions, None).unwrap();
    let mut gpu_result: FlowResult = FlowResult::with_capacity(MAX_KEYPOINTS);
    gpu_tracker
        .track(
            &gpu_prev,
            &gpu_next,
            &gpu_patches,
            &guesses,
            &mut gpu_result,
        )
        .unwrap();

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
    let mut agreed: usize = 0;
    let mut worst_position: f32 = 0.0;
    for index in 0..count {
        if cpu_result.is_valid(index) == gpu_result.is_valid(index) {
            agreed += 1;
        }
        if cpu_result.is_valid(index) && gpu_result.is_valid(index) {
            let difference =
                cpu_result.transform(index).translation - gpu_result.transform(index).translation;
            worst_position = worst_position.max(difference.norm());
        }
    }
    println!(
        "tracker: {tracked} of {count} kept, worst recovered shift GPU {worst_shift:.4} px \
         / CPU {worst_cpu_shift:.4} px, worst lane-to-lane position \
         {worst_position:.3e} px, converged flag agreed on {agreed} of {count}"
    );
    // A Gauss-Newton fixed point reached from the same start converges to the
    // same place; fused multiply-add moves the last steps, not the answer.
    assert!(
        worst_position < 0.01,
        "positions differ by up to {worst_position} px between the lanes"
    );
    // The flag can differ only where a patch sits on a threshold — the tap
    // count, the increment norm, the two-pixel border — so the bound is a
    // fraction, not equality (decisions D09/D45: corner and track choice moves
    // trajectories, and that is what the clip-level gate measures).
    assert!(
        agreed * 100 >= count * 98,
        "the two lanes agreed on the converged flag for {agreed} of {count} patches"
    );
}

/// A textured 8-bit field with fine detail, so FAST has plenty to find: the
/// smooth plane-wave texture above gives almost no corners.
fn cornered_image(width: usize, height: usize) -> ImageU16 {
    let mut image: ImageU16 = ImageU16::zeros(width, height).unwrap();
    let mut state: u32 = 0x1234_5678;
    for y in 0..height {
        for x in 0..width {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let wave: i32 = ((x as f64 / 11.0).sin() * 60.0 + (y as f64 / 7.0).cos() * 50.0) as i32;
            let noise: i32 = (state >> 24) as i32 / 4;
            // The detector reads `pixel >> 8`, so the 8-bit value goes in the
            // high byte.
            image.set(x, y, ((128 + wave + noise).clamp(0, 255) as u16) << 8);
        }
    }
    image
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
        let image: ImageU16 = cornered_image(width, height);
        let mut cpu: CpuCornerScan = CpuCornerScan::default();
        let mut gpu: GpuCornerScan<_> = GpuCornerScan::new(cuda_client());
        cpu.scan(0, &image).unwrap();
        gpu.scan(0, &image).unwrap();

        // The row bands the detector asks for: one per grid row of a 50-pixel
        // cell, at every rung of the shipped ladder.
        let mut total: usize = 0;
        for band_y in (3..height - 3).step_by(50) {
            for threshold in [40i32, 20, 10, 5, 1] {
                let expected: Vec<FastCorner> = cpu.band(band_y, 44, threshold).unwrap().to_vec();
                let actual: &[FastCorner] = gpu.band(band_y, 44, threshold).unwrap();
                assert_eq!(
                    actual.len(),
                    expected.len(),
                    "{width}x{height} band {band_y} threshold {threshold}: \
                     GPU {} corners, kornia {}",
                    actual.len(),
                    expected.len()
                );
                for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
                    assert_eq!(
                        (got.xy, got.response),
                        (want.xy, want.response),
                        "{width}x{height} band {band_y} threshold {threshold}, corner {index}"
                    );
                }
                total += expected.len();
            }
        }
        println!("{width}x{height}: {total} corners over every band and rung, identical");
    }
}

/// The scanner is reused frame after frame, so the second frame's bands must be
/// the second frame's.
#[test]
fn a_reused_corner_scan_carries_only_the_newest_frame() {
    let first: ImageU16 = cornered_image(512, 128);
    let second: ImageU16 = ImageU16::zeros(512, 128).unwrap();

    let mut gpu: GpuCornerScan<_> = GpuCornerScan::new(cuda_client());
    gpu.scan(0, &first).unwrap();
    assert!(
        !gpu.band(3, 44, 5).unwrap().is_empty(),
        "the textured frame has corners"
    );
    gpu.scan(0, &second).unwrap();
    assert!(
        gpu.band(3, 44, 5).unwrap().is_empty(),
        "a black frame has none"
    );
}

/// The detector reads the pyramid's level 0, and uploads nothing.
///
/// The corner scanner and the pyramid builder are handed the same pixels once
/// per camera per frameset — the detector's input *is* level 0
/// (`keypoints.cpp:152`, `image_pyr.h:73`) — and until the camera index reached
/// both seams the GPU scanner had no way to know that, so it uploaded the frame
/// a second time. This is the test that the sharing is exact rather than merely
/// cheaper: the same corners at every rung, from a scanner that uploaded zero
/// frames, on two cameras of different geometry so the per-camera table is
/// actually indexed.
#[test]
fn the_gpu_corner_scan_reads_the_pyramid_and_uploads_nothing() {
    let frames: [ImageU16; 2] = [cornered_image(960, 240), cornered_image(512, 192)];
    let client = cuda_client();

    // The lane the frontend runs: one builder, one scanner, one client, the
    // level-0 table between them.
    let mut builder = GpuPyramidBuilder::new(client.clone(), &[[0.0, 0.0]]);
    let mut shared: GpuCornerScan<_> = GpuCornerScan::new(client.clone());
    shared.share_level0(builder.level0_table());
    // The lane before this change: the scanner uploads its own copy.
    let mut alone: GpuCornerScan<_> = GpuCornerScan::new(client);

    let mut pyramids: Vec<_> = frames
        .iter()
        .map(|frame| builder.allocate(frame.width(), frame.height(), 3).unwrap())
        .collect();
    for (camera, frame) in frames.iter().enumerate() {
        builder.build(camera, frame, &mut pyramids[camera]).unwrap();
    }
    for (camera, frame) in frames.iter().enumerate() {
        shared.scan(camera, frame).unwrap();
        alone.scan(camera, frame).unwrap();
        let mut total: usize = 0;
        for band_y in (3..frame.height() - 3).step_by(50) {
            for threshold in [40i32, 20, 10, 5, 1] {
                let expected: Vec<FastCorner> = alone.band(band_y, 44, threshold).unwrap().to_vec();
                let actual: &[FastCorner] = shared.band(band_y, 44, threshold).unwrap();
                assert_eq!(
                    actual.len(),
                    expected.len(),
                    "camera {camera} band {band_y} threshold {threshold}"
                );
                for (got, want) in actual.iter().zip(expected.iter()) {
                    assert_eq!((got.xy, got.response), (want.xy, want.response));
                }
                total += expected.len();
            }
        }
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

    // A frame whose geometry does not match the published entry is refused
    // rather than read: the fallback upload is what keeps a stale table safe.
    let odd: ImageU16 = cornered_image(256, 128);
    shared.scan(0, &odd).unwrap();
    assert_eq!(shared.frame_uploads(), 1);
}

/// The per-frame path keeps CubeCL's pool bounded.
///
/// The lane held 670 MiB to 1.4 GiB over idle for about 6 MB of pyramids and
/// patch storage, which is irrelevant on a 32 GB card and decisive on the cap's
/// shared 8 GB. The cause is not the data: `cubecl-cuda` sizes its pools from
/// the device — `max_page_size = total / 4`, then `MemoryConfiguration::SubSlices`
/// lays a geometric ladder of pools down to 8 MB pages — and `RuntimeOptions`
/// is built inside `DeviceService::init`, so a client cannot ask for anything
/// smaller. What a caller controls is how many allocations per frame it hands
/// the pool and how many distinct sizes they come in. This measures that, over
/// enough framesets that a leak would show, and holds the reserved bytes to a
/// ceiling the 8 GB lane can afford.
#[test]
fn the_per_frame_path_holds_the_pool_flat() {
    const FRAMES: usize = 200;
    /// Two 960x960 four-level pyramids are 4.7 MB and the scan buffers 3.9 MB.
    /// A pool that reuses its pages sits far under this; one that grows a page
    /// per frame passes it before frame twenty.
    const RESERVED_CEILING: u64 = 256 * 1024 * 1024;

    let client = cuda_client();
    let mut builder = GpuPyramidBuilder::new(client.clone(), Pattern51::OFFSETS);
    let mut scanner: GpuCornerScan<_> = GpuCornerScan::new(client.clone());
    scanner.share_level0(builder.level0_table());

    let frames: [ImageU16; 2] = [cornered_image(960, 960), cornered_image(960, 960)];
    let mut pyramids: Vec<_> = frames
        .iter()
        .map(|frame| builder.allocate(frame.width(), frame.height(), 3).unwrap())
        .collect();

    let mut worst: u64 = 0;
    for frame_index in 0..FRAMES {
        for (camera, frame) in frames.iter().enumerate() {
            builder.build(camera, frame, &mut pyramids[camera]).unwrap();
            scanner.scan(camera, frame).unwrap();
            // One band, so the download and the host-side walk run too.
            scanner.band(3, 44, 20).unwrap();
        }
        if let Ok(usage) = client.memory_usage() {
            worst = worst.max(usage.bytes_reserved);
            if frame_index == 0 || frame_index == 9 || frame_index + 1 == FRAMES {
                println!(
                    "frame {}: {} allocs, {:.2} MiB in use, {:.2} MiB reserved",
                    frame_index + 1,
                    usage.number_allocs,
                    usage.bytes_in_use as f64 / (1024.0 * 1024.0),
                    usage.bytes_reserved as f64 / (1024.0 * 1024.0),
                );
            }
        }
    }
    println!(
        "worst reserved over {FRAMES} framesets: {:.2} MiB",
        worst as f64 / (1024.0 * 1024.0)
    );
    assert!(
        worst < RESERVED_CEILING,
        "CubeCL reserved {worst} bytes over {FRAMES} framesets of two 960x960 cameras"
    );
}
