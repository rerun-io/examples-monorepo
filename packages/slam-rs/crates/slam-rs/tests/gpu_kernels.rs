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

use kornia_image::Image;
use kornia_staging_gpu::runtime::gpu_client;
use kornia_staging_imgproc::optical_flow::patch_se2::AffineCompact2f;
use kornia_staging_imgproc::optical_flow::patch_se2::Pattern51;
use kornia_staging_imgproc::optical_flow::patch_tracker::{FlowResult, FlowTransforms, PointsSoA, PatchTrackerPlan, PatchSoA};
use kornia_staging_slam::tracking::optical_flow::{
    PatchTracker, TrackInput, TrackPhase,
};
use nalgebra::Vector2;
use slam_rs::frontend::parallel::WorkPool;
use slam_rs::gpu::{GpuPatchSources, GpuPatchTracker, GpuPyramidBuilder};
use slam_rs::pyramid::{CpuPyramidBuilder, Pyramid, PyramidBuilder};

const LANE_POSITION_BOUND: f32 = 1e-3;
mod common;

use common::gpu::{LEVELS, MAX_ITERATIONS, MAX_KEYPOINTS, MAX_RECOVERED_DIST2, guesses_at};
use common::{grid_positions, textured_image};

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

    let mut builder = GpuPyramidBuilder::new(gpu_client().unwrap(), Default::default());
    let mut pyramid = builder.allocate(64, 48, LEVELS).unwrap();
    builder.build(0, &strided, &mut pyramid).unwrap();
    let mut actual = slam_rs::image::empty();
    pyramid.copy_level_into(0, &mut actual).unwrap();
    assert_eq!(actual.as_slice(), strided.as_slice());
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
        lane1.push(Vector2::new(point[0] + 0.25, point[1] - 0.25));
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

    let inputs: Vec<_> = (0..2).map(|lane| TrackInput { ids: (0..points[lane].len() as u64).collect(), positions: points[lane].clone(), guesses: guesses[lane].clone() }).collect();
    let mut alone = Vec::new();
    for input in &inputs {
        let mut slot = [0];
        tracker.submit_batch(std::slice::from_ref(&prev), std::slice::from_ref(&next), TrackPhase::Temporal(std::slice::from_ref(input)), &mut patches, &mut slot).unwrap();
        // A second submit must preserve the first result until collection.
        assert!(matches!(tracker.submit_batch(std::slice::from_ref(&prev), std::slice::from_ref(&next), TrackPhase::Temporal(std::slice::from_ref(input)), &mut patches, &mut slot), Err(slam_rs::frontend::flow::FrontendError::Tracker(kornia_staging_imgproc::optical_flow::patch_tracker::TrackerError::PendingBatch))));
        tracker.collect().unwrap();
        alone.push(tracker.result(slot[0]).clone());
    }
    let mut passes = [0; 2];
    let mut prev_second = builder.allocate(SIZE, SIZE, LEVELS).unwrap();
    let mut next_second = builder.allocate(SIZE, SIZE, LEVELS).unwrap();
    builder.build(1, &first, &mut prev_second).unwrap();
    builder.build(1, &second, &mut next_second).unwrap();
    tracker.submit_batch(&[prev, prev_second], &[next, next_second], TrackPhase::Temporal(&inputs), &mut patches, &mut passes).unwrap();
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
        let mut tracker = PatchTrackerPlan::<Pattern51>::new(
            33,
            LEVELS + 1,
            MAX_ITERATIONS,
            MAX_RECOVERED_DIST2,
            WorkPool::new(1).unwrap().rayon_pool(),
        )
        .unwrap();
        let mut patches = PatchSoA::<Pattern51>::new(33, LEVELS + 1).unwrap();
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
    let inputs: Vec<_> = positions
        .into_iter()
        .zip(guesses)
        .map(|(positions, guesses)| TrackInput {
            ids: (0..positions.len() as u64).collect(),
            positions,
            guesses,
        })
        .collect();
    let mut slots = [0; 4];
    tracker
        .submit_batch(
            &gpu_pyramids[0],
            &gpu_pyramids[1],
            TrackPhase::Temporal(&inputs),
            &mut patches,
            &mut slots,
        )
        .unwrap();
    tracker.collect().unwrap();
    for camera in 0..inputs.len() {
        let actual = tracker.result(slots[camera]);
        let cpu = &expected[camera];
        for point in 0..33 {
            assert_eq!(
                actual.is_valid(point),
                cpu.is_valid(point),
                "camera {camera}, point {point}"
            );
            if cpu.is_valid(point) {
                let delta = Vector2::from(actual.transform(point).translation)
                    - Vector2::from(cpu.transform(point).translation);
                assert!(
                    delta.norm() < LANE_POSITION_BOUND,
                    "camera {camera}, point {point}, error {}",
                    delta.norm()
                );
            }
        }
    }
}

use kornia_staging_imgproc::test_fixtures::packed_texture;
