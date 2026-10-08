use criterion::{criterion_group, criterion_main, Criterion};
use kornia_image::{Image, ImageSize};
use kornia_staging_imgproc::optical_flow::{
    patch_se2::{AffineCompact2f, Pattern51},
    patch_tracker::*,
};
use kornia_staging_imgproc::pyramid::PyramidPlanU16;
use kornia_staging_slam::tracking::optical_flow::{
    CpuPatchTracker, PatchTracker, TrackInput, TrackPhase,
};
use std::hint::black_box;
fn tracking(c: &mut Criterion) {
    let pixels = (0..320 * 240)
        .map(|i| (i as u32).wrapping_mul(7919) as u16)
        .collect();
    let image = Image::new(
        ImageSize {
            width: 320,
            height: 240,
        },
        pixels,
    )
    .unwrap();
    let mut frame = PyramidPlanU16::new(image.size(), 0).unwrap();
    frame.run(&image).unwrap();
    let mut positions = PointsSoA::default();
    let mut guesses = FlowTransforms::default();
    for y in (20..220).step_by(20) {
        for x in (20..300).step_by(20) {
            positions.push([x as f32, y as f32]);
            guesses.push(&AffineCompact2f::at([x as f32, y as f32]));
        }
    }
    let input = TrackInput {
        ids: (0..positions.len() as u64).collect(),
        positions: positions.clone(),
        guesses,
    };
    for threads in [1, 4] {
        let mut tracker = CpuPatchTracker::<Pattern51>::new(
            positions.len(),
            1,
            5,
            0.04,
            (threads > 1).then(|| {
                std::sync::Arc::new(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .unwrap(),
                )
            }),
        )
        .unwrap();
        let mut patches = tracker.make_patches().unwrap();
        patches.build(&frame, &positions, None).unwrap();
        let mut slots = [0];
        c.bench_function(&format!("patch_tracker_140_points_{threads}t"), |b| {
            b.iter(|| {
                tracker
                    .submit_batch(
                        std::slice::from_ref(black_box(&frame)),
                        std::slice::from_ref(black_box(&frame)),
                        TrackPhase::Temporal(std::slice::from_ref(black_box(&input))),
                        &mut patches,
                        &mut slots,
                    )
                    .unwrap();
                tracker.collect().unwrap();
                black_box(tracker.result(slots[0]));
            })
        });
    }
}
criterion_group!(benches, tracking);
criterion_main!(benches);
