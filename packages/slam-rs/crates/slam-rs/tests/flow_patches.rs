//! Cached templates must track exactly like templates rebuilt at the same positions.
#![allow(clippy::unwrap_used)]

use kornia_staging_imgproc::optical_flow::patch_se2::AffineCompact2f;
use kornia_staging_imgproc::optical_flow::patch_se2::Pattern51;
use kornia_staging_imgproc::optical_flow::patch_tracker::{FlowResult, PointsSoA, PatchTrackerPlan};
use kornia_staging_slam::tracking::optical_flow::{
    CpuPatchTracker, PatchTracker, TrackInput, TrackPhase,
};
use nalgebra::Vector2;
use slam_rs::frontend::parallel::WorkPool;
use slam_rs::pyramid::{CpuPyramidBuilder, PyramidBuilder};

mod common;

#[test]
fn cached_templates_track_like_rebuilt_templates_with_new_points_between_them() {
    let mut slots = [0; 4];
    let image = common::dotted_image(0);
    let mut builder = CpuPyramidBuilder::new();
    let mut pyramid = builder.allocate(image.width(), image.height(), 2).unwrap();
    builder.build(0, &image, &mut pyramid).unwrap();
    let shifted = common::dotted_image(1);
    let mut target = builder.allocate(image.width(), image.height(), 2).unwrap();
    builder.build(0, &shifted, &mut target).unwrap();

    for threads in [1, 4] {
        for temporal in [true, false] {
            let mut tracker = CpuPatchTracker::<Pattern51>::new(
                13,
                3,
                5,
                0.09,
                WorkPool::new(threads).unwrap().rayon_pool(),
            )
            .unwrap();
            let mut patches = tracker.make_patches().unwrap();
            let mut first = TrackInput::default();
            for (id, point) in [
                [45.25, 45.5],
                [85.0, 85.0],
                [125.5, 125.25],
                [1.0, 1.0],
                [85.25, 45.5],
                [125.25, 85.5],
                [45.25, 125.5],
                [165.25, 125.5],
                [125.25, 165.5],
            ]
            .into_iter()
            .enumerate()
            {
                first.ids.push(id as u64);
                first.positions.push(Vector2::from(point));
                first
                    .guesses
                    .push(&AffineCompact2f::at(Vector2::from(point)));
            }
            let destinations = [first.guesses.clone()];
            let phase = if temporal {
                TrackPhase::Temporal(std::slice::from_ref(&first))
            } else {
                TrackPhase::Matching {
                    ids: &first.ids,
                    positions: &first.positions,
                    destinations: &destinations,
                }
            };
            tracker
                .submit_batch(
                    std::slice::from_ref(&pyramid),
                    &[pyramid.clone(), pyramid.clone()],
                    phase,
                    &mut patches,
                    &mut slots[..1],
                )
                .unwrap();
            tracker.collect().unwrap();
            let result = tracker.result(slots[0]);
            assert!(!result.is_empty());
            let mut old_positions = PointsSoA::default();
            for index in 0..first.positions.len() {
                old_positions.push(if temporal {
                    result.transform(index).translation
                } else {
                    first.positions.get(index)
                });
            }

            // Reorder columns across four-point groups, retain the invalid
            // border point, and mix in two new ids in a partial final group.
            let mut second = TrackInput::default();
            for (id, point) in [
                (8, old_positions.get(8)),
                (20, [65.0, 65.0]),
                (0, old_positions.get(0)),
                (3, old_positions.get(3)),
                (2, old_positions.get(2)),
                (21, [65.25, 105.5]),
                (5, old_positions.get(5)),
            ] {
                second.ids.push(id);
                second.positions.push(point);
                second.guesses.push(&AffineCompact2f::at(point));
            }
            let mut fresh = tracker.make_patches().unwrap();
            fresh.build(&pyramid, &second.positions, None).unwrap();
            let mut expected = FlowResult::default();
            PatchTrackerPlan::<Pattern51>::new(13, 3, 5, 0.09, WorkPool::new(threads).unwrap().rayon_pool()).unwrap()
                .track(&pyramid, &target, &fresh, &second.guesses, &mut expected)
                .unwrap();
            tracker
                .submit_batch(
                    std::slice::from_ref(&pyramid),
                    std::slice::from_ref(&target),
                    TrackPhase::Temporal(std::slice::from_ref(&second)),
                    &mut patches,
                    &mut slots[..(std::slice::from_ref(&second)).len()],
                )
                .unwrap();
            tracker.collect().unwrap();
            assert!(!expected.is_empty());
            assert_eq!(tracker.result(slots[0]), &expected);
        }
    }
}
