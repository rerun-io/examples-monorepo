//! Reusing backward templates must give the same public tracking result as
//! rebuilding them at the committed positions.
#![allow(clippy::unwrap_used)]

use nalgebra::Vector2;
use slam_rs::frontend::parallel::WorkPool;
use slam_rs::frontend::patterns::Pattern51;
use slam_rs::frontend::se2::AffineCompact2f;
use slam_rs::frontend::tracker::{
    CpuPatchTracker, FlowResult, FlowTransforms, PatchSoA, PatchTracker, PointsSoA, SourcePatches,
};
use slam_rs::pyramid::{CpuPyramidBuilder, PyramidBuilder};

mod common;

#[test]
fn compacted_templates_track_like_rebuilt_templates_with_new_points_between_them() {
    let image = common::dotted_image(0);
    let mut builder = CpuPyramidBuilder::new();
    let mut pyramid = builder.allocate(image.width(), image.height(), 2).unwrap();
    builder.build(0, &image, &mut pyramid).unwrap();
    let shifted = common::dotted_image(1);
    let mut target = builder.allocate(image.width(), image.height(), 2).unwrap();
    builder.build(0, &shifted, &mut target).unwrap();
    let mut old_positions = PointsSoA::default();
    for point in [
        [45.25, 45.5],
        [85.0, 85.0],
        [125.5, 125.25],
        [1.0, 1.0],
        [85.25, 45.5],
        [125.25, 85.5],
        [45.25, 125.5],
        [165.25, 125.5],
        [125.25, 165.5],
    ] {
        old_positions.push(Vector2::from(point));
    }
    let mut old = PatchSoA::<Pattern51>::new(13, 3).unwrap();
    old.build(&pyramid, &old_positions, None).unwrap();

    // Reorder and compact old columns around two new detections. An invalid
    // border template must stay invalid, including its per-level flags. Copies
    // cross four-point groups and use each store's partial final group.
    let mut positions = PointsSoA::default();
    for point in [
        old_positions.get(8),
        Vector2::new(65.0, 65.0),
        old_positions.get(0),
        old_positions.get(3),
        old_positions.get(2),
        Vector2::new(65.25, 105.5),
        old_positions.get(5),
    ] {
        positions.push(point);
    }
    let mut rebuilt = PatchSoA::<Pattern51>::new(7, 3).unwrap();
    rebuilt.build(&pyramid, &positions, None).unwrap();
    let mut reused = PatchSoA::<Pattern51>::new(7, 3).unwrap();
    reused
        .build(
            &pyramid,
            &positions,
            Some(&[false, true, false, false, false, true, false]),
        )
        .unwrap();
    reused
        .copy_columns_from(&old, &[(0, 8), (2, 0), (3, 3), (4, 2), (6, 5)])
        .unwrap();

    let mut guesses = FlowTransforms::default();
    for index in 0..positions.len() {
        guesses.push(&AffineCompact2f::at(positions.get(index)));
        assert_eq!(reused.position(index), rebuilt.position(index));
        for level in 0..3 {
            assert_eq!(reused.valid(level, index), rebuilt.valid(level, index));
        }
    }
    for threads in [1, 4] {
        let mut tracker =
            CpuPatchTracker::<Pattern51>::new(7, 3, 5, 0.09, WorkPool::new(threads).unwrap())
                .unwrap();
        let mut expected = FlowResult::default();
        let mut actual = FlowResult::default();
        tracker
            .track(&pyramid, &target, &rebuilt, &guesses, &mut expected)
            .unwrap();
        tracker
            .track(&pyramid, &target, &reused, &guesses, &mut actual)
            .unwrap();
        assert!(!expected.is_empty());
        assert_eq!(actual, expected);
    }
}
