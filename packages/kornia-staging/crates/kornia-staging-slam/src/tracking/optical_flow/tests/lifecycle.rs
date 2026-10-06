use super::*;

#[test]
fn more_keypoints_than_capacity_is_refused() {
    let levels: usize = 1;
    let scene: Fixture = fixture(0.0, 0.0, levels);
    let mut tracker: PatchTrackerPlan<Pattern51> = plan(2, levels, 1);
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

/// A patch set shallower than the tracker used to index past the end of its
/// validity array; it is a typed error now.
#[test]
fn a_shallow_patch_set_is_refused() {
    for threads in [1, 4] {
        let levels: usize = 3;
        let scene: Fixture = fixture(0.0, 0.0, levels);
        let mut shallow: PatchSoA<Pattern51> = PatchSoA::new(scene.positions.len(), 1).unwrap();
        shallow.build(&scene.prev, &scene.positions, None).unwrap();

        let mut tracker: PatchTrackerPlan<Pattern51> = plan(scene.positions.len(), levels, threads);
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
    let shallow: PyramidPlanU16 = pyramid_of(&shifted_image(160, 160, 0.0, 0.0), 1);
    let mut tracker: PatchTrackerPlan<Pattern51> = plan(scene.positions.len(), levels, 1);
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
/// Reject impossible storage sizes before allocation can panic or abort.
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
            PatchTrackerPlan::<Pattern51>::new(capacity, 4, 5, 0.04, pool(1)).unwrap_err(),
            TrackerError::CapacityTooLarge {
                capacity,
                ceiling: MAX_CAPACITY,
            }
        );
    }
}

/// A pyramid deeper than the ceiling is refused before the allocation.
///
/// A count that fits usize can still exceed available memory; reject it before allocation.
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
            linear: Matrix2::new(0.5, -0.25, 0.25, 0.5).into(),
            translation: Vector2::new(-3.0, 4.5).into(),
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

/// The write sequence is usable on its own: reset, set, finish.
#[test]
fn the_flow_result_writing_surface_compacts_what_it_is_given() {
    let mut out: FlowResult = FlowResult::default();
    out.reset(5);
    assert!(out.is_empty());
    out.set_track(1, true, &AffineCompact2f::at(Vector2::new(3.0, 4.0)));
    out.set_track(4, true, &AffineCompact2f::at(Vector2::new(-1.0, 0.5)));
    out.set_track(2, false, &AffineCompact2f::identity());
    out.finish();

    assert_eq!(out.tracked(), &[1, 4]);
    assert_eq!(out.transform(1).translation, [3.0, 4.0]);
    assert_eq!(out.transform(4).translation, [-1.0, 0.5]);
    assert!(!out.is_valid(0));
    assert!(!out.is_valid(2));

    // A reset clears the survivors without dropping the allocation.
    out.reset(3);
    assert!(out.tracked().is_empty());
    assert!(!out.is_valid(1));
}

/// A zero-capacity patch set is valid (`PatchSoA::new` accepts it); building it from no positions is a no-op on one worker and
/// on a pool, as it was before the per-level chunked build.
#[test]
fn an_empty_patch_set_builds_on_one_worker_and_on_a_pool() {
    let levels: usize = 3;
    let base: Image<u16, 1> = shifted_image(160, 160, 0.0, 0.0);
    let prev: PyramidPlanU16 = pyramid_of(&base, levels);
    let positions: PointsSoA = PointsSoA::default();
    let mut single: PatchSoA<Pattern51> = PatchSoA::new(0, levels + 1).unwrap();
    single.build(&prev, &positions, None).unwrap();
    let mut pooled: PatchSoA<Pattern51> = PatchSoA::new(0, levels + 1).unwrap().with_pool(pool(4));
    pooled.build(&prev, &positions, None).unwrap();
    assert_eq!((single.len(), pooled.len()), (0, 0));
}

#[test]
fn invalid_tracker_settings_are_rejected_before_sampling() {
    assert!(PatchTrackerPlan::<Pattern51>::new(1, 0, 5, 0.04, pool(1)).is_err());
    assert!(PatchTrackerPlan::<Pattern51>::new(1, 1, 0, 0.04, pool(1)).is_err());
    assert!(PatchTrackerPlan::<Pattern51>::new(1, 1, 5, f32::NAN, pool(1)).is_err());
}

#[test]
fn invalid_convergence_thresholds_are_refused() {
    let mut tracker = plan(1, 1, 1);
    for threshold in [f32::NAN, f32::INFINITY, -1.0, 0.0] {
        assert!(tracker.set_klt_exit_step_px(Some(threshold)).is_err());
    }
    assert!(tracker.set_klt_exit_step_px(Some(0.1)).is_ok());
    assert!(tracker.set_klt_exit_step_px(None).is_ok());
}
