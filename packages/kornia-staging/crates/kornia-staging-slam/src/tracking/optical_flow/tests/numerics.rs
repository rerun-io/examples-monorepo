use super::*;

/// A permissive threshold applies exactly one valid update at each level,
/// including the backward pass, rather than dropping the converged update.
#[test]
fn convergence_exit_matches_one_step_when_every_update_is_small() {
    let levels = 3;
    let scene = fixture(0.6, -0.4, levels);
    let count = scene.positions.len();
    let mut early = plan(count, levels, 1);
    early.set_klt_exit_step_px(Some(1000.0)).unwrap();
    let mut once = PatchTrackerPlan::<Pattern51>::new(count, levels + 1, 1, 0.04, pool(1)).unwrap();
    let mut expected = FlowResult::with_capacity(count);
    let mut actual = FlowResult::with_capacity(count);
    once.track(
        &scene.prev,
        &scene.next,
        &scene.patches,
        &scene.transforms,
        &mut expected,
    )
    .unwrap();
    early
        .track(
            &scene.prev,
            &scene.next,
            &scene.patches,
            &scene.transforms,
            &mut actual,
        )
        .unwrap();
    assert!(!expected.is_empty());
    assert_eq!(actual.tracked(), expected.tracked());
    for index in expected.tracked() {
        assert_eq!(
            actual.transform(*index as usize),
            expected.transform(*index as usize)
        );
    }
}

/// Invalid camera projections keep shared stereo source slots but must never track.
#[test]
fn finite_out_of_image_guesses_are_rejected_without_losing_source_slots() {
    let levels = 3;
    let mut scene = fixture(0.0, 0.0, levels);
    let count = scene.positions.len();
    let mut rejected = scene.transforms.get(0);
    rejected.translation = [-1.0e6; 2];
    scene.transforms.set(0, &rejected);
    let mut tracker = plan(count, levels, 1);
    let mut result = FlowResult::with_capacity(count);
    tracker
        .track(
            &scene.prev,
            &scene.next,
            &scene.patches,
            &scene.transforms,
            &mut result,
        )
        .unwrap();
    assert!(!result.is_valid(0));
    assert!(!result.tracked().is_empty());
    assert_eq!(scene.positions.len(), scene.transforms.len());
}

#[test]
fn an_integer_shift_is_recovered() {
    let levels: usize = 3;
    let scene: Fixture = fixture(2.0, -1.0, levels);
    let mut tracker: PatchTrackerPlan<Pattern51> = plan(scene.positions.len(), levels, 1);
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
        let moved: Vector2<f32> = Vector2::from(out.transform(index).translation)
            - Vector2::from(scene.positions.get(index));
        // An integer shift moves the samples themselves, so the two
        // bilinear reconstructions are exact translates of each other and
        // the residual's fixed point is the true shift.
        assert!(
            (moved.x - 2.0).abs() < 0.01 && (moved.y + 1.0).abs() < 0.01,
            "patch {index} moved by {moved:?}, expected (2, -1)"
        );
    }
}

/// Subpixel tracking has interpolation bias as well as convergence error.
/// Bilinear values and unit-step central-difference gradients can put the fixed
/// point beside the true shift. The error depends on fractional shift: integer
/// shifts recover exactly while half-pixel shifts are harder. Shorter texture
/// wavelengths raise bias; longer ones weaken conditioning. Check the median
/// and cap the tail instead of equating bias with optimizer convergence.
fn sub_pixel_shift_error(dx: f32, dy: f32) -> (f32, f32, usize, usize) {
    let levels: usize = 3;
    let scene: Fixture = fixture(dx, dy, levels);
    let mut tracker: PatchTrackerPlan<Pattern51> = plan(scene.positions.len(), levels, 1);
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
        let moved: Vector2<f32> = Vector2::from(out.transform(index).translation)
            - Vector2::from(scene.positions.get(index));
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

/// The forward-backward gate is
/// what rejects a track onto an unrelated image.
#[test]
fn a_mismatched_pair_is_rejected() {
    let levels: usize = 3;
    let scene: Fixture = fixture(0.0, 0.0, levels);
    // A different texture entirely, not a shift of the first.
    let mut other: Image<u16, 1> = Image::from_size_val(
        kornia_image::ImageSize {
            width: 160,
            height: 160,
        },
        0u16,
    )
    .unwrap();
    for y in 0..160 {
        for x in 0..160 {
            let value: f64 = 25_000.0
                + 9_000.0 * ((x as f64) * 0.61).cos()
                + 6_000.0 * ((y as f64) * 0.47).sin();
            other.set_pixel(x, y, 0, value as u16).unwrap();
        }
    }
    let unrelated: PyramidPlanU16 = pyramid_of(&other, levels);

    let mut tracker: PatchTrackerPlan<Pattern51> = plan(scene.positions.len(), levels, 1);
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
    plan(scene.positions.len(), levels, 1)
        .track(
            &scene.prev,
            &scene.next,
            &scene.patches,
            &scene.transforms,
            &mut single,
        )
        .unwrap();

    let mut wide: FlowResult = FlowResult::with_capacity(scene.positions.len());
    plan(scene.positions.len(), levels, 4)
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
    let mut tracker: PatchTrackerPlan<Pattern51> = plan(scene.positions.len(), levels, 4);

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

#[test]
fn nonfinite_guesses_are_rejected_before_warp_composition() {
    let scene = fixture(0.6, 1.4, 3);
    for value in [f32::NAN, f32::INFINITY, f32::MAX] {
        let mut guesses = scene.transforms.clone();
        let mut bad = guesses.get(0);
        bad.linear = [[value, value], [value, -value]];
        guesses.set(0, &bad);
        let mut tracker = plan(scene.positions.len(), 3, 1);
        let mut result = FlowResult::default();
        tracker
            .track(
                &scene.prev,
                &scene.next,
                &scene.patches,
                &guesses,
                &mut result,
            )
            .unwrap();
        assert!(!result.is_valid(0));
        assert!(result
            .transform(0)
            .coefficients()
            .iter()
            .all(|v| v.is_finite()));
        assert!(!result.tracked().is_empty());
    }
}
