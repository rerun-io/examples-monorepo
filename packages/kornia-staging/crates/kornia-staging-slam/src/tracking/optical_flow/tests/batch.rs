use super::*;

#[test]
fn temporal_batch_matches_individual_cameras_across_empty_lanes() {
    let mut slots = [0; 4];
    let levels = 3;
    let count = 9;
    let scene = fixture(0.0, 0.0, levels);
    let mut positions = PointsSoA::default();
    for index in 0..count {
        positions.push(scene.positions.get(index));
    }
    // Distinct source textures and destination shifts expose stale targets in
    // either tracking direction. Empty lanes precede and separate live lanes.
    let phases = [(0.0, 0.0), (9.0, -7.0), (4.0, 3.0), (-13.0, 11.0)];
    let shifts = [(0.0, 0.0), (0.6, 1.4), (0.0, 0.0), (-0.9, 0.4)];
    let prev: Vec<_> = phases
        .iter()
        .map(|&(x, y)| pyramid_of(&shifted_image(160, 160, x, y), levels))
        .collect();
    let next: Vec<_> = phases
        .iter()
        .zip(shifts)
        .map(|(&(x, y), (dx, dy))| pyramid_of(&shifted_image(160, 160, x + dx, y + dy), levels))
        .collect();

    for threads in [1, 4] {
        let inputs: Vec<_> = (0..4)
            .map(|camera| {
                if camera % 2 == 0 {
                    batch_input(&[], &PointsSoA::default())
                } else {
                    batch_input(&(0..count as u64).collect::<Vec<_>>(), &positions)
                }
            })
            .collect();
        let mut batched = tracker(count, levels, threads);
        let mut patches = batched.make_patches().unwrap();
        batched
            .submit_batch(
                &prev,
                &next,
                TrackPhase::Temporal(&inputs),
                &mut patches,
                &mut slots[..inputs.len()],
            )
            .unwrap();
        batched.collect().unwrap();

        for (camera, input) in inputs.iter().enumerate() {
            let mut ordinary = plan(count, levels, threads);
            let mut source = PatchSoA::<Pattern51>::new(count, levels + 1).unwrap();
            source.build(&prev[camera], &input.positions, None).unwrap();
            let mut expected = FlowResult::with_capacity(count);
            ordinary
                .track(
                    &prev[camera],
                    &next[camera],
                    &source,
                    &input.guesses,
                    &mut expected,
                )
                .unwrap();
            assert_eq!(expected.len(), input.positions.len());
            let actual = batched.result(slots[camera]);
            assert_eq!(
                actual.tracked(),
                expected.tracked(),
                "camera {camera}, threads {threads}"
            );
            for &index in expected.tracked() {
                assert_eq!(
                    actual.transform(index as usize),
                    expected.transform(index as usize),
                    "camera {camera}, point {index}, threads {threads}"
                );
            }
        }
    }
}

#[test]
fn submit_batch_refuses_matching_lanes_with_different_template_counts() {
    let mut slots = [0; 4];
    let scene = fixture(0.0, 0.0, 3);
    let mut cpu = tracker(32, 3, 1);
    let mut patches = cpu.make_patches().unwrap();
    for count in [2, 8] {
        let inputs: [TrackInput; 2] = std::array::from_fn(|lane| {
            let len = if lane == 0 { 3 } else { count };
            let mut positions = PointsSoA::default();
            for index in 0..len {
                positions.push(scene.positions.get(index));
            }
            let ids: Vec<_> = (0..len).map(|id| id as u64).collect();
            batch_input(&ids, &positions)
        });
        let next = [scene.next.clone(), scene.next.clone(), scene.next.clone()];
        assert_eq!(
            cpu.submit_batch(
                std::slice::from_ref(&scene.prev),
                &next,
                TrackPhase::Matching {
                    ids: &inputs[0].ids,
                    positions: &inputs[0].positions,
                    destinations: &inputs
                        .iter()
                        .map(|input| input.guesses.clone())
                        .collect::<Vec<_>>()
                },
                &mut patches,
                &mut slots[..inputs.len()]
            ),
            Err(TrackerError::LengthMismatch {
                first_name: "patches",
                first: 3,
                second_name: "transforms",
                second: count,
            })
        );
    }
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
    batch: TrackBatch,
    capacity: usize,
    num_levels: usize,
}

impl PatchTracker for EchoTracker {
    type Error = TrackerError;
    fn set_klt_exit_step_px(&mut self, threshold: Option<f32>) -> Result<(), TrackerError> {
        validate_exit_step(threshold)
    }

    fn batch(&self) -> &TrackBatch {
        &self.batch
    }
    fn batch_mut(&mut self) -> &mut TrackBatch {
        &mut self.batch
    }

    type Pattern = Pattern51;
    type Pyramid = PyramidPlanU16;
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

    fn submit_batch(
        &mut self,
        prev: &[PyramidPlanU16],
        next: &[PyramidPlanU16],
        phase: TrackPhase<'_>,
        _patches: &mut PatchSoA<Pattern51>,
        slots: &mut [usize],
    ) -> Result<(), TrackerError> {
        let inputs = phase.validate(
            self.capacity,
            self.num_levels,
            slots.len(),
            matches!(phase, TrackPhase::Matching { .. }).then_some(_patches.num_levels()),
            |i| prev.get(i).map(|p| p.levels().len()),
            |i| next.get(i).map(|p| p.levels().len()),
        )?;
        super::super::submit_each(inputs, slots, |lane| {
            let (slot, out) = self.batch.submit_slot(self.capacity);
            out.reset(lane.guesses.len());
            for i in 0..lane.guesses.len() {
                out.set_track(i, true, &lane.guesses.get(i));
            }
            let (valid, [m00, ..]) = out.parts_mut();
            assert_eq!(m00.len(), valid.len());
            out.finish();
            Ok(slot)
        })
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
    let input = TrackInput {
        ids: (0..scene.positions.len() as u64).collect(),
        positions: scene.positions.clone(),
        guesses: scene.transforms.clone(),
    };
    let mut slots = [0];
    echo.submit_batch(
        std::slice::from_ref(&scene.prev),
        std::slice::from_ref(&scene.next),
        TrackPhase::Temporal(&[input]),
        &mut echo.make_patches().unwrap(),
        &mut slots,
    )
    .unwrap();
    echo.collect().unwrap();
    let out = echo.result(slots[0]);

    assert_eq!(out.len(), scene.positions.len());
    for index in out.tracked() {
        let index: usize = *index as usize;
        assert!(out.is_valid(index));
        assert_eq!(out.transform(index), scene.transforms.get(index));
    }
}

#[test]
fn stereo_batch_refuses_shallow_patches_before_mutation() {
    let mut slots = [0; 4];
    let scene = fixture(0.0, 0.0, 1);
    let mut tracker = tracker(scene.positions.len(), 1, 1);
    let mut shallow = PatchSoA::<Pattern51>::new(scene.positions.len(), 1).unwrap();
    let ids: Vec<u64> = (0..scene.positions.len() as u64).collect();
    let inputs = [batch_input(&ids, &scene.positions)];
    let next = [scene.next.clone(), scene.next.clone()];
    let before = format!("{tracker:?}{shallow:?}{inputs:?}");
    let error = tracker
        .submit_batch(
            std::slice::from_ref(&scene.prev),
            &next,
            TrackPhase::Matching {
                ids: &inputs[0].ids,
                positions: &inputs[0].positions,
                destinations: &inputs
                    .iter()
                    .map(|input| input.guesses.clone())
                    .collect::<Vec<_>>(),
            },
            &mut shallow,
            &mut slots[..inputs.len()],
        )
        .unwrap_err();
    assert_eq!(
        error,
        TrackerError::LevelMismatch {
            what: "the patch set",
            expected: 2,
            actual: 1,
        }
    );
    assert_eq!(format!("{tracker:?}{shallow:?}{inputs:?}"), before);
}

#[test]
fn single_input_batch_equals_track_and_checks_before_mutation() {
    let scene = fixture(0.6, -0.4, 3);
    let ids: Vec<_> = (0..scene.positions.len() as u64).collect();
    for threads in [1, 4] {
        let mut tracker = tracker(ids.len(), 3, threads);
        let mut input = batch_input(&ids, &scene.positions);
        input.guesses.clone_from(&scene.transforms);
        let mut patches = tracker.make_patches().unwrap();
        let mut expected = FlowResult::default();
        plan(ids.len(), 3, threads)
            .track(
                &scene.prev,
                &scene.next,
                &scene.patches,
                &input.guesses,
                &mut expected,
            )
            .unwrap();
        let mut slots = [usize::MAX];
        tracker
            .submit_batch(
                std::slice::from_ref(&scene.prev),
                std::slice::from_ref(&scene.next),
                TrackPhase::Temporal(std::slice::from_ref(&input)),
                &mut patches,
                &mut slots,
            )
            .unwrap();
        tracker.collect().unwrap();
        assert_eq!(tracker.result(slots[0]), &expected);
        input.ids.pop();
        let before = format!("{tracker:?}");
        assert!(tracker
            .submit_batch(
                std::slice::from_ref(&scene.prev),
                std::slice::from_ref(&scene.next),
                TrackPhase::Temporal(&[input]),
                &mut patches,
                &mut slots
            )
            .is_err());
        assert_eq!(before, format!("{tracker:?}"));
        assert_eq!(slots, [0]);
    }
}
