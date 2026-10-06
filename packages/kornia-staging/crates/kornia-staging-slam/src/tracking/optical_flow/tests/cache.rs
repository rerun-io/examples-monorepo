use super::*;

/// Equal ids and position bits must not reuse templates from a different image.
#[test]
fn submit_batch_rebuilds_templates_when_the_previous_pyramid_changes() {
    let mut slots = [0; 4];
    for rebuild in 0..2 {
        let scene = fixture(0.6, 1.4, 3);
        let mut cached = tracker(scene.positions.len(), 3, 1);
        let mut patches = cached.make_patches().unwrap();
        let ids: Vec<_> = (0..scene.positions.len()).map(|id| id as u64).collect();
        let first = [batch_input(&ids, &scene.positions)];
        cached
            .submit_batch(
                std::slice::from_ref(&scene.prev),
                std::slice::from_ref(&scene.next),
                TrackPhase::Temporal(&first),
                &mut patches,
                &mut slots[..first.len()],
            )
            .unwrap();
        cached.collect().unwrap();
        let result = cached.result(slots[0]);
        assert_eq!(result.len(), ids.len());
        let mut positions = PointsSoA::default();
        for index in 0..ids.len() {
            positions.push(result.transform(index).translation);
        }

        let image = shifted_image(160, 160, 31.0, -23.0);
        // Separate pyramids can have the same local build count. A clone can
        // also be rebuilt in place with a new build generation.
        let mut c = scene.next.clone();
        if rebuild == 0 {
            c = pyramid_of(&image, 3);
        } else {
            c.run(&image).unwrap();
        }
        let second = [batch_input(&ids, &positions)];
        cached
            .submit_batch(
                std::slice::from_ref(&c),
                std::slice::from_ref(&c),
                TrackPhase::Temporal(&second),
                &mut patches,
                &mut slots[..second.len()],
            )
            .unwrap();
        cached.collect().unwrap();

        let mut fresh = plan(ids.len(), 3, 1);
        let mut source = PatchSoA::<Pattern51>::new(ids.len(), 4).unwrap();
        source.build(&c, &positions, None).unwrap();
        let mut expected = FlowResult::default();
        fresh
            .track(&c, &c, &source, &second[0].guesses, &mut expected)
            .unwrap();
        assert_eq!(expected.len(), ids.len(), "all points track on C");
        assert_eq!(cached.result(slots[0]), &expected, "rebuild path {rebuild}");
    }
}
