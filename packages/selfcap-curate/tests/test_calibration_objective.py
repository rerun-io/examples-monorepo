import numpy as np

from selfcap_curate.calibration_objective import block_weights, robust_loss


def test_duplicating_observations_preserves_total_robust_cost_and_view_shares():
    views = np.array([0, 0, 1], dtype=np.int64)
    confidence = np.array([.9, .6, .8])
    z = np.array([1.0, 100.0, 4.0])
    weights = block_weights(views, confidence, 3.0)
    np.testing.assert_allclose([weights[:2].sum(), weights[2]], [1.5, 1.5])
    replicated = block_weights(np.repeat(views, 7), np.repeat(confidence, 7), 3.0)
    for kind in ('huber', 'cauchy'):
        original = robust_loss(weights, kind)(z)[0].sum()
        duplicated = robust_loss(replicated, kind)(np.repeat(z, 7))[0].sum()
        np.testing.assert_allclose(original, duplicated, rtol=1e-14)


def test_normalized_camera_fit_is_invariant_to_repeated_hand_observations():
    from dataclasses import replace

    from selfcap_curate.calibration_objective import ObjectiveShares
    from selfcap_curate.exo_calibration import Camera
    from selfcap_curate.exo_landmarks import Landmark, Landmarks
    from selfcap_curate.exo_refine import RefinementProblem, solve
    from selfcap_curate.quest_fit import Correspondences

    rng = np.random.default_rng(15)
    camera = Camera(np.array([.1, -.05, .02, .1, .2, 3., 850., 960., 540., 0., 0.]))
    points = rng.uniform(-.3, .3, (30, 3))
    pixels = camera.project(points)[:, :2]
    hand = Correspondences(points, pixels, np.ones(len(points)))
    targets = points[:5] + [.0, .2, .1]
    image = camera.project(targets)[:, :2] + [3., -2.]
    labels = Landmarks([Landmark(0, float(i), 'quest0', float(u), float(v), .8) for i, (u,v) in enumerate(image)], 'synthetic', 'conflicting anchors', '2026-10-30')
    problem = RefinementProblem((camera,), (hand,), labels, targets, np.tile(np.eye(3), (5, 1, 1)))
    first = solve(problem, focal_source='fixed', fixed_focals=(850.,), objective=ObjectiveShares())
    repeated = Correspondences(np.tile(points, (7,1)), np.tile(pixels, (7,1)), np.ones(len(points)*7))
    second = solve(replace(problem, hands=(repeated,)), focal_source='fixed', fixed_focals=(850.,), objective=ObjectiveShares())
    np.testing.assert_allclose(first.cameras[0].parameters, second.cameras[0].parameters, atol=2e-6)


def test_body_polish_cost_has_fixed_shares_when_samples_are_repeated():
    from selfcap_curate.calibration_objective import ObjectiveShares
    from selfcap_curate.exo_calibration import BodyObservation, Camera, bundle
    from selfcap_curate.quest_fit import Correspondences

    rng = np.random.default_rng(9)
    xyz = rng.normal(0., .2, (20, 3))
    cameras = {str(i): Camera(np.array([0., i*.1, 0., i*.4, 0., 3., 850., 960., 540., 0., 0.])) for i in range(3)}
    hands = {p: Correspondences(xyz, c.project(xyz)[:, :2] + [2., -1.], np.ones(20)) for p,c in cameras.items()}
    body = [{p: BodyObservation(c.project(xyz[:13])[:, :2] + [float(i), 0.], np.ones(13)) for i,(p,c) in enumerate(cameras.items())}]
    repeated = {p: Correspondences(np.tile(d.world,(5,1)),np.tile(d.image,(5,1)),np.ones(100)) for p,d in hands.items()}
    first = bundle(hands, body, cameras, max_nfev=1, objective=ObjectiveShares())
    second = bundle(repeated, body*7, cameras, max_nfev=1, objective=ObjectiveShares())
    np.testing.assert_allclose(first.cost, second.cost, rtol=1e-12)
