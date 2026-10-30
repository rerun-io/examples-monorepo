"""Body camera-pair selection rejects alternating high-score false matches."""

import numpy as np
from test_keypoints_geometry import cameras

from selfcap_curate.keypoints import Keypoints3D, SelectedView, project, triangulate_frames
from selfcap_curate.temporal_keypoints import stable_body


def test_body_path_stays_on_consistent_views_and_leaves_hands_unchanged() -> None:
    cams = cameras()
    frames = 61
    true = np.tile([0.1, -0.2, 4.0], (frames, 133, 1)).astype(np.float32)
    false = true.copy()
    false[:, :, 0] += (0.25 * (-1.0) ** np.arange(frames))[:, None]
    uv = np.stack([
        np.stack([project(camera, points.astype(np.float64))[:, :2] for points in (true if view < 2 else false)])
        for view, camera in enumerate(cams)
    ], axis=1)
    scores = np.broadcast_to(np.array([0.85, 0.85, 0.99, 0.99])[None, :, None], uv.shape[:3]).copy()
    raw = triangulate_frames(cams, uv, scores)
    selected = {str(i): SelectedView(uv[:, i].astype(np.float32), scores[:, i]) for i in range(4)}
    result = Keypoints3D(np.arange(frames) / 24.0, raw.xyz, raw.n_views, raw.confidence, selected,
                        dict(zip(selected, cams, strict=True)), 24.0, raw.inliers, raw.max_reprojection_px)
    stable = stable_body(result, raw.xyz)
    np.testing.assert_allclose(stable[:, 6], true[:, 6], atol=0.01)
    np.testing.assert_array_equal(stable[:, 91:], raw.xyz[:, 91:])


def test_short_gaps_are_bridged_with_reduced_confidence_but_long_gaps_stay_missing() -> None:
    from selfcap_curate.temporal_keypoints import bridge_short_gaps

    positions = np.zeros((30, 1, 3), dtype=np.float32)
    positions[:, 0, 0] = np.arange(30)
    confidence = np.full((30, 1), 0.9, dtype=np.float32)
    positions[2:5] = np.nan
    positions[10:24] = np.nan
    filled, scores, bridged = bridge_short_gaps(positions, confidence, 24.0)
    np.testing.assert_allclose(filled[2:5, 0, 0], [2, 3, 4])
    assert bridged[2:5].all()
    assert (scores[2:5] < 0.9).all()
    assert np.isnan(filled[10:24]).all()
    assert not bridged[10:24].any()
