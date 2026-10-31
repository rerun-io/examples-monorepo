"""Bone-length gate: a joint that stretches its bones far from the session's median length is dropped."""

import numpy as np

from selfcap_curate.keypoints import drop_bone_outliers

REST = {5: (-0.2, 1.4, 0.0), 6: (0.2, 1.4, 0.0), 7: (-0.25, 1.1, 0.0), 8: (0.25, 1.1, 0.0), 9: (-0.25, 0.85, 0.05), 10: (0.25, 0.85, 0.05),
        11: (-0.12, 0.9, 0.0), 12: (0.12, 0.9, 0.0), 13: (-0.12, 0.5, 0.0), 14: (0.12, 0.5, 0.0), 15: (-0.12, 0.1, 0.0), 16: (0.12, 0.1, 0.0),
        3: (-0.08, 1.62, -0.05), 4: (0.08, 1.62, -0.05)}


def skeleton(frames: int = 200, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xyz = np.full((frames, 133, 3), np.nan, np.float32)
    for joint, point in REST.items():
        xyz[:, joint] = np.asarray(point) + rng.normal(0.0, 0.008, (frames, 3))  # 8 mm triangulation noise
    xyz[:, :, 0] += np.linspace(0.0, 1.0, frames)[:, None]  # the person walks 1 m
    return xyz, np.full((frames, 133), 0.8, np.float32)


def test_a_flung_wrist_is_dropped_and_the_rest_of_the_frame_kept() -> None:
    xyz, confidence = skeleton()
    xyz[50, 9] += (0.6, 0.3, 0.2)  # a wrong two-view point 70 cm off
    xyz[80, 15] += (0.5, 0.0, 0.3)  # an ankle 58 cm off: the shin 1.8x its length
    out = drop_bone_outliers(xyz, confidence)
    assert np.isnan(out[50, 9]).all() and np.isnan(out[80, 15]).all()
    assert np.isfinite(out[50, [5, 7, 10, 11]]).all() and np.isfinite(out[80, [13, 16]]).all()
    untouched = np.ones(len(xyz), bool)
    untouched[[50, 80]] = False
    assert np.array_equal(np.isnan(out[untouched]), np.isnan(xyz[untouched]))


def test_a_clean_session_is_left_alone() -> None:
    xyz, confidence = skeleton(seed=4)
    assert np.array_equal(np.isnan(drop_bone_outliers(xyz, confidence)), np.isnan(xyz))


def test_a_bridged_joint_that_breaks_its_bones_is_not_written(monkeypatch) -> None:
    """Gap bridging and jump replacement run after the triangulation's bone check; a bridged knee 60 cm off (a044c36b:
    thighs up to 1.5 m in 9.5% of frames, 80% of them bridged) must still be dropped."""
    from test_keypoints_geometry import cameras

    import selfcap_curate.temporal_keypoints as temporal
    from selfcap_curate.keypoints import Keypoints3D, SelectedView, project, smooth_keypoints

    xyz, confidence = skeleton(frames=120)
    xyz[:, :, 2] += 4.0  # in front of the test cameras
    cams = cameras()[:2]
    selected = {str(i): SelectedView(np.stack([project(c, np.nan_to_num(f.astype(np.float64)))[:, :2] for f in xyz]).astype(np.float32), np.ones((120, 133)))
                for i, c in enumerate(cams)}
    result = Keypoints3D(np.arange(120) / 25.0, xyz, np.full((120, 133), 2, np.uint8), confidence, selected, {str(i): c for i, c in enumerate(cams)},
                         25.0, np.ones((120, 2, 133), bool), np.zeros((120, 133), np.float32))
    real = temporal.stable_body

    def bridged_wrong(result, points):  # noqa: ANN001, ANN202
        out = real(result, points).copy()
        out[60, 13] += (0.0, -0.6, 0.0)  # a bridged left knee: the thigh 2.5x its length
        return out

    monkeypatch.setattr(temporal, "stable_body", bridged_wrong)
    out = smooth_keypoints(result)
    assert np.isnan(out.xyz[60, 13]).all() and out.confidence[60, 13] == 0.0
    assert np.isfinite(out.xyz[60, [11, 15]]).all() and np.isfinite(out.xyz[59, 13]).all()
