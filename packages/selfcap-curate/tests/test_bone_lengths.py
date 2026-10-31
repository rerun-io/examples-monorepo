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
