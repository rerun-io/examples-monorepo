"""Fixed sample selection must not depend on model predictions."""

import numpy as np
import pytest

from posekit.eval.cached_hand_gt import select_eval_indices


def test_sample_spans_visible_frames_without_replacement() -> None:
    valid = np.zeros((10, 2, 21), dtype=bool)
    valid[[1, 3, 5, 7, 9], 0, :] = True
    assert select_eval_indices(valid, max_frames=3).tolist() == [1, 5, 9]


def test_sample_requires_wrist_and_fifteen_joints() -> None:
    valid = np.ones((3, 2, 21), dtype=bool)
    valid[0, :, 0] = False
    valid[1, :, :8] = False
    assert select_eval_indices(valid, max_frames=100).tolist() == [2]


def test_empty_sample_is_explicit() -> None:
    with pytest.raises(ValueError, match="visible"):
        select_eval_indices(np.zeros((2, 2, 21), dtype=bool), max_frames=100)


@pytest.mark.integration
@pytest.mark.parametrize("dataset", ["dataforge-umetrack", "dataforge-hot3d-quest3", "dataforge-gen-humanego-subset", "gsi-six-view"])
def test_cached_labels_project_back_to_the_source_camera(dataset: str) -> None:
    from pathlib import Path

    from posekit.eval.cached_hand_gt import load_cached_hand_gt
    from posekit.models.egohand_geometry import project

    files = sorted((Path.home() / "egohand-3060/cache" / dataset / "eval").glob("*.rrd"))
    if not files:
        pytest.skip(f"Missing native catalog RRD cache for {dataset}")
    segment = load_cached_hand_gt(files[0], max_frames=10)
    assert 0 < len(segment.sample_indices) <= 10
    errors = []
    for xyz, uv, valid in zip(segment.xyz_camera, segment.uv, segment.valid, strict=True):
        predicted = project(segment.camera, xyz.reshape(-1, 3).astype(np.float64)).reshape(2, 21, 2)
        errors.extend(np.linalg.norm(predicted - uv, axis=-1)[valid].tolist())
    assert np.median(errors) < 1.0
