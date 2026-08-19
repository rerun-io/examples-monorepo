"""Public-contract tests for MammaNet display skeleton helpers."""

from pathlib import Path

import numpy as np
import pytest

from mamma.landmarks.skeleton import SkeletonRegressor, body_models_available, load_skeleton_regressor, skeleton_strips


def test_body_model_loader_owns_missing_asset_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The skeleton seam reports the Pixi recovery task when SMPL-X assets are absent."""
    monkeypatch.setattr("mamma.landmarks.skeleton.DEFAULT_BODY_MODELS_DIR", tmp_path)
    load_skeleton_regressor.cache_clear()

    assert not body_models_available()
    with pytest.raises(RuntimeError, match="_download-mamma-body-models"):
        load_skeleton_regressor()


def test_skeleton_strips_follow_regressor_bones() -> None:
    """Display strips contain the child-parent pairs stored by the regressor."""
    joints_xy = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    regressor = SkeletonRegressor(
        weights=np.zeros((3, 512), dtype=np.float32),
        parents=(-1, 0, 1),
        bones=np.asarray([[1, 0], [2, 1]], dtype=np.int64),
    )

    strips = skeleton_strips(joints_xy, regressor)

    np.testing.assert_array_equal(strips, [[[3.0, 4.0], [1.0, 2.0]], [[5.0, 6.0], [3.0, 4.0]]])
