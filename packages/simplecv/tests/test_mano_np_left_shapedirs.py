"""MANOLayerNP's opt-in left-hand shape fix: the left model's first shape direction ships with its x not mirrored."""

from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Float32
from numpy import ndarray

from simplecv.ops.mano.mano_np import MANOLayerNP

MANO_ROOT: Path = Path(__file__).resolve().parents[1] / "simplecv" / "data"
pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def mano_models() -> None:
    if not (MANO_ROOT / "MANO_LEFT.pkl").is_file() or not (MANO_ROOT / "MANO_RIGHT.pkl").is_file():
        pytest.skip(f"MANO_LEFT.pkl and MANO_RIGHT.pkl required under {MANO_ROOT}")


def posed(layer: MANOLayerNP) -> tuple[Float32[ndarray, "2 778 3"], Float32[ndarray, "2 21 3"]]:
    """Two poses of the layer's hand: rest, and a bent one."""
    poses: Float32[ndarray, "2 48"] = np.zeros((2, 48), dtype=np.float32)
    poses[1, 3:] = 0.2
    return layer(poses, np.zeros((2, 3), dtype=np.float32))


def test_the_left_fix_is_the_hand_applied_flip_of_the_first_shape_direction() -> None:
    betas: Float32[ndarray, "10"] = np.linspace(-1.5, 1.5, 10, dtype=np.float32)
    by_hand: MANOLayerNP = MANOLayerNP(side="left", betas=betas, mano_root_dir=MANO_ROOT, use_pca=False)
    shapedirs: Float32[ndarray, "778 3 10"] = by_hand._mano_layer.th_shapedirs.copy()  # the patch GenRobot used to apply
    shapedirs[:, 0, :] *= -1.0
    by_hand._mano_layer.th_shapedirs = shapedirs
    fixed: MANOLayerNP = MANOLayerNP(side="left", betas=betas, mano_root_dir=MANO_ROOT, use_pca=False, fix_left_shapedirs=True)
    for ours, theirs in zip(posed(fixed), posed(by_hand), strict=True):
        np.testing.assert_array_equal(ours, theirs)
    plain: MANOLayerNP = MANOLayerNP(side="left", betas=betas, mano_root_dir=MANO_ROOT, use_pca=False)
    assert not np.allclose(posed(plain)[0], posed(fixed)[0])


def test_the_fix_leaves_the_right_hand_alone() -> None:
    betas: Float32[ndarray, "10"] = np.linspace(-1.5, 1.5, 10, dtype=np.float32)
    plain: MANOLayerNP = MANOLayerNP(side="right", betas=betas, mano_root_dir=MANO_ROOT, use_pca=False)
    flagged: MANOLayerNP = MANOLayerNP(side="right", betas=betas, mano_root_dir=MANO_ROOT, use_pca=False, fix_left_shapedirs=True)
    for ours, theirs in zip(posed(flagged), posed(plain), strict=True):
        np.testing.assert_array_equal(ours, theirs)
