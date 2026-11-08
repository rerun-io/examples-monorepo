"""auto_orient_and_center_poses under the dev env's beartype claw."""

from typing import ForwardRef

import numpy as np
from jaxtyping import Float64
from numpy import ndarray

from simplecv.camera_orient_utils import OrientResults, auto_orient_and_center_poses


def test_orient_results_fields_are_resolved_types() -> None:
    """Stringified NamedTuple hints are unresolvable for beartype >= 0.23 (mast3r-slam-dev crashed on every call)."""
    assert not any(isinstance(hint, str | ForwardRef) for hint in OrientResults.__annotations__.values())


def test_auto_orient_and_center_poses_returns_unpackable_results() -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    world_T_cam: Float64[ndarray, "5 4 4"] = np.tile(np.eye(4), (5, 1, 1))
    world_T_cam[:, :3, 3] = rng.normal(size=(5, 3))
    oriented, transform = auto_orient_and_center_poses(world_T_cam.astype(np.float32), method="up", center_method="poses")
    assert oriented.shape == (5, 3, 4)
    assert transform.shape == (3, 4)
    np.testing.assert_allclose(oriented[:, :, 3].mean(axis=0), 0.0, atol=1e-9)
