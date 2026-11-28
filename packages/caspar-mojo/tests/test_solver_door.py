"""The Python door's checks that need no GPU and no built extension."""

import dataclasses

import numpy as np
import pytest
from reference.synthetic import make_problem

from caspar_mojo.problem import BaOptions, CameraModel
from caspar_mojo.solver import solve


def test_full_system_pcg_is_refused() -> None:
    problem = make_problem(CameraModel.PINHOLE, n_frames=3, n_points=13, seed=0)[0]
    with pytest.raises(NotImplementedError):
        solve(problem, BaOptions(reduced_camera_system=False))


def test_mixed_camera_models_are_refused() -> None:
    problem = make_problem(CameraModel.PINHOLE, n_frames=3, n_points=13, seed=0, rig_cameras=2)[0]
    mixed = dataclasses.replace(problem, camera_model=np.array([CameraModel.PINHOLE, CameraModel.SIMPLE_RADIAL], dtype=np.int32))
    with pytest.raises(ValueError, match="one camera model per solve"):
        solve(mixed)
