"""Aligning a run to its pose prior: the similarity is recovered despite outliers and applied exactly to poses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap

from colsfm.geometry import rigid3d_from_matrix
from colsfm.prior_alignment import PriorAlignment, _transform_tum, fit_prior_alignment


def _yaw(degrees: float) -> np.ndarray:
    angle: float = np.radians(degrees)
    return np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])


def test_the_similarity_is_recovered_and_a_few_outliers_do_not_move_it() -> None:
    rng = np.random.default_rng(0)
    estimate = rng.uniform(-10.0, 10.0, size=(200, 3))
    prior = 1.25 * estimate @ _yaw(30.0).T + np.array([1.0, -2.0, 0.5])
    prior[:5] += 8.0  # five cameras the run got wrong

    alignment: PriorAlignment = fit_prior_alignment(estimate, prior)

    assert abs(alignment.scale - 1.25) < 1e-9
    assert abs(alignment.rotation_deg - 30.0) < 1e-6
    np.testing.assert_allclose(alignment.translation_m, [1.0, -2.0, 0.5], atol=1e-9)
    assert alignment.num_cameras == 200 and alignment.num_fitted == 195
    assert alignment.residual_p90_m < 1e-9


def test_tum_poses_move_with_the_similarity_and_keep_their_stamps(tmp_path: Path) -> None:
    alignment = PriorAlignment(
        scale=2.0, rotation_deg=90.0, rotation_xyzw=[0.0, 0.0, float(np.sqrt(0.5)), float(np.sqrt(0.5))], translation_m=[1.0, 0.0, 0.0],
        residual_median_m=0.0, residual_p90_m=0.0, num_cameras=3, num_fitted=3,
    )  # fmt: skip
    path: Path = tmp_path / "poses.tum"
    path.write_text("2.2623050000000000 1.0 0.0 0.0 0.0 0.0 0.0 1.0\n")

    _transform_tum(path, alignment.sim3d())

    fields: list[float] = [float(value) for value in path.read_text().split()]
    assert fields[0] == 2.262305
    np.testing.assert_allclose(fields[1:4], [1.0, 2.0, 0.0], atol=1e-12)  # 2 * Rz(90) @ (1, 0, 0) + (1, 0, 0)
    rotated = pycolmap.Rotation3d(np.asarray(fields[4:8])).matrix()
    np.testing.assert_allclose(rotated, rigid3d_from_matrix(_yaw(90.0), np.zeros(3)).rotation.matrix(), atol=1e-12)
