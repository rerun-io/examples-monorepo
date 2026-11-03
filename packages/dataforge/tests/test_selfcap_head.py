"""SelfCap read-side head axes; stored poses are never rewritten."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dataforge.selfcap_schema import rdf_head


@pytest.mark.parametrize("yaw", [-0.7, 0.0, 0.8])
def test_rub_head_axes_become_rdf_without_mutating_the_source(yaw: float) -> None:
    rdf = Rotation.from_euler("y", [[yaw], [yaw + 0.1]]).as_matrix() @ np.diag([1.0, -1.0, -1.0])
    source = Rotation.from_matrix(rdf @ np.diag([1.0, -1.0, -1.0])).as_quat()
    before = source.copy()
    converted = rdf_head(source)
    assert converted.convention == "RUB"
    np.testing.assert_allclose(Rotation.from_quat(converted.quaternions).as_matrix(), rdf, atol=1e-12)
    np.testing.assert_array_equal(source, before)
    again = rdf_head(converted.quaternions)
    assert again.convention == "RDF"
    np.testing.assert_array_equal(again.quaternions, converted.quaternions)


def test_rdf_and_untracked_rows_keep_their_values() -> None:
    source = np.array([[1.0, 0.0, 0.0, 0.0], [np.nan] * 4, [0.0] * 4])
    result = rdf_head(source)
    assert result.convention == "RDF"
    np.testing.assert_array_equal(result.quaternions, source)


def test_an_empty_head_track_stays_empty() -> None:
    assert rdf_head(np.empty((0, 4), dtype=np.float64)).quaternions.shape == (0, 4)


def test_normalized_reads_keep_the_recorded_basis_for_camera_extrinsics() -> None:
    raw = Rotation.from_euler("y", [[0.2], [0.4]]).as_quat()
    axes = rdf_head(raw)
    again = rdf_head(axes.quaternions, axes.convention)
    assert again.convention == "RUB"
    np.testing.assert_array_equal(again.recorded_T_rdf, np.diag([1.0, -1.0, -1.0, 1.0]))
    mount = np.eye(4)
    mount[:3, 3] = [0.1, 0.2, 0.3]
    raw_poses = np.tile(np.eye(4), (2, 1, 1))
    raw_poses[:, :3, :3] = Rotation.from_quat(raw).as_matrix()
    normalized_poses = raw_poses.copy()
    normalized_poses[:, :3, :3] = Rotation.from_quat(axes.quaternions).as_matrix()
    np.testing.assert_allclose(raw_poses @ mount, normalized_poses @ np.linalg.inv(axes.recorded_T_rdf) @ mount, atol=1e-12)
