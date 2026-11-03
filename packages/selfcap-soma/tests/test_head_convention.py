"""Head tracks stored with y up and z back (SelfCap captures of 2025-12-11/12) reach the fit in RDF axes."""

import numpy as np
from dataforge.selfcap_schema import rdf_head
from scipy.spatial.transform import Rotation


def test_a_y_up_z_back_head_track_is_turned_to_rdf_and_an_rdf_track_is_kept() -> None:
    forward = np.array([0.0, -0.5, 1.0]) / np.linalg.norm([0.0, -0.5, 1.0])
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    level = np.c_[right, np.cross(forward, right), forward]  # RDF camera in a Y-up world, looking ahead and down
    rdf = np.repeat(Rotation.from_matrix(level).as_quat()[None], 10, axis=0)
    rub = np.repeat(Rotation.from_matrix(level @ np.diag([1.0, -1.0, -1.0])).as_quat()[None], 10, axis=0)
    assert np.allclose(Rotation.from_quat(rdf_head(rub).quaternions).as_matrix(), level, atol=1e-9)
    assert np.array_equal(rdf_head(rdf).quaternions, rdf)
