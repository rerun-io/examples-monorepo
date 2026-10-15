"""Types through a real Rerun catalog read (in-process server): ``rerun_roundtrip`` gives each value back (integration lane)."""

import numpy as np
import pytest
import torch
from dagkit.testing import rerun_roundtrip
from dagkit.time import STATIC
from mamma_fakes import assert_same

from mamma.types import BodyTopology, Landmarks3d, MultiViewTracks, SmplxFits

pytestmark = pytest.mark.integration
PEOPLE = "/world/mamma/people"


def test_fits_and_landmarks_come_back_from_a_catalog_read() -> None:
    rng = np.random.default_rng(0)
    fits = SmplxFits(7, (0, 2), *(rng.standard_normal(s).astype(np.float32) for s in ((2, 5, 3), (2, 165), (2, 16), (2, 3))))
    ((t_ns, back),) = rerun_roundtrip(fits, PEOPLE, "video_time", 1_000_000)
    assert t_ns == 1_000_000
    assert_same(back, fits)
    valid = rng.random((1, 6)) > 0.3
    cloud = Landmarks3d(7, (1,), np.where(valid[..., None], rng.standard_normal((1, 6, 3)), 0.0).astype(np.float32), valid)
    assert_same(rerun_roundtrip(cloud, PEOPLE, "video_time", 1_000_000)[0][1], cloud)
    topology = BodyTopology((0,), np.arange(9, dtype=np.int32).reshape(3, 3))
    ((t_ns, faces),) = rerun_roundtrip(topology, PEOPLE, "video_time", STATIC)
    assert t_ns == STATIC and faces.persons == (0,) and np.array_equal(faces.faces, topology.faces)


def test_tracks_come_back_from_a_catalog_read() -> None:
    masks = torch.zeros((2, 6, 8), dtype=torch.bool)
    masks[0, 1:4, 2:5] = True
    tracks = MultiViewTracks(
        3, ("rig_01/cam_00", "rig_03/cam_00"), np.array([0, 1]), np.array([0, 0]), masks,
        np.array([[2.0, 1.0, 4.0, 3.0], [np.nan] * 4], dtype=np.float32), np.array([0.9, 0.1], dtype=np.float32), np.full((2, 2), 2.0, dtype=np.float32),
    )  # fmt: skip
    ((t_ns, back),) = rerun_roundtrip(tracks, "/world", "video_time", 5_000)
    assert t_ns == 5_000
    assert_same(back, tracks)
