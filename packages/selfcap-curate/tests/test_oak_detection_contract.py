"""EgoHand calibration layers provide the same COCO hands to SOMA evidence reads."""

from pathlib import Path

import numpy as np
import pytest
from rerun.server import Server

soma_catalog = pytest.importorskip("selfcap_soma.catalog", reason="requires selfcap-soma on PYTHONPATH for the cross-package contract")
evaluation_io = pytest.importorskip("selfcap_soma.evaluation_io", reason="requires selfcap-soma for the cross-package contract")

from selfcap_curate.detection_io import write_oak_egohand_layer
from selfcap_curate.records import OakDetections


@pytest.mark.integration
def test_oak_layer_round_trips_both_hands_into_soma_image_evidence(tmp_path: Path) -> None:
    times = np.array([1.0, 1.04])
    xy = np.full((2, 133, 2), np.nan)
    xy[:, 91:112] = [101.0, 202.0]
    xy[:, 112:133] = [303.0, 404.0]
    scores = np.where(np.isfinite(xy).all(-1), 0.9, 0.0)
    xy[1, 91] = [999.0, 888.0]
    scores[1, 91] = 0.1
    path = tmp_path / "egohand.rrd"
    write_oak_egohand_layer(path, "episode", 4, {"rgb": OakDetections(times, xy, scores)})
    with Server(port=0, datasets={"selfcap": [path]}) as server:
        view = server.client().get_dataset("selfcap").filter_segments("episode")
        catalog = soma_catalog.SegmentCatalog(view, view.schema().column_names(), soma_catalog.RigPaths((), "/world/rig_04", "/world/rig_05"))
        tracks = evaluation_io.read_egohand_rgb(catalog, times)
    assert len(tracks) == 2
    left = next(track for track in tracks if "left" in track.name)
    right = next(track for track in tracks if "right" in track.name)
    np.testing.assert_array_equal(left.times, times)
    np.testing.assert_array_equal(left.points[0], np.tile([101.0, 202.0], (21, 1)))
    np.testing.assert_array_equal(right.points, np.tile([303.0, 404.0], (2, 21, 1)))
    assert np.isnan(left.points[1, 0]).all()
