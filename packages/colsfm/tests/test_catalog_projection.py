"""Which cuSFM projection a catalog camera's distortion model maps onto, and which models colsfm refuses."""

from __future__ import annotations

import numpy as np
import pytest
from simplecv.catalog_calibration import CatalogDistortion

from colsfm.catalog import projection_of


def test_known_models_map_onto_their_projection_and_a_camera_without_one_is_a_pinhole() -> None:
    assert projection_of("/world/rig_00/cam_02", None) == ("PINHOLE", ())
    assert projection_of("cam", CatalogDistortion("kannala_brandt", np.array([0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0]))) == (
        "OPENCV_FISHEYE",
        pytest.approx((0.1, 0.2, 0.3, 0.4)),
    )


@pytest.mark.parametrize("model", ["mei_v3", "mei_v6", "mei_v3_unified", "mei_v6_unified"])
def test_a_mei_camera_is_refused_by_name_and_not_read_as_a_pinhole(model: str) -> None:
    """cuSFM has no unified-sphere model: a Mei lens read as a pinhole (its Pinhole is gamma or the paraxial focal)
    would map with the wrong projection everywhere off the axis. colsfm reads the cube faces, which are pinholes."""
    with pytest.raises(ValueError, match=rf"/world/rig_00/cam_00.*{model}.*cube faces"):
        projection_of("/world/rig_00/cam_00", CatalogDistortion(model, np.zeros(17)))
