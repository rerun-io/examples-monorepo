"""Fast tests for X-Lens catalog sampling and output geometry."""

import numpy as np
import pytest
from simplecv.rerun_log_utils import RerunTyroConfig

from monopriors.apis.rig_depth_catalog import RigDepthCatalogConfig, nearest_time_index


@pytest.fixture(scope="module")
def rr_config() -> RerunTyroConfig:
    """Provide a headless recording configuration without a viewer process."""
    return RerunTyroConfig(headless=True)


def test_catalog_config_accepts_model_patch_grid(rr_config: RerunTyroConfig) -> None:
    """The four outward cameras use the larger patch-aligned 16:9 grid."""
    config = RigDepthCatalogConfig(rr_config=rr_config)
    assert config.cams == ("cam_00", "cam_01", "cam_04", "cam_05")
    assert (config.width, config.height) == (1120, 630)
    assert config.fuse is True
    assert config.log_points is False
    assert config.fusion_voxel_m == 0.04
    assert config.fusion_max_depth_m == 6.0
    assert config.mesh_every == 300


@pytest.mark.parametrize(("width", "height"), [(895, 504), (896, 503), (896, 490)])
def test_catalog_config_rejects_invalid_output_geometry(rr_config: RerunTyroConfig, width: int, height: int) -> None:
    """Both dimensions must be patch-aligned and preserve the source aspect ratio."""
    with pytest.raises(ValueError, match="multiples of 14|16:9"):
        RigDepthCatalogConfig(rr_config=rr_config, width=width, height=height)


def test_nearest_time_index_chooses_closest_sample() -> None:
    """Pose and video sampling use nearest timestamps, including either endpoint."""
    times = np.array([0, 10, 20], dtype="timedelta64[ns]")
    assert nearest_time_index(times, 1) == 0
    assert nearest_time_index(times, 16) == 2
    assert nearest_time_index(times, -5) == 0
    assert nearest_time_index(times, 50) == 2
