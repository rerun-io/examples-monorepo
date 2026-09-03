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
    """The Robocap output size is 16:9 and divisible by the ViT patch size."""
    config = RigDepthCatalogConfig(rr_config=rr_config)
    assert (config.width, config.height) == (896, 504)
    assert config.fuse is False


@pytest.mark.parametrize(("width", "height"), [(895, 504), (896, 503), (896, 490)])
def test_catalog_config_rejects_invalid_output_geometry(rr_config: RerunTyroConfig, width: int, height: int) -> None:
    """Both dimensions must be patch-aligned and preserve the source aspect ratio."""
    with pytest.raises(ValueError, match="multiples of 14|16:9"):
        RigDepthCatalogConfig(rr_config=rr_config, width=width, height=height)


def test_catalog_config_rejects_fisheye_tsdf(rr_config: RerunTyroConfig) -> None:
    """The pinhole-only shared fuser cannot consume fisheye depth."""
    with pytest.raises(ValueError, match="fisheye TSDF"):
        RigDepthCatalogConfig(rr_config=rr_config, fuse=True)


def test_nearest_time_index_chooses_closest_sample() -> None:
    """Pose and video sampling use nearest timestamps, including either endpoint."""
    times = np.array([0, 10, 20], dtype="timedelta64[ns]")
    assert nearest_time_index(times, 1) == 0
    assert nearest_time_index(times, 16) == 2
    assert nearest_time_index(times, -5) == 0
    assert nearest_time_index(times, 50) == 2
