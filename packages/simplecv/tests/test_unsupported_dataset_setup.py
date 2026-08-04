from pathlib import Path

import pytest

from simplecv.data.exoego.ego_dex import EgoDexConfig
from simplecv.data.exoego.rrd_exoego import RRDExoEgoConfig


@pytest.mark.parametrize(
    ("config", "expected_message"),
    [
        (EgoDexConfig(root_directory=Path("/does/not/exist")), "EgoDex dataset setup is disabled"),
        (RRDExoEgoConfig(rrd_path=Path("/does/not/exist.rrd")), "RRD dataset setup is disabled"),
    ],
)
def test_unsupported_dataset_fails_during_setup(config: EgoDexConfig | RRDExoEgoConfig, expected_message: str) -> None:
    with pytest.raises(RuntimeError, match=expected_message):
        config.setup()
