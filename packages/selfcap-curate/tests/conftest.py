"""Shared test assets and private-server factories for every test lane.

The helper modules import selfcap_curate, so each fixture imports its helper: pytest-beartype instruments
selfcap_curate only if this file has not imported it.
"""

import os

import pytest

# beartype runs under pytest only; code that cannot run under it (torch.compile, ONNX export) reads this and stays eager.
os.environ.setdefault("PIXI_DEV_MODE", "1")


@pytest.fixture
def fixture_sync():
    from frozen_v2 import fixture_sync

    return fixture_sync


@pytest.fixture
def private_catalog():
    from frozen_v2 import private_catalog

    return private_catalog


@pytest.fixture
def private_server():
    from rrd_fixtures import private_server

    return private_server


@pytest.fixture
def synthetic_base():
    from rrd_fixtures import synthetic_base

    return synthetic_base
