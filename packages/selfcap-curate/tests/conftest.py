"""Shared test assets and private-server factories for every test lane."""

import pytest
from frozen_v2 import fixture_sync as make_sync
from frozen_v2 import private_catalog as catalog
from rrd_fixtures import private_server as server
from rrd_fixtures import synthetic_base as base


@pytest.fixture
def fixture_sync():
    return make_sync


@pytest.fixture
def private_catalog():
    return catalog


@pytest.fixture
def private_server():
    return server


@pytest.fixture
def synthetic_base():
    return base
