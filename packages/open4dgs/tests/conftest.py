"""Shared fixtures. No workspace package is imported at module level: pytest-beartype hooks imports after this file loads."""

import os
import socket
from urllib.parse import urlparse

import pytest

# beartype runs under pytest only; code that cannot run under it (torch.compile of jaxtyping-annotated functions: the
# batched prior terms) reads this and stays eager.
os.environ.setdefault("PIXI_DEV_MODE", "1")

SPARK_CATALOG_URL: str = "rerun+http://dgx-spark:51235"
"""The read-only Spark catalog that holds the MAMMA datasets."""


@pytest.fixture(scope="session")
def spark_catalog_url() -> str:
    """The Spark catalog URL, or a skip when its port does not answer."""
    parsed = urlparse(SPARK_CATALOG_URL.removeprefix("rerun+"))
    try:
        with socket.create_connection((parsed.hostname or "", parsed.port or 51235), timeout=3.0):
            pass
    except OSError:
        pytest.skip(f"the Spark catalog {SPARK_CATALOG_URL} is unreachable")
    return SPARK_CATALOG_URL
