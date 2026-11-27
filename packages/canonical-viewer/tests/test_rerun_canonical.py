"""The built rerun-canonical binary is the stock rerun command line at the SDK's release."""

import subprocess

import pytest
import rerun as rr

from canonical_viewer.pixels import CANONICAL_VIEWER

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def binary() -> str:
    if not CANONICAL_VIEWER.is_file():
        pytest.skip(f"{CANONICAL_VIEWER} is missing: run `pixi run -e canonical-viewer-dev canonical-viewer-build`")
    return str(CANONICAL_VIEWER)


def test_version_is_the_sdk_release(binary: str) -> None:
    version: str = subprocess.run([binary, "--version"], check=True, capture_output=True, text=True).stdout
    assert version.split()[1] == rr.__version__, version


@pytest.mark.parametrize("command", [["mcp", "--help"], ["rrd", "--help"], ["--help"]])
def test_stock_subcommands_exist(binary: str, command: list[str]) -> None:
    subprocess.run([binary, *command], check=True, capture_output=True, timeout=60)
