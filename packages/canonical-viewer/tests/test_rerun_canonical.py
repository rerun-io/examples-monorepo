"""The built rerun-canonical binary is the stock rerun command line at the SDK's release."""

import subprocess
from pathlib import Path

import pytest
import rerun as rr

pytestmark = pytest.mark.integration


def test_version_is_the_sdk_release(canonical_viewer: Path) -> None:
    version: str = subprocess.run([canonical_viewer, "--version"], check=True, capture_output=True, text=True).stdout
    assert version.split()[1] == rr.__version__, version


@pytest.mark.parametrize("command", [["mcp", "--help"], ["rrd", "--help"], ["--help"]])
def test_stock_subcommands_exist(canonical_viewer: Path, command: list[str]) -> None:
    subprocess.run([canonical_viewer, *command], check=True, capture_output=True, timeout=60)
