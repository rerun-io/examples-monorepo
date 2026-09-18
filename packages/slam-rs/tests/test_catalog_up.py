"""Starting the local catalog is one operation with a real readiness check, not a shell probe of a port."""

import os
import socket
from pathlib import Path

import pytest

pytest.importorskip("rerun.catalog", reason="requires the rerun.catalog dependency in this environment")

from rerun.catalog import CatalogClient

from slam_rs.apis.catalog_up import ensure_catalog

pytestmark = pytest.mark.integration

def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_starts_a_catalog_once_and_reuses_it_after(tmp_path: Path) -> None:
    """First call starts a server and records the server's own pid; the second call finds it and starts nothing."""
    port: int = free_port()
    pid: int | None = ensure_catalog(port=port, root=tmp_path, timeout_s=20.0)
    try:
        assert pid is not None and alive(pid)
        assert int((tmp_path / "server.pid").read_text()) == pid
        assert (tmp_path / "server.log").exists()
        assert CatalogClient(f"rerun+http://127.0.0.1:{port}").dataset_names() == []
        assert ensure_catalog(port=port, root=tmp_path, timeout_s=20.0) is None
        assert int((tmp_path / "server.pid").read_text()) == pid
    finally:
        if pid is not None:
            os.kill(pid, 15)


def test_a_catalog_that_never_answers_is_an_error_and_leaves_no_pid_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A server that starts but never becomes a catalog fails loudly, pointing at its log, and claims no ownership."""
    monkeypatch.setattr("slam_rs.apis.catalog_up.SERVER_COMMAND", ("sleep", "30"))
    with pytest.raises(RuntimeError, match="server.log"):
        ensure_catalog(port=free_port(), root=tmp_path, timeout_s=1.0)
    assert not (tmp_path / "server.pid").exists()
