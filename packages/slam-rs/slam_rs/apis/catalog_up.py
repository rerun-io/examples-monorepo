"""Start a local catalog and record its PID only after it answers a catalog query."""

import os
import resource
import signal
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from beartype.roar import BeartypeException
from rerun.catalog import CatalogClient

SERVER_COMMAND: tuple[str, ...] = ("rerun", "server")


def _catalog_ready(port: int) -> bool:
    """Probe the catalog API, allowing configuration errors to surface.

    The catalog this module starts is plain HTTP on the loopback, so a token the
    SDK has stored for some other server must not veto the probe; this is the
    same override every local-catalog task in `pixi.toml` sets.
    """
    os.environ.setdefault("RERUN_INSECURE_SKIP_HOST_CHECK", "1")
    try:
        CatalogClient(f"rerun+http://127.0.0.1:{port}").dataset_names()
    except BeartypeException:
        raise
    except ConnectionError:
        return False
    return True


def ensure_catalog(port: int, root: Path, timeout_s: float) -> int | None:
    """Reuse a ready catalog or start one, returning the new server's PID.

    Raises:
        RuntimeError: If the new catalog does not become ready within the timeout.
    """
    if _catalog_ready(port):
        print(f"Reusing the catalog on 127.0.0.1:{port}.")
        return None

    root.mkdir(parents=True, exist_ok=True)
    try:
        limits: tuple[int, int] = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (max(limits[0], 524288), limits[1]))
    except (OSError, ValueError) as error:
        print(f"Could not raise the open-file limit: {error}; fine for the sample, raise it before registering all 64 recordings.")

    log_path: Path = root / "server.log"
    with log_path.open("a") as log:
        proc: subprocess.Popen[bytes] = subprocess.Popen(
            [*SERVER_COMMAND, "--port", str(port)],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    ready: bool = False
    deadline: float = time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline:
            if _catalog_ready(port):
                pid: int = proc.pid
                # The Python CLI wrapper launches the actual server as its child.
                try:
                    children: subprocess.CompletedProcess[str] = subprocess.run(
                        ["pgrep", "-P", str(proc.pid)], capture_output=True, text=True, check=False
                    )
                    if children.returncode == 0 and children.stdout.strip():
                        pid = int(children.stdout.split()[0])
                except FileNotFoundError:
                    pass
                (root / "server.pid").write_text(str(pid))
                ready = True
                print(f"The catalog is up on 127.0.0.1:{port} and stays running. Stop it with: kill $(cat {root / 'server.pid'})")
                return pid
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
        raise RuntimeError(f"The catalog did not start; see {log_path}")
    finally:
        if not ready:
            # Terminate only this launch's process group, including the wrapper's child.
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


@dataclass
class Config:
    """Local catalog startup settings."""

    port: int = 51235
    """Port on which the catalog listens."""
    root: Path = Path("data/msd-rrd")
    """Directory for the server log and PID file."""
    timeout_s: float = 10.0
    """Maximum time to wait for catalog readiness, in seconds."""


def main(config: Config) -> None:
    """Start or reuse the configured local catalog."""
    ensure_catalog(config.port, config.root, config.timeout_s)
