"""Mount a directory of RRDs as a local Rerun dataset served by ``rr.server.Server``."""

from __future__ import annotations

from pathlib import Path

import rerun as rr


def mount_catalog(rrd_dir: Path, *, dataset_name: str = "vslam") -> rr.server.Server:
    """Spin up a local catalog server exposing every ``*.rrd`` under ``rrd_dir``.

    Use as a context manager::

        with mount_catalog(Path("data/slam-evals/rrd")) as server:
            ds = server.client().get_dataset("vslam")

    The server picks a random free port and binds to ``0.0.0.0`` by default;
    pass it through to ``rr.server.Server`` if a fixed port is needed.
    """
    rrd_dir = rrd_dir.expanduser().resolve()
    rrds = sorted(str(p) for p in rrd_dir.rglob("*.rrd"))
    if not rrds:
        raise FileNotFoundError(f"no RRDs found under {rrd_dir}")
    return rr.server.Server(datasets={dataset_name: rrds})
