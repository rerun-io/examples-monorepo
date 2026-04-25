"""Mount a directory of RRDs as a local Rerun dataset served by ``rr.server.Server``."""

from __future__ import annotations

from pathlib import Path

import rerun as rr


def mount_catalog(
    rrd_dir: Path,
    *,
    dataset_name: str = "vslam",
    port: int | None = None,
) -> rr.server.Server:
    """Spin up a local catalog server exposing every ``*.rrd`` under ``rrd_dir``.

    Use as a context manager::

        with mount_catalog(Path("data/slam-evals/rrd"), port=9987) as server:
            print(server.url())  # rerun+http://127.0.0.1:9987 — connect viewer here
            ds = server.client().get_dataset("vslam")

    Pass ``port`` for a stable, viewer-connectable URL; defaults to a random
    free port chosen by ``rr.server.Server``.
    """
    rrd_dir = rrd_dir.expanduser().resolve()
    rrds = sorted(str(p) for p in rrd_dir.rglob("*.rrd"))
    if not rrds:
        raise FileNotFoundError(f"no RRDs found under {rrd_dir}")
    return rr.server.Server(datasets={dataset_name: rrds}, port=port)
