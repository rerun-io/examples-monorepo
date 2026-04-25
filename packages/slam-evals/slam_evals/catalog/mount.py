"""Mount a directory of RRDs as a local Rerun dataset served by ``rr.server.Server``.

Optionally registers a default blueprint at the *catalog* level (idiomatic
per the rerun docs), so the viewer picks it up for every recording in the
dataset without each RRD needing a baked-in copy.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb


def mount_catalog(
    rrd_dir: Path,
    *,
    dataset_name: str = "vslam",
    port: int | None = None,
    blueprint: rrb.Blueprint | None = None,
    application_id: str = "slam-evals",
) -> rr.server.Server:
    """Spin up a local catalog server exposing every ``*.rrd`` under ``rrd_dir``.

    Use as a context manager::

        with mount_catalog(Path("data/slam-evals/rrd"), port=9987,
                           blueprint=build_blueprint()) as server:
            print(server.url())  # rerun+http://127.0.0.1:9987 — connect viewer here
            ds = server.client().get_dataset("vslam")

    Parameters
    ----------
    rrd_dir:
        Directory containing ``*.rrd`` files, recursed.
    dataset_name:
        Logical name the catalog exposes the recordings under.
    port:
        Pin a stable, viewer-connectable port; default picks a free random one.
    blueprint:
        Optional default blueprint to register on the dataset entry. Saved
        once to a ``.rbl`` tempfile and pointed at via
        ``DatasetEntry.register_blueprint``. The viewer picks this up for
        every recording in the dataset.
    application_id:
        Application id stamped into the saved ``.rbl``. Must match the
        ``application_id`` used at ingest time (slam-evals' default is
        ``"slam-evals"``); otherwise rerun won't apply the blueprint.
    """
    rrd_dir = rrd_dir.expanduser().resolve()
    rrds = sorted(str(p) for p in rrd_dir.rglob("*.rrd"))
    if not rrds:
        raise FileNotFoundError(f"no RRDs found under {rrd_dir}")

    server = rr.server.Server(datasets={dataset_name: rrds}, port=port)

    if blueprint is not None:
        # ``register_blueprint`` wants a URI to a real .rbl on disk. We write
        # it next to /tmp and let the OS clean up; lifetime > server lifetime.
        rbl_path = Path(tempfile.mkdtemp(prefix=f"{dataset_name}-")) / f"{dataset_name}.rbl"
        blueprint.save(application_id, path=str(rbl_path))
        server.client().get_dataset(dataset_name).register_blueprint(str(rbl_path), set_default=True)

    return server
