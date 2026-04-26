"""Mount per-sequence layer ``.rrd`` directories as a Rerun catalog dataset.

Each sequence on disk is laid out as one directory per ``<dataset>/<seq>/``
containing 3-7 ``.rrd`` files (one per source data stream). All layer files
share the sequence's ``recording_id``, so registering each one under its
filename stem as ``layer_name`` collapses them into a single segment per
sequence at query/view time. See ``docs/schema.md`` for the schema.
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
    """Spin up a local catalog server, register every layer ``.rrd`` under ``rrd_dir``.

    Use as a context manager::

        with mount_catalog(Path("data/slam-evals/rrd"), port=9987,
                           blueprint=build_blueprint()) as server:
            print(server.url())  # rerun+http://127.0.0.1:9987 — connect viewer here
            ds = server.client().get_dataset("vslam")
            df = ds.segment_table().to_pandas()

    Parameters
    ----------
    rrd_dir:
        Directory containing layer files at
        ``<rrd_dir>/<dataset>/<seq>/<layer_name>.rrd``. Globbed recursively;
        each ``.rrd`` becomes a layer named after its filename stem.
    dataset_name:
        Logical name the catalog exposes the segments under.
    port:
        Pin a stable, viewer-connectable port; default picks a free random one.
    blueprint:
        Optional default blueprint to register on the dataset. Saved once
        to a ``.rbl`` tempfile and pointed at via
        ``DatasetEntry.register_blueprint``. The viewer picks this up for
        every segment in the dataset.
    application_id:
        Application id stamped into the saved ``.rbl``. Must match the
        ``application_id`` used at ingest time (``"slam-evals"`` by default);
        otherwise rerun won't apply the blueprint.
    """
    rrd_dir = rrd_dir.expanduser().resolve()
    layer_files = sorted(rrd_dir.rglob("*.rrd"))
    if not layer_files:
        raise FileNotFoundError(f"no RRDs found under {rrd_dir}")

    # Start with an empty dataset, then explicitly register each layer file
    # under its filename stem. Files sharing a recording_id collapse into
    # one segment automatically.
    server = rr.server.Server(datasets={dataset_name: []}, port=port)
    dataset = server.client().get_dataset(dataset_name)

    for layer_file in layer_files:
        dataset.register(
            [layer_file.resolve().as_uri()],
            layer_name=layer_file.stem,
        ).wait()

    if blueprint is not None:
        # ``register_blueprint`` wants a URI to a real .rbl on disk. We write
        # it next to /tmp and let the OS clean up; lifetime > server lifetime.
        rbl_path = Path(tempfile.mkdtemp(prefix=f"{dataset_name}-")) / f"{dataset_name}.rbl"
        blueprint.save(application_id, path=str(rbl_path))
        dataset.register_blueprint(str(rbl_path), set_default=True)

    return server
