"""Mount + refresh per-sequence layer ``.rrd`` directories as a Rerun catalog dataset.

Each sequence on disk is laid out as one directory per ``<dataset>/<seq>/``
containing 3-7 ``.rrd`` files (one per source data stream). All layer files
share the sequence's ``recording_id``, so registering each one under its
filename stem as ``layer_name`` collapses them into a single segment per
sequence at query/view time. See ``docs/schema.md`` for the schema.

Two entry points share the same per-layer-group registration helper:

* ``mount_catalog`` starts a new in-process gRPC server, registers every
  matching layer file, and returns the long-lived ``Server`` to the
  caller. Used at startup.

* ``refresh_catalog`` connects to an already-running catalog server via
  ``CatalogClient`` and re-registers a subset of files using
  ``OnDuplicateSegmentLayer.REPLACE``. Used to push edits without
  restarting the server (avoids the 30-40 s parse cost of the full
  corpus on every dev-loop iteration).
"""

from __future__ import annotations

import atexit
import tempfile
import weakref
from collections import defaultdict
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb
from rerun.catalog import CatalogClient, OnDuplicateSegmentLayer
from tqdm import tqdm


def _filter_layer_files(
    rrd_dir: Path,
    *,
    datasets: tuple[str, ...] = (),
    layers: tuple[str, ...] = (),
    only: tuple[str, ...] = (),
) -> list[Path]:
    """Apply ingest-style filters to the recursive ``*.rrd`` glob.

    Files live at ``<rrd_dir>/<dataset>/<seq>/<layer>.rrd``, so:

    * ``datasets``: keep only files whose grandparent directory matches.
    * ``layers``: keep only files whose stem (= layer name) matches.
    * ``only``: keep only files whose ``<dataset>/<seq>`` slug matches.

    Empty filter tuples mean "no restriction" for that axis, mirroring
    ``tools/ingest.py``'s ``IngestConfig`` semantics.
    """
    files = sorted(rrd_dir.rglob("*.rrd"))
    if datasets:
        ds_set = set(datasets)
        files = [f for f in files if f.parent.parent.name in ds_set]
    if layers:
        layer_set = set(layers)
        files = [f for f in files if f.stem in layer_set]
    if only:
        only_set = set(only)
        files = [f for f in files if f"{f.parent.parent.name}/{f.parent.name}" in only_set]
    return files


def _register_layer_groups(
    dataset: object,
    layer_files: list[Path],
    *,
    on_duplicate: OnDuplicateSegmentLayer,
    show_progress: bool,
) -> None:
    """Register ``layer_files`` against ``dataset``, batched per layer name.

    The server has a fast path when ``register`` is called with a
    single ``layer_name`` and many URIs — it processes the URIs as one
    bulk operation instead of one (uri, layer_name) pair at a time.
    Group files by their filename stem (which is also their layer
    name) and issue one register call per group; for ~500 files this
    drops the round-trip overhead from ~10 s to under 1 s. Server-side
    .rrd parsing dominates the rest, and grouping doesn't speed that
    up.
    """
    layers_by_stem: dict[str, list[Path]] = defaultdict(list)
    for f in layer_files:
        layers_by_stem[f.stem].append(f)

    iterator = tqdm(
        sorted(layers_by_stem.items()),
        desc="register",
        unit="layer-group",
        disable=not show_progress,
    )
    for stem, files in iterator:
        iterator.set_postfix_str(f"{stem} ({len(files)} files)")
        uris = [f.resolve().as_uri() for f in files]
        # pyrefly can't see ``register`` because we accept ``dataset`` as
        # ``object`` (DatasetEntry vs the catalog-client variant — both
        # quack the same on this method).
        dataset.register(uris, layer_name=stem, on_duplicate=on_duplicate).wait()  # type: ignore[attr-defined]


def mount_catalog(
    rrd_dir: Path,
    *,
    dataset_name: str = "vslam",
    port: int | None = None,
    blueprint: rrb.Blueprint | None = None,
    application_id: str = "slam-evals",
    show_progress: bool = True,
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
    show_progress:
        Render a tqdm bar over the per-layer-group ``dataset.register(...)``
        loop. Default on; turn off for non-interactive callers (tests).
    """
    rrd_dir = rrd_dir.expanduser().resolve()
    layer_files = sorted(rrd_dir.rglob("*.rrd"))
    if not layer_files:
        raise FileNotFoundError(f"no RRDs found under {rrd_dir}")

    print(f"Mounting catalog from {rrd_dir} ({len(layer_files)} layer files)…", flush=True)
    server = rr.server.Server(datasets={dataset_name: []}, port=port)
    dataset = server.client().get_dataset(dataset_name)

    _register_layer_groups(
        dataset,
        layer_files,
        on_duplicate=OnDuplicateSegmentLayer.ERROR,
        show_progress=show_progress,
    )

    if blueprint is not None:
        # ``register_blueprint`` is implemented internally as
        # ``blueprint_dataset.register(uri, …)`` and needs a real URL with a
        # scheme — passing a bare ``str(rbl_path)`` raises
        # ``ValueError: Could not parse URL: relative URL without a base``.
        # ``Path.as_uri()`` gives the right ``file://…`` form.
        #
        # We use ``TemporaryDirectory`` (kept alive for the server's lifetime
        # via ``weakref.finalize``) plus an ``atexit`` belt-and-suspenders
        # so the rbl tempdir is removed when the server is GCed *or* when
        # the process exits. Using ``mkdtemp`` directly leaked one
        # directory per mount.
        tmp_dir = tempfile.TemporaryDirectory(prefix=f"{dataset_name}-")
        weakref.finalize(server, tmp_dir.cleanup)
        atexit.register(tmp_dir.cleanup)
        rbl_path = Path(tmp_dir.name) / f"{dataset_name}.rbl"
        blueprint.save(application_id, path=str(rbl_path))
        dataset.register_blueprint(rbl_path.resolve().as_uri(), set_default=True)

    return server


def refresh_catalog(
    rrd_dir: Path,
    *,
    catalog_url: str,
    dataset_name: str = "vslam",
    datasets: tuple[str, ...] = (),
    layers: tuple[str, ...] = (),
    only: tuple[str, ...] = (),
    show_progress: bool = True,
) -> int:
    """Push a subset of layer files into an already-running catalog server.

    Connects to ``catalog_url`` (typically ``rerun+http://127.0.0.1:9987``)
    via ``CatalogClient`` and re-registers the matching layer files with
    ``OnDuplicateSegmentLayer.REPLACE`` so existing entries are swapped
    in-place. Use this to apply edits to the running catalog without
    paying the ~30-40 s cold-mount cost.

    Parameters mirror ``tools/ingest.py``'s filter flags so the same
    invocation pattern works (``--datasets``, ``--layers``, ``--only``).

    Returns the number of files registered. Zero means no files matched
    the filters; the caller should treat that as a no-op rather than an
    error so ``--datasets X --layers Y`` is safe to script.
    """
    rrd_dir = rrd_dir.expanduser().resolve()
    layer_files = _filter_layer_files(rrd_dir, datasets=datasets, layers=layers, only=only)
    if not layer_files:
        print(f"No layer files matched filters under {rrd_dir}; nothing to refresh.")
        return 0

    print(
        f"Refreshing catalog at {catalog_url} ({len(layer_files)} layer files)…",
        flush=True,
    )
    client = CatalogClient(catalog_url)
    dataset = client.get_dataset(dataset_name)

    _register_layer_groups(
        dataset,
        layer_files,
        on_duplicate=OnDuplicateSegmentLayer.REPLACE,
        show_progress=show_progress,
    )
    return len(layer_files)
