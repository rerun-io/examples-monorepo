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
from tqdm import tqdm


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
        Render a tqdm bar over the per-layer ``dataset.register(...)`` loop.
        Default on; turn off for non-interactive callers (tests).
    """
    rrd_dir = rrd_dir.expanduser().resolve()
    layer_files = sorted(rrd_dir.rglob("*.rrd"))
    if not layer_files:
        raise FileNotFoundError(f"no RRDs found under {rrd_dir}")

    # Start with an empty dataset, then register every layer file under its
    # filename stem in a single batched call. Files sharing a recording_id
    # collapse into one segment automatically.
    #
    # The batched ``register(uris, layer_name=names)`` form is what the SDK's
    # ``RegistrationHandle`` API was designed for — one client call, per-URI
    # error reporting via ``iter_results()``, no manual loop bookkeeping.
    # Note that the wall-clock cost (~40 s for 500 files / 25 GB on the local
    # benchmark) is dominated by server-side .rrd parsing, not client
    # roundtrips, so batching here is a code-clarity win rather than a
    # latency win — by the time ``register()`` returns, the handle is
    # already fully resolved and ``iter_results`` exits in milliseconds.
    print(f"Mounting catalog from {rrd_dir} ({len(layer_files)} layer files)…", flush=True)
    server = rr.server.Server(datasets={dataset_name: []}, port=port)
    dataset = server.client().get_dataset(dataset_name)

    uris = [f.resolve().as_uri() for f in layer_files]
    layer_names = [f.stem for f in layer_files]
    handle = dataset.register(uris, layer_name=layer_names)

    errors: list[tuple[str, str]] = []
    with tqdm(total=len(layer_files), desc="register", unit="layer", disable=not show_progress) as bar:
        for result in handle.iter_results():
            bar.update(1)
            if result.is_error and result.error is not None:
                errors.append((result.uri, result.error))

    if errors:
        msg = "\n  ".join(f"{u}: {e}" for u, e in errors)
        raise RuntimeError(f"failed to register {len(errors)}/{len(layer_files)} layer(s):\n  {msg}")

    if blueprint is not None:
        # ``register_blueprint`` is implemented internally as
        # ``blueprint_dataset.register(uri, …)`` and needs a real URL with a
        # scheme — passing a bare ``str(rbl_path)`` raises
        # ``ValueError: Could not parse URL: relative URL without a base``.
        # ``Path.as_uri()`` gives the right ``file://…`` form.
        rbl_path = Path(tempfile.mkdtemp(prefix=f"{dataset_name}-")) / f"{dataset_name}.rbl"
        blueprint.save(application_id, path=str(rbl_path))
        dataset.register_blueprint(rbl_path.resolve().as_uri(), set_default=True)

    return server
