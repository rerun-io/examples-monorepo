"""Mount per-sequence layer ``.rrd`` directories as a Rerun catalog dataset.

Each sequence on disk is laid out as one directory per ``<dataset>/<seq>/``
containing 3-7 ``.rrd`` files (one per source data stream). All layer files
share the sequence's ``recording_id``, so registering each one under its
filename stem as ``layer_name`` collapses them into a single segment per
sequence at query/view time. See ``docs/schema.md`` for the schema.
"""

from __future__ import annotations

import tempfile
from collections import defaultdict
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

    # Start with an empty dataset, then register one batch per layer name
    # (calibration, groundtruth, view_coordinates, rgb_<i>, depth_<i>,
    # imu_<i>). Files sharing a recording_id collapse into one segment
    # automatically — the catalog uses ``recording_id`` from the .rrd
    # itself as the segment key, regardless of how we group at register
    # time, so grouping by layer name is purely a transport optimisation.
    #
    # Per-call register is ~30-100 ms of fixed overhead and the server
    # has a fast path when ``layer_name`` is a single string — it can
    # process the URIs as one bulk operation instead of one (uri,
    # layer_name) pair at a time. Per advice from rerun engineers: for
    # ~500 files this drops mount time from ~40 s to a few seconds.
    # tqdm shows per-layer-group progress so the operator can see what's
    # happening; ``iter_results`` would give per-file granularity but
    # exits instantly because the SDK's ``register`` is synchronous.
    print(f"Mounting catalog from {rrd_dir} ({len(layer_files)} layer files)…", flush=True)
    server = rr.server.Server(datasets={dataset_name: []}, port=port)
    dataset = server.client().get_dataset(dataset_name)

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
        dataset.register(uris, layer_name=stem).wait()

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
