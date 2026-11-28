"""Write a catalog layer atomically and register it: one writer and one registrar for every package.

``serve_local_segment`` serves a base recording and named layer files of its segment on a private in-process catalog.
``register_named_blueprint`` registers a blueprint under a name and retires the blueprint that held it.
``segment_properties`` reads one segment's property group from the segment table.

A layer is a recording file with its segment's recording id; the catalog stacks a segment's layers by that id. A target
file either exists complete or not at all: half-written files poison batch runs and the catalog server, and saving over
an rrd that a server has registered truncates what the server reads. So ``atomic_recording`` writes a temp file beside
the target and replaces the target only on a clean exit; re-runs write a fresh temp file and the catalog re-registers.

Recording properties and layers: when the catalog merges a segment's layers, distinct property groups
(``property:masks:*``, ``property:moge2_prior:*``) sit side by side, but a column that two layers both carry has one
value, the one of the layer registered last (checked on rerun 0.42: no error, the newest registration wins, so a
``REPLACE`` re-registration of an old layer flips it). Rerun's own ``RecordingInfo`` (name, wall-clock ``start_time``)
is such a column: a derived layer passes ``send_properties=False`` so that the base layer's stays, and names its own
property group after the layer.
"""

import os
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
import rerun.chunk as rrc
from datafusion import col, lit
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer, RegisteredBlueprint

LENS_MASK_LAYER: str = "lens_mask"
"""A 360 recording's lens masks: written by posekit's lens mask step, read by SLAM and the face masks."""
DATASET_RETRY_DELAY_S: float = 1.0
"""Wait before the one retry of ``ensure_dataset`` (another worker may be creating the same dataset)."""


@contextmanager
def atomic_write(target: Path, *, exclusive: bool = False) -> Iterator[Path]:
    """Yield a temp path beside ``target`` that replaces it only on clean exit.

    Args:
        target: Final location; its parent directory is created if missing.
        exclusive: Never overwrite: ``target`` must not exist, at entry or when the temp file is published.

    Yields:
        A temp path in ``target.parent`` for the caller to write into.

    Raises:
        FileExistsError: If ``exclusive`` and ``target`` exists.
    """
    if exclusive and (target.exists() or target.is_symlink()):
        raise FileExistsError(f"refusing to write {target}: it exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(dir=target.parent, suffix=f"{target.suffix}.tmp")
    os.close(descriptor)
    temp_path: Path = Path(temp_name)
    try:
        yield temp_path
        temp_path.chmod(0o644)  # mkstemp temps are 0600; NFS mounts and the catalog server need world-readable files
        if exclusive:
            os.link(temp_path, target)  # unlike os.replace, refuses a target that appeared since entry
        else:
            os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)


@contextmanager
def atomic_recording(
    target: Path,
    *,
    application_id: str,
    recording_id: str,
    default_blueprint: rrb.Blueprint | None = None,
    send_properties: bool = True,
    optimize: bool = False,
) -> Iterator[rr.RecordingStream]:
    """Yield a recording saved to a temp file that replaces ``target`` on success.

    The file has a footer, so the catalog serves each chunk whole by byte span.

    Args:
        target: Final location, replaced atomically on a clean exit; on an exception it is left as it was.
        application_id: Rerun application id; one per package.
        recording_id: Shared across a segment's layers: it is what stacks them.
        default_blueprint: Layout embedded in the file, if any.
        send_properties: Whether Rerun adds its own ``RecordingInfo`` (the name and the wall-clock start time). A
            **base** layer wants it. A **derived** layer passes ``False``: it is the same recording as the base it
            stacks onto, and its ``RecordingInfo`` would replace the base's when it registers (module docstring).
            Explicit ``send_property`` calls land either way.
        optimize: Compact column batches for object-store publication. Only use on layers whose full payload fits
            in memory; mesh layers keep their lazy-loading boundaries.
    """
    # The recording stream is the inner context on purpose: it flushes and closes before atomic_write reaches its os.replace.
    with atomic_write(target) as temp_path:
        with rr.RecordingStream(application_id=application_id, recording_id=recording_id, send_properties=send_properties) as recording:
            recording.save(temp_path, default_blueprint=default_blueprint, write_footer=True)
            yield recording
        if optimize:
            # Work on the unpublished artifact; a failed optimization must leave the old target intact.
            with atomic_write(temp_path) as optimized:
                rrc.RrdReader(temp_path).stream().collect(optimize=rrc.OptimizationProfile.OBJECT_STORE).write_rrd(
                    optimized, application_id=application_id, recording_id=recording_id
                )


def ensure_dataset(client: CatalogClient, name: str) -> DatasetEntry:
    """The dataset ``name`` on ``client``'s catalog, created when absent.

    Retries once after ``DATASET_RETRY_DELAY_S``: the first failure may be another worker creating the same dataset.
    """
    try:
        return client.create_dataset(name, exist_ok=True)
    except (ConnectionError, RuntimeError, ValueError):
        time.sleep(DATASET_RETRY_DELAY_S)
        return client.create_dataset(name, exist_ok=True)


def segment_properties(entry: DatasetEntry, segment: str, group: str) -> dict[str, list[object]]:
    """One segment's ``property:<group>:*`` cells from the segment table, keyed by field name; empty cells are left out.

    Each cell stays the list the catalog holds (one value per instance), so list-valued properties keep every value.

    Raises:
        ValueError: If ``entry`` has no segment ``segment``.
    """
    table: pa.Table = entry.segment_table().filter(col("rerun_segment_id") == lit(segment)).to_arrow_table()
    if table.num_rows == 0:
        raise ValueError(f"no segment {segment!r} in dataset {entry.name!r}")
    prefix: str = f"property:{group}:"
    row: dict[str, object] = table.slice(0, 1).to_pylist()[0]
    return {name.removeprefix(prefix): cell for name, cell in row.items() if name.startswith(prefix) and isinstance(cell, list) and cell}


def registered_layer_files(entry: DatasetEntry, segment: str) -> dict[str, Path]:
    """Each layer of ``segment`` that the catalog holds as a local file, by layer name (the base is ``base``).

    For a stage that reads layer files: the catalog server reads them at the same paths, so they are this host's too.
    """
    rows: list[dict[str, str]] = (
        entry.manifest().filter(col("rerun_segment_id") == lit(segment)).select("rerun_layer_name", "rerun_storage_url").to_arrow_table().to_pylist()
    )
    return {row["rerun_layer_name"]: Path(unquote(urlparse(row["rerun_storage_url"]).path)) for row in rows if row["rerun_storage_url"].startswith("file://")}


def register_layer(entry: DatasetEntry, paths: Sequence[Path], layer: str | None, *, replace: bool = True) -> None:
    """Register recording files as ``layer`` of their segments and wait until the catalog has them.

    Args:
        entry: The dataset to register on.
        paths: Recording files, one per segment (their recording ids name the segments).
        layer: Layer name; None registers the catalog's default layer (``base``), as a whole segment.
        replace: Replace a segment's existing copy of the layer; otherwise a duplicate is an error.
    """
    if not paths:
        return
    on_duplicate: OnDuplicateSegmentLayer = OnDuplicateSegmentLayer.REPLACE if replace else OnDuplicateSegmentLayer.ERROR
    uris: list[str] = [path.resolve().as_uri() for path in paths]
    if layer is None:
        entry.register(uris, on_duplicate=on_duplicate).wait()
    else:
        entry.register(uris, layer_name=layer, on_duplicate=on_duplicate).wait()


def register_named_blueprint(entry: DatasetEntry, uri: str, name: str, *, table: bool = False, set_default: bool = True) -> RegisteredBlueprint:
    """Register a blueprint file as the blueprint ``name`` of ``entry``, retiring the one that held the name.

    Names are unique per dataset, so the holder gives its name up first, and it is unregistered only once the new
    blueprint is registered: a failed registration gives the holder its name back and leaves it registered.
    Registering the holder's own file again replaces it in place under the same id, so nothing is unregistered then.

    Args:
        entry: The dataset to register on.
        uri: The blueprint file, as a URI the server can read.
        name: The blueprint's name.
        table: Register a segment-table blueprint instead of a recording blueprint.
        set_default: Also make it the default blueprint of its kind.

    Returns:
        The registered blueprint.

    Raises:
        LookupError: When a blueprint holds ``name`` but the dataset has no blueprint dataset to retire it from.
    """
    holder: RegisteredBlueprint | None = next((blueprint for blueprint in entry.blueprints() if blueprint.name == name), None)
    blueprints: DatasetEntry | None = None if holder is None else entry.blueprint_dataset()
    if holder is not None:
        if blueprints is None:
            raise LookupError(f"blueprint {holder.id} holds {name!r} on dataset {entry.name!r}, which has no blueprint dataset to retire it from")
        holder.rename(f"{name} (retiring {holder.id})")
    try:
        if table:
            registered: RegisteredBlueprint = entry.register_table_blueprint(uri, name, set_default=set_default)
        else:
            registered = entry.register_blueprint(uri, name, set_default=set_default)
    except BaseException:
        if holder is not None:
            holder.rename(name)
        raise
    if holder is not None and blueprints is not None and holder.id != registered.id:
        blueprints.unregister(segments_to_drop=[holder.id], layers_to_drop=[]).wait()
    return registered


@dataclass(frozen=True, slots=True)
class LocalSegment:
    """One segment served from local files by ``serve_local_segment``."""

    url: str
    """The private catalog's URL (an OS-assigned port, never a shared catalog's)."""
    dataset: DatasetEntry
    """The dataset that holds the segment."""
    segment: str
    """The segment id: the base recording's recording id."""


@contextmanager
def serve_local_segment(base: Path, layers: Mapping[str, Path], *, dataset: str) -> Iterator[LocalSegment]:
    """Serve a base recording and named layers of its segment on a private in-process catalog, while the context is open.

    Args:
        base: The base recording; its recording id names the segment.
        layers: Layer name -> recording file of the same segment (``register_layer``).
        dataset: Name of the one dataset on the private catalog.

    Yields:
        The served segment.

    Raises:
        ValueError: When the base holds more than one recording, or a layer's recording id is not the base's.
    """
    with rr.server.Server(host="127.0.0.1", port=0, datasets={dataset: [str(base)]}) as server:
        entry: DatasetEntry = server.client().get_dataset(dataset)
        for name, path in layers.items():
            register_layer(entry, [path], name)
        segments: list[str] = sorted(entry.segment_ids())
        if len(segments) != 1:
            raise ValueError(f"{base} and layers {sorted(layers)} hold segments {segments}; every layer must carry the base's recording id")
        yield LocalSegment(url=server.url(), dataset=entry, segment=segments[0])
