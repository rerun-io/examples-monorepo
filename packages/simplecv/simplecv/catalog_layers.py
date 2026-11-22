"""Write a catalog layer atomically and register it: one writer and one registrar for every package.

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
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb
import rerun.chunk as rrc
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer

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
