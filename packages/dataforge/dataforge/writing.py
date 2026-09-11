"""Atomic publication of layer artifacts: tmp file → ``os.replace``.

Half-written files poison batch runs and the catalog server, so a target either
exists complete or not at all ("exists = done"). Never save over an rrd (or an
``.rbl``) that a catalog server has registered — truncation poisons the server;
re-runs write a fresh tmp and atomically replace, and the catalog re-registers.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import rerun as rr
import rerun.blueprint as rrb

from dataforge import schema
from dataforge.identity import SequenceIdentity

APPLICATION_ID: str = "dataforge"
"""Rerun application id of every recording this package writes; one package, one app."""


def should_skip(target: Path, *, force: bool) -> bool:
    """Idempotency check: an existing target is done unless ``--force``."""
    return target.exists() and not force


@contextmanager
def atomic_write(target: Path) -> Iterator[Path]:
    """Yield a temp path beside ``target`` that replaces it only on clean exit.

    Args:
        target: Final location; its parent directory is created if missing.

    Yields:
        A temp path in ``target.parent`` for the caller to write into.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(dir=target.parent, suffix=f"{target.suffix}.tmp")
    os.close(descriptor)
    temp_path: Path = Path(temp_name)
    try:
        yield temp_path
        temp_path.chmod(0o644)  # mkstemp temps are 0600; NFS mounts and the catalog server need world-readable files
        os.replace(temp_path, target)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


@contextmanager
def recording_to(
    path: Path,
    *,
    application_id: str = APPLICATION_ID,
    recording_id: str,
    default_blueprint: rrb.Blueprint | None = None,
    send_properties: bool = True,
) -> Iterator[rr.RecordingStream]:
    """Yield a recording saved to exactly ``path``; the caller owns publication.

    The lower half of ``atomic_recording``, for a converter writing **several**
    layers that must be published together: it stages each one under its own
    ``atomic_write`` and writes into the temp paths, which ``atomic_recording``'s
    single-target contract cannot express. The recording is flushed and closed on
    exit, so ``path`` is complete and readable by the time the caller returns —
    which is what lets a derived layer read the staged base rrd it was just
    handed.

    Args:
        path: Exact file to save into; nothing is replaced or cleaned up here.
        application_id: Rerun application id; one per package.
        recording_id: Shared across a sequence's layers — it is what stacks them.
        default_blueprint: Layout embedded in the file, if any.
        send_properties: Whether Rerun adds its own ``RecordingInfo`` (the name
            and the wall-clock start time). See ``atomic_recording``.
    """
    with rr.RecordingStream(application_id=application_id, recording_id=recording_id, send_properties=send_properties) as recording:
        recording.save(path, default_blueprint=default_blueprint)
        yield recording


@contextmanager
def atomic_recording(
    target: Path,
    *,
    application_id: str = APPLICATION_ID,
    recording_id: str,
    default_blueprint: rrb.Blueprint | None = None,
    send_properties: bool = True,
) -> Iterator[rr.RecordingStream]:
    """Yield a recording saved to a temp file that replaces ``target`` on success.

    Args:
        target: Final location, replaced atomically on a clean exit.
        application_id: Rerun application id; one per package.
        recording_id: Shared across a sequence's layers — it is what stacks them.
        default_blueprint: Layout embedded in the file, if any.
        send_properties: Whether Rerun adds its own ``RecordingInfo``. A **base**
            layer wants it: its wall-clock ``start_time`` is when the capture was
            converted, and the viewer shows it. A **derived** layer passes
            ``False`` — it is the same recording as the base it stacks onto, so a
            second name and a second start time are duplicate, and its own
            ``start_time`` would be whenever it was last rebuilt. Explicit
            ``send_property``/``send_recording_name`` calls still land either way
            (verified on rerun 0.37.0), so a layer's own property group is
            unaffected.
    """
    # The recording stream is the inner context on purpose: it flushes and closes
    # before atomic_write reaches its os.replace.
    with (
        atomic_write(target) as temp_path,
        recording_to(
            temp_path,
            application_id=application_id,
            recording_id=recording_id,
            default_blueprint=default_blueprint,
            send_properties=send_properties,
        ) as recording,
    ):
        yield recording


def send_capture_properties(
    recording: rr.RecordingStream,
    identity: SequenceIdentity,
    *,
    num_cameras: int,
    num_frames: int,
    **extra: Any,
) -> None:
    """Name the recording and stamp it with the dataforge capture/convert properties.

    Args:
        recording: Destination recording stream.
        identity: Sequence identity; its ``recording_id`` becomes the recording name.
        num_cameras: Cameras actually logged in this recording.
        num_frames: Longest per-camera video sample count.
        **extra: Dataset-specific capture keys (``start_time_ns``, ``task``, …);
            ``None`` values are dropped rather than written as empty values.
    """
    recording.send_recording_name(identity.recording_id)
    present: dict[str, Any] = {key: value for key, value in extra.items() if value is not None}
    recording.send_property(
        "capture",
        rr.AnyValues(schema=schema.DATAFORGE_SCHEMA_VERSION, num_frames=num_frames, num_cameras=num_cameras, **present),
    )
    recording.send_property("convert", rr.AnyValues(version="1"))
