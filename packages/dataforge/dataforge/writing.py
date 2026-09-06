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
def atomic_recording(
    target: Path,
    *,
    application_id: str = APPLICATION_ID,
    recording_id: str,
    default_blueprint: rrb.Blueprint | None = None,
) -> Iterator[rr.RecordingStream]:
    """Yield a recording saved to a temp file that replaces ``target`` on success."""
    # The recording stream is the inner context on purpose: it flushes and closes
    # before atomic_write reaches its os.replace.
    with (
        atomic_write(target) as temp_path,
        rr.RecordingStream(application_id=application_id, recording_id=recording_id, send_properties=True) as recording,
    ):
        recording.save(temp_path, default_blueprint=default_blueprint)
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
