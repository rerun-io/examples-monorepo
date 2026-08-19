"""Atomic publication of layer rrds: tmp file → ``os.replace``.

Half-written files poison batch runs and the catalog server, so a target rrd
either exists complete or not at all ("exists = done"). Never save over an
rrd that a catalog server has registered — truncation poisons the server;
re-runs write a fresh tmp and atomically replace, and the catalog re-registers.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb


def should_skip(target: Path, *, force: bool) -> bool:
    """Idempotency check: an existing target is done unless ``--force``."""
    return target.exists() and not force


@contextmanager
def atomic_recording(
    target: Path,
    *,
    application_id: str,
    recording_id: str,
    default_blueprint: rrb.Blueprint | None = None,
) -> Iterator[rr.RecordingStream]:
    """Yield a recording saved to a temp file that replaces ``target`` on success."""
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(dir=target.parent, suffix=".rrd.tmp")
    os.close(descriptor)
    temp_path: Path = Path(temp_name)
    try:
        with rr.RecordingStream(application_id=application_id, recording_id=recording_id, send_properties=True) as recording:
            recording.save(temp_path, default_blueprint=default_blueprint)
            yield recording
        temp_path.chmod(0o644)  # mkstemp temps are 0600; NFS mounts and the catalog server need world-readable files
        os.replace(temp_path, target)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
