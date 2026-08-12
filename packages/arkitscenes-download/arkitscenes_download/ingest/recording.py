"""Atomic publication of layer RRDs."""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb


@contextmanager
def atomic_recording(output_path: Path, recording_id: str, *, send_properties: bool, default_blueprint: rrb.Blueprint | None = None):
    """Yield a recording that is finalized before atomically replacing its target."""
    descriptor, temp_name = tempfile.mkstemp(dir=output_path.parent, suffix=".rrd.tmp")
    os.close(descriptor)
    temp_path: Path = Path(temp_name)
    try:
        with rr.RecordingStream(application_id="arkitscenes", recording_id=recording_id, send_properties=send_properties) as recording:
            recording.save(temp_path, default_blueprint=default_blueprint)
            yield recording
        temp_path.chmod(0o644)  # mkstemp temps are 0600; NFS mounts and the catalog server need world-readable files
        os.replace(temp_path, output_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
