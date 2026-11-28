"""``exit_now``: the CLI ends without interpreter teardown, and its saved recording is still complete."""

import subprocess
import sys
from pathlib import Path

import numpy as np
from rerun.chunk import RrdReader

SCRIPT: str = """
import sys
import numpy as np
import rerun as rr
from fdanyone_rerun.apis.run import exit_now

recording = rr.RecordingStream("exit-now-test")
recording.save(sys.argv[1])
for frame in range(50):
    recording.set_time("frame", sequence=frame)
    recording.log("points", rr.Points3D(np.full((1000, 3), frame, np.float32)))
print("logged", flush=False)
exit_now(recording)
raise SystemExit(3)  # never reached
"""


def test_exit_now_closes_the_recording_and_exits_zero(tmp_path: Path) -> None:
    path: Path = tmp_path / "run.rrd"
    done = subprocess.run([sys.executable, "-c", SCRIPT, str(path)], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "logged"  # stdout flushed before the exit
    rows: int = 0
    last: float = -1.0
    for chunk in RrdReader(path).stream():
        batch = chunk.to_record_batch()
        if "Points3D:positions" in batch.schema.names:
            rows += batch.num_rows
            last = max(last, float(np.max(batch.column("Points3D:positions").to_pylist()[-1])))
    assert rows == 50
    assert last == 49.0
