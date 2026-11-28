"""A stalled reader of the live proxy does not block the logger (patches/rerun-0.44.0-proxy-slow-reader.patch).

A proxy with a 200 MB budget, one reader frozen with SIGSTOP, and a logger that sends about 600 MB: with the patch the
logger flushes in seconds (1.4 s on the 5090 box); stock Rerun blocks it on the frozen reader (27.3 s). A reader that
connects afterwards still receives the proxy's history.
"""

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

PRODUCER: str = """
import sys, time
import numpy as np
import rerun as rr
rec = rr.RecordingStream("proxy-slow-reader-check")
rec.connect_grpc(sys.argv[1])
blob = np.random.default_rng(0).integers(0, 255, size=(1000, 1000), dtype=np.uint8)
start = time.monotonic()
for i in range(600):
    rec.set_time("step", sequence=i)
    rec.log("blob", rr.Tensor(blob))
rec.flush()
print(time.monotonic() - start)
rec.disconnect()
"""


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_stalled_reader_does_not_block_the_logger(canonical_viewer: Path, tmp_path: Path) -> None:
    proxy_port: int = free_port()
    url: str = f"rerun+http://127.0.0.1:{proxy_port}/proxy"
    proxy = subprocess.Popen([canonical_viewer, "--serve-grpc", "--port", str(proxy_port), "--server-memory-limit", "200MB"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
    reader: subprocess.Popen[bytes] | None = None
    late_reader: subprocess.Popen[bytes] | None = None
    try:
        time.sleep(2)
        reader = subprocess.Popen([canonical_viewer, url, "--save", str(tmp_path / "stalled.rrd"), "--port", str(free_port())],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
        time.sleep(2)
        os.kill(reader.pid, signal.SIGSTOP)
        flushed: str = subprocess.run([sys.executable, "-c", PRODUCER, url], check=True, capture_output=True, text=True, timeout=120).stdout
        assert float(flushed) < 10.0, f"the logger took {float(flushed):.1f} s with a stalled reader (stock Rerun: about 27 s)"
        # The proxy keeps the last 200 MB (its budget) for readers that connect later.
        late: Path = tmp_path / "late.rrd"
        late_reader = subprocess.Popen([canonical_viewer, url, "--save", str(late), "--port", str(free_port())],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
        deadline: float = time.monotonic() + 30.0
        while time.monotonic() < deadline and not (late.is_file() and late.stat().st_size > 100_000_000):
            time.sleep(0.5)
        assert late.is_file() and late.stat().st_size > 100_000_000, "a reader that connects later did not receive the recording"
    finally:
        for process in (late_reader, reader, proxy):
            if process is not None:
                os.kill(process.pid, signal.SIGCONT)
                process.kill()
                process.wait()
