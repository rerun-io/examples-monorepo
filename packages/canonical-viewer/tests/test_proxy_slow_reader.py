"""A stalled reader of the live proxy does not block the logger (patches/rerun-0.44.0-proxy-slow-reader.patch).

A proxy with a 200 MB budget, one reader frozen with SIGSTOP once it has saved a first message, and a logger that sends
about 600 MB: with the patch the logger flushes in seconds (1.4 s on the 5090 box); stock Rerun blocks it on the frozen
reader (27-28 s). A reader that connects afterwards still receives the proxy's history.
"""

import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from canonical_viewer.pixels import free_port

pytestmark = pytest.mark.integration

PRODUCER: str = """
import sys, time
import numpy as np
import rerun as rr
url, count = sys.argv[1], int(sys.argv[2])
rec = rr.RecordingStream("proxy-slow-reader-check")
rec.connect_grpc(url)
blob = np.random.default_rng(0).integers(0, 255, size=(1000, 1000), dtype=np.uint8)
start = time.monotonic()
for i in range(count):
    rec.set_time("step", sequence=i)
    rec.log("blob", rr.Tensor(blob))
rec.flush()
print(time.monotonic() - start)
rec.disconnect()
"""


def wait_for(condition: Callable[[], bool], what: str, logs: list[Path], timeout_s: float = 30.0) -> None:
    """Poll until condition() holds; fail with the children's logs after timeout_s."""
    deadline: float = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            pytest.fail(f"{what} within {timeout_s:.0f} s\n" + "\n".join(f"--- {log.name}\n{log.read_text()}" for log in logs))
        time.sleep(0.2)


def accepts(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) == 0


def size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def test_stalled_reader_does_not_block_the_logger(canonical_viewer: Path, tmp_path: Path) -> None:
    proxy_port: int = free_port()
    url: str = f"rerun+http://127.0.0.1:{proxy_port}/proxy"
    logs: list[Path] = [tmp_path / name for name in ("proxy.log", "stalled.log", "late.log")]
    processes: list[subprocess.Popen[bytes]] = []

    def start(log: Path, *args: str) -> subprocess.Popen[bytes]:
        with log.open("w") as stream:
            process = subprocess.Popen([canonical_viewer, *args], stdout=stream, stderr=subprocess.STDOUT)
        processes.append(process)
        return process

    def produce(count: int) -> float:
        return float(subprocess.run([sys.executable, "-c", PRODUCER, url, str(count)], check=True, capture_output=True, text=True, timeout=120).stdout)

    try:
        start(logs[0], "--serve-grpc", "--port", str(proxy_port), "--server-memory-limit", "200MB")
        wait_for(lambda: accepts(proxy_port), "the proxy did not listen", logs)
        stalled_rrd: Path = tmp_path / "stalled.rrd"
        reader = start(logs[1], url, "--save", str(stalled_rrd), "--port", str(free_port()))
        # The reader is subscribed once it has saved the first 1 MB message; only then is freezing it a stall.
        produce(1)
        wait_for(lambda: size(stalled_rrd) > 1_000_000, "the reader did not save the first message", logs)
        os.kill(reader.pid, signal.SIGSTOP)
        flushed: float = produce(600)
        assert flushed < 10.0, f"the logger took {flushed:.1f} s with a stalled reader (stock Rerun: about 27 s)"
        # The proxy keeps the last 200 MB (its budget) for readers that connect later.
        late_rrd: Path = tmp_path / "late.rrd"
        late = start(logs[2], url, "--save", str(late_rrd), "--port", str(free_port()))
        wait_for(lambda: late.poll() is not None or size(late_rrd) > 100_000_000, "a reader that connects later did not receive the history", logs)
        assert size(late_rrd) > 100_000_000, "the late reader exited early\n" + logs[2].read_text()
    finally:
        for process in processes:
            if process.poll() is None:
                os.kill(process.pid, signal.SIGCONT)
                process.kill()
            process.wait()
