"""Peak GPU memory of a process, sampled with NVML on a background thread (run-layout.md protocol)."""

import threading
import time
from dataclasses import dataclass, field

import pynvml


@dataclass(slots=True)
class PeakVramSampler:
    """Sample the GPU memory NVML reports for `pid` every `interval_s` and keep the maximum.

    Use as a context manager around the process's lifetime; `peak_bytes` holds the maximum seen.
    """

    pid: int
    """Process to watch (the trainer itself, not a launcher)."""
    interval_s: float = 0.005
    """Sampling period in seconds."""
    peak_bytes: int = 0
    """Largest used GPU memory seen for `pid`, summed over devices."""
    samples: int = 0
    """Number of samples taken."""
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def _sample_once(self, handles: list) -> None:
        used: int = 0
        for handle in handles:
            for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if process.pid == self.pid and process.usedGpuMemory is not None:
                    used += int(process.usedGpuMemory)
        self.peak_bytes = max(self.peak_bytes, used)
        self.samples += 1

    def _run(self) -> None:
        pynvml.nvmlInit()
        try:
            handles: list = [pynvml.nvmlDeviceGetHandleByIndex(index) for index in range(pynvml.nvmlDeviceGetCount())]
            while not self._stop.is_set():
                self._sample_once(handles)
                time.sleep(self.interval_s)
            self._sample_once(handles)
        finally:
            pynvml.nvmlShutdown()

    def __enter__(self) -> "PeakVramSampler":
        self._thread = threading.Thread(target=self._run, name=f"nvml-peak-{self.pid}", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def peak_gib(self) -> float:
        return self.peak_bytes / 2**30


def gpu_name() -> str:
    pynvml.nvmlInit()
    try:
        name: str | bytes = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
        return name.decode() if isinstance(name, bytes) else name
    finally:
        pynvml.nvmlShutdown()
