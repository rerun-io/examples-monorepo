"""Peak GPU memory of a trainer: the one sampler for every trainer we run (docs/run-layout.md protocol).

`run_measured` starts the trainer as a direct child and samples it with NVML on a background thread every
5 ms: the sum over GPUs of the `usedGpuMemory` NVML reports for the trainer's pid (its own allocations: the
CUDA context and torch's caching allocator, or Brush's wgpu/Vulkan buffers; never other processes). Each
change of that value is kept with its wall-clock time, so a runner can cut the peak at the end of training:
Brush trains, evaluates, exports and then renders the test views in one process (cut at its
`training_end_unix_seconds`); the upstream Hexels trainer runs with --skip_postprocess and the 2D fits render
nothing afterwards, so their whole process counts. `gpu_lock` is the one flock every GPU job on a host
holds, so timings and peaks do not share the GPU.
"""

import fcntl
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pynvml


@dataclass(slots=True)
class PeakVramSampler:
    """Sample the GPU memory NVML reports for `pid` every `interval_s`; a context manager around the process's lifetime."""

    pid: int
    """Process to watch (the trainer itself, not a launcher)."""
    interval_s: float = 0.005
    """Sampling period in seconds."""
    changes: list[tuple[float, int]] = field(default_factory=list)
    """(Unix seconds, bytes summed over GPUs) each time the sampled value changed; only samples in which NVML
    reported a usage for the pid count."""
    samples: int = 0
    """Number of samples taken."""
    available: bool = True
    """False when NVML could not start (no NVIDIA driver): nothing was sampled."""
    error: str | None = None
    """The NVML error that stopped sampling during the run; the peak is then unknown."""
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def _sample_once(self, handles: list) -> None:
        used: int | None = None
        for handle in handles:
            for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if process.pid == self.pid and process.usedGpuMemory is not None:
                    used = (used or 0) + int(process.usedGpuMemory)
        if used is not None and (not self.changes or self.changes[-1][1] != used):
            self.changes.append((time.time(), used))
        self.samples += 1

    def _run(self) -> None:
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError:
            self.available = False
            return
        try:
            handles: list = [pynvml.nvmlDeviceGetHandleByIndex(index) for index in range(pynvml.nvmlDeviceGetCount())]
            while not self._stop.is_set():
                self._sample_once(handles)
                time.sleep(self.interval_s)
            self._sample_once(handles)
        except pynvml.NVMLError as error:
            self.error = f"{type(error).__name__}: {error}"
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

    def peak_gib(self, until_unix_s: float | None = None) -> float | None:
        """Largest sample taken at or before `until_unix_s` (all samples for None), GiB.

        None when the peak is unknown: NVML was unavailable or raised during the run, or it reported no usage
        for the pid by then.
        """
        if not self.available or self.error is not None:
            return None
        used: list[int] = [used for when, used in self.changes if until_unix_s is None or when <= until_unix_s]
        return max(used) / 2**30 if used else None


@dataclass(frozen=True, slots=True)
class Measured:
    """A trainer run as a direct child that exited with status 0."""

    seconds: float
    """Wall time from start to exit."""
    vram: PeakVramSampler
    """Its GPU memory samples."""


def run_measured(command: list[str], cwd: Path | None, log: Path) -> Measured:
    """Run `command` as a direct child under `PeakVramSampler`; the command line, then its stdout and stderr, are appended to `log`.

    Raises:
        RuntimeError: The command exited with a nonzero status.
    """
    started: float = time.perf_counter()
    with log.open("a") as stream:
        stream.write(" ".join(command) + "\n")
        stream.flush()
        process = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT)
        with PeakVramSampler(pid=process.pid) as sampler:
            returncode: int = process.wait()
    if returncode != 0:
        raise RuntimeError(f"{Path(command[0]).name} exited with {returncode}; see {log}")
    return Measured(seconds=time.perf_counter() - started, vram=sampler)


@contextmanager
def gpu_lock(path: Path | None) -> Iterator[None]:
    """Hold an exclusive `flock(path)` for the block, so GPU jobs sharing the file do not overlap; no lock for None."""
    if path is None:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def gpu_name() -> str | None:
    """Name of GPU 0, None without NVML."""
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        return None
    try:
        name: str | bytes = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
        return name.decode() if isinstance(name, bytes) else name
    finally:
        pynvml.nvmlShutdown()
