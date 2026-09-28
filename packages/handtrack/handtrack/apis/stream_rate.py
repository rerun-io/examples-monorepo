"""Measure the catalog stream's sustained rate: batches drawn as fast as they come, after a warm-up, over minutes.

Reports images (DetNet) and crops (KeyNet) delivered per second, the main thread's wait fraction, NVDEC and GPU
utilisation and GPU memory (NVML), and the producer-side time split. In joint mode the loop draws one DetNet batch, then
KeyNet batches while ``keynet_ready()`` (the schedule a joint trainer uses). Run it in the prod env for real numbers.
"""

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pynvml
import torch

from handtrack.data.batches import DetNetBatch, KeyNetBatch
from handtrack.data.stream import CatalogStream, StreamConfig, StreamStats


@dataclass(frozen=True, slots=True)
class Config:
    """Stream-rate measurement."""

    stream: StreamConfig = field(default_factory=StreamConfig)
    warmup_s: float = 30.0
    """Seconds of drawing before the measurement starts (pools fill, decoders warm up)."""
    seconds: float = 120.0
    """Length of the measured window."""
    label: str = ""
    """Free text stored with the result."""
    output: Path | None = None
    """Append the result as one JSON line here."""


@dataclass(frozen=True, slots=True)
class RateResult:
    """One measured window."""

    label: str
    datasets: list[str]
    nets: str
    producers: int
    seconds: float
    detnet_images_per_s: float
    keynet_crops_per_s: float
    decoded_images_per_s: float
    """Pool images decoded per second (after the validity rules)."""
    main_wait_fraction: float
    nvdec_percent: float
    gpu_percent: float
    gpu_memory_used_mib: int
    """Peak whole-GPU memory used (NVML), every process included."""
    process_memory_mib: int
    """Peak GPU memory of this process (NVML): what the process costs the GPU, CUDA context included."""
    torch_peak_allocated_mib: int
    torch_peak_reserved_mib: int
    """Peak memory held by torch's caching allocator (allocated plus cached blocks)."""
    epochs_turned: int
    segments: int
    query_s_per_segment: float
    label_s_per_segment: float
    decode_s_per_segment: float
    keynet_kind_share: list[float]
    overwritten: list[int]


class _Sampler:
    """NVML samples of GPU, NVDEC and memory every half second on a background thread."""

    def __init__(self, device_index: int) -> None:
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.samples: list[tuple[float, float, float, float]] = []
        self._stop: threading.Event = threading.Event()
        self._thread: threading.Thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(0.5):
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            decoder: float = float(pynvml.nvmlDeviceGetDecoderUtilization(self._handle)[0])
            used: float = float(pynvml.nvmlDeviceGetMemoryInfo(self._handle).used) / 2**20
            mine: float = sum(float(p.usedGpuMemory or 0) for p in pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle) if p.pid == os.getpid()) / 2**20
            self.samples.append((float(utilization.gpu), decoder, used, mine))

    def stop(self) -> tuple[float, float, float, float]:
        """(mean GPU %, mean NVDEC %, peak MiB of the GPU, peak MiB of this process)."""
        self._stop.set()
        self._thread.join()
        if not self.samples:
            return 0.0, 0.0, 0.0, 0.0
        return (
            sum(s[0] for s in self.samples) / len(self.samples),
            sum(s[1] for s in self.samples) / len(self.samples),
            max(s[2] for s in self.samples),
            max(s[3] for s in self.samples),
        )


def main(config: Config) -> None:
    stream_config: StreamConfig = config.stream
    detnet_on: bool = stream_config.nets in ("detnet", "both")
    keynet_on: bool = stream_config.nets in ("keynet", "both")
    with CatalogStream(stream_config) as stream:
        epoch: int = 0
        stream.start_epoch(epoch)
        started: float = time.perf_counter()
        measuring: bool = False
        window_start: float = started
        images: int = 0
        crops: int = 0
        epochs_turned: int = 0
        before: StreamStats = StreamStats()
        sampler: _Sampler | None = None
        while True:
            now: float = time.perf_counter()
            if not measuring and now - started >= config.warmup_s:
                measuring = True
                window_start = now
                images, crops, epochs_turned = 0, 0, 0
                before = StreamStats(**asdict(stream.stats))
                sampler = _Sampler(stream.device.index or 0)
            if measuring and now - window_start >= config.seconds:
                break
            ended: bool = False
            if detnet_on:
                detnet: DetNetBatch | None = stream.next_detnet_batch()
                if detnet is None:
                    ended = True
                else:
                    images += detnet.pooled.shape[0]
            if keynet_on and not ended:
                # Joint mode: KeyNet batches while ready; KeyNet alone: one blocking draw per turn.
                while not detnet_on or stream.keynet_ready():
                    keynet: KeyNetBatch | None = stream.next_keynet_batch()
                    if keynet is None:
                        ended = not detnet_on
                        break
                    crops += keynet.crops.shape[0]
                    if not detnet_on:
                        break
            torch.cuda.synchronize(stream.device)
            if ended:
                epoch += 1
                epochs_turned += 1
                stream.start_epoch(epoch)
        elapsed: float = time.perf_counter() - window_start
        gpu: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0) if sampler is None else sampler.stop()
        after: StreamStats = stream.stats
        segments: int = max(1, after.segments - before.segments)
        kinds: list[int] = [a - b for a, b in zip(after.keynet_kinds, before.keynet_kinds, strict=True)]
        result: RateResult = RateResult(
            label=config.label,
            datasets=list(stream_config.datasets),
            nets=stream_config.nets,
            producers=stream_config.producers,
            seconds=round(elapsed, 1),
            detnet_images_per_s=round(images / elapsed, 1),
            keynet_crops_per_s=round(crops / elapsed, 1),
            decoded_images_per_s=round((after.images_decoded - before.images_decoded) / elapsed, 1),
            main_wait_fraction=round((after.wait_s - before.wait_s) / elapsed, 3),
            nvdec_percent=round(gpu[1], 1),
            gpu_percent=round(gpu[0], 1),
            gpu_memory_used_mib=int(gpu[2]),
            process_memory_mib=int(gpu[3]),
            torch_peak_allocated_mib=int(torch.cuda.max_memory_allocated(stream.device) / 2**20),
            torch_peak_reserved_mib=int(torch.cuda.max_memory_reserved(stream.device) / 2**20),
            epochs_turned=epochs_turned,
            segments=after.segments - before.segments,
            query_s_per_segment=round((after.query_s - before.query_s) / segments, 3),
            label_s_per_segment=round((after.label_s - before.label_s) / segments, 3),
            decode_s_per_segment=round((after.decode_s - before.decode_s) / segments, 3),
            keynet_kind_share=[round(k / max(1, sum(kinds)), 3) for k in kinds],
            overwritten=list(stream.overwritten()),
        )
    line: str = json.dumps(asdict(result))
    print(line, flush=True)
    if config.output is not None:
        with config.output.open("a") as handle:
            handle.write(line + "\n")
