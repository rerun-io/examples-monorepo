"""Multiprocess NVDEC decode: one persistent worker process per camera.

torchcodec 0.10 (conda-forge ceiling) caps in-process multi-stream CUDA decode
at ~140 cam-fps: its decoder cache holds a single NVDEC instance per
(codec, resolution), so identical streams thrash with ~60 ms re-inits (fixed
upstream in 0.11, PR #1232 — unavailable here). Separate processes sidestep
the shared cache entirely: ~400 cam-fps measured for 4x 4K HEVC.

Workers are spawned once at construction (torch import + NVDEC open happen
outside any timed region) and serve decode jobs over command queues. Frames
travel as pinned CPU uint8 (CUDA-IPC across spawn deadlocked; two PCIe copies
of resized frames are ~0.2 ms/frame — noise next to NVDEC).

NOTE: ``spawn`` re-imports ``__main__`` — drive this from a real script with
an ``if __name__ == "__main__"`` guard, never from stdin/REPL.
"""

from __future__ import annotations

import queue
from pathlib import Path

import torch
import torch.multiprocessing as mp
from jaxtyping import UInt8


def _decode_worker(video_path: str, resize_hw: tuple[int, int], cmd_queue, out_queue) -> None:
    from simplecv.video_io import TorchCodecVideoReader

    try:
        reader = TorchCodecVideoReader(Path(video_path), device="cuda", resize_hw=resize_hw)
    except Exception as exc:
        # Forward init failures (corrupt video, NVDEC/CUDA unavailable, OOM) to the
        # parent with the full traceback so construction fails fast instead of the
        # parent blocking forever on a "ready" that will never arrive.
        import traceback

        out_queue.put(("error", repr(exc), traceback.format_exc()))
        return
    out_queue.put(("ready", None, None))
    while True:
        job = cmd_queue.get()
        if job is None:
            return
        start_frame, frame_end, chunk_size = job
        for chunk_start in range(start_frame, frame_end, chunk_size):
            chunk_stop: int = min(chunk_start + chunk_size, frame_end)
            chunk: UInt8[torch.Tensor, "b 3 h w"] = reader.get_frames_in_range(chunk_start, chunk_stop)
            out_queue.put(("chunk", chunk_start, chunk.to("cpu").pin_memory()))
        out_queue.put(("done", None, None))


def _recv_or_die(out_queue, proc, timeout: float = 30.0):
    """Block for one message from a decode worker; raise if it dies first.

    The timeout is only a liveness-poll interval: a slow-but-alive worker keeps
    waiting, while a worker that exited fails fast instead of hanging forever on
    a message that will never arrive.
    """
    while True:
        try:
            return out_queue.get(timeout=timeout)
        except queue.Empty:
            if not proc.is_alive():
                raise RuntimeError("decode worker died (see worker stderr)") from None


class MultiprocessDecoder:
    """Persistent per-camera decode workers; chunk iterator per job."""

    def __init__(
        self,
        video_paths: list[Path],
        resize_hw: tuple[int, int],
        chunk_size: int = 24,
        queue_depth: int = 3,
    ) -> None:
        self.video_paths: list[Path] = video_paths
        self.resize_hw: tuple[int, int] = resize_hw
        self.chunk_size: int = chunk_size
        ctx = mp.get_context("spawn")
        self._cmd_queues = [ctx.Queue() for _ in video_paths]
        self._out_queues = [ctx.Queue(maxsize=queue_depth) for _ in video_paths]
        self._procs = [
            ctx.Process(
                target=_decode_worker,
                args=(str(path), resize_hw, cmd_q, out_q),
                daemon=True,
            )
            for path, cmd_q, out_q in zip(video_paths, self._cmd_queues, self._out_queues, strict=True)
        ]
        for proc in self._procs:
            proc.start()

        # Wait for NVDEC open (construction time). A worker that fails init (or
        # dies) must fail the build, not hang it — clean up the others first.
        for proc, out_q in zip(self._procs, self._out_queues, strict=True):
            try:
                kind, payload, tb = _recv_or_die(out_q, proc)
            except RuntimeError:
                self.close()
                raise
            if kind == "error":
                self.close()
                raise RuntimeError(f"decode worker failed to initialize ({payload}):\n{tb}")
            assert kind == "ready", f"unexpected startup message from decode worker: {kind!r}"

    def iter_chunks(self, start_frame: int, frame_end: int):
        """Yield ``(chunk_start, [per-camera UInt8 CUDA tensors])`` in order.

        ``frame_end`` is an exclusive end index (decode covers
        ``range(start_frame, frame_end)``), not a frame count.
        """
        for cmd_q in self._cmd_queues:
            cmd_q.put((start_frame, frame_end, self.chunk_size))

        while True:
            items = [_recv_or_die(out_q, proc) for proc, out_q in zip(self._procs, self._out_queues, strict=True)]
            kinds = {item[0] for item in items}
            if kinds == {"done"}:
                return
            assert kinds == {"chunk"}, f"decode workers out of sync: {kinds}"
            chunk_start: int = items[0][1]
            assert all(item[1] == chunk_start for item in items)
            yield chunk_start, [item[2].to("cuda", non_blocking=True) for item in items]

    def close(self) -> None:
        for cmd_q in self._cmd_queues:
            cmd_q.put(None)
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        self._procs = []
