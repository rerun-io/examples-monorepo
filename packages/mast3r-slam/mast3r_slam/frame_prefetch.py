"""Read the next frames (decode, resize, build the ``Frame``) on a worker thread while the current one is tracked.

The worker runs its GPU work (NVDEC decode, resize, normalise) on its own CUDA stream and hands each frame over
with a CUDA event, so the tracker's stream waits for exactly that frame's work and nothing else. Its host work
(the copy of the cropped image for logging, the frame's shape tensors) leaves the tracking thread too.
"""

import contextlib
import queue
import threading
from types import TracebackType

import lietorch_mojo as lietorch
import torch

from mast3r_slam.dataloader import MonocularDataset
from mast3r_slam.frame import Frame, create_frame


class FramePrefetcher:
    """Frames ``0 .. n_frames - 1`` of a dataset, read ``depth`` ahead; use as a context manager.

    The frames come with an identity pose: the tracker sets the real one, which depends on the frame before.
    """

    def __init__(self, dataset: MonocularDataset, n_frames: int, device: str = "cuda:0", depth: int = 2) -> None:
        self._dataset: MonocularDataset = dataset
        self._n_frames: int = n_frames
        self._device: str = device
        self._ready: queue.Queue[tuple[Frame, torch.cuda.Event] | BaseException] = queue.Queue(maxsize=depth)
        self._stop: threading.Event = threading.Event()
        self._thread: threading.Thread = threading.Thread(target=self._run, name="frame-prefetch", daemon=True)

    def __enter__(self) -> "FramePrefetcher":
        self._thread.start()
        return self

    def __exit__(self, _exc_type: type[BaseException] | None, _exc_val: BaseException | None, _exc_tb: TracebackType | None) -> None:
        self._stop.set()
        # Unblock a worker waiting on a full queue.
        while self._thread.is_alive():
            with contextlib.suppress(queue.Empty):
                self._ready.get_nowait()
            self._thread.join(timeout=0.01)

    def _run(self) -> None:
        stream: torch.cuda.Stream = torch.cuda.Stream(device=self._device)
        try:
            with torch.cuda.stream(stream):
                for i in range(self._n_frames):
                    if self._stop.is_set():
                        return
                    frame: Frame = create_frame(i, self._dataset.read_resized(i), lietorch.Sim3.Identity(1, device=self._device), device=self._device)
                    ready: torch.cuda.Event = torch.cuda.Event()
                    ready.record(stream)
                    self._put((frame, ready))
        except BaseException as exc:  # re-raised on the tracking thread by get(), beartype violations included
            self._put(exc)

    def _put(self, item: tuple[Frame, torch.cuda.Event] | BaseException) -> None:
        while not self._stop.is_set():
            try:
                self._ready.put(item, timeout=0.05)
                return
            except queue.Full:
                continue

    def get(self) -> Frame:
        """The next frame, ready to use on the current CUDA stream.

        Raises:
            BaseException: Whatever reading the frame raised on the worker.
        """
        item: tuple[Frame, torch.cuda.Event] | BaseException = self._ready.get()
        if isinstance(item, BaseException):
            raise item
        frame: Frame = item[0]
        current: torch.cuda.Stream = torch.cuda.current_stream(self._device)
        current.wait_event(item[1])
        # The worker's stream allocated these; keep the allocator from reusing them while this stream still reads.
        for tensor in (frame.rgb_tensor, frame.img_shape, frame.img_true_shape):
            tensor.record_stream(current)
        return frame
