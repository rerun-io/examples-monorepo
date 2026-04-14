"""DBoW2-based image retrieval for classical loop closure.

Runs the DBoW2 vocabulary tree in a separate process for async operation.
Requires the ``dpretrieval`` workspace package.
"""

import os
import time
from multiprocessing import Process, Queue, Value

import numpy as np
from einops import parse_shape

try:
    import dpretrieval

    _cls = dpretrieval.DPRetrieval  # Verify the class exists
    del _cls
except Exception as _err:
    raise ModuleNotFoundError("Couldn't load dpretrieval. It may not be installed.") from _err


NMS = 50
RAD = 50


def _dbow_loop(in_queue: Queue, out_queue: Queue, vocab_path: str, ready: Value) -> None:
    """Run DBoW retrieval in a background process."""
    dbow = dpretrieval.DPRetrieval(vocab_path, 50)
    ready.value = 1
    while True:
        n, image = in_queue.get()
        dbow.insert_image(image)
        q = dbow.query(n)
        out_queue.put((n, q))


class RetrievalDBOW:
    """DBoW2-based image retrieval for loop closure detection."""

    def __init__(self, vocab_path: str = "ORBvoc.txt") -> None:
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(
                "Missing the ORB vocabulary. Please download and un-tar it from "
                "https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/Vocabulary/ORBvoc.txt.tar.gz "
                "and place it in the working directory."
            )

        self.image_buffer: dict[int, np.ndarray] = {}
        self.stored_indices: np.ndarray = np.zeros(100000, dtype=bool)

        self.prev_loop_closes: list[tuple[int, int]] = []
        self.found: list[tuple[int, int]] = []

        self.in_queue: Queue = Queue(maxsize=20)
        self.out_queue: Queue = Queue(maxsize=20)
        ready = Value("i", 0)
        self.proc: Process = Process(target=_dbow_loop, args=(self.in_queue, self.out_queue, vocab_path, ready))
        self.proc.start()
        self.being_processed: int = 0
        while not ready.value:
            time.sleep(0.01)

    def keyframe(self, k: int) -> None:
        tmp = dict(self.image_buffer)
        self.image_buffer.clear()
        for n, v in tmp.items():
            if n != k:
                key = (n - 1) if (n > k) else n
                self.image_buffer[key] = v

    def save_up_to(self, c: int) -> None:
        for n in list(self.image_buffer):
            if n <= c:
                assert not self.stored_indices[n]
                img = self.image_buffer.pop(n)
                self.in_queue.put((n, img))
                self.stored_indices[n] = True
                self.being_processed += 1

    def confirm_loop(self, i: int, j: int) -> None:
        assert i > j
        self.prev_loop_closes.append((i, j))

    def _repetition_check(self, idx: int, num_repeat: int) -> tuple[int, int] | None:
        if len(self.found) < num_repeat:
            return None
        latest = self.found[-num_repeat:]
        (b, _), (i, j), _ = latest
        if (1 + idx - b) == num_repeat:
            return (i, max(j, 1))
        return None

    def detect_loop(self, thresh: float, num_repeat: int = 1) -> tuple[int, int] | None:
        while self.being_processed > 0:
            x = self._detect_loop(thresh, num_repeat)
            if x is not None:
                return x
        return None

    def _detect_loop(self, thresh: float, num_repeat: int = 1) -> tuple[int, int] | None:
        assert self.being_processed > 0
        i, (score, j, _) = self.out_queue.get()
        self.being_processed -= 1
        if score < thresh:
            return None
        assert i > j

        dists_sq = [(np.square(i - a) + np.square(j - b)) for a, b in self.prev_loop_closes]
        if min(dists_sq, default=np.inf) < np.square(NMS):
            return None

        self.found.append((i, j))
        return self._repetition_check(i, num_repeat)

    def __call__(self, image: np.ndarray, n: int) -> None:
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        assert parse_shape(image, "_ _ RGB") == dict(RGB=3)
        self.image_buffer[n] = image

    def close(self) -> None:
        self.proc.terminate()
        self.proc.join()
