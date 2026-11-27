"""A stand-in for simplecv's ``ChunkedVideoDecoder`` in tests that read sample times only."""

import numpy as np
from simplecv.catalog_video_reader import ChunkedVideoDecoder, StreamIndex


class TimesOnly(ChunkedVideoDecoder):
    """A video's sample times (int64 ns) without a catalog; nothing decodes."""

    def __init__(self, t_ns: np.ndarray) -> None:
        times = np.asarray(t_ns, dtype=np.int64).astype("timedelta64[ns]")
        self._index = StreamIndex("/stub/video", times, np.ones(len(times), dtype=bool), "av1")

    @property
    def index(self) -> StreamIndex:
        return self._index
