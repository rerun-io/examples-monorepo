"""Bounded encoder ordering, overlap, timing and cleanup without GPU work."""

from functools import partial
from pathlib import Path
from threading import Event

import pytest

from dataforge import video_encoding
from dataforge.timing import SequenceTimer
from dataforge.video_encoding import parallel_clips


@pytest.mark.parametrize("fail", [False, True])
def test_parallel_clips_overlap_and_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool) -> None:
    logged = Event()
    clock = [10.0]
    monkeypatch.setattr(video_encoding, "perf_counter", lambda: clock[0])
    clips = [tmp_path / f"{i}.mp4" for i in range(3)]

    def encode(index: int) -> None:
        if index:
            assert logged.wait(5.0), "encoder waited for logging to finish"
        clips[index].write_bytes(b"clip")
        if index == 1 and fail:
            raise ValueError("encoder failed")

    timer = SequenceTimer()
    seen = []
    try:
        with parallel_clips([(clip, partial(encode, i)) for i, clip in enumerate(clips)], timer) as ready:
            for clip in ready:
                seen.append(clip.name)
                assert clip.read_bytes() == b"clip"
                clip.unlink()
                logged.set()
            clock[0] = 100.0  # caller work after encode completion is excluded
    except ValueError as error:
        assert fail and str(error) == "encoder failed"
    else:
        assert not fail
    assert seen == (["0.mp4"] if fail else ["0.mp4", "1.mp4", "2.mp4"])
    assert list(tmp_path.iterdir()) == []
    assert timer.stage_s == {"transcode": 0.0}
