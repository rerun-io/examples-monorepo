"""PriorLog's page-locked ring and its shutdown (its logging is covered by test_capture)."""

import pytest
import torch

from metrictrail.prior_log import RING_SLOTS, HostRing, PriorFrame, PriorLog


@pytest.mark.integration
def test_the_ring_copies_into_page_locked_buffers_it_allocates_once() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device (page-locked memory)")
    ring = HostRing()
    copies: list[torch.Tensor] = [ring.copy(torch.full((2, 3), float(i), device="cuda")) for i in range(RING_SLOTS + 1)]
    torch.cuda.synchronize()
    assert len(ring.buffers) == RING_SLOTS and all(copy.is_pinned() for copy in copies)
    assert copies[RING_SLOTS].data_ptr() == copies[0].data_ptr()  # the first slot, reused
    assert [float(copy[0, 0]) for copy in copies[1:RING_SLOTS]] == [float(i) for i in range(1, RING_SLOTS)]


def test_close_stops_the_logger_thread_even_when_logging_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The failure reaches the caller, and the thread is stopped and joined all the same (no thread left waiting)."""
    import rerun as rr

    def failing_log(self: PriorLog, item: PriorFrame) -> None:
        raise RuntimeError("normals encoder failed")

    monkeypatch.setattr(PriorLog, "log", failing_log)
    priors = PriorLog(depth_range_m=(0.3, 3.0), video_time_ns=None, fps=30.0)
    priors.submit(rr.RecordingStream("t", recording_id="priors"), 0, None, torch.full((4, 4), 2.0), None)
    thread = priors.thread
    assert thread is not None
    with pytest.raises(RuntimeError, match="encoder failed"):
        priors.close()
    assert priors.thread is None and not thread.is_alive()
