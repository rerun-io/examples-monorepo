"""PriorLog's page-locked ring (the logger thread itself is covered by test_capture)."""

import pytest
import torch

from metrictrail.prior_log import RING_SLOTS, HostRing


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
