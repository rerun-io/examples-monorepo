"""The queue residency set (lietorch_mojo.metal) must not keep PyTorch's released MPS heaps alive."""

import pytest
import torch

pytestmark = [pytest.mark.integration, pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU (MPS)")]


def test_covered_buffers_are_released_after_pytorch_frees_them() -> None:
    from lietorch_mojo import metal

    metal.install_queue_residency()
    torch.mps.empty_cache()
    baseline: int = torch.mps.driver_allocated_memory()
    for i in range(24):
        # 8 tensors of 4-52 MB with changing sizes, so PyTorch keeps creating and releasing heaps.
        tensors: list[torch.Tensor] = [torch.empty((1 + (i * 7 + k) % 13) * 2**20, device="mps") for k in range(8)]
        metal.gpu_addresses(tensors)
        del tensors
        torch.mps.synchronize()
        torch.mps.empty_cache()
    metal.gpu_addresses([torch.empty(1, device="mps")])  # the next launch's buffers replace the old ones in the set
    torch.mps.synchronize()
    torch.mps.empty_cache()
    # One launch's worth of heaps may stay; without release 24 rounds hold several GB.
    assert torch.mps.driver_allocated_memory() - baseline < 256 * 2**20
