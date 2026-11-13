"""Prepared graph topology keeps update outputs while avoiding host reads during neural inference."""

import pytest
from jaxtyping import Float32, Int64

torch = pytest.importorskip("torch")
pytest.importorskip("dpvo.mojo_backend._dpvo_mojo", reason="requires the DPVO Mojo extension")

from torch import Tensor  # noqa: E402

from dpvo.net import Update, UpdateTopology  # noqa: E402


def test_prepared_topology_matches_an_ordinary_update_without_scalar_reads() -> None:
    torch.manual_seed(17)
    model: Update = Update(p=3).eval()
    hidden: Float32[Tensor, "1 6 384"] = torch.randn(1, 6, 384)
    context: Float32[Tensor, "1 6 384"] = torch.randn(1, 6, 384)
    correlations: Float32[Tensor, "1 6 882"] = torch.randn(1, 6, 882)
    source: Int64[Tensor, "6"] = torch.tensor([2, 1, 2, 1, 1, 2])
    target: Int64[Tensor, "6"] = torch.tensor([1, 0, 0, 2, 0, 1])
    patch: Int64[Tensor, "6"] = torch.tensor([5, 2, 5, 3, 3, 4])
    with torch.no_grad():
        expected_hidden, (expected_delta, expected_weight, _) = model(hidden, context, correlations, None, source, target, patch)
        topology: UpdateTopology = UpdateTopology.from_edges(source, target, patch)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            actual_hidden, (actual_delta, actual_weight, _) = model(hidden, context, correlations, None, source, target, patch, topology=topology)
    torch.testing.assert_close(actual_hidden, expected_hidden, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_delta, expected_delta, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_weight, expected_weight, rtol=0.0, atol=0.0)
    operations: set[str] = {event.key for event in profile.key_averages()}
    assert "aten::_unique2" not in operations
    assert "aten::_local_scalar_dense" not in operations
