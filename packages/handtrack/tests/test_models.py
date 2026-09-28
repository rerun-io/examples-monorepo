"""Table 4/5 architecture contracts and masked training losses."""

import math
from collections.abc import Iterator

import pytest

pytest.importorskip("torch", reason="model tests need the handtrack environment")

import torch
from jaxtyping import Bool, Float32
from torch import Tensor, nn
from torch.utils.flop_counter import FlopCounterMode

from handtrack.models.detnet import Detections, DetNetF, DetNetLoss, DetNetOutput, decode_detections, detnet_loss
from handtrack.models.keynet import KeyNetF, KeyNetLoss, KeyNetOutput, keynet_loss


@pytest.fixture(autouse=True)
def cpu_threads() -> Iterator[None]:
    """Keep small CPU convolutions from oversubscribing this host."""
    previous: int = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_detnet_table_and_pooled_equivalence() -> None:
    model: DetNetF = DetNetF().eval()
    frame: Float32[Tensor, "b 1 480 640"] = torch.rand(1, 1, 480, 640)
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_448_376
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        output: DetNetOutput = model(frame)
    macs: float = counter.get_total_flops() / 2
    print(f"DetNet-F: 1,448,376 parameters; {macs:,.0f} MACs")
    assert macs == pytest.approx(137.09e6, rel=0.005)
    assert output.center.shape == (1, 2, 2)
    assert output.radius.shape == output.presence_logit.shape == (1, 2)
    with torch.no_grad():
        assert model.backbone(frame).shape == (1, 160, 4, 5)
        pooled: DetNetOutput = model.forward_pooled(torch.nn.functional.avg_pool2d(frame, 4))
    torch.testing.assert_close(output.center, pooled.center, rtol=0, atol=0)
    torch.testing.assert_close(output.radius, pooled.radius, rtol=0, atol=0)
    torch.testing.assert_close(output.presence_logit, pooled.presence_logit, rtol=0, atol=0)


def test_keynet_table_and_intermediate_sizes() -> None:
    model: KeyNetF = KeyNetF().eval()
    crop: Float32[Tensor, "b 1 96 96"] = torch.rand(1, 1, 96, 96)
    keypoints: Float32[Tensor, "b 63"] = torch.rand(1, 63)
    assert sum(parameter.numel() for parameter in model.presence_head.parameters()) == 161
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_948_326 + 161
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        output: KeyNetOutput = model(crop, keypoints)
    macs: float = counter.get_total_flops() / 2
    print(f"KeyNet-F: 1,948,487 parameters (Table 5: 1,948,326); {macs:,.0f} MACs")
    assert macs == pytest.approx(161.67e6, rel=0.005)
    assert output.heatmaps.shape == (1, 21, 18, 18)
    assert output.distance.shape == (1, 21, 18)
    assert output.presence_logit.shape == (1,)
    with torch.no_grad():
        assert model.image(crop).shape == (1, 64, 12, 12)
        assert model.keypoints(keypoints).shape == (1, 4608)
        fused: Float32[Tensor, "b 160 6 6"] = model.fused(torch.zeros(1, 96, 12, 12))
        assert fused.shape == (1, 160, 6, 6)
        assert model.heatmap_head[:3](fused).shape == (1, 63, 8, 8)
        assert model.heatmap_head[:4](fused).shape == (1, 42, 16, 16)
        assert model.heatmap_head(fused).shape == (1, 21, 18, 18)
        assert isinstance(model.heatmap_head[3], nn.ConvTranspose2d)
        assert model.heatmap_head[3].bias is not None


def test_detnet_loss_masks_reduction_and_presence_weight() -> None:
    output: DetNetOutput = DetNetOutput(torch.zeros(3, 2, 2), torch.zeros(3, 2), torch.zeros(3, 2, requires_grad=True))
    circle: Float32[Tensor, "b 2 3"] = torch.tensor([[[1.0] * 3, [2.0] * 3], [[3.0] * 3, [float("nan")] * 3], [[float("nan")] * 3] * 2])
    circle_mask: Bool[Tensor, "b 2"] = torch.tensor([[True, True], [True, False], [False, False]])
    presence_mask: Bool[Tensor, "b 2"] = torch.tensor([[True, True], [False, True], [False, False]])
    presence: Float32[Tensor, "b 2"] = torch.zeros(3, 2)
    loss: DetNetLoss = detnet_loss(output, circle, presence, presence_mask, circle_mask)
    # Left mean = (1 + 9)/2, right mean = 4; two BCE terms = 2*log(2).
    assert loss.circle.item() == pytest.approx(9.0)
    assert loss.presence.item() == pytest.approx(2 * math.log(2))
    assert loss.total.item() == pytest.approx(9.0 + 200 * math.log(2))
    assert not loss.circle.requires_grad and not loss.presence.requires_grad
    circle[~circle_mask] = 1234.0
    presence[~presence_mask] = float("nan")
    torch.testing.assert_close(loss.total, detnet_loss(output, circle, presence, presence_mask, circle_mask).total)
    loss.total.backward()
    assert output.presence_logit.grad is not None
    assert torch.count_nonzero(output.presence_logit.grad[~presence_mask]) == 0


def test_detnet_perfect_and_empty_losses() -> None:
    output: DetNetOutput = DetNetOutput(
        torch.zeros(2, 2, 2, requires_grad=True), torch.zeros(2, 2, requires_grad=True), torch.full((2, 2), -1000.0, requires_grad=True)
    )
    for enabled in (True, False):
        mask: Bool[Tensor, "b 2"] = torch.full((2, 2), enabled)
        loss: DetNetLoss = detnet_loss(output, torch.zeros(2, 2, 3), torch.zeros(2, 2), mask, mask)
        assert loss.total.item() == loss.circle.item() == loss.presence.item() == 0.0
        loss.total.backward()
    assert output.center.grad is not None and torch.isfinite(output.center.grad).all()


def test_decode_detections_pixels_and_strict_threshold() -> None:
    output: DetNetOutput = DetNetOutput(
        torch.tensor([[[0.5, 0.25], [0.1, 0.75]]]), torch.tensor([[0.125, 0.05]]), torch.tensor([[0.0, math.log(3.0)]])
    )
    decoded: Detections = decode_detections(output)
    torch.testing.assert_close(decoded.circle, torch.tensor([[[320.0, 120.0, 80.0], [64.0, 360.0, 32.0]]]))
    torch.testing.assert_close(decoded.box, torch.tensor([[[240.0, 40.0, 400.0, 200.0], [32.0, 328.0, 96.0, 392.0]]]))
    torch.testing.assert_close(decoded.probability, torch.tensor([[0.5, 0.75]]))
    assert decoded.present.dtype == torch.bool
    assert decoded.present.tolist() == [[False, True]]
    assert decode_detections(output, threshold=0.8).present.tolist() == [[False, False]]


def test_keynet_loss_masks_weights_and_detached_logging() -> None:
    output: KeyNetOutput = KeyNetOutput(
        torch.zeros(3, 21, 18, 18, requires_grad=True), torch.zeros(3, 21, 18, requires_grad=True), torch.zeros(3, requires_grad=True)
    )
    heatmaps: Float32[Tensor, "b 21 18 18"] = torch.ones(3, 21, 18, 18)
    distance: Float32[Tensor, "b 21 18"] = torch.full((3, 21, 18), 2.0)
    heatmaps[1] = 3.0
    distance[1] = 4.0
    positive: Bool[Tensor, "b"] = torch.tensor([True, True, False])
    presence_mask: Bool[Tensor, "b"] = torch.tensor([True, False, True])
    presence: Float32[Tensor, "b"] = torch.zeros(3)
    loss: KeyNetLoss = keynet_loss(output, heatmaps, distance, presence, positive, presence_mask, 2.0)
    assert loss.heatmap.item() == pytest.approx(5.0)
    assert loss.distance.item() == pytest.approx(10.0)
    assert loss.presence.item() == pytest.approx(math.log(2))
    assert loss.total.item() == pytest.approx(5.0 + 0.05 * 10.0 + 2.0 * math.log(2))
    assert not any(term.requires_grad for term in (loss.heatmap, loss.distance, loss.presence))
    heatmaps[~positive] = float("nan")
    distance[~positive] = float("nan")
    presence[~presence_mask] = float("nan")
    torch.testing.assert_close(loss.total, keynet_loss(output, heatmaps, distance, presence, positive, presence_mask, 2.0).total)
    loss.total.backward()
    assert output.heatmaps.grad is not None and torch.count_nonzero(output.heatmaps.grad[~positive]) == 0
    assert output.distance.grad is not None and torch.count_nonzero(output.distance.grad[~positive]) == 0
    assert output.presence_logit.grad is not None and torch.count_nonzero(output.presence_logit.grad[~presence_mask]) == 0


def test_keynet_perfect_and_empty_losses() -> None:
    output: KeyNetOutput = KeyNetOutput(
        torch.zeros(2, 21, 18, 18, requires_grad=True), torch.zeros(2, 21, 18, requires_grad=True), torch.full((2,), -1000.0, requires_grad=True)
    )
    for enabled in (True, False):
        mask: Bool[Tensor, "b"] = torch.full((2,), enabled)
        loss: KeyNetLoss = keynet_loss(output, torch.zeros_like(output.heatmaps), torch.zeros_like(output.distance), torch.zeros(2), mask, mask, 1.0)
        assert loss.total.item() == loss.heatmap.item() == loss.distance.item() == loss.presence.item() == 0.0
        loss.total.backward()
    assert output.heatmaps.grad is not None and torch.isfinite(output.heatmaps.grad).all()


def test_training_backward_reaches_both_network_inputs() -> None:
    detnet: DetNetF = DetNetF().train()
    frame: Float32[Tensor, "b 1 480 640"] = torch.rand(2, 1, 480, 640, requires_grad=True)
    det_output: DetNetOutput = detnet(frame)
    mask: Bool[Tensor, "b 2"] = torch.ones(2, 2, dtype=torch.bool)
    detnet_loss(det_output, torch.rand(2, 2, 3), torch.ones(2, 2), mask, mask).total.backward()
    assert frame.grad is not None and torch.isfinite(frame.grad).all() and frame.grad.abs().sum() > 0
    keynet: KeyNetF = KeyNetF().train()
    crop: Float32[Tensor, "b 1 96 96"] = torch.rand(2, 1, 96, 96, requires_grad=True)
    keypoints: Float32[Tensor, "b 63"] = torch.rand(2, 63, requires_grad=True)
    key_output: KeyNetOutput = keynet(crop, keypoints)
    positive: Bool[Tensor, "b"] = torch.ones(2, dtype=torch.bool)
    keynet_loss(
        key_output, torch.ones_like(key_output.heatmaps), torch.ones_like(key_output.distance), torch.ones(2), positive, positive, 1.0
    ).total.backward()
    for source in (crop, keypoints):
        assert source.grad is not None and torch.isfinite(source.grad).all() and source.grad.abs().sum() > 0
