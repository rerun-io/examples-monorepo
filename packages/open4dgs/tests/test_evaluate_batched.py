"""The batched ``score_camera`` against the per-frame ``score_frame`` path it replaced in ``evaluate_cameras`` (CPU, synthetic)."""

import functools
import math

import pytest
import torch

pytest.importorskip("trigs", reason="TRiGS comes from recipes/open4dgs-trigs-src (open4dgs envs)")


def _clip(num_frames: int, height: int, width: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """UInt8 frames, soft masks with a person rectangle per frame (two frames share a box size, one is empty), renders, alphas."""
    gen = torch.Generator().manual_seed(seed)
    frames = torch.randint(0, 256, (num_frames, height, width, 3), generator=gen, dtype=torch.uint8)
    masks = torch.zeros(num_frames, height, width)
    boxes = [(10, 8, 60, 70), (14, 8, 64, 70), (5, 3, 50, 61), (20, 12, 75, 80), None, (0, 0, width, height), (30, 20, 72, 66)]
    for f in range(num_frames):
        box = boxes[f % len(boxes)]
        if box is None:
            continue
        x0, y0, x1, y1 = box
        masks[f, y0:y1, x0:x1] = (torch.rand(y1 - y0, x1 - x0, generator=gen) * 0.6 + 0.45).clamp(max=1.0)  # soft, mostly above 0.5
    renders = torch.rand(num_frames, height, width, 3, generator=gen)
    renders[1] = frames[1].float() / 255.0 * masks[1, ..., None]  # exact render: crop_raw PSNR is inf
    alphas = (torch.rand(num_frames, height, width, 1, generator=gen) + masks[..., None] * 0.5).clamp(0, 1)
    return frames, masks, renders, alphas


def _per_frame_reference(camera, frame_indices, renders, alphas, frames, masks, metrics, with_baselines):  # noqa: ANN001, ANN202
    """The pre-batching ``evaluate_cameras`` loop body (one camera): ``score_frame`` through the torchmetrics LPIPS module."""
    from open4dgs.evaluate import BASELINES, baseline_renders

    rows, floors = [], {kind: [] for kind in BASELINES} if with_baselines else {}
    for f, frame_index in enumerate(frame_indices):
        gt = frames[f].float() / 255.0
        mask = masks[f].float()
        rows.append(metrics.frame(camera, frame_index, renders[f], alphas[f][..., 0], gt, mask))
        if with_baselines:
            for kind, (floor_rgb, floor_alpha) in baseline_renders(gt, mask).items():
                floors[kind].append(metrics.frame(camera, frame_index, floor_rgb, floor_alpha, gt, mask))
    return rows, floors


def _assert_rows_close(new, old) -> None:  # noqa: ANN001
    from open4dgs.evaluate import METRIC_NAMES

    assert len(new) == len(old)
    for a, b in zip(new, old, strict=True):
        assert (a.camera, a.frame_index) == (b.camera, b.frame_index)
        for name in METRIC_NAMES:
            x, y = getattr(a, name), getattr(b, name)
            if math.isnan(y) or math.isinf(y):
                assert (math.isnan(x) and math.isnan(y)) or x == y, (a.frame_index, name, x, y)
            else:
                assert x == pytest.approx(y, abs=1e-5, rel=0.0), (a.frame_index, name, x, y)


@pytest.mark.parametrize("with_baselines", [False, True])
def test_score_camera_matches_score_frame(monkeypatch: pytest.MonkeyPatch, with_baselines: bool) -> None:
    import torchmetrics.image.lpip as lpip_module

    from open4dgs import evaluate

    # Randomly initialised AlexNet LPIPS (no downloaded weights); both paths use the same network instance.
    monkeypatch.setattr(lpip_module, "_NoTrainLpips", functools.partial(lpip_module._NoTrainLpips, pretrained=False, pnet_rand=True))
    torch.manual_seed(0)
    metrics = evaluate.Metrics(torch.device("cpu"))
    monkeypatch.setattr(evaluate, "EVAL_BATCH_FRAMES", 4)  # 11 frames: batches of 4, 4, 3

    frames, masks, renders, alphas = _clip(11, 88, 96, seed=3)
    frame_indices = list(range(100, 111))
    new = evaluate.score_camera(
        "cam", frame_indices, lambda f: (renders[f], alphas[f]), frames, masks, metrics, keep_renders=True, with_baselines=with_baselines
    )
    rows, floors = _per_frame_reference("cam", frame_indices, renders, alphas, frames, masks, metrics, with_baselines)
    _assert_rows_close(new.metrics, rows)
    assert set(new.baselines) == set(floors)
    for kind in floors:
        _assert_rows_close(new.baselines[kind], floors[kind])
    assert torch.equal(new.render, (renders * 255.0).round().to(torch.uint8))
    assert math.isnan(new.metrics[4].crop_raw_psnr) and math.isnan(new.metrics[4].alpha_iou)  # empty mask
    assert new.metrics[1].crop_raw_psnr == math.inf and new.metrics[0].crop_raw_lpips > 0.0


def _fp32_flags() -> tuple[str, str, str]:
    return (torch.backends.cudnn.conv.fp32_precision, torch.backends.cudnn.rnn.fp32_precision, torch.backends.cuda.matmul.fp32_precision)


def test_ieee_fp32_sets_full_float32_and_restores_the_flags() -> None:
    from open4dgs.evaluate import ieee_fp32

    before = _fp32_flags()
    try:
        # A non-default state (TF32 everywhere, as torch.set_float32_matmul_precision("high") + cuDNN's default give).
        torch.backends.cudnn.conv.fp32_precision = torch.backends.cudnn.rnn.fp32_precision = "tf32"
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        with ieee_fp32():
            assert _fp32_flags() == ("ieee", "ieee", "ieee")
        assert _fp32_flags() == ("tf32", "tf32", "tf32")
        with pytest.raises(RuntimeError), ieee_fp32():
            raise RuntimeError("scoring failed")
        assert _fp32_flags() == ("tf32", "tf32", "tf32")
    finally:
        torch.backends.cudnn.conv.fp32_precision, torch.backends.cudnn.rnn.fp32_precision, torch.backends.cuda.matmul.fp32_precision = before


def test_scoring_runs_without_tf32_and_leaves_the_training_flags_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both scoring paths see TF32 off inside; outside (where the trainer runs) the flags are exactly what they were."""
    import torchmetrics.image.lpip as lpip_module

    from open4dgs import evaluate

    monkeypatch.setattr(lpip_module, "_NoTrainLpips", functools.partial(lpip_module._NoTrainLpips, pretrained=False, pnet_rand=True))
    metrics = evaluate.Metrics(torch.device("cpu"))
    seen: list[tuple[str, str, str]] = []
    batch_lpips = evaluate._batch_lpips

    def spy_batch(*args):  # noqa: ANN002, ANN202
        seen.append(_fp32_flags())
        return batch_lpips(*args)

    monkeypatch.setattr(evaluate, "_batch_lpips", spy_batch)
    training_flags = _fp32_flags()
    assert training_flags[:2] == ("tf32", "tf32")  # PyTorch's default for cuDNN: what the trainer runs with
    frames, masks, renders, alphas = _clip(3, 88, 96, seed=1)
    evaluate.score_camera("cam", [0, 1, 2], lambda f: (renders[f], alphas[f]), frames, masks, metrics, with_baselines=True)
    assert _fp32_flags() == training_flags

    def spy_lpips(pred, target):  # noqa: ANN001, ANN202
        seen.append(_fp32_flags())
        return metrics.lpips(pred, target)

    evaluate.score_frame("cam", 0, renders[0], alphas[0][..., 0], frames[0].float() / 255.0, masks[0], spy_lpips)
    metrics.frame("cam", 0, renders[0], alphas[0][..., 0], frames[0].float() / 255.0, masks[0])
    assert _fp32_flags() == training_flags
    assert seen and all(flags == ("ieee", "ieee", "ieee") for flags in seen), seen
