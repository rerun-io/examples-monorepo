"""Pure policy tests for catalog fine-tuning."""

from pathlib import Path

import pytest
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from zipdepth.apis.train_catalog import (
    CompileMode,
    TrainCatalogConfig,
    _adamw_optimizer,
    _load_json_config,
    _model_to_device,
    resolve_amp_dtype,
    resolve_max_lr,
)
from zipdepth.catalog.segments import split_holdout_segments
from zipdepth.loss import MetricDepthLoss, ZipDepthLoss
from zipdepth.training.trainer import ZipDepthTrainer


def test_upstream_train_config_loads_shipped_values_and_fallbacks(tmp_path: Path) -> None:
    """Ignore unused upstream keys while preserving every catalog-training fallback."""
    shipped = _load_json_config(Path("configs/default.json"))

    assert shipped.optimizer.lr == 1e-3
    assert shipped.optimizer.weight_decay == 0.05
    assert shipped.scheduler.pct_start == 0.05
    assert shipped.loss.alpha_grad == 2.0
    assert shipped.amp.dtype == "bfloat16"
    assert shipped.model.upsample_unfold is True
    assert shipped.training.max_step_checkpoints == 5

    minimal_path: Path = tmp_path / "minimal.json"
    minimal_path.write_text("{}")
    minimal = _load_json_config(minimal_path)

    assert minimal.optimizer.lr == 1e-3
    assert minimal.optimizer.weight_decay == 0.01
    assert minimal.scheduler.div_factor == 25.0
    assert minimal.scheduler.final_div_factor == 1000.0
    assert minimal.loss.alpha_ssi == 1.0
    assert minimal.amp.enabled is True
    assert minimal.model.upsample_unfold is True
    assert minimal.training.max_step_checkpoints == 5


def test_catalog_training_defaults_to_parallel_segment_producers() -> None:
    """Keep enough catalog producers and bounded sample prefetch for GPU training."""
    config: TrainCatalogConfig = TrainCatalogConfig()

    assert config.shuffle_buffer_size == 256
    assert config.num_producers == 6
    assert config.prefetch_samples == 256
    assert not hasattr(config, "prefetch")
    assert not hasattr(config, "gpu_augment")


def test_runtime_accelerator_defaults_match_existing_training_objects() -> None:
    """Keep the v4 model layout, AdamW arguments, compile mode, and JSON AMP default."""
    config: TrainCatalogConfig = TrainCatalogConfig()
    expected_model: nn.Conv2d = nn.Conv2d(3, 4, kernel_size=3)
    actual_model: nn.Conv2d = nn.Conv2d(3, 4, kernel_size=3)
    actual_model.load_state_dict(expected_model.state_dict())
    expected_optimizer: optim.AdamW = optim.AdamW(expected_model.parameters(), lr=1e-5, weight_decay=0.05)
    actual_optimizer: optim.AdamW = _adamw_optimizer(actual_model, lr=1e-5, weight_decay=0.05, fused=False)

    assert config.compile_mode == "reduce-overhead"
    assert config.channels_last is False
    assert config.fused_adamw is False
    assert config.amp_dtype is None
    assert _model_to_device(actual_model, torch.device("cpu"), channels_last=False) is actual_model
    assert actual_model.weight.stride() == expected_model.weight.stride()
    assert actual_model.state_dict().keys() == expected_model.state_dict().keys()
    assert actual_optimizer.defaults == expected_optimizer.defaults
    assert actual_optimizer.state_dict() == expected_optimizer.state_dict()
    assert resolve_amp_dtype(config.amp_dtype, "bfloat16") == "bfloat16"


def test_runtime_accelerator_flags_change_constructed_objects() -> None:
    """Apply channels-last layout, fused AdamW, and an explicit AMP dtype."""
    model: nn.Conv2d = nn.Conv2d(3, 4, kernel_size=3)
    prepared: nn.Module = _model_to_device(model, torch.device("cpu"), channels_last=True)
    optimizer: optim.AdamW = _adamw_optimizer(prepared, lr=1e-5, weight_decay=0.05, fused=True)

    assert model.weight.is_contiguous(memory_format=torch.channels_last)
    assert optimizer.defaults["fused"] is True
    assert resolve_amp_dtype("float16", "bfloat16") == "float16"


@pytest.mark.parametrize("compile_mode", ["default", "reduce-overhead", "max-autotune"])
def test_compile_mode_selects_torch_compile(monkeypatch: pytest.MonkeyPatch, compile_mode: CompileMode) -> None:
    """Compile with the selected production mode."""
    model: nn.Linear = nn.Linear(1, 1)
    optimizer: optim.SGD = optim.SGD(model.parameters(), lr=0.1)
    observed_modes: list[str] = []

    def fake_compile(module: nn.Module, *, mode: str) -> nn.Module:
        observed_modes.append(mode)
        return module

    monkeypatch.delenv("PIXI_DEV_MODE", raising=False)
    monkeypatch.setattr(torch, "compile", fake_compile)

    ZipDepthTrainer(model, [], optimizer, None, "cpu", use_amp=False, compile_mode=compile_mode)

    assert observed_modes == [compile_mode]


def test_compile_can_be_disabled_by_flag_or_dev_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep compilation off when requested and whenever beartype instrumentation is active."""
    model: nn.Linear = nn.Linear(1, 1)
    optimizer: optim.SGD = optim.SGD(model.parameters(), lr=0.1)
    observed_modes: list[str] = []

    def fake_compile(module: nn.Module, *, mode: str) -> nn.Module:
        observed_modes.append(mode)
        return module

    monkeypatch.delenv("PIXI_DEV_MODE", raising=False)
    monkeypatch.setattr(torch, "compile", fake_compile)
    ZipDepthTrainer(model, [], optimizer, None, "cpu", use_amp=False, compile_mode="off")
    monkeypatch.setenv("PIXI_DEV_MODE", "1")
    ZipDepthTrainer(model, [], optimizer, None, "cpu", use_amp=False, compile_mode="max-autotune")

    assert observed_modes == []


def test_catalog_training_defaults_to_metric_with_an_explicit_ssi_lane() -> None:
    """Make prompted metric distillation the default without removing relative training."""
    assert TrainCatalogConfig().target_mode == "metric"
    assert TrainCatalogConfig(target_mode="ssi").target_mode == "ssi"


def test_catalog_training_exposes_only_the_fixed_metric_loss_weight() -> None:
    """Keep the fixed metric objective free of default-off tuning knobs."""
    config: TrainCatalogConfig = TrainCatalogConfig()

    assert config.metric_gradient_weight == 0.5
    assert not hasattr(config, "metric_grad_scale_weights")
    assert not hasattr(config, "metric_grad_max_depth_m")
    assert not hasattr(config, "metric_grad_pool_mask_threshold")
    assert not hasattr(config, "metric_edge_weight_alpha")


def test_trainer_selects_the_objective_without_mixing_ssi_into_metric() -> None:
    """Use the fixed metric criterion only for the prompted lane."""
    model: nn.Linear = nn.Linear(1, 1)
    optimizer: optim.SGD = optim.SGD(model.parameters(), lr=0.1)

    ssi_trainer: ZipDepthTrainer = ZipDepthTrainer(model, [], optimizer, None, "cpu", use_amp=False, target_mode="ssi")
    metric_trainer: ZipDepthTrainer = ZipDepthTrainer(model, [], optimizer, None, "cpu", use_amp=False, target_mode="metric")

    assert isinstance(ssi_trainer.criterion, ZipDepthLoss)
    assert isinstance(metric_trainer.criterion, MetricDepthLoss)


def test_catalog_training_uses_one_explicit_optimizer_step_schedule() -> None:
    """Size the run directly without a fabricated epoch or frame-count model."""
    config: TrainCatalogConfig = TrainCatalogConfig()

    assert config.total_steps == 70_000
    assert not hasattr(config, "epochs")
    assert not hasattr(config, "max_steps")


@pytest.mark.parametrize(("skip_batches", "expected_skip"), [(0, 0), (None, 3)])
def test_trainer_accepts_an_explicit_resume_skip_override(skip_batches: int | None, expected_skip: int) -> None:
    """Bypass decoded-batch replay for unordered streams without changing the upstream default."""
    model: nn.Linear = nn.Linear(1, 1)
    optimizer: optim.SGD = optim.SGD(model.parameters(), lr=0.1)
    trainer: ZipDepthTrainer = ZipDepthTrainer(
        student=model,
        train_loader=[{"image": model.weight.detach()}] * 10,
        optimizer=optimizer,
        scheduler=None,
        device="cpu",
        use_amp=False,
        is_distributed=True,
        rank=1,
    )
    trainer.global_step = 3
    observed_skips: list[int] = []

    def fake_train_epoch(epoch: int, num_epochs: int, pbar: object, skip_batches: int = 0, max_steps: int = 0) -> float:
        del epoch, num_epochs, pbar, max_steps
        observed_skips.append(skip_batches)
        trainer.global_step = 4
        return 0.0

    trainer.train_epoch = fake_train_epoch  # type: ignore[method-assign]

    trainer.train(num_epochs=1, max_steps=4, skip_batches=skip_batches)

    assert observed_skips == [expected_skip]


def test_holdout_split_is_deterministic_disjoint_and_complete() -> None:
    """Reserve one stable holdout without losing or duplicating segments."""
    segment_ids: list[str] = [f"segment-{index}" for index in range(12)]

    first: tuple[list[str], list[str]] = split_holdout_segments(segment_ids, holdout_count=4, seed=17)
    second: tuple[list[str], list[str]] = split_holdout_segments(list(reversed(segment_ids)), holdout_count=4, seed=17)

    train_ids: list[str] = first[0]
    holdout_ids: list[str] = first[1]
    assert first == second
    assert set(train_ids).isdisjoint(holdout_ids)
    assert set(train_ids) | set(holdout_ids) == set(segment_ids)
    assert len(train_ids) + len(holdout_ids) == len(segment_ids)


@pytest.mark.parametrize(
    ("override", "from_scratch", "expected"),
    [
        (None, False, 1e-5),
        (None, True, 1e-3),
        (2e-5, False, 2e-5),
        (2e-5, True, 2e-5),
    ],
)
def test_max_lr_defaults_follow_initialization_policy(override: float | None, from_scratch: bool, expected: float) -> None:
    """Use one hundredth of upstream LR for fine-tuning and full LR from scratch."""
    actual: float = resolve_max_lr(config_lr=1e-3, override=override, from_scratch=from_scratch)

    assert actual == pytest.approx(expected)


def test_trainer_pins_wrapped_batchnorm_after_entering_train_mode() -> None:
    """Freeze BatchNorm through a wrapper without replacing the model's train method."""
    inner: nn.Sequential = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.Sequential(nn.BatchNorm1d(3)))
    wrapped: nn.Sequential = nn.Sequential(inner)
    optimizer: optim.SGD = optim.SGD(wrapped.parameters(), lr=0.1)
    trainer: ZipDepthTrainer = ZipDepthTrainer(
        student=wrapped,
        train_loader=DataLoader(TensorDataset(torch.empty(0)), batch_size=1),
        optimizer=optimizer,
        scheduler=None,
        device="cpu",
        use_amp=False,
        pin_batchnorm_eval=True,
    )
    original_train = wrapped.train
    batchnorms: list[nn.modules.batchnorm._BatchNorm] = [
        module for module in wrapped.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]

    with tqdm(disable=True) as progress:
        trainer.train_epoch(epoch=0, num_epochs=1, pbar=progress)

    assert wrapped.train == original_train
    assert wrapped.training is True
    assert inner[0].training is True
    assert all(not module.training for module in batchnorms)
