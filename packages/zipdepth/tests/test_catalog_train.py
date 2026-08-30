"""Pure policy tests for catalog fine-tuning."""

import pytest
from torch import nn, optim

from zipdepth.apis.train_catalog import TrainCatalogConfig, pin_batchnorm_eval, resolve_max_lr, split_holdout_segments
from zipdepth.loss import MetricDepthLoss, ZipDepthLoss
from zipdepth.training.trainer import ZipDepthTrainer


def test_catalog_training_defaults_to_parallel_segment_producers() -> None:
    """Keep enough catalog producers and bounded sample prefetch for GPU training."""
    config: TrainCatalogConfig = TrainCatalogConfig()

    assert config.shuffle_buffer_size == 256
    assert config.num_producers == 6
    assert config.prefetch_samples == 256
    assert not hasattr(config, "prefetch")
    assert not hasattr(config, "gpu_augment")


def test_catalog_training_defaults_to_metric_with_an_explicit_ssi_lane() -> None:
    """Make prompted metric distillation the default without removing relative training."""
    assert TrainCatalogConfig().target_mode == "metric"
    assert TrainCatalogConfig(target_mode="ssi").target_mode == "ssi"


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


def test_pin_batchnorm_eval_overrides_only_the_model_root_train_method() -> None:
    """Keep BatchNorm frozen across root mode changes without altering its own API."""
    model: nn.Sequential = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.Sequential(nn.BatchNorm1d(3)))
    batchnorms: list[nn.modules.batchnorm._BatchNorm] = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]

    pinned: int = pin_batchnorm_eval(model)
    model.train(False)
    model.train(True)

    assert pinned == 2
    assert model.training is True
    assert model[0].training is True
    assert all(not module.training for module in batchnorms)
    batchnorms[0].train(True)
    assert batchnorms[0].training is True
    model.train(True)
    assert all(not module.training for module in batchnorms)

