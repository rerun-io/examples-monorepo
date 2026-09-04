"""Pure policy tests for catalog fine-tuning."""

from pathlib import Path

import pytest
import torch
from monopriors.models.zipdepth_checkpoint import RANGE_MARGIN_KEY
from serde.json import from_json, to_json
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from zipdepth.apis.train_catalog import (
    CompileMode,
    ConstantCooldownLR,
    ResolutionStage,
    ResolvedTrainCatalogConfig,
    TrainCatalogConfig,
    TrainingRecipe,
    UpstreamTrainConfig,
    _adamw_optimizer,
    _build_scheduler,
    _load_json_config,
    _model_to_device,
    parse_stage_schedule,
    resolve_amp_dtype,
    resolve_max_lr,
    resolve_training_recipe,
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
    """Keep the existing loader, producer count, and bounded sample prefetch by default."""
    config: TrainCatalogConfig = TrainCatalogConfig()

    assert config.dataloader == "current"
    assert config.rerun_fetch_block_size == 512
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
    name: str
    tensor: torch.Tensor
    for name, tensor in actual_model.state_dict().items():
        assert torch.equal(tensor, expected_model.state_dict()[name])
    assert actual_optimizer.defaults == expected_optimizer.defaults
    assert actual_optimizer.state_dict() == expected_optimizer.state_dict()
    assert resolve_amp_dtype(config.amp_dtype, "bfloat16") == "bfloat16"


def test_runtime_accelerator_flags_change_constructed_objects() -> None:
    """Apply channels-last layout, fused AdamW, and an explicit AMP dtype."""
    model: nn.Conv2d = nn.Conv2d(3, 4, kernel_size=3)
    prepared: nn.Module = _model_to_device(model, torch.device("cpu"), channels_last=True)
    optimizer: optim.AdamW = _adamw_optimizer(prepared, lr=1e-5, weight_decay=0.05, fused=True)
    trainer: ZipDepthTrainer = ZipDepthTrainer(
        prepared,
        [],
        optimizer,
        None,
        "cpu",
        use_amp=False,
        amp_dtype=resolve_amp_dtype("float16", "bfloat16"),
        compile_mode="off",
        channels_last=True,
    )

    assert model.weight.is_contiguous(memory_format=torch.channels_last)
    assert optimizer.defaults["fused"] is True
    assert trainer.amp_dtype is torch.float16
    assert trainer.channels_last is True


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


def test_range_margin_defaults_to_the_prompt_bounded_head_and_survives_resolved_config() -> None:
    """Record the ultrawide output-range margin in the run's resolved_config.json."""
    assert TrainCatalogConfig().range_margin == 0.0
    config: TrainCatalogConfig = TrainCatalogConfig(range_margin=0.25)
    upstream: UpstreamTrainConfig = _load_json_config(Path("configs/default.json"))

    resolved: ResolvedTrainCatalogConfig = ResolvedTrainCatalogConfig(
        config=config,
        rank_count=1,
        steps_per_epoch=10,
        total_steps=10,
        resolved_max_lr=1e-5,
        recipe=resolve_training_recipe(config, upstream.scheduler, upstream.amp),
    )

    restored: ResolvedTrainCatalogConfig = from_json(ResolvedTrainCatalogConfig, to_json(resolved))
    assert restored.config.range_margin == 0.25


def test_trainer_records_the_range_margin_in_every_checkpoint(tmp_path: Path) -> None:
    """Let inference rebuild the trained head without repeating the training flag."""
    model: nn.Linear = nn.Linear(1, 1)
    optimizer: optim.SGD = optim.SGD(model.parameters(), lr=0.1)
    trainer: ZipDepthTrainer = ZipDepthTrainer(model, [], optimizer, None, "cpu", use_amp=False, range_margin=0.25)
    checkpoint: Path = tmp_path / "step_1.pth"

    trainer.save_checkpoint(str(checkpoint), epoch=0, loss=0.0)

    saved: dict[str, object] = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved[RANGE_MARGIN_KEY] == 0.25


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


def test_scheduler_defaults_match_existing_onecycle_state() -> None:
    """Construct the same OneCycle scheduler and optimizer state as today's recipe."""
    config: TrainCatalogConfig = TrainCatalogConfig()
    expected_model: nn.Linear = nn.Linear(1, 1)
    actual_model: nn.Linear = nn.Linear(1, 1)
    expected_optimizer: optim.AdamW = optim.AdamW(expected_model.parameters(), lr=1e-5, weight_decay=0.05)
    actual_optimizer: optim.AdamW = optim.AdamW(actual_model.parameters(), lr=1e-5, weight_decay=0.05)
    expected: optim.lr_scheduler.OneCycleLR = optim.lr_scheduler.OneCycleLR(
        expected_optimizer,
        max_lr=1e-5,
        total_steps=100,
        pct_start=0.05,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1000.0,
    )
    actual: optim.lr_scheduler.LRScheduler = _build_scheduler(
        actual_optimizer,
        schedule=config.schedule,
        max_lr=1e-5,
        total_steps=100,
        warmup_pct=0.05,
        div_factor=25.0,
        final_div_factor=1000.0,
        cooldown_pct=config.cooldown_pct,
    )

    assert config.schedule == "onecycle"
    assert config.warmup_pct is None
    assert config.final_div_factor is None
    assert config.cooldown_pct == 0.1
    assert isinstance(actual, optim.lr_scheduler.OneCycleLR)
    assert actual.state_dict() == expected.state_dict()
    assert actual_optimizer.state_dict() == expected_optimizer.state_dict()


def test_onecycle_overrides_change_the_lr_trajectory() -> None:
    """Pass warmup percentage and final division through to OneCycleLR."""
    default_model: nn.Linear = nn.Linear(1, 1)
    tuned_model: nn.Linear = nn.Linear(1, 1)
    default_optimizer: optim.SGD = optim.SGD(default_model.parameters(), lr=1e-3)
    tuned_optimizer: optim.SGD = optim.SGD(tuned_model.parameters(), lr=1e-3)
    default_scheduler: optim.lr_scheduler.LRScheduler = _build_scheduler(
        default_optimizer,
        schedule="onecycle",
        max_lr=1e-3,
        total_steps=100,
        warmup_pct=0.05,
        div_factor=25.0,
        final_div_factor=1000.0,
        cooldown_pct=0.1,
    )
    tuned_scheduler: optim.lr_scheduler.LRScheduler = _build_scheduler(
        tuned_optimizer,
        schedule="onecycle",
        max_lr=1e-3,
        total_steps=100,
        warmup_pct=0.25,
        div_factor=25.0,
        final_div_factor=10.0,
        cooldown_pct=0.1,
    )
    default_lrs: list[float] = [default_optimizer.param_groups[0]["lr"]]
    tuned_lrs: list[float] = [tuned_optimizer.param_groups[0]["lr"]]
    for _ in range(100):
        default_optimizer.step()
        tuned_optimizer.step()
        default_scheduler.step()
        tuned_scheduler.step()
        default_lrs.append(default_optimizer.param_groups[0]["lr"])
        tuned_lrs.append(tuned_optimizer.param_groups[0]["lr"])

    assert tuned_lrs != default_lrs
    assert tuned_lrs[20] > default_lrs[20]
    assert tuned_lrs[-1] > default_lrs[-1]


def test_constant_cooldown_is_flat_then_linear() -> None:
    """Hold max LR before cooling linearly over the requested final fraction."""
    model: nn.Linear = nn.Linear(1, 1)
    optimizer: optim.SGD = optim.SGD(model.parameters(), lr=1.0)
    scheduler: optim.lr_scheduler.LRScheduler = _build_scheduler(
        optimizer,
        schedule="constant-cooldown",
        max_lr=1.0,
        total_steps=10,
        warmup_pct=0.05,
        div_factor=25.0,
        final_div_factor=10.0,
        cooldown_pct=0.3,
    )
    used_lrs: list[float] = []
    for _ in range(10):
        used_lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert isinstance(scheduler, ConstantCooldownLR)
    assert used_lrs[:7] == [1.0] * 7
    assert used_lrs[7:] == pytest.approx([0.7, 0.4, 0.1])


def test_stage_schedule_defaults_to_one_unchanged_resolution() -> None:
    """Keep the configured height and width for every step by default."""
    config: TrainCatalogConfig = TrainCatalogConfig()

    stages: list[ResolutionStage] = parse_stage_schedule(config.stage_schedule, config.total_steps, config.height, config.width)

    assert config.stage_schedule is None
    assert stages == [ResolutionStage(height=768, width=1024, start_step=0, end_step=70_000)]


def test_stage_schedule_assigns_each_fraction_to_a_resolution() -> None:
    """Convert progressive fractions into exact, contiguous optimizer-step ranges."""
    stages: list[ResolutionStage] = parse_stage_schedule("384x512:0.3,768x1024:0.7", total_steps=10, height=1, width=1)

    assert stages == [
        ResolutionStage(height=384, width=512, start_step=0, end_step=3),
        ResolutionStage(height=768, width=1024, start_step=3, end_step=10),
    ]


@pytest.mark.parametrize(
    "stage_schedule",
    ["", "384x512", "384x512:0.0,768x1024:1.0", "384x512:0.4,768x1024:0.5"],
)
def test_stage_schedule_rejects_invalid_or_incomplete_fractions(stage_schedule: str) -> None:
    """Reject malformed stages, empty stages, and fractions that do not cover the run."""
    with pytest.raises(ValueError):
        parse_stage_schedule(stage_schedule, total_steps=10, height=768, width=1024)


def test_checkpoint_carries_the_progressive_resolution_stage(tmp_path: Path) -> None:
    """Persist the active stage so a checkpoint records where training stopped."""
    model: nn.Linear = nn.Linear(1, 1)
    optimizer: optim.SGD = optim.SGD(model.parameters(), lr=0.1)
    trainer: ZipDepthTrainer = ZipDepthTrainer(model, [], optimizer, None, "cpu", use_amp=False)
    trainer.training_stage = 1
    checkpoint: Path = tmp_path / "stage.pth"

    trainer.save_checkpoint(checkpoint, epoch=0, loss=0.5)
    saved: object = torch.load(checkpoint, map_location="cpu", weights_only=True)

    assert isinstance(saved, dict)
    assert saved["training_stage"] == 1


def test_fast_preset_only_raises_the_peak_learning_rate() -> None:
    """The hill-climb winner: fast = v4 with max_lr config/10 instead of config/100; an explicit --max-lr still wins."""
    assert resolve_max_lr(config_lr=1e-3, override=None, from_scratch=False, preset="v4") == pytest.approx(1e-5)
    assert resolve_max_lr(config_lr=1e-3, override=None, from_scratch=False, preset="fast") == pytest.approx(1e-4)
    assert resolve_max_lr(config_lr=1e-3, override=3e-5, from_scratch=False, preset="fast") == pytest.approx(3e-5)
    assert resolve_max_lr(config_lr=1e-3, override=None, from_scratch=True, preset="fast") == pytest.approx(1e-3)


def test_v4_and_fast_presets_share_every_runtime_control() -> None:
    """Keep v4 bit-compatible; fast shares the whole recipe except the peak learning rate."""
    upstream: UpstreamTrainConfig = _load_json_config(Path("configs/default.json"))

    v4: TrainingRecipe = resolve_training_recipe(TrainCatalogConfig(preset="v4"), upstream.scheduler, upstream.amp)
    fast: TrainingRecipe = resolve_training_recipe(TrainCatalogConfig(preset="fast"), upstream.scheduler, upstream.amp)

    assert v4 == TrainingRecipe(
        compile_mode="reduce-overhead",
        channels_last=False,
        fused_adamw=False,
        amp_dtype="bfloat16",
        schedule="onecycle",
        warmup_pct=0.05,
        final_div_factor=1000.0,
        cooldown_pct=0.1,
        stage_schedule=None,
    )
    assert fast == v4


def test_explicit_training_knobs_override_the_named_preset() -> None:
    """Let a hill-climb arm replace any individual value without defining a new preset."""
    upstream: UpstreamTrainConfig = _load_json_config(Path("configs/default.json"))
    config: TrainCatalogConfig = TrainCatalogConfig(
        preset="fast",
        compile_mode="off",
        channels_last=True,
        fused_adamw=True,
        amp_dtype="float16",
        schedule="constant-cooldown",
        warmup_pct=0.2,
        final_div_factor=20.0,
        cooldown_pct=0.3,
        stage_schedule="384x512:0.3,768x1024:0.7",
    )

    recipe: TrainingRecipe = resolve_training_recipe(config, upstream.scheduler, upstream.amp)

    assert recipe == TrainingRecipe(
        compile_mode="off",
        channels_last=True,
        fused_adamw=True,
        amp_dtype="float16",
        schedule="constant-cooldown",
        warmup_pct=0.2,
        final_div_factor=20.0,
        cooldown_pct=0.3,
        stage_schedule="384x512:0.3,768x1024:0.7",
    )


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
