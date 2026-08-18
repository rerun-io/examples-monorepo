"""Direct 7,000-step gsplat trainer with the settled GausSurf losses."""

import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import torch
from gsplat import export_splats
from gsplat.strategy import DefaultStrategy
from plyfile import PlyData
from torchmetrics.functional.image import structural_similarity_index_measure

from gauss_surf.gaussurf_losses import (
    GausSurfLosses,
    GausSurfTargets,
    compute_gaussurf_losses,
    linear_ramp_weight,
    normal_contradiction_gate,
    prepare_gaussurf_targets,
)
from gauss_surf.gsplat_schedule import downscale_factor, opacity_reset_callback_step, sh_degree, should_refine, should_reset_opacity
from gauss_surf.train_gsplat.cache import GpuTrainingCache
from gauss_surf.train_gsplat.core import median_scale_regularization, opacity_export_mask, positive_neighbor_distances, save_checkpoint
from gauss_surf.train_gsplat.renderer import RenderOutput, render_splats, render_splats_two_call

ITERATIONS: int = 7_000
BASE_MEANS_LR: float = 1.6e-4
BASE_MEANS_LR_FINAL: float = 1.6e-6
BASE_DEPTH_REGULARIZATION_WEIGHT: float = 3.2
BASE_FLAT_REGULARIZATION_WEIGHT: float = 1.0
TrainingRenderer: TypeAlias = Callable[..., RenderOutput]


def select_training_renderer(*, fused_training: bool) -> TrainingRenderer:
    """Select the default two-call path or the explicit fused quality-round path."""
    return render_splats if fused_training else render_splats_two_call


@dataclass(frozen=True, slots=True)
class MetricTrainingConstants:
    """The four scale-dependent constants for metric-world training."""

    means_lr: float
    """Initial Gaussian-mean learning rate in metres."""
    means_lr_final: float
    """Final Gaussian-mean learning rate in metres."""
    depth_regularization_weight: float
    """Dense metric-depth loss weight."""
    flat_regularization_weight: float
    """Metric Gaussian-flatness loss weight."""


def metric_training_constants(scene_scale: float) -> MetricTrainingConstants:
    """Scale exactly the four dimensional training constants."""
    if not math.isfinite(scene_scale) or scene_scale <= 0.0:
        raise ValueError(f"scene scale must be positive and finite, got {scene_scale}")
    return MetricTrainingConstants(
        means_lr=BASE_MEANS_LR * scene_scale,
        means_lr_final=BASE_MEANS_LR_FINAL * scene_scale,
        depth_regularization_weight=BASE_DEPTH_REGULARIZATION_WEIGHT / scene_scale,
        flat_regularization_weight=BASE_FLAT_REGULARIZATION_WEIGHT / scene_scale,
    )


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Direct training artifacts and scalar outcomes."""

    splats: torch.nn.ParameterDict
    """Final live Gaussian parameters."""
    training_wall_seconds: float
    """Synchronized training-loop wall time."""
    gaussian_count: int
    """Final in-memory Gaussian count."""
    exported_gaussian_count: int
    """Gaussian count after the opacity export filter."""
    refinement_steps: tuple[int, ...]
    """Iterations at which refinement ran."""
    reset_steps: tuple[int, ...]
    """Iterations at which opacity reset ran."""
    strategy_trajectory: tuple["StrategyEvent", ...]
    """Pre/post counts and opacity maxima for every strategy event."""
    final_losses: dict[str, float]
    """Final iteration loss terms."""


@dataclass(frozen=True, slots=True)
class StrategyEvent:
    """One observed refinement or opacity-reset event."""

    step: int
    """Zero-based training iteration."""
    kind: Literal["refine", "reset"]
    """Strategy operation that ran at this iteration."""
    gaussian_count_before: int
    """Gaussian count before the strategy callback."""
    gaussian_count_after: int
    """Gaussian count after the strategy callback."""
    opacity_max_before: float
    """Maximum post-sigmoid opacity before the callback."""
    opacity_max_after: float
    """Maximum post-sigmoid opacity after the callback."""


def camera_sample_indices(*, num_train_data: int, iteration_count: int, seed: int) -> tuple[int, ...]:
    """Return the accepted seeded, shuffled, without-replacement camera stream."""
    if num_train_data <= 0 or iteration_count < 0:
        raise ValueError("training camera count must be positive and iteration count non-negative")
    sample_random: random.Random = random.Random(seed)
    sampled_indices: list[int] = []
    while len(sampled_indices) < iteration_count:
        epoch_indices: list[int] = list(range(num_train_data))
        sample_random.shuffle(epoch_indices)
        sampled_indices.extend(epoch_indices[: iteration_count - len(sampled_indices)])
    return tuple(sampled_indices)


def read_metric_seed(seed_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read metric-world seed vertices and colors without normalization."""
    ply_data: PlyData = PlyData.read(str(seed_path), mmap="r")
    vertex: Any = ply_data["vertex"]
    points_n3: np.ndarray = np.stack((vertex["x"], vertex["y"], vertex["z"]), axis=-1).astype(np.float32)
    colors_n3: np.ndarray = np.stack((vertex["red"], vertex["green"], vertex["blue"]), axis=-1).astype(np.float32) / 255.0
    return points_n3, colors_n3


def _random_quaternions(count: int) -> torch.Tensor:
    """Match the accepted uniform random quaternion initialization."""
    u: torch.Tensor = torch.rand(count)
    v: torch.Tensor = torch.rand(count)
    w: torch.Tensor = torch.rand(count)
    return torch.stack(
        (
            torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
            torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
            torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
            torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
        ),
        dim=-1,
    )


def initialize_splats(
    seed_path: Path,
    device: torch.device,
    constants: MetricTrainingConstants,
) -> tuple[torch.nn.ParameterDict, dict[str, torch.optim.Optimizer]]:
    """Initialize seeds with sklearn three-neighbor mean scales."""
    from sklearn.neighbors import NearestNeighbors  # pyrefly: ignore  # copied runtime package; see Part 8 report

    points_n3, colors_n3 = read_metric_seed(seed_path)
    neighbor_model = NearestNeighbors(n_neighbors=4, algorithm="auto", metric="euclidean").fit(points_n3)
    distances_n4: np.ndarray
    distances_n4, _indices_n4 = neighbor_model.kneighbors(points_n3)
    average_distance_n1: np.ndarray = distances_n4[:, 1:].astype(np.float32).mean(axis=-1, keepdims=True)
    average_distance_n1 = positive_neighbor_distances(average_distance_n1)
    count: int = len(points_n3)
    colors_n16x3: torch.Tensor = torch.zeros((count, 16, 3), dtype=torch.float32)
    colors_n16x3[:, 0] = (torch.from_numpy(colors_n3) - 0.5) / 0.28209479177387814
    parameter_values: dict[str, torch.Tensor] = {
        "means": torch.from_numpy(points_n3),
        "scales": torch.from_numpy(np.log(average_distance_n1).repeat(3, axis=1)),
        "quats": _random_quaternions(count),
        "opacities": torch.logit(torch.full((count, 1), 0.1)),
        "sh0": colors_n16x3[:, :1],
        "shN": colors_n16x3[:, 1:],
    }
    splats = torch.nn.ParameterDict(
        {name: torch.nn.Parameter(value.to(device)) for name, value in parameter_values.items()}
    )
    learning_rates: dict[str, float] = {
        "means": constants.means_lr,
        "scales": 0.005,
        "quats": 0.001,
        "opacities": 0.05,
        "sh0": 0.0025,
        "shN": 0.0025 / 20.0,
    }
    optimizers: dict[str, torch.optim.Optimizer] = {
        name: torch.optim.Adam([{"params": splats[name], "lr": learning_rate, "name": name}], eps=1e-15)
        for name, learning_rate in learning_rates.items()
    }
    return splats, optimizers


def _losses(
    *,
    step: int,
    splats: torch.nn.ParameterDict,
    output: RenderOutput,
    target_rgb_hw3: torch.Tensor,
    target_depth_hw1: torch.Tensor,
    target_normal_3hw: torch.Tensor,
    constants: MetricTrainingConstants,
) -> dict[str, torch.Tensor]:
    """Compute the unchanged appearance, depth, plane, and normal recipe."""
    target_rgb_hw3 = target_rgb_hw3[..., :3]
    l1: torch.Tensor = (target_rgb_hw3 - output.rgb_hw3).abs().mean()
    ssim_value = structural_similarity_index_measure(
        target_rgb_hw3.permute(2, 0, 1).unsqueeze(0),
        output.rgb_hw3.permute(2, 0, 1).unsqueeze(0),
        data_range=1.0,
    )
    if not isinstance(ssim_value, torch.Tensor):
        raise RuntimeError("SSIM metric unexpectedly returned a full image")
    ssim: torch.Tensor = ssim_value
    valid_depth_hw1: torch.Tensor = target_depth_hw1 > 0.0
    depth_error_hw1: torch.Tensor = (output.surface_depth_hw1 - target_depth_hw1).abs()
    depth_loss: torch.Tensor = (depth_error_hw1 * valid_depth_hw1).sum() / valid_depth_hw1.sum().clamp_min(1)
    result: dict[str, torch.Tensor] = {
        "main_loss": 0.8 * l1 + 0.2 * (1.0 - ssim),
        "depth_loss": constants.depth_regularization_weight * depth_loss,
    }
    if step < 1_400:
        return result

    target_normal_valid_1hw: torch.Tensor = torch.linalg.vector_norm(target_normal_3hw, dim=0, keepdim=True) > 0.5
    targets: GausSurfTargets = prepare_gaussurf_targets(
        prior_normal_3hw=target_normal_3hw,
        prior_valid_1hw=target_normal_valid_1hw,
        height=output.rgb_hw3.shape[0],
        width=output.rgb_hw3.shape[1],
    )
    normal_gate_hw1: torch.Tensor | None = None
    if step >= 2_625:
        normal_gate_hw1 = normal_contradiction_gate(
            output.direct_normal_hw3,
            targets.prior_normal_hw3,
            targets.prior_valid_hw1,
            maximum_angle_degrees=30.0,
        )
    normal_losses: GausSurfLosses = compute_gaussurf_losses(
        direct_normal_hw3=output.direct_normal_hw3,
        depth_normal_hw3=output.depth_normal_hw3,
        prior_normal_hw3=targets.prior_normal_hw3,
        prior_valid_hw1=targets.prior_valid_hw1,
        surface_valid_hw1=output.surface_valid_hw1.bool(),
        normal_prior_gate_hw1=normal_gate_hw1,
    )
    normal_ramp: float = linear_ramp_weight(step=step, start_step=1_400, ramp_steps=875)
    final_consistency_ramp: float = linear_ramp_weight(step=step, start_step=5_500, ramp_steps=500)
    consistency_weight: float = 0.5 + (0.55 - 0.5) * final_consistency_ramp
    scale_exp_n3: torch.Tensor = torch.exp(splats["scales"])
    result.update(
        {
            "gaussurf_normal_prior_loss": normal_losses.normal_prior_cosine * normal_ramp,
            "gaussurf_normal_consistency_loss": normal_losses.depth_normal_cosine * consistency_weight * normal_ramp,
            "flat_loss": constants.flat_regularization_weight * scale_exp_n3.amin(dim=-1).mean(),
        }
    )
    if step % 10 == 0:
        result["scale_reg"] = 0.1 * median_scale_regularization(splats["scales"], maximum_ratio=2.0)
    return result


def train(
    cache: GpuTrainingCache,
    *,
    bundle_dir: Path,
    run_dir: Path,
    seed: int,
    opacity_reset_every: int,
    fused_training: bool = False,
) -> TrainingResult:
    """Train the structural-parity recipe and write checkpoint/PLY artifacts."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device: torch.device = cache.wide_rgb_nhw3.device
    constants: MetricTrainingConstants = metric_training_constants(cache.scene_scale)
    splats, optimizers = initialize_splats(bundle_dir / "seed.ply", device, constants)
    means_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=(constants.means_lr_final / constants.means_lr) ** (1.0 / ITERATIONS)
    )
    num_train_data: int = len(cache.train_indices)
    strategy = DefaultStrategy(
        prune_opa=0.1,
        grow_grad2d=0.0008,
        grow_scale3d=0.01,
        grow_scale2d=0.05,
        prune_scale3d=0.5,
        prune_scale2d=0.15,
        refine_scale2d_stop_iter=5_500,
        refine_start_iter=500,
        refine_stop_iter=5_500,
        reset_every=3_000,
        refine_every=100,
        pause_refine_after_reset=num_train_data + 100,
        absgrad=True,
        revised_opacity=False,
        verbose=True,
    )
    strategy.check_sanity(splats, optimizers)
    strategy_state: dict[str, Any] = strategy.initialize_state(scene_scale=cache.scene_scale)
    sample_indices: tuple[int, ...] = camera_sample_indices(
        num_train_data=num_train_data,
        iteration_count=ITERATIONS,
        seed=seed,
    )
    training_renderer: TrainingRenderer = select_training_renderer(fused_training=fused_training)
    training_log_path: Path = run_dir / "training.log"
    run_dir.mkdir(parents=True, exist_ok=False)
    refinement_steps: list[int] = []
    reset_steps: list[int] = []
    strategy_trajectory: list[StrategyEvent] = []
    final_losses: dict[str, float] = {}
    torch.cuda.synchronize(device)
    started_at: float = time.perf_counter()
    with training_log_path.open("w", encoding="utf-8", buffering=1) as training_log:
        for step in range(ITERATIONS):
            sample_index: int = sample_indices[step]
            factor: int = downscale_factor(step)
            camera, target_rgb_hw3, target_depth_hw1, target_normal_3hw = cache.sample(sample_index, factor)
            background_3: torch.Tensor = torch.rand(3, dtype=torch.float32, device=device)
            output: RenderOutput = training_renderer(
                splats,
                camera,
                downscale=factor,
                sh_degree=sh_degree(step),
                background_3=background_3,
                absgrad=True,
            )
            strategy.step_pre_backward(splats, optimizers, strategy_state, step, output.appearance_info)
            loss_terms: dict[str, torch.Tensor] = _losses(
                step=step,
                splats=splats,
                output=output,
                target_rgb_hw3=target_rgb_hw3,
                target_depth_hw1=target_depth_hw1,
                target_normal_3hw=target_normal_3hw,
                constants=constants,
            )
            stacked_loss_terms: torch.Tensor = torch.stack(tuple(loss_terms.values()))
            if not bool(torch.isfinite(stacked_loss_terms).all().item()):
                nonfinite_terms: dict[str, float] = {
                    name: float(value.detach().item()) for name, value in loss_terms.items() if not bool(torch.isfinite(value).item())
                }
                all_terms: dict[str, float] = {name: float(value.detach().item()) for name, value in loss_terms.items()}
                message = f"step={step} nonfinite_loss_terms={nonfinite_terms} all_loss_terms={all_terms}"
                print(message, flush=True)
                training_log.write(message + "\n")
                raise FloatingPointError(message)
            total_loss: torch.Tensor = stacked_loss_terms.sum()
            total_loss.backward()
            for optimizer in optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            means_scheduler.step()
            scheduled_refinement: bool = should_refine(
                step=step,
                num_train_frames=num_train_data,
                refine_every=strategy.refine_every,
                reset_every=strategy.reset_every,
                refine_start=strategy.refine_start_iter,
                refine_stop=strategy.refine_stop_iter,
            )
            scheduled_reset: bool = should_reset_opacity(
                step=step,
                reset_every=opacity_reset_every,
                refine_stop=strategy.refine_stop_iter,
            )
            gaussian_count_before: int = len(splats["means"])
            opacity_max_before: float | None = None
            if scheduled_refinement or scheduled_reset:
                opacity_max_before = float(torch.sigmoid(splats["opacities"]).max().item())
            strategy_callback_step: int = opacity_reset_callback_step(
                step=step,
                refinement_cycle_every=strategy.reset_every,
                opacity_reset_every=opacity_reset_every,
            )
            strategy.step_post_backward(splats, optimizers, strategy_state, strategy_callback_step, output.appearance_info, packed=False)
            if scheduled_refinement or scheduled_reset:
                if opacity_max_before is None:
                    raise RuntimeError("strategy event opacity was not sampled")
                event_kind: Literal["refine", "reset"] = "reset" if scheduled_reset else "refine"
                event = StrategyEvent(
                    step=step,
                    kind=event_kind,
                    gaussian_count_before=gaussian_count_before,
                    gaussian_count_after=len(splats["means"]),
                    opacity_max_before=opacity_max_before,
                    opacity_max_after=float(torch.sigmoid(splats["opacities"]).max().item()),
                )
                strategy_trajectory.append(event)
                message: str = (
                    f"Step {step}: {event_kind} completed; "
                    f"gaussians={event.gaussian_count_before}->{event.gaussian_count_after}; "
                    f"opacity_max={event.opacity_max_before:.9f}->{event.opacity_max_after:.9f}."
                )
                print(message, flush=True)
                training_log.write(message + "\n")
            if scheduled_refinement:
                refinement_steps.append(step)
            if scheduled_reset:
                reset_steps.append(step)
            if step + 1 == ITERATIONS:
                final_losses = {name: float(value.detach().item()) for name, value in loss_terms.items()}
            if step % 100 == 0 or step + 1 == ITERATIONS:
                elapsed: float = time.perf_counter() - started_at
                message = (
                    f"step={step} loss={float(total_loss.detach().item()):.6f} "
                    f"gaussians={len(splats['means'])} factor={factor} sh={sh_degree(step)} wall={elapsed:.3f}"
                )
                print(message, flush=True)
                training_log.write(message + "\n")
    torch.cuda.synchronize(device)
    training_wall_seconds: float = time.perf_counter() - started_at
    checkpoint_path: Path = run_dir / "checkpoints" / "step-000006999.pt"
    save_checkpoint(
        checkpoint_path,
        step=ITERATIONS - 1,
        splats=splats,
        metadata={"seed": seed, "holdout_sha256": cache.holdout_sha256, "iterations": ITERATIONS},
    )
    keep_n: torch.Tensor = opacity_export_mask(splats["opacities"].squeeze(-1))
    export_splats(
        means=splats["means"][keep_n],
        scales=splats["scales"][keep_n],
        quats=splats["quats"][keep_n],
        opacities=splats["opacities"][keep_n].squeeze(-1),
        sh0=splats["sh0"][keep_n],
        shN=splats["shN"][keep_n],
        format="ply",
        save_to=str(run_dir / "splat.ply"),
    )
    return TrainingResult(
        splats=splats,
        training_wall_seconds=training_wall_seconds,
        gaussian_count=len(splats["means"]),
        exported_gaussian_count=int(keep_n.sum().item()),
        refinement_steps=tuple(refinement_steps),
        reset_steps=tuple(reset_steps),
        strategy_trajectory=tuple(strategy_trajectory),
        final_losses=final_losses,
    )
