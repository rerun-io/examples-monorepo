"""Read an upstream Hexels training run (config.json + checkpoints/ckpt_*.pt) into a `HexelCloud`.

The checkpoint is read as data (`torch.load(weights_only=True)`); the activations are the
`Hexels3DModel` properties of hexels@698d1af (`hexels/model.py` L.180-193), written out here so the
export runs wherever torch runs, without the CUDA build. The integration test
`test_export_matches_upstream_pack` pins this against upstream `pack_for_rasterizer`.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Bool, Float32
from numpy import ndarray
from serde import SerdeError, serde
from serde.json import from_json

from hexels_bench.hexel_ply import HexelCloud, HexelVariant


@serde
@dataclass(frozen=True, slots=True)
class UpstreamRunConfig:
    """The fields of upstream's `config.json` (Hexels3DOptions dump) that rendering depends on."""

    source_path: str
    """Scene directory the run trained on (upstream-relative, e.g. data/dtu/scan24)."""
    images: str | None
    """Image subdirectory of the scene (images, images_2, images_4); None means `images` (upstream default)."""
    resolution: int
    """Upstream resolution flag: a downsampling factor, or -1 for automatic."""
    eval: bool
    """True when the run held out a test split."""
    llffhold: int
    """Every llffhold-th camera (sorted by name, from index 0) is a test camera."""
    color_model: str
    """Spatial color model; the hexel PLY covers `linear` only."""
    view_model: str
    """none (V1), linear (V2), sh3 (V3), or lobe."""
    use_view_grads: bool
    """Whether the view term is active."""
    softness_mode: str
    """iso (one beta per hexel) or aniso (one per edge orientation)."""
    scale_modifier: float
    """Global scale multiplier at render time."""
    prefilter_edge_on: bool
    """Screen-space edge prefilter at render time."""
    white_background: bool
    """Background color: white when true, black otherwise."""
    iterations: int
    """Training steps of the run."""


def load_run_config(path: Path) -> UpstreamRunConfig:
    """Load upstream's config.json; unknown fields are upstream's business."""
    try:
        return from_json(UpstreamRunConfig, path.read_text())
    except SerdeError as error:
        raise ValueError(f"{path}: {error}") from error


def variant_of(config: UpstreamRunConfig) -> HexelVariant:
    """Upstream `infer_appearance_layout` (hexels/compact_checkpoint.py L.59), restricted to what the PLY holds."""
    problems: list[str] = []
    if config.color_model != "linear":
        problems.append(f"color_model={config.color_model}")
    if config.softness_mode != "iso":
        problems.append(f"softness_mode={config.softness_mode}")
    if config.scale_modifier != 1.0:
        problems.append(f"scale_modifier={config.scale_modifier}")
    if config.prefilter_edge_on:
        problems.append("prefilter_edge_on")
    if problems:
        raise ValueError("hexel PLY cannot represent " + ", ".join(problems))
    if config.view_model == "none":
        return "v1"
    if config.view_model == "linear" and config.use_view_grads:
        return "v2"
    if config.view_model == "sh3" and config.use_view_grads:
        return "v3"
    raise ValueError(f"no hexel PLY variant for view_model={config.view_model}, use_view_grads={config.use_view_grads}")


@dataclass(frozen=True, slots=True)
class UpstreamCheckpoint:
    """A loaded upstream checkpoint: its step and the render-ready hexels."""

    step: int
    """Training iteration stored in the checkpoint."""
    cloud: HexelCloud
    """Active hexels, activated."""


def _array(state: dict[str, torch.Tensor], name: str) -> ndarray:
    tensor: torch.Tensor | None = state.get(name)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"checkpoint model state has no tensor {name!r}")
    return tensor.detach().to(device="cpu").numpy()


def load_checkpoint(checkpoint: Path, config: UpstreamRunConfig) -> UpstreamCheckpoint:
    """Read `ckpt_*.pt` and apply upstream's activations to the active base hexels."""
    variant: HexelVariant = variant_of(config)
    payload: dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "deployment" in payload:
        raise ValueError(f"{checkpoint}: compact deployment checkpoints are not supported; use the FP32 original")
    state: dict[str, torch.Tensor] = payload["model"]
    if "child_parent_idx" in state and state["child_parent_idx"].numel() > 0:
        raise ValueError(f"{checkpoint}: checkpoints with child hexels are not supported")
    active: Bool[ndarray, "p"] = (
        _array(state, "active_mask").astype(bool) if "active_mask" in state else np.ones(state["means3d"].shape[0], dtype=bool)
    )
    if not active.any():
        raise ValueError(f"{checkpoint}: every hexel is inactive")

    # Activations in float32 torch ops, exactly the module properties.
    log_scales: torch.Tensor = state["log_scales"].float()
    quats: torch.Tensor = torch.nn.functional.normalize(state["rotation_quats"].float(), dim=1)
    view_grads: Float32[ndarray, "p k 3"] = _array(state, "view_grads").astype(np.float32)
    expected_basis: int = {"v1": view_grads.shape[1], "v2": 3, "v3": 15}[variant]
    if view_grads.shape[1] != expected_basis:
        raise ValueError(f"{checkpoint}: view_grads basis {view_grads.shape[1]} does not match {variant}")
    color_grads: Float32[ndarray, "p 2 3"] = _array(state, "color_grads").astype(np.float32)
    cloud = HexelCloud(
        variant=variant,
        centers_xyz=_array(state, "means3d").astype(np.float32)[active],
        quats_wxyz=quats.numpy()[active],
        scales_uv=torch.exp(log_scales).numpy()[active],
        opacities=torch.sigmoid(state["opacity_logits"].float()).numpy()[active],
        sharpness=torch.exp(state["log_softness"].float()).numpy()[active],
        base_rgb=_array(state, "colors").astype(np.float32)[active],
        grad_u_rgb=color_grads[active, 0, :],
        grad_v_rgb=color_grads[active, 1, :],
        view_linear=view_grads[active] if variant == "v2" else None,
        sh_coeffs=view_grads[active] if variant == "v3" else None,
    )
    return UpstreamCheckpoint(step=int(payload.get("iteration", -1)), cloud=cloud)


def find_checkpoint(model_dir: Path) -> Path:
    """The single `checkpoints/ckpt_*.pt` of an upstream run directory (the highest step if several)."""
    candidates: list[Path] = sorted((model_dir / "checkpoints").glob("ckpt_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"{model_dir}/checkpoints has no ckpt_*.pt")
    return candidates[-1]
