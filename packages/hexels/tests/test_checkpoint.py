import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hexels_bench.apis.export import export_run
from hexels_bench.checkpoint import UpstreamRunConfig, load_checkpoint, load_run_config, variant_of
from hexels_bench.hexel_ply import HexelPly, read_hexel_ply

VIEW_MODELS: dict[str, tuple[str, bool, int]] = {"v1": ("none", False, 3), "v2": ("linear", True, 3), "v3": ("sh3", True, 15)}


def write_config(path: Path, variant: str, **overrides: object) -> Path:
    view_model, use_view_grads, _ = VIEW_MODELS[variant]
    config: dict[str, object] = {
        "source_path": "data/dtu/scan24", "images": "images", "resolution": 2, "eval": True, "llffhold": 8,
        "color_model": "linear", "view_model": view_model, "use_view_grads": use_view_grads, "softness_mode": "iso",
        "scale_modifier": 1.0, "prefilter_edge_on": False, "white_background": False, "iterations": 30000,
        "lr_decay_milestones": [15001, 20001, 25001],  # an upstream field the reader ignores
    }  # fmt: skip
    config.update(overrides)
    path.write_text(json.dumps(config))
    return path


def upstream_state(count: int, basis: int, seed: int = 0) -> dict[str, torch.Tensor]:
    """Synthetic `Hexels3DModel.state_dict()` with upstream's tensor names and shapes."""
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    return {
        "means3d": randn(count, 3), "log_scales": randn(count, 2), "rotation_quats": randn(count, 4),
        "colors": randn(count, 3), "color_grads": randn(count, 2, 3), "color_quads": randn(count, 3, 3),
        "view_grads": randn(count, basis, 3), "specular_ks": randn(count, 3), "specular_log_s": randn(count),
        "opacity_logits": randn(count), "log_softness": randn(count),
        "active_mask": torch.tensor([index % 3 != 1 for index in range(count)]),
        "child_parent_idx": torch.empty(0, dtype=torch.long),
    }  # fmt: skip


@pytest.mark.parametrize("variant", ["v1", "v2", "v3"])
def test_activations_match_upstream_properties(tmp_path: Path, variant: str) -> None:
    state: dict[str, torch.Tensor] = upstream_state(7, VIEW_MODELS[variant][2])
    torch.save({"iteration": 30000, "model": state}, tmp_path / "ckpt_030000.pt")
    loaded = load_checkpoint(tmp_path / "ckpt_030000.pt", load_run_config(write_config(tmp_path / "config.json", variant)))
    active: np.ndarray = state["active_mask"].numpy()
    assert loaded.step == 30000
    assert loaded.cloud.count == int(active.sum()) == 5
    cloud = loaded.cloud
    np.testing.assert_array_equal(cloud.centers_xyz, state["means3d"].numpy()[active])
    np.testing.assert_allclose(cloud.scales_uv, np.exp(state["log_scales"].numpy()[active]), rtol=1e-6)
    np.testing.assert_allclose(cloud.opacities, 1.0 / (1.0 + np.exp(-state["opacity_logits"].numpy()[active])), rtol=1e-6)
    np.testing.assert_allclose(cloud.sharpness, np.exp(state["log_softness"].numpy()[active]), rtol=1e-6)
    quats: np.ndarray = state["rotation_quats"].numpy()[active]
    np.testing.assert_allclose(cloud.quats_wxyz, quats / np.linalg.norm(quats, axis=1, keepdims=True), rtol=1e-6)
    np.testing.assert_array_equal(cloud.grad_u_rgb, state["color_grads"].numpy()[active, 0])
    np.testing.assert_array_equal(cloud.grad_v_rgb, state["color_grads"].numpy()[active, 1])
    view: np.ndarray = state["view_grads"].numpy()[active]
    if variant == "v2":
        assert cloud.view_linear is not None
        np.testing.assert_array_equal(cloud.view_linear, view)
    if variant == "v3":
        assert cloud.sh_coeffs is not None
        np.testing.assert_array_equal(cloud.sh_coeffs, view)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"color_model": "quadratic"}, "color_model"),
        ({"softness_mode": "aniso"}, "softness_mode"),
        ({"scale_modifier": 2.0}, "scale_modifier"),
        ({"prefilter_edge_on": True}, "prefilter_edge_on"),
        ({"view_model": "lobe"}, "no hexel PLY variant"),
    ],
)
def test_unrepresentable_configs_are_rejected(tmp_path: Path, overrides: dict[str, object], message: str) -> None:
    config: UpstreamRunConfig = load_run_config(write_config(tmp_path / "config.json", "v1", **overrides))
    with pytest.raises(ValueError, match=message):
        variant_of(config)


def test_children_are_rejected(tmp_path: Path) -> None:
    state: dict[str, torch.Tensor] = upstream_state(4, 3)
    state["child_parent_idx"] = torch.tensor([0, 1])
    torch.save({"iteration": 10, "model": state}, tmp_path / "ckpt.pt")
    with pytest.raises(ValueError, match="child"):
        load_checkpoint(tmp_path / "ckpt.pt", load_run_config(write_config(tmp_path / "config.json", "v1")))


def test_export_run_writes_readable_ply(tmp_path: Path) -> None:
    (tmp_path / "checkpoints").mkdir()
    torch.save({"iteration": 7000, "model": upstream_state(6, 15)}, tmp_path / "checkpoints" / "ckpt_007000.pt")
    write_config(tmp_path / "config.json", "v3")
    written: HexelPly = export_run(tmp_path, tmp_path / "model.ply")
    loaded: HexelPly = read_hexel_ply(tmp_path / "model.ply")
    assert loaded.cloud.variant == "v3"
    assert loaded.step == 7000
    assert loaded.source.endswith("ckpt_007000.pt")
    np.testing.assert_array_equal(loaded.cloud.to_rows(), written.cloud.to_rows())
