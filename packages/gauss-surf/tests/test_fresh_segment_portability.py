"""Contracts that let the direct trainer accept a fresh catalog segment."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gauss_surf.train_gsplat.cache import GpuTrainingCache, TrainingCamera
from gauss_surf.train_gsplat.evaluation import evaluate_holdout
from gauss_surf.train_gsplat.metadata import holdout_wide_indices, metadata_files_complete
from gauss_surf.train_gsplat.renderer import RenderOutput


def _camera(index: int) -> TrainingCamera:
    """Build one tiny wide holdout camera."""
    return TrainingCamera(
        stem=f"wide_{index:06d}",
        camera="wide",
        timestamp_ns=index,
        width=16,
        height=16,
        viewmat_44=torch.eye(4),
        K_33=torch.eye(3),
        cache_index=index,
        holdout=True,
    )


def test_fresh_metadata_contract_uses_segment_length(tmp_path: Path) -> None:
    """Fresh metadata uses a dynamic holdout cadence."""
    assert holdout_wide_indices(663) == tuple(range(7, 663, 8))
    assert len(holdout_wide_indices(663)) == 82
    assert metadata_files_complete(tmp_path) is False


def test_generated_metadata_rejects_an_interrupted_mixed_generation(tmp_path: Path) -> None:
    """An auto-generated bundle is invalid when one file differs from its completed generation."""
    file_contents: dict[str, bytes] = {
        "transforms.json": b"generation-a transforms\n",
        "cameras_all.json": b"generation-a cameras\n",
        "seed.ply": b"generation-a seed\n",
    }
    for name, content in file_contents.items():
        (tmp_path / name).write_bytes(content)
    (tmp_path / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": {name: {"sha256": hashlib.sha256(content).hexdigest()} for name, content in file_contents.items()},
            }
        ),
        encoding="utf-8",
    )
    assert metadata_files_complete(tmp_path) is True

    (tmp_path / "cameras_all.json").write_bytes(b"stale cameras from generation-b\n")

    assert metadata_files_complete(tmp_path) is False


def test_explicit_legacy_metadata_bundle_remains_readable(tmp_path: Path) -> None:
    """An explicit legacy bundle may omit the generation manifest."""
    for name in ("transforms.json", "cameras_all.json", "seed.ply"):
        (tmp_path / name).touch()

    assert metadata_files_complete(tmp_path) is False
    assert metadata_files_complete(tmp_path, allow_legacy=True) is True


def test_holdout_evaluation_reads_dynamic_targets_from_gpu_cache(monkeypatch: Any) -> None:
    """Evaluation neither reads bundle images nor requires exactly 81 holdouts."""
    cameras: tuple[TrainingCamera, ...] = (_camera(0), _camera(1))
    wide_rgb_nhw3: torch.Tensor = torch.stack(
        (
            torch.full((16, 16, 3), 64, dtype=torch.uint8),
            torch.full((16, 16, 3), 192, dtype=torch.uint8),
        )
    )
    cache = GpuTrainingCache(
        wide_rgb_nhw3=wide_rgb_nhw3,
        wide_depth_nhw=torch.empty((2, 16, 16), dtype=torch.uint16),
        wide_normal_nhw3=torch.empty((2, 16, 16, 3), dtype=torch.uint8),
        uw_rgb_nhw3=torch.empty((0, 16, 16, 3), dtype=torch.uint8),
        uw_depth_nhw=torch.empty((0, 16, 16), dtype=torch.uint16),
        uw_normal_nhw3=torch.empty((0, 16, 16, 3), dtype=torch.uint8),
        cameras=cameras,
        train_indices=torch.empty(0, dtype=torch.int64),
        holdout_indices=(0, 1),
        holdout_sha256="test",
        scene_scale=1.0,
    )

    def fake_render(
        _splats: torch.nn.ParameterDict | dict[str, torch.Tensor],
        camera: TrainingCamera,
        **_kwargs: Any,
    ) -> RenderOutput:
        target: torch.Tensor = cache.wide_rgb_nhw3[camera.cache_index].to(torch.float32) / 255.0
        rgb_hw3: torch.Tensor = (target - 0.1).clamp(0.0, 1.0)
        scalar_hw1: torch.Tensor = torch.ones((16, 16, 1))
        normal_hw3: torch.Tensor = torch.zeros((16, 16, 3))
        return RenderOutput(
            rgb_hw3=rgb_hw3,
            alpha_hw1=scalar_hw1,
            center_depth_hw1=scalar_hw1,
            plane_features_hw4=torch.zeros((16, 16, 4)),
            surface_depth_hw1=scalar_hw1,
            surface_valid_hw1=scalar_hw1.to(torch.bool),
            direct_normal_hw3=normal_hw3,
            depth_normal_hw3=normal_hw3,
            background_3=torch.zeros(3),
            appearance_info={},
        )

    monkeypatch.setattr("gauss_surf.train_gsplat.evaluation.render_splats", fake_render)
    monkeypatch.setattr(
        "gauss_surf.train_gsplat.evaluation.structural_similarity_index_measure",
        lambda *_args, **_kwargs: torch.tensor(0.75),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)

    metrics: dict[str, float] = evaluate_holdout({"means": torch.zeros((1, 3))}, cache)

    assert np.isfinite(metrics["psnr"])
    assert metrics["ssim"] == 0.75
