"""EgoHand files come from pinned Hugging Face revisions unless a local path overrides them."""

from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("egohand_inference", reason="EgoHand adapters require the optional posekit-egohand feature (egohand-dev environment)")

from posekit.models import egohand
from posekit.models.egohand import EgoHandConfig, EgoHandDetectorConfig


def test_the_detector_downloads_its_pinned_onnx(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, str]] = []
    monkeypatch.setattr(egohand, "hf_hub_download", lambda **kwargs: calls.append(kwargs) or "/cache/hand_detector.onnx")
    assert EgoHandDetectorConfig().weights_path() == Path("/cache/hand_detector.onnx")
    assert calls == [{"repo_id": "nvidia/RT-DETR-Hand-Detector-v1", "filename": "hand_detector.onnx", "revision": "55572ddc93a5916c3e486811dc6e62bb2b7bdb9d"}]


def test_egohand_downloads_its_pinned_safetensors_and_soma_files(monkeypatch: pytest.MonkeyPatch) -> None:
    files: list[dict[str, str]] = []
    snapshots: list[dict[str, Any]] = []
    monkeypatch.setattr(egohand, "hf_hub_download", lambda **kwargs: files.append(kwargs) or "/cache/egohand.safetensors")
    monkeypatch.setattr(egohand, "snapshot_download", lambda **kwargs: snapshots.append(kwargs) or "/cache/soma")
    config: EgoHandConfig = EgoHandConfig()
    assert config.weights_path() == Path("/cache/egohand.safetensors")
    assert config.soma_path() == Path("/cache/soma")
    assert files == [{"repo_id": "pablovela5620/egohand", "filename": "egohand_s060000.safetensors", "revision": "24a325fde719fdb985c7fb4606115b0c1912ab7d"}]
    assert snapshots == [
        {
            "repo_id": "nvidia/SOMA-X",
            "revision": "f81a01796f652260ff930d5318d6c6b845a505ed",
            "allow_patterns": [
                "SOMAHand.npz", "SOMA_neutral.npz", "SOMA_procedural_transforms.json", "SOMA_template_rig.usda",
                "MHR/mhr_model_lod1.pt", "MHR/base_body_lod1.obj", "MHR/SOMA_wrap_lod1.obj",
            ],
        }
    ]


def test_local_paths_override_the_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(**_: str) -> str:
        raise AssertionError("a local path must not download")

    monkeypatch.setattr(egohand, "hf_hub_download", refuse)
    monkeypatch.setattr(egohand, "snapshot_download", refuse)
    assert EgoHandDetectorConfig(Path("/local/detector.onnx")).weights_path() == Path("/local/detector.onnx")
    config: EgoHandConfig = EgoHandConfig(Path("/local/egohand.safetensors"), Path("/local/soma"))
    assert (config.weights_path(), config.soma_path()) == (Path("/local/egohand.safetensors"), Path("/local/soma"))


def test_the_model_loads_from_safetensors_with_its_cfg_in_the_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No torch.load: the cfg is YAML metadata, the backbone is skipped when an engine replaces it."""
    from egohand.models import egohand as upstream
    from egohand_inference.pipeline import predictor

    built: list[Any] = []

    class Model(torch.nn.Module):
        def __init__(self, cfg: Any) -> None:
            super().__init__()
            built.append(cfg)
            self.head = torch.nn.Parameter(torch.zeros(2))

        def validate_camera_depth_checkpoint(self, checkpoint: Any) -> None:
            built.append(checkpoint["hyper_parameters"]["cfg"].camera_depth_mode)

    def refuse(*_: Any, **__: Any) -> None:
        raise AssertionError("no pickle")

    monkeypatch.setattr(upstream, "EgoHand", Model)
    monkeypatch.setattr(predictor.torch, "load", refuse)
    path: Path = tmp_path / "weights.safetensors"
    cfg: str = "_target_: egohand.models.egohand.EgoHand\ncamera_depth_mode: log_lambda\nbackbone_weights: gone\nsoma_hand:\n  data_root: gone\n"
    save_file({"head": torch.tensor([1.0, 2.0]), "backbone.layer": torch.ones(3)}, path, metadata={"cfg": cfg})

    engine: torch.nn.Module = torch.nn.Identity()
    model, target = predictor.load_egohand_model(path, tmp_path / "soma", backbone=engine)

    assert target == "egohand.models.egohand.EgoHand"
    assert isinstance(model, Model)
    assert model.backbone is engine and not model.training
    assert built[0].backbone_weights is None and built[0].soma_hand.data_root == str(tmp_path / "soma")
    assert built[0].backbone == {"_target_": "torch.nn.Identity"}
    assert built[1] == "log_lambda"
    assert model.head.tolist() == [1.0, 2.0]
