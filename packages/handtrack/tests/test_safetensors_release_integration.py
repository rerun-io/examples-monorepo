"""The DetNet-F safetensors in the robocap-live sample bundle against the torch files they came from: the same outputs, the same
next SGD step on resume, and ONNX graphs exported from them that match the bundle's.

Assets: the bundle (``pixi run -e robocap-live robocap-live-download-sample``, or ``HANDTRACK_DETNET_RELEASE`` = a
``models/detnet`` directory) and the torch weights under ``~/handtrack-data``. Each test skips with the missing file named.
"""
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from torch import Tensor

from handtrack.apis.export_detnet import ExportDetNetConfig
from handtrack.apis.export_detnet import main as export_detnet
from handtrack.data.array_table import write_arrays
from handtrack.models.detnet import DetNetF, DetNetOutput
from handtrack.train.checkpoint import TrainingState, load_checkpoint
from handtrack.train.safetensors_checkpoint import ModelCard, file_sha256, load_model, load_training_state

pytestmark = pytest.mark.integration

RELEASE: Path = Path(os.environ.get("HANDTRACK_DETNET_RELEASE", Path(__file__).parents[2] / "robocap-live/data/robocap-live-sample/models/detnet"))
DATA: Path = Path.home() / "handtrack-data"
SOURCES: dict[str, tuple[Path, Path | None]] = {
    "mix1-upright": (DATA / "recipe/runs/robocap-mix-1/detnet/last.weights.pt", DATA / "recipe/runs/robocap-mix-1/detnet/last.pt"),
    "0929": (DATA / "checkpoints/detnet-demo-0929.weights.pt", None),
}


def need(*paths: Path) -> None:
    for path in paths:
        if not path.exists():
            pytest.skip(f"asset absent: {path}")


def pooled(seed: int, count: int = 8) -> Tensor:
    return torch.rand(count, 1, 120, 160, generator=torch.Generator().manual_seed(seed))


def outputs(model: DetNetF, batch: Tensor) -> list[Tensor]:
    with torch.no_grad():
        out: DetNetOutput = model.eval().forward_pooled(batch)
    return [out.center, out.radius, out.presence_logit]


@pytest.mark.parametrize("version", sorted(SOURCES))
def test_the_safetensors_weights_give_the_torch_weights_outputs(version: str) -> None:
    weights, _ = SOURCES[version]
    need(RELEASE / version / "model.safetensors", weights)
    from_torch: DetNetF = DetNetF()
    from_torch.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    from_safetensors: DetNetF = DetNetF()
    card: ModelCard = load_model(RELEASE / version / "model.safetensors", from_safetensors)
    assert (card.version, card.source_sha256) == (version, file_sha256(weights))
    for want, got in zip(outputs(from_torch, pooled(0)), outputs(from_safetensors, pooled(0)), strict=True):
        assert torch.equal(got, want)


def test_resuming_mix1_from_its_training_state_takes_the_same_next_step_as_last_pt() -> None:
    _, checkpoint = SOURCES["mix1-upright"]
    assert checkpoint is not None
    need(RELEASE / "mix1-upright/model.safetensors", RELEASE / "mix1-upright/training_state.safetensors", checkpoint)
    models: list[DetNetF] = [DetNetF(), DetNetF()]
    optimisers: list[torch.optim.SGD] = [torch.optim.SGD(model.parameters(), lr=1.0, momentum=0.5) for model in models]
    state: TrainingState = load_checkpoint(checkpoint, {"detnet": models[0]}, {"detnet": optimisers[0]})
    load_model(RELEASE / "mix1-upright/model.safetensors", models[1])
    assert load_training_state(RELEASE / "mix1-upright/training_state.safetensors", models[1], optimisers[1]) == state
    assert optimisers[1].param_groups[0]["lr"] == optimisers[0].param_groups[0]["lr"]
    for model, optimiser in zip(models, optimisers, strict=True):
        optimiser.param_groups[0]["lr"] = 1e-3  # the cosine ended at 0, where every next step is a no-op; a fine-tune's rate
        model.train()
        optimiser.zero_grad(set_to_none=True)
        out: DetNetOutput = model.forward_pooled(pooled(1))
        (out.center.square().sum() + out.radius.sum() + out.presence_logit.sum()).backward()
        optimiser.step()
    for name, tensor in models[0].state_dict().items():
        assert torch.equal(models[1].state_dict()[name], tensor), name


@pytest.mark.parametrize("version", sorted(SOURCES))
def test_onnx_exported_from_the_safetensors_matches_the_bundles_onnx(version: str, tmp_path: Path) -> None:
    released: Path = RELEASE / version
    need(released / "model.safetensors", released / "detnet_full.onnx", released / "detnet_pooled_b1.onnx")
    frames: np.ndarray = (pooled(2).numpy()[:, 0] * 255).astype(np.uint8)
    write_arrays(tmp_path / "heldout.parquet", {"pooled_u8": frames, "dataset": np.zeros(len(frames), dtype=np.int64)})
    export_detnet(ExportDetNetConfig(weights=released / "model.safetensors", heldout=tmp_path / "heldout.parquet", out_dir=tmp_path / "onnx", threads=4))
    feed: np.ndarray = (frames.astype(np.float32) / 255.0)[:, None]
    feeds: dict[str, np.ndarray] = {"detnet_pooled_b1.onnx": feed[:1], "detnet_full.onnx": np.repeat(np.repeat(feed, 4, axis=2), 4, axis=3)}
    for name, image in feeds.items():
        ours = ort.InferenceSession(str(tmp_path / "onnx" / name), providers=["CPUExecutionProvider"]).run(None, {"image": image})
        theirs = ort.InferenceSession(str(released / name), providers=["CPUExecutionProvider"]).run(None, {"image": image})
        for a, b in zip(ours, theirs, strict=True):
            assert np.abs(a - b).max() <= 1e-5, f"{version} {name}: max |ours - released| = {np.abs(a - b).max()}"
