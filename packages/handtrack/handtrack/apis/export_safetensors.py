"""Write one DetNet-F release as safetensors: ``model.safetensors`` (weights and model card) and, from the run's ``last.pt``,
``training_state.safetensors`` (SGD momentum, RNG, progress) for an exact resume.

The card's eval numbers come from a ``detector_eval.json`` (``handtrack.apis.detector_eval``): recall at IoU 0.5 and 0.3 and
precision of the named model, per dataset, both hands.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from serde.json import from_json

from handtrack.apis.detector_eval import DetectorEvalReport
from handtrack.apis.export_nets_onnx import read_checked_state
from handtrack.eval.boxes import BoxCounts
from handtrack.models.detnet import DetNetF
from handtrack.train.checkpoint import TrainingState, load_checkpoint
from handtrack.train.safetensors_checkpoint import ModelCard, save_model, save_training_state


@dataclass(frozen=True, slots=True)
class ExportSafetensorsConfig:
    """Convert a DetNet-F ``*.weights.pt`` (with its ``.sha256`` sidecar) and optionally the run's ``last.pt``."""

    weights: Path
    out_dir: Path
    """Receives ``model.safetensors`` and, with ``checkpoint``, ``training_state.safetensors``."""
    version: str
    """Release name, e.g. ``mix1-upright``."""
    input_convention: Literal["upright", "legacy"]
    run_id: str
    checkpoint: Path | None = None
    """The run's ``last.pt``; its DetNet weights must equal ``weights``."""
    step: int | None = None
    """Steps of ``weights``; default the checkpoint's step, else unknown."""
    git_sha: str | None = None
    eval_json: Path | None = None
    eval_model: str = ""
    """The model's name in ``eval_json``."""
    note: str = ""


def eval_numbers(path: Path, model: str) -> dict[str, float]:
    report: DetectorEvalReport = from_json(DetectorEvalReport, path.read_text())
    numbers: dict[str, float] = {}
    for dataset, result in report.datasets.items():
        if model not in result.models:
            raise ValueError(f"{path}: no model {model!r} in {dataset} (has {sorted(result.models)})")
        both: BoxCounts = result.models[model].both
        for metric, value in (("recall@0.5", both.recall), ("recall@0.3", both.recall_at(0.3)), ("precision", both.precision)):
            if value is not None:
                numbers[f"{dataset}/{metric}"] = round(value, 4)
    return numbers


def main(config: ExportSafetensorsConfig) -> ModelCard:
    state, sha256 = read_checked_state(config.weights)
    model: DetNetF = DetNetF()
    model.load_state_dict(state)
    step: int | None = config.step
    progress: TrainingState | None = None
    cuda_rng: list[torch.Tensor] = []
    optimiser: torch.optim.SGD = torch.optim.SGD(model.parameters(), lr=1.0, momentum=0.9)  # load_checkpoint restores the group
    if config.checkpoint is not None:
        resumed: DetNetF = DetNetF()
        optimiser = torch.optim.SGD(resumed.parameters(), lr=1.0, momentum=0.9)
        progress = load_checkpoint(config.checkpoint, {"detnet": resumed}, {"detnet": optimiser})
        cuda_rng = torch.load(config.checkpoint, map_location="cpu", weights_only=True)["cuda_rng"]
        for name, tensor in resumed.state_dict().items():
            if not torch.equal(tensor, state[name]):
                raise ValueError(f"{config.checkpoint}: {name} differs from {config.weights}")
        model = resumed
        step = progress.step if step is None else step
    numbers: dict[str, float] = eval_numbers(config.eval_json, config.eval_model) if config.eval_json is not None else {}
    card: ModelCard = ModelCard(config.version, config.input_convention, config.run_id, step, config.git_sha, sha256, numbers, note=config.note)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    save_model(config.out_dir / "model.safetensors", model, card)
    if progress is not None:
        save_training_state(config.out_dir / "training_state.safetensors", model, optimiser, progress, cuda_rng=cuda_rng)
    print(f"{config.out_dir}: {config.version} step {step}, {len(numbers)} eval numbers, training state: {progress is not None}")
    return card
