"""Portable checkpoints of one network as safetensors: the weights with a model card, and the SGD state for an exact resume.

``model.safetensors`` is the ``state_dict`` (parameters and BatchNorm buffers, ``num_batches_tracked`` included) with a
:class:`ModelCard` as JSON under the ``handtrack.model_card`` metadata key. Fine-tuning needs this file alone.

``training_state.safetensors`` holds what ``last.pt`` adds to the weights: one ``momentum_buffer/<parameter name>`` tensor per
parameter, the torch RNG states (``rng/cpu``, ``rng/cuda/<i>``), the :class:`~handtrack.train.checkpoint.TrainingState` JSON
(``handtrack.training_state``: step, epoch, history, the run's resolved config) and the SGD hyperparameters (``handtrack.sgd``).
The learning-rate schedule has no state of its own: the trainer computes the rate from the step (``Trainer.schedule_step``).
"""
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from serde import SerdeError, serde
from serde.json import from_json, to_json
from torch import Tensor, nn
from torch.optim import SGD

from handtrack.train.checkpoint import TrainingState

MODEL_CARD_KEY: str = "handtrack.model_card"
TRAINING_STATE_KEY: str = "handtrack.training_state"
SGD_KEY: str = "handtrack.sgd"


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ModelCard:
    """What a weights file is: which version, which input, where it came from and how it scored."""

    version: str
    """The release name, e.g. ``mix1-upright`` or ``0929``."""
    input_convention: Literal["upright", "legacy"]
    """upright: cameras mounted upside down are turned 180 degrees before the net; legacy: every camera as calibrated."""
    run_id: str
    """The training run, e.g. ``robocap-mix-1``."""
    step: int | None
    """Optimiser steps of the weights; None when the run did not record it."""
    git_sha: str | None
    """The commit the run trained from; None when unknown."""
    source_sha256: str
    """sha256 of the torch weights file these tensors were read from."""
    eval: dict[str, float] = field(default_factory=dict)
    """Held-out scores, ``<dataset>/<metric>`` (detector_eval: recall@0.5, recall@0.3, precision)."""
    net: Literal["detnet-f"] = "detnet-f"
    note: str = ""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class SgdState:
    """``torch.optim.SGD``'s one parameter group, without its parameters; the momentum buffers are tensors in the file."""

    lr: float
    """The rate the last step used (the schedule's value at that step)."""
    momentum: float
    dampening: float
    weight_decay: float
    nesterov: bool
    maximize: bool
    parameters: list[str]
    """Parameter names in the optimiser's order (``named_parameters``)."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_model(path: Path, model: nn.Module, card: ModelCard) -> None:
    """Write the model's ``state_dict`` and its card."""
    tensors: dict[str, Tensor] = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
    save_file(tensors, path, metadata={MODEL_CARD_KEY: to_json(card)})


def read_metadata(path: Path) -> dict[str, str]:
    with safe_open(path, framework="pt") as stream:
        return stream.metadata() or {}


def load_model(path: Path, model: nn.Module) -> ModelCard:
    """Load every tensor into ``model`` (strict: a missing or extra name is an error) and return the card."""
    try:
        card: ModelCard = from_json(ModelCard, read_metadata(path)[MODEL_CARD_KEY])
    except (KeyError, SerdeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: no valid {MODEL_CARD_KEY} metadata: {error}") from error
    try:
        model.load_state_dict(load_file(path))
    except RuntimeError as error:  # load_state_dict reports missing, unexpected and mis-shaped tensors this way
        raise ValueError(f"{path}: {error}") from error
    return card


def save_training_state(path: Path, model: nn.Module, optimiser: SGD, state: TrainingState, cuda_rng: list[Tensor] | None = None) -> None:
    """Write the SGD momentum buffers, the RNG states and the training progress of ``model``'s optimiser.

    ``cuda_rng`` defaults to this process's CUDA generators when CUDA is initialised; pass a checkpoint's own to convert it on a
    host without a GPU.
    """
    if len(optimiser.param_groups) != 1:
        raise ValueError("one SGD parameter group expected")
    group: dict = optimiser.param_groups[0]
    names: dict[int, str] = {id(parameter): name for name, parameter in model.named_parameters()}
    order: list[str] = [names[id(parameter)] for parameter in group["params"]]
    tensors: dict[str, Tensor] = {}
    for name, parameter in zip(order, group["params"], strict=True):
        buffer: Tensor | None = optimiser.state.get(parameter, {}).get("momentum_buffer")
        if buffer is not None:
            tensors[f"momentum_buffer/{name}"] = buffer.detach().cpu().contiguous()
    tensors["rng/cpu"] = torch.get_rng_state()
    if cuda_rng is None and torch.cuda.is_initialized():
        cuda_rng = torch.cuda.get_rng_state_all()
    for index, rng in enumerate(cuda_rng or []):
        tensors[f"rng/cuda/{index}"] = rng
    sgd: SgdState = SgdState(float(group["lr"]), float(group["momentum"]), float(group["dampening"]), float(group["weight_decay"]),
                             bool(group["nesterov"]), bool(group.get("maximize", False)), order)
    save_file(tensors, path, metadata={TRAINING_STATE_KEY: to_json(state), SGD_KEY: to_json(sgd)})


def load_training_state(path: Path, model: nn.Module, optimiser: SGD) -> TrainingState:
    """Restore ``optimiser`` (built over ``model.parameters()``) and the RNG states; return the progress to resume from."""
    metadata: dict[str, str] = read_metadata(path)
    try:
        state: TrainingState = from_json(TrainingState, metadata[TRAINING_STATE_KEY])
        sgd: SgdState = from_json(SgdState, metadata[SGD_KEY])
    except (KeyError, SerdeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: no valid training-state metadata: {error}") from error
    names: list[str] = [name for name, _ in model.named_parameters()]
    if names != sgd.parameters:
        raise ValueError(f"{path}: the optimiser's parameters differ from this model's")
    tensors: dict[str, Tensor] = load_file(path)
    optimiser.load_state_dict({
        "state": {index: {"momentum_buffer": tensors[f"momentum_buffer/{name}"]} for index, name in enumerate(names)
                  if f"momentum_buffer/{name}" in tensors},
        "param_groups": [{"lr": sgd.lr, "momentum": sgd.momentum, "dampening": sgd.dampening, "weight_decay": sgd.weight_decay,
                          "nesterov": sgd.nesterov, "maximize": sgd.maximize, "foreach": None, "differentiable": False, "fused": None,
                          "params": list(range(len(names)))}],
    })
    torch.set_rng_state(tensors["rng/cpu"])
    cuda: list[Tensor] = [tensors[f"rng/cuda/{index}"] for index in range(len(tensors)) if f"rng/cuda/{index}" in tensors]
    if cuda and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda)
    return state

