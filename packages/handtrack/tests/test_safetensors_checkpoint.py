"""DetNet-F as safetensors: the weights file gives the network the torch weights give, and the training state resumes SGD exactly."""
from pathlib import Path

import pytest
import torch
from torch import Tensor

from handtrack.models.detnet import DetNetF, DetNetOutput
from handtrack.train.checkpoint import TrainingState, load_checkpoint, save_checkpoint
from handtrack.train.safetensors_checkpoint import ModelCard, load_model, load_training_state, save_model, save_training_state

CARD: ModelCard = ModelCard(version="mix1-upright", input_convention="upright", run_id="robocap-mix-1", step=2, git_sha="e388ad89b",
                            source_sha256="0" * 64, eval={"dataforge-umetrack/recall@0.5": 0.5})


def batch(seed: int) -> Tensor:
    return torch.rand(4, 1, 120, 160, generator=torch.Generator().manual_seed(seed))


def sgd_step(model: DetNetF, optimiser: torch.optim.SGD, pooled: Tensor) -> None:
    """One training step; train mode also moves the BatchNorm running statistics."""
    model.train()
    optimiser.zero_grad(set_to_none=True)
    output: DetNetOutput = model.forward_pooled(pooled)
    (output.center.square().sum() + output.radius.sum() + output.presence_logit.sum()).backward()
    optimiser.step()


def trained(steps: int) -> tuple[DetNetF, torch.optim.SGD]:
    torch.manual_seed(0)
    model: DetNetF = DetNetF()
    optimiser: torch.optim.SGD = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for step in range(steps):
        sgd_step(model, optimiser, batch(step))
    return model, optimiser


def fresh() -> tuple[DetNetF, torch.optim.SGD]:
    torch.manual_seed(1)  # other initial weights: everything must come from the file
    model: DetNetF = DetNetF()
    return model, torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.1)


def test_the_weights_file_holds_batchnorm_buffers_and_gives_the_same_outputs(tmp_path: Path) -> None:
    model, _ = trained(2)
    save_model(tmp_path / "model.safetensors", model, CARD)
    loaded, _ = fresh()
    assert load_model(tmp_path / "model.safetensors", loaded) == CARD
    for name, tensor in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor), name
    model.eval()
    loaded.eval()
    with torch.no_grad():
        want: DetNetOutput = model.forward_pooled(batch(9))
        got: DetNetOutput = loaded.forward_pooled(batch(9))
    assert torch.equal(got.center, want.center) and torch.equal(got.radius, want.radius) and torch.equal(got.presence_logit, want.presence_logit)


def test_a_weights_file_missing_a_tensor_is_refused(tmp_path: Path) -> None:
    model, _ = trained(0)
    save_model(tmp_path / "model.safetensors", model, CARD)
    from safetensors.torch import load_file, save_file

    tensors: dict[str, Tensor] = load_file(tmp_path / "model.safetensors")
    del tensors["backbone.1.running_mean"]
    save_file(tensors, tmp_path / "partial.safetensors", metadata={"handtrack.model_card": "{}"})
    with pytest.raises(ValueError, match="partial.safetensors"):
        load_model(tmp_path / "partial.safetensors", fresh()[0])


def test_resuming_from_the_training_state_takes_the_same_next_step_as_the_torch_checkpoint(tmp_path: Path) -> None:
    model, optimiser = trained(2)
    state: TrainingState = TrainingState(epoch=1, step=2, config_json='{"nets": "detnet"}', epoch_steps={"detnet": 2}, best={"detnet": 0.25})
    save_checkpoint(tmp_path / "last.pt", {"detnet": model}, {"detnet": optimiser}, state)
    save_model(tmp_path / "model.safetensors", model, CARD)
    save_training_state(tmp_path / "training_state.safetensors", model, optimiser, state)

    from_torch, torch_optimiser = fresh()
    load_checkpoint(tmp_path / "last.pt", {"detnet": from_torch}, {"detnet": torch_optimiser})
    from_safetensors, safetensors_optimiser = fresh()
    load_model(tmp_path / "model.safetensors", from_safetensors)
    assert load_training_state(tmp_path / "training_state.safetensors", from_safetensors, safetensors_optimiser) == state

    sgd_step(model, optimiser, batch(2))
    sgd_step(from_torch, torch_optimiser, batch(2))
    sgd_step(from_safetensors, safetensors_optimiser, batch(2))
    for name, tensor in model.state_dict().items():
        assert torch.equal(from_torch.state_dict()[name], tensor), f"torch checkpoint: {name}"
        assert torch.equal(from_safetensors.state_dict()[name], tensor), f"safetensors: {name}"
