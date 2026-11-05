"""SAM 3D Body stage profiling and output widths on the real model (CUDA GPU, gated checkpoint in the Hugging Face cache)."""

from pathlib import Path
from typing import get_args

import pytest
import torch

pytest.importorskip("yacs", reason="SAM 3D Body needs the posekit-body Pixi environment (posekit-sam3d-body feature)")

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from jaxtyping import UInt8
from torch import Tensor

from posekit.models.sam3d_body import (
    MHR_PARAMETER_WIDTHS,
    SAM3D_BODY_FILES,
    SAM3D_BODY_REVISION,
    Sam3dBodyConfig,
    Sam3dBodyPredictor,
    Sam3dBodyReconstruction,
    Sam3dBodyStage,
)
from posekit.predictions import BoxDetections

pytestmark = pytest.mark.integration

WRAPPED: tuple[str, ...] = ("forward_step", "run_keypoint_prompt", "mhr_forward")
EVERY_FRAME: frozenset[str] = frozenset(get_args(Sam3dBodyStage)) - {"decoder_prompt"}
"""Stages of every full-mode frame; upstream runs the keypoint-prompted decoder pass only on some inputs."""


@pytest.fixture(scope="module")
def predictor() -> Sam3dBodyPredictor:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    config: Sam3dBodyConfig = Sam3dBodyConfig(mode="full")
    try:
        checkpoint: Path = Path(
            snapshot_download(repo_id=config.hf_repo_id, revision=SAM3D_BODY_REVISION, allow_patterns=list(SAM3D_BODY_FILES), local_files_only=True)
        )
    except LocalEntryNotFoundError:
        pytest.skip(f"missing SAM 3D Body checkpoint {config.hf_repo_id}@{SAM3D_BODY_REVISION} in the Hugging Face cache")
    return Sam3dBodyConfig(mode="full", checkpoint_dir=checkpoint).setup()


@pytest.fixture(scope="module")
def frame() -> UInt8[Tensor, "1 h w 3"]:
    generator: torch.Generator = torch.Generator().manual_seed(0)
    return torch.randint(0, 256, (1, 480, 640, 3), dtype=torch.uint8, generator=generator).cuda()


def person(count: int = 1) -> BoxDetections:
    return BoxDetections(
        torch.tensor([[160.0, 40.0, 480.0, 440.0]] * count, device="cuda"),
        torch.ones(count, device="cuda"),
        torch.zeros(count, dtype=torch.int64, device="cuda"),
    )


def instrumented(predictor: Sam3dBodyPredictor) -> bool:
    """Whether any stage wrapper or hook is installed on the model."""
    model: torch.nn.Module = predictor.estimator.model
    wrappers: bool = any(name in vars(owner) for owner in (model, model.head_pose, model.head_pose_hand) for name in WRAPPED)
    modules: tuple[torch.nn.Module, ...] = tuple(model.get_submodule(name) for name in ("backbone", "decoder", "decoder_hand"))
    return wrappers or any(module._forward_pre_hooks or module._forward_hooks for module in modules)


def test_stage_profiling_instruments_the_model_only_while_on(predictor: Sam3dBodyPredictor, frame: Tensor) -> None:
    with torch.inference_mode():
        predictor(frame, person())  # the first full-mode call differs slightly from every later one
        assert not instrumented(predictor)
        plain: Sam3dBodyReconstruction = predictor(frame, person())
        assert predictor.stage_ms == {} and not instrumented(predictor)

        predictor.profile_stages = True
        try:
            profiled: Sam3dBodyReconstruction = predictor(frame, person())
        finally:
            predictor.profile_stages = False
    assert not instrumented(predictor)
    assert EVERY_FRAME <= set(predictor.stage_ms) <= set(get_args(Sam3dBodyStage))
    assert all(len(values) == 1 for values in predictor.stage_ms.values())
    assert predictor.stage_ms["total"][0] > 0.0
    torch.testing.assert_close(profiled.vertices, plain.vertices, atol=1e-5, rtol=0)
    predictor.stage_ms.clear()


def test_mhr_widths_match_the_model(predictor: Sam3dBodyPredictor, frame: Tensor) -> None:
    with torch.inference_mode():
        people: Sam3dBodyReconstruction = predictor(frame, person(2))
        nobody: Sam3dBodyReconstruction = predictor(frame, BoxDetections.empty("cuda"))
    assert people.num_people == 2 and nobody.num_people == 0
    for name, width in MHR_PARAMETER_WIDTHS.items():
        assert tuple(getattr(people, name).shape) == (2, width), name
    for row, empty in ((people, nobody), (people.keypoints, nobody.keypoints)):
        for name in type(row).__dataclass_fields__:
            value = getattr(row, name)
            if isinstance(value, Tensor) and name != "faces":
                assert getattr(empty, name).shape[1:] == value.shape[1:], name
    assert torch.equal(people.take(torch.tensor([1], device="cuda")).vertices, people.vertices[1:])
