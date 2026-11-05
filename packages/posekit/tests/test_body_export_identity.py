"""Alternate checkpoints must not use exports named for the pinned weights."""

from pathlib import Path

import pytest

from posekit.models.sam3d_body import SAM3D_BODY_REVISION
from posekit.models.sam3d_body_fast import Sam3dBodyFastConfig


@pytest.mark.parametrize(
    "checkpoint,repo,revision",
    [
        (Path("/alternate-weights"), "facebook/sam-3d-body-dinov3", SAM3D_BODY_REVISION),
        (None, "example/alternate-body", SAM3D_BODY_REVISION),
        (None, "facebook/sam-3d-body-dinov3", "different-revision"),
    ],
)
def test_alternate_checkpoint_requires_explicit_exports(checkpoint: Path | None, repo: str, revision: str) -> None:
    with pytest.raises(ValueError, match="checkpoint.*export"):
        Sam3dBodyFastConfig(checkpoint_dir=checkpoint, hf_repo_id=repo, hf_revision=revision, backbone_precision="bf16")


def test_alternate_checkpoint_accepts_explicit_exports_or_torch() -> None:
    Sam3dBodyFastConfig(checkpoint_dir=Path("/alternate-weights"), backbone_onnx=None, decoder_onnx_dir=None)
    Sam3dBodyFastConfig(
        checkpoint_dir=Path("/alternate-weights"),
        backbone_precision="bf16",
        backbone_onnx=Path("/custom/quantized.onnx"),
        backbone_source_onnx=Path("/custom/source.onnx"),
        decoder_onnx_dir=Path("/custom/decoders"),
    )
