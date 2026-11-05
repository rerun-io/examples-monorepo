"""Alternate checkpoints must not use exports named for the pinned weights."""
from dataclasses import replace
from pathlib import Path

import pytest

from posekit.models.sam3d_body_fast import Sam3dBodyFastConfig


@pytest.mark.parametrize('selector', [
    {'checkpoint_dir': Path('/alternate-weights')},
    {'hf_repo_id': 'example/alternate-body'},
    {'hf_revision': 'different-revision'},
])
def test_alternate_checkpoint_requires_explicit_exports(selector: dict[str, str | Path]) -> None:
    with pytest.raises(ValueError, match='checkpoint.*export'):
        replace(Sam3dBodyFastConfig(backbone_precision='bf16'), **selector)

def test_alternate_checkpoint_accepts_explicit_exports_or_torch() -> None:
    Sam3dBodyFastConfig(checkpoint_dir=Path('/alternate-weights'), backbone_onnx=None, decoder_onnx_dir=None)
    Sam3dBodyFastConfig(checkpoint_dir=Path('/alternate-weights'), backbone_precision='bf16', backbone_onnx=Path('/custom/quantized.onnx'),
        backbone_source_onnx=Path('/custom/source.onnx'), decoder_onnx_dir=Path('/custom/decoders'))
