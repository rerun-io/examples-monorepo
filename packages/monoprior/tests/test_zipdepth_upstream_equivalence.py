"""Prove that the annotated ZipDepth fork stays numerically identical to upstream."""

import importlib.util
import re
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, TypeAlias

import pytest
import torch
from jaxtyping import Float, Num
from torch import nn

from monopriors.third_party.zipdepth.architecture import MODEL_CONFIGS, ZipDepth, create_model
from monopriors.third_party.zipdepth.model_utils import fuse_remaining_conv_bn, strip_state_dict_prefixes

REFERENCE_DIR: Path = Path(__file__).parent / "reference_data" / "zipdepth"
TensorMap: TypeAlias = dict[str, Num[torch.Tensor, "..."]]
GradientMap: TypeAlias = dict[str, Float[torch.Tensor, "..."] | None]


def _load_upstream_module(name: str, path: Path) -> ModuleType:
    """Load one pristine upstream source fixture.

    Args:
        name: Unique module name for the import.
        path: Path to the pristine Python source file.

    Returns:
        The loaded upstream module.
    """
    spec: importlib.machinery.ModuleSpec | None = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load upstream fixture {path}")
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_tensor_maps_equal(left: TensorMap, right: TensorMap) -> None:
    """Assert exact equality for two named tensor mappings.

    Args:
        left: First mapping of names to numeric tensors of arbitrary shape.
        right: Second mapping of names to numeric tensors of arbitrary shape.
    """
    assert list(left) == list(right)
    for name, left_tensor in left.items():
        right_tensor: Num[torch.Tensor, "..."] = right[name]
        assert torch.equal(left_tensor, right_tensor), name


def test_annotated_model_matches_upstream_through_training_and_fusion() -> None:
    upstream_architecture: ModuleType = _load_upstream_module("zipdepth_upstream_architecture", REFERENCE_DIR / "upstream_architecture.py")
    upstream_model_utils: ModuleType = _load_upstream_module("zipdepth_upstream_model_utils", REFERENCE_DIR / "upstream_model_utils.py")
    upstream_create_model: Callable[..., nn.Module] = upstream_architecture.create_model
    upstream_fuse_remaining_conv_bn: Callable[[nn.Module], int] = upstream_model_utils.fuse_remaining_conv_bn

    torch.manual_seed(41)
    upstream_model: Any = upstream_create_model(variant="small")
    ours_model: ZipDepth = create_model(variant="small")
    upstream_state_dict: TensorMap = dict(upstream_model.state_dict())
    ours_state_dict: TensorMap = dict(ours_model.state_dict())
    assert list(upstream_state_dict) == list(ours_state_dict)
    ours_model.load_state_dict(upstream_state_dict, strict=True)

    upstream_model.train()
    ours_model.train()
    train_generator: torch.Generator = torch.Generator().manual_seed(42)
    train_input_b3hw: Float[torch.Tensor, "2 3 96 128"] = torch.rand(2, 3, 96, 128, generator=train_generator)
    upstream_train_output_b1hw: Float[torch.Tensor, "2 1 h_out w_out"] = upstream_model(train_input_b3hw)
    ours_train_output_b1hw: Float[torch.Tensor, "2 1 h_out w_out"] = ours_model(train_input_b3hw)
    assert torch.equal(upstream_train_output_b1hw, ours_train_output_b1hw)

    upstream_train_output_b1hw.mean().backward()
    ours_train_output_b1hw.mean().backward()
    upstream_gradients: GradientMap = {name: parameter.grad for name, parameter in upstream_model.named_parameters()}
    ours_gradients: GradientMap = {name: parameter.grad for name, parameter in ours_model.named_parameters()}
    assert list(upstream_gradients) == list(ours_gradients)
    for name, upstream_gradient in upstream_gradients.items():
        ours_gradient: Float[torch.Tensor, "..."] | None = ours_gradients[name]
        assert (upstream_gradient is None) == (ours_gradient is None), name
        if upstream_gradient is not None and ours_gradient is not None:
            assert torch.equal(upstream_gradient, ours_gradient), name

    upstream_bn_buffers: TensorMap = {
        name: buffer for name, buffer in upstream_model.named_buffers() if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }
    ours_bn_buffers: TensorMap = {
        name: buffer for name, buffer in ours_model.named_buffers() if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }
    _assert_tensor_maps_equal(upstream_bn_buffers, ours_bn_buffers)

    upstream_fused_model: nn.Module = upstream_model.fuse_for_inference()
    ours_fused_model: ZipDepth = ours_model.fuse_for_inference()
    eval_generator: torch.Generator = torch.Generator().manual_seed(43)
    eval_input_b3hw: Float[torch.Tensor, "1 3 101 97"] = torch.rand(1, 3, 101, 97, generator=eval_generator)
    with torch.no_grad():
        upstream_fused_output_b1hw: Float[torch.Tensor, "1 1 h_out w_out"] = upstream_fused_model(eval_input_b3hw)
        ours_fused_output_b1hw: Float[torch.Tensor, "1 1 h_out w_out"] = ours_fused_model(eval_input_b3hw)
    assert torch.equal(upstream_fused_output_b1hw, ours_fused_output_b1hw)

    upstream_fused_count: int = upstream_fuse_remaining_conv_bn(upstream_fused_model)
    ours_fused_count: int = fuse_remaining_conv_bn(ours_fused_model)
    assert upstream_fused_count == ours_fused_count
    with torch.no_grad():
        upstream_remaining_fused_output_b1hw: Float[torch.Tensor, "1 1 h_out w_out"] = upstream_fused_model(eval_input_b3hw)
        ours_remaining_fused_output_b1hw: Float[torch.Tensor, "1 1 h_out w_out"] = ours_fused_model(eval_input_b3hw)
    assert torch.equal(upstream_remaining_fused_output_b1hw, ours_remaining_fused_output_b1hw)


def test_state_dict_prefix_stripping_matches_upstream() -> None:
    upstream_model_utils: ModuleType = _load_upstream_module("zipdepth_upstream_model_utils_prefixes", REFERENCE_DIR / "upstream_model_utils.py")
    upstream_strip_state_dict_prefixes: Callable[[TensorMap], TensorMap] = upstream_model_utils.strip_state_dict_prefixes
    prefixed_state_dict: TensorMap = {
        "module._orig_mod.encoder.weight": torch.tensor([1.0, 2.0]),
        "_orig_mod.module.decoder.bias": torch.tensor([3.0]),
    }
    upstream_stripped_state_dict: TensorMap = upstream_strip_state_dict_prefixes(prefixed_state_dict)
    ours_stripped_state_dict: TensorMap = strip_state_dict_prefixes(prefixed_state_dict)
    _assert_tensor_maps_equal(upstream_stripped_state_dict, ours_stripped_state_dict)


def test_unknown_variants_raise_explicit_errors() -> None:
    expected_message: str = f"unknown ZipDepth variant 'mystery'; expected one of {sorted(MODEL_CONFIGS)}"
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        ZipDepth(variant="mystery")
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        create_model(variant="ZIP_MYSTERY")
