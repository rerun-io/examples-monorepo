"""Prove that the owned Fast-FoundationStereo inference fork stays numerically equivalent to upstream."""

import importlib.util
import pickle
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
import torch
from conftest import requires_cuda, slow_cuda
from jaxtyping import Float32
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from monopriors.models.stereo_depth.fast_foundationstereo import download_fast_foundationstereo_checkpoint
from monopriors.third_party.fast_foundationstereo import extractor as owned_extractor
from monopriors.third_party.fast_foundationstereo import foundation_stereo as owned_foundation_stereo
from monopriors.third_party.fast_foundationstereo import submodule as owned_submodule
from monopriors.third_party.fast_foundationstereo.foundation_stereo import FastFoundationStereo, normalize_image

REFERENCE_DIR: Path = Path(__file__).parent / "reference_data" / "fast_foundationstereo"
UPSTREAM_PACKAGE: str = "core"
OWNED_PACKAGE: str = "monopriors.third_party.fast_foundationstereo"
TIMM_CREATE_MODEL: Callable[..., nn.Module] = cast(Callable[..., nn.Module], owned_extractor.timm.create_model)


def _load_upstream_module(qualified_name: str, fixture_name: str) -> ModuleType:
    """Load one pristine upstream source fixture as a synthetic package module.

    Args:
        qualified_name: Synthetic fully qualified module name.
        fixture_name: Upstream module basename without the ``upstream_`` prefix.

    Returns:
        The loaded upstream module.
    """
    path: Path = REFERENCE_DIR / f"upstream_{fixture_name}.py"
    spec: importlib.machinery.ModuleSpec | None = importlib.util.spec_from_file_location(qualified_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load upstream fixture {path}")
    module: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


def _load_upstream_package() -> dict[str, ModuleType]:
    """Load the mutually importing pristine fixtures under the upstream package names.

    Returns:
        Loaded upstream modules keyed by source basename.
    """
    package: ModuleType = ModuleType(UPSTREAM_PACKAGE)
    package.__path__ = [str(REFERENCE_DIR)]
    sys.modules[UPSTREAM_PACKAGE] = package

    utils_package: ModuleType = ModuleType(f"{UPSTREAM_PACKAGE}.utils")
    utils_package.__path__ = [str(REFERENCE_DIR)]
    sys.modules[f"{UPSTREAM_PACKAGE}.utils"] = utils_package
    modules: dict[str, ModuleType] = {
        "utils": _load_upstream_module(f"{UPSTREAM_PACKAGE}.utils.utils", "utils"),
    }

    upstream_globals: ModuleType = ModuleType("Utils")
    upstream_globals.AMP_DTYPE = torch.float16
    sys.modules["Utils"] = upstream_globals

    for name in ("submodule", "extractor", "update", "geometry", "foundation_stereo", "distill_block"):
        modules[name] = _load_upstream_module(f"{UPSTREAM_PACKAGE}.{name}", name)
    return modules


def _inference_config() -> DictConfig:
    """Build the released YAML configuration with small, float32 test dimensions.

    Returns:
        OmegaConf configuration accepted by the pristine and owned constructors.
    """
    return OmegaConf.create(
        {
            "corr_levels": 2,
            "corr_radius": 4,
            "hidden_dims": [128],
            "low_memory": 0,
            "max_disp": 64,
            "mixed_precision": False,
            "n_downsample": 2,
            "n_gru_layers": 1,
            "slow_fast_gru": False,
            "valid_iters": 4,
            "vit_size": "vitl",
            "normalize": True,
            "image_size": [64, 96],
            "cv_group": 8,
        }
    )


def _disable_pretrained_and_compilation(monkeypatch: pytest.MonkeyPatch, upstream_modules: dict[str, ModuleType]) -> None:
    """Keep the CPU equivalence test offline and avoid TorchDynamo startup cost.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        upstream_modules: Loaded pristine fixture modules.
    """

    def create_model_without_pretrained(model_name: str, *args: Any, **kwargs: Any) -> nn.Module:
        kwargs["pretrained"] = False
        return TIMM_CREATE_MODEL(model_name, *args, **kwargs)

    monkeypatch.setattr(owned_extractor.timm, "create_model", create_model_without_pretrained)
    upstream_submodule: ModuleType = upstream_modules["submodule"]
    upstream_foundation: ModuleType = upstream_modules["foundation_stereo"]
    for function_name, eager_name in (
        ("build_gwc_volume_optimized_pytorch1", "build_gwc_volume_optimized_pytorch1_eager"),
        ("build_concat_volume_optimized_pytorch1", "build_concat_volume_optimized_pytorch1_eager"),
    ):
        upstream_compiled: Any = getattr(upstream_submodule, function_name)
        monkeypatch.setattr(upstream_foundation, function_name, upstream_compiled._torchdynamo_orig_callable)
        monkeypatch.setattr(owned_foundation_stereo, function_name, getattr(owned_submodule, eager_name))


def _build_equal_models(monkeypatch: pytest.MonkeyPatch) -> tuple[nn.Module, FastFoundationStereo]:
    """Build pristine and owned config models with one identical state dict.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Evaluation-mode upstream and owned models with identical parameters.
    """
    upstream_modules: dict[str, ModuleType] = _load_upstream_package()
    _disable_pretrained_and_compilation(monkeypatch, upstream_modules)
    upstream_factory: Callable[[DictConfig], nn.Module] = cast(
        Callable[[DictConfig], nn.Module], upstream_modules["foundation_stereo"].FastFoundationStereo
    )

    torch.manual_seed(41)
    upstream_model: nn.Module = upstream_factory(_inference_config()).eval()
    torch.manual_seed(41)
    owned_model: FastFoundationStereo = FastFoundationStereo(_inference_config()).eval()
    upstream_state: dict[str, Tensor] = dict(upstream_model.state_dict())
    assert list(upstream_state) == list(owned_model.state_dict())
    owned_model.load_state_dict(upstream_state, strict=True)
    return upstream_model, owned_model


def test_config_model_matches_upstream_forward_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """The owned config architecture is bit-identical for iterative and hierarchical inference."""
    upstream_model: nn.Module
    owned_model: FastFoundationStereo
    upstream_model, owned_model = _build_equal_models(monkeypatch)
    generator: torch.Generator = torch.Generator().manual_seed(42)
    left_13hw: Float32[Tensor, "1 3 64 96"] = torch.rand((1, 3, 64, 96), generator=generator, dtype=torch.float32) * 255.0
    right_13hw: Float32[Tensor, "1 3 64 96"] = torch.rand((1, 3, 64, 96), generator=generator, dtype=torch.float32) * 255.0

    with torch.inference_mode():
        for iterations in (1, 4):
            upstream_disparity_11hw: Float32[Tensor, "1 1 64 96"] = upstream_model(
                left_13hw,
                right_13hw,
                iters=iterations,
                test_mode=True,
                optimize_build_volume="pytorch1",
            )
            owned_disparity_11hw: Float32[Tensor, "1 1 64 96"] = owned_model(
                left_13hw,
                right_13hw,
                iters=iterations,
                test_mode=True,
                optimize_build_volume="pytorch1",
            )
            assert torch.equal(upstream_disparity_11hw, owned_disparity_11hw)

        upstream_hierarchical_11hw: Float32[Tensor, "1 1 64 96"] = upstream_model.run_hierachical(
            left_13hw,
            right_13hw,
            iters=1,
            test_mode=True,
        )
        owned_hierarchical_11hw: Float32[Tensor, "1 1 64 96"] = owned_model.run_hierachical(
            left_13hw,
            right_13hw,
            iters=1,
            test_mode=True,
        )
    assert torch.equal(upstream_hierarchical_11hw, owned_hierarchical_11hw)


def _pickle_module(target_package: str) -> type:
    """Build a torch-load pickle adapter that targets pristine or owned modules.

    Args:
        target_package: ``core`` for pristine fixtures or the owned package path.

    Returns:
        Module-like class accepted by ``torch.load``.
    """

    class RemappingUnpickler(pickle.Unpickler):
        """Remap both historical checkpoint package prefixes."""

        def find_class(self, module: str, name: str) -> Any:
            if module.startswith("core."):
                module = f"{target_package}.{module.removeprefix('core.')}"
            elif module.startswith("foundation_stereo_ori."):
                module = f"{target_package}.{module.removeprefix('foundation_stereo_ori.')}"
            return super().find_class(module, name)

    class PickleModule:
        """Module-like adapter accepted by ``torch.load``."""

        Unpickler = RemappingUnpickler
        load = staticmethod(pickle.load)

    return PickleModule


def _load_pickled_model(checkpoint: Path, target_package: str) -> nn.Module:
    """Load the released NAS-pruned module against one implementation package.

    Args:
        checkpoint: Released pickled checkpoint.
        target_package: Synthetic pristine or owned implementation package.

    Returns:
        Evaluation-mode float32 model on CUDA.
    """
    serialized: object = torch.load(
        checkpoint,
        map_location="cpu",
        pickle_module=_pickle_module(target_package),
        weights_only=False,
    )
    if not isinstance(serialized, nn.Module):
        raise TypeError(f"Expected a pickled nn.Module, got {type(serialized).__name__}")
    model: nn.Module = serialized
    model.args.normalize = True
    model.args.mixed_precision = False
    model.args.valid_iters = 1
    model.args.max_disp = 416
    return model.float().cuda().eval()


@slow_cuda
@requires_cuda
def test_released_checkpoint_matches_upstream_and_gwc_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """The released pickle is exact across implementations, and its GWC kernels stay within tolerance."""
    upstream_modules: dict[str, ModuleType] = _load_upstream_package()
    _disable_pretrained_and_compilation(monkeypatch, upstream_modules)
    checkpoint: Path = download_fast_foundationstereo_checkpoint()
    upstream_model: nn.Module = _load_pickled_model(checkpoint, UPSTREAM_PACKAGE)
    owned_model: nn.Module = _load_pickled_model(checkpoint, OWNED_PACKAGE)
    assert list(upstream_model.state_dict()) == list(owned_model.state_dict())

    generator: torch.Generator = torch.Generator().manual_seed(43)
    left_13hw: Float32[Tensor, "1 3 256 320"] = torch.rand((1, 3, 256, 320), generator=generator, dtype=torch.float32).cuda() * 255.0
    right_13hw: Float32[Tensor, "1 3 256 320"] = torch.rand((1, 3, 256, 320), generator=generator, dtype=torch.float32).cuda() * 255.0
    with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True), torch.inference_mode():
        upstream_disparity_11hw: Float32[Tensor, "1 1 256 320"] = upstream_model(
            left_13hw,
            right_13hw,
            iters=1,
            test_mode=True,
            optimize_build_volume="pytorch1",
        )
        owned_disparity_11hw: Float32[Tensor, "1 1 256 320"] = owned_model(
            left_13hw,
            right_13hw,
            iters=1,
            test_mode=True,
            optimize_build_volume="pytorch1",
        )
        normalized_pair_23hw: Float32[Tensor, "2 3 256 320"] = normalize_image(torch.cat((left_13hw, right_13hw), dim=0))
        feature_pair: list[Float32[Tensor, "2 _channels _height _width"]] = owned_model.feature(normalized_pair_23hw)
        left_features_1chw: Float32[Tensor, "1 channels h4 w4"] = feature_pair[0][:1]
        right_features_1chw: Float32[Tensor, "1 channels h4 w4"] = feature_pair[0][1:]
        eager_gwc: Callable[..., Tensor] = owned_submodule.build_gwc_volume_optimized_pytorch1_eager
        pytorch_volume_1gdhw: Float32[Tensor, "1 groups disparities h4 w4"] = eager_gwc(
            left_features_1chw,
            right_features_1chw,
            416 // 4,
            owned_model.cv_group,
            normalize=True,
        )
        triton_volume_1gdhw: Float32[Tensor, "1 groups disparities h4 w4"] = owned_submodule.build_gwc_volume_triton(
            left_features_1chw,
            right_features_1chw,
            416 // 4,
            owned_model.cv_group,
            normalize=True,
        )

    output_max_abs: float = (upstream_disparity_11hw - owned_disparity_11hw).abs().max().item()
    assert torch.equal(upstream_disparity_11hw, owned_disparity_11hw), f"released checkpoint max abs diff: {output_max_abs:.9g}"
    max_abs_difference: float = float((pytorch_volume_1gdhw - triton_volume_1gdhw).abs().max())
    assert torch.allclose(pytorch_volume_1gdhw, triton_volume_1gdhw, atol=2e-5, rtol=1e-4), (
        f"Triton/PyTorch GWC max abs diff {max_abs_difference:.8f}"
    )
