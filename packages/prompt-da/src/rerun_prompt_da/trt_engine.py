"""ONNX export and TensorRT engine caching for Prompt Depth Anything.

prompt-da owns the whole acceleration path: it imports the torch PromptDA
network from monopriors, exports the ONNX interchange graph itself, and builds
dynamic-batch TensorRT engines from it. ONNX files are portable and cached per
model/resolution; engines are machine-local (TensorRT-version- and SM-specific)
and rebuilt from ONNX on each target machine, following the posekit/mamma
convention.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import torch

TrtPrecision: TypeAlias = Literal["fp32", "fp16", "bf16"]
ModelType: TypeAlias = Literal["large", "small", "small-transparent"]

DEFAULT_CACHE_DIR: Path = Path(os.environ.get("PROMPTDA_TRT_CACHE", "~/.cache/prompt-da")).expanduser()
"""Cache root holding ``onnx/`` (portable) and ``trt/`` (machine-local) artifacts."""

PROMPT_DEPTH_HW: tuple[int, int] = (192, 256)
"""ARKit LiDAR prompt-depth resolution PromptDA was trained on."""

ONNX_OPSET: int = 17
"""Legacy-exporter opset; TRT 10.13's parser chokes on dynamo exports (see mamma)."""

ONNX_EXPORT_VERSION: int = 1
"""Bump whenever the export recipe or the vendored PromptDA implementation changes,
so cached ONNX graphs from older code are not silently reused."""


@dataclass(frozen=True, slots=True)
class TrtBuildConfig:
    """How to build (and cache-key) a PromptDA TensorRT engine."""

    max_batch_size: int = 8
    """Largest batch the dynamic-batch engine accepts (profile max)."""
    opt_batch_size: int = 8
    """Batch size TensorRT tunes kernels for (dynamic profile optimum)."""
    precision: TrtPrecision = "fp16"
    """Builder precision flag. ``fp32`` leaves the builder defaults untouched."""
    workspace_gib: float = 8.0
    """Workspace memory pool limit handed to the builder."""
    builder_optimization_level: int = 3
    """TensorRT builder optimization level (0-5)."""


def export_promptda_onnx(
    model_type: ModelType = "large",
    image_hw: tuple[int, int] = (756, 1008),
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Export the monopriors PromptDA network to a dynamic-batch ONNX graph.

    The graph takes ``image`` (float32 ``[B,3,H,W]``, RGB in [0,1]) and
    ``prompt_depth`` (float32 ``[B,1,192,256]``, meters) and returns ``depth``
    (float32 ``[B,1,H,W]``, meters). Only the batch axis is dynamic; H and W
    must be multiples of the DINOv2 patch size (14).

    Args:
        model_type: PromptDA checkpoint variant (monopriors ``NAME_TO_HFNAME`` key).
        image_hw: Static (height, width) the graph is exported at.
        cache_dir: Cache root; the file lands in ``cache_dir / "onnx"``.

    Returns:
        Path to the cached ONNX file (exported on first use).
    """
    height, width = image_hw
    if height % 14 != 0 or width % 14 != 0:
        raise ValueError(f"PromptDA image size must be a multiple of the 14px patch size, got {image_hw}.")

    from huggingface_hub import hf_hub_download
    from monopriors.models.depth_completion.prompt_da import NAME_TO_HFNAME
    from monopriors.third_party.promptda.promptda import PromptDA

    # Resolve the checkpoint first so its HF snapshot revision is part of the
    # cache identity — an updated checkpoint or export recipe (ONNX_EXPORT_VERSION)
    # must not silently reuse a stale graph.
    ckpt_path = Path(hf_hub_download(repo_id=NAME_TO_HFNAME[model_type], repo_type="model", filename="model.ckpt"))
    ckpt_rev: str = _checkpoint_revision(ckpt_path)
    onnx_dir: Path = cache_dir / "onnx"
    onnx_path: Path = onnx_dir / f"promptda-{model_type}_{height}x{width}_op{ONNX_OPSET}_v{ONNX_EXPORT_VERSION}_{ckpt_rev}.onnx"
    if onnx_path.exists():
        return onnx_path

    print(f"[prompt-da] exporting ONNX (one-time, may take a minute): {onnx_path.name}")
    model = PromptDA.from_pretrained(str(ckpt_path)).to("cuda").eval()
    # Trace at batch 2 so no op accidentally specializes on batch 1.
    dummy_image: torch.Tensor = torch.zeros((2, 3, height, width), dtype=torch.float32, device="cuda")
    dummy_prompt: torch.Tensor = torch.rand((2, 1, *PROMPT_DEPTH_HW), dtype=torch.float32, device="cuda") + 0.5
    onnx_dir.mkdir(parents=True, exist_ok=True)
    # pid-unique temp + atomic rename: concurrent exporters may duplicate work
    # but can never clobber each other's in-flight writes or publish a
    # truncated file.
    tmp_path: Path = onnx_path.with_name(f"{onnx_path.name}.part{os.getpid()}")
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (dummy_image, dummy_prompt),
            str(tmp_path),
            input_names=["image", "prompt_depth"],
            output_names=["depth"],
            opset_version=ONNX_OPSET,
            do_constant_folding=True,
            dynamic_axes={
                "image": {0: "batch"},
                "prompt_depth": {0: "batch"},
                "depth": {0: "batch"},
            },
            dynamo=False,
        )
    tmp_path.rename(onnx_path)
    del model
    torch.cuda.empty_cache()
    return onnx_path


def cached_engine_path(onnx_path: Path, config: TrtBuildConfig, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Return the machine-local cache path for an engine built from this ONNX file.

    The name encodes everything that invalidates an engine: ONNX content hash,
    batch range, precision, TensorRT version, and GPU compute capability.

    Args:
        onnx_path: ONNX interchange file the engine is built from.
        config: Build configuration contributing to the cache key.
        cache_dir: Cache root; the engine lands in ``cache_dir / "trt"``.

    Returns:
        Deterministic engine path inside ``cache_dir / "trt"``.
    """
    if not 1 <= config.opt_batch_size <= config.max_batch_size:
        raise ValueError(f"opt_batch_size must be within [1, max_batch_size], got opt={config.opt_batch_size} max={config.max_batch_size}.")
    trt: Any = _import_tensorrt()
    capability: tuple[int, int] = torch.cuda.get_device_capability()
    onnx_hash: str = _onnx_content_hash(onnx_path)[:12]
    name: str = (
        f"{onnx_path.stem}_b1-{config.opt_batch_size}-{config.max_batch_size}_{config.precision}"
        f"_trt{trt.__version__}_sm{capability[0]}{capability[1]}_{onnx_hash}.engine"
    )
    return cache_dir / "trt" / name


def build_engine(onnx_path: Path, engine_path: Path, config: TrtBuildConfig) -> None:
    """Build a dynamic-batch TensorRT engine from ONNX and write it plus a manifest.

    Args:
        onnx_path: ONNX interchange file with a dynamic batch axis; the
            optimization profile spans ``1..config.max_batch_size`` with the
            optimum at ``config.opt_batch_size``.
        engine_path: Output engine path (a ``.json`` manifest is written beside it).
        config: Precision/batch/workspace build options.

    Raises:
        RuntimeError: If ONNX parsing or engine serialization fails, or an
            input has dynamic non-batch dimensions.
    """
    trt: Any = _import_tensorrt()
    logger: Any = trt.Logger(trt.Logger.WARNING)
    builder: Any = trt.Builder(logger)
    network: Any = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser: Any = trt.OnnxParser(network, logger)
    # parse_from_file (not parse(bytes)) so external weight data, if any,
    # resolves relative to the model file.
    if not parser.parse_from_file(str(onnx_path.expanduser())):
        errors: str = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {onnx_path}:\n{errors}")
    builder_config: Any = builder.create_builder_config()
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(config.workspace_gib * (1 << 30)))
    builder_config.builder_optimization_level = int(config.builder_optimization_level)
    if config.precision == "fp16":
        builder_config.set_flag(trt.BuilderFlag.FP16)
    elif config.precision == "bf16":
        builder_config.set_flag(trt.BuilderFlag.BF16)
    profile: Any = builder.create_optimization_profile()
    for idx in range(int(network.num_inputs)):
        tensor: Any = network.get_input(idx)
        shape: tuple[int, ...] = tuple(int(dim) for dim in tensor.shape)
        if any(dim < 0 for dim in shape[1:]):
            raise RuntimeError(f"ONNX input {tensor.name!r} has dynamic non-batch dims {shape}; per-sample shapes must be static.")
        if shape[0] >= 0:
            raise RuntimeError(f"ONNX input {tensor.name!r} has static batch {shape[0]}; re-export with a dynamic batch axis.")
        per_sample: tuple[int, ...] = shape[1:]
        profile.set_shape(tensor.name, (1, *per_sample), (config.opt_batch_size, *per_sample), (config.max_batch_size, *per_sample))
    builder_config.add_optimization_profile(profile)
    serialized: Any = builder.build_serialized_network(network, builder_config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path}.")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    # pid-unique temp + atomic rename so an interrupted or concurrent build
    # never leaves a truncated engine at the final path (ensure_engine treats
    # existence as readiness).
    tmp_engine_path: Path = engine_path.with_name(f"{engine_path.name}.part{os.getpid()}")
    tmp_engine_path.write_bytes(bytes(serialized))
    tmp_engine_path.rename(engine_path)
    manifest: dict[str, Any] = {
        "onnx_path": str(onnx_path),
        "onnx_sha256": _onnx_content_hash(onnx_path),
        "engine_path": str(engine_path),
        "portable_engine": False,
        "rebuild_from_onnx_on_target_machine": True,
        "max_batch_size": config.max_batch_size,
        "opt_batch_size": config.opt_batch_size,
        "precision": config.precision,
        "workspace_gib": config.workspace_gib,
        "builder_optimization_level": config.builder_optimization_level,
        "tensorrt_version": str(trt.__version__),
        "cuda_device_name": torch.cuda.get_device_name(),
        "cuda_compute_capability": list(torch.cuda.get_device_capability()),
    }
    engine_path.with_suffix(engine_path.suffix + ".json").write_text(json.dumps(manifest, indent=2) + "\n")


def ensure_engine(onnx_path: Path, config: TrtBuildConfig, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Return a cached engine for this ONNX file, building it on first use.

    Args:
        onnx_path: ONNX interchange file the engine is built from.
        config: Precision/batch/workspace build options.
        cache_dir: Cache root; the engine lands in ``cache_dir / "trt"``.

    Returns:
        Path to a ready-to-load engine matching this machine and config.
    """
    engine_path: Path = cached_engine_path(onnx_path, config, cache_dir=cache_dir)
    if not engine_path.exists():
        print(f"[prompt-da] building TensorRT engine (one-time, may take minutes): {engine_path.name}")
        build_engine(onnx_path, engine_path, config)
    return engine_path


def _checkpoint_revision(ckpt_path: Path) -> str:
    """Short revision identifying a resolved HF checkpoint file.

    Args:
        ckpt_path: Checkpoint path returned by ``hf_hub_download`` (its HF cache
            layout is ``…/snapshots/<commit>/model.ckpt``).

    Returns:
        First 8 chars of the snapshot commit, or a size-based tag for
        checkpoints outside the HF cache layout.
    """
    parts: tuple[str, ...] = ckpt_path.parts
    if "snapshots" in parts:
        return parts[parts.index("snapshots") + 1][:8]
    return f"size{ckpt_path.stat().st_size}"


def _onnx_content_hash(onnx_path: Path) -> str:
    """Hex SHA-256 of an ONNX model, covering its external weight file if present.

    Args:
        onnx_path: ONNX model file (large exports may keep weights in a
            sibling ``<name>.onnx.data`` file).

    Returns:
        Hex digest string over the proto and any external data.
    """
    digest = hashlib.sha256()
    for path in (onnx_path.expanduser(), onnx_path.expanduser().with_suffix(".onnx.data")):
        if not path.exists():
            continue
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _import_tensorrt() -> Any:
    """Import TensorRT lazily so non-TRT code paths can import this module.

    Returns:
        Imported TensorRT Python module.

    Raises:
        RuntimeError: If TensorRT bindings are not installed in the active Pixi environment.
    """
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError("TensorRT Python bindings are not installed in this Pixi environment.") from exc
    return trt
