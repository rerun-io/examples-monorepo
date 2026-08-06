"""TensorRT engine building and machine-local caching (the hub layer).

Engines are never committed or downloaded: they are sm-/version-specific
artifacts built once from a model's ONNX interchange file into a local cache
directory, with a JSON manifest recording how each engine was produced.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

DEFAULT_TRT_CACHE_DIR: Path = Path(os.environ.get("TRTKIT_TRT_CACHE", "~/.cache/trtkit/trt")).expanduser()
"""Machine-local engine cache; override with the ``TRTKIT_TRT_CACHE`` env var."""


@dataclass(frozen=True, slots=True)
class TrtBuildConfig:
    """How to build (and cache-key) a TensorRT engine from an ONNX file."""

    max_batch_size: int = 32
    """Largest batch a dynamic-batch engine accepts (profile max). Static-batch
    ONNX graphs must match this value and yield a static engine."""
    opt_batch_size: int = 8
    """Batch size TensorRT tunes kernels for (dynamic profile optimum)."""
    allow_tf32: bool = True
    """Allow TF32 math for fp32-typed layers (TensorRT's default). Disable when
    a model needs strict fp32 numerics."""
    workspace_gib: float = 8.0
    """Workspace memory pool limit handed to the builder."""
    builder_optimization_level: int = 3
    """TensorRT builder optimization level (0-5)."""


def cached_engine_path(onnx_path: Path, config: TrtBuildConfig, *, cache_dir: Path = DEFAULT_TRT_CACHE_DIR) -> Path:
    """Return the machine-local cache path for an engine built from this ONNX file.

    The name encodes everything that invalidates an engine: ONNX content hash,
    batch, a disabled-TF32 marker, workspace, optimization level, TensorRT
    version, and GPU compute capability. Compute dtype is not a knob: every
    build is strongly typed, so precision lives in the ONNX graph (and thus in
    the content hash).

    Args:
        onnx_path: ONNX interchange file the engine is built from.
        config: Build configuration contributing to the cache key.
        cache_dir: Engine cache root.

    Returns:
        Deterministic engine path inside ``cache_dir``.
    """
    if not 1 <= config.opt_batch_size <= config.max_batch_size:
        raise ValueError(f"opt_batch_size must be within [1, max_batch_size], got opt={config.opt_batch_size} max={config.max_batch_size}.")
    import tensorrt as trt
    capability: tuple[int, int] = torch.cuda.get_device_capability()
    onnx_hash: str = _onnx_content_hash(onnx_path)[:12]
    precision_key: str = "strong" if config.allow_tf32 else "strong-notf32"
    name: str = (
        f"{onnx_path.stem}_b1-{config.opt_batch_size}-{config.max_batch_size}_{precision_key}"
        f"_w{config.workspace_gib:g}o{config.builder_optimization_level}"
        f"_trt{trt.__version__}_sm{capability[0]}{capability[1]}_{onnx_hash}.engine"
    )
    return cache_dir / name


def build_engine(onnx_path: Path, engine_path: Path, config: TrtBuildConfig) -> None:
    """Build a TensorRT engine from ONNX and write it plus a manifest.

    Args:
        onnx_path: ONNX interchange file. Its batch dimension may be dynamic;
            a dynamic profile spans ``1..config.max_batch_size`` (opt at
            ``config.opt_batch_size``) so callers run true batch sizes.
        engine_path: Output engine path (a ``.json`` manifest is written beside it).
        config: Precision/batch/workspace build options.

    Raises:
        RuntimeError: If ONNX parsing or engine serialization fails.
    """
    import tensorrt

    trt: Any = tensorrt  # the compiled bindings have incomplete stubs
    logger: Any = trt.Logger(trt.Logger.WARNING)
    builder: Any = trt.Builder(logger)
    # TensorRT 11 removed weak typing (and the EXPLICIT_BATCH flag with it):
    # every network is strongly typed and compute dtypes come from the graph.
    network_flags: int = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network: Any = builder.create_network(network_flags)
    parser: Any = trt.OnnxParser(network, logger)
    # parse_from_file (not parse(bytes)) so ONNX external weight data resolves
    # relative to the model file (dynamo exports of large models use it).
    if not parser.parse_from_file(str(onnx_path.expanduser())):
        errors: str = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {onnx_path}:\n{errors}")
    builder_config: Any = builder.create_builder_config()
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(config.workspace_gib * (1 << 30)))
    builder_config.builder_optimization_level = int(config.builder_optimization_level)
    if not config.allow_tf32:
        builder_config.clear_flag(trt.BuilderFlag.TF32)
    profile: Any = builder.create_optimization_profile()
    has_dynamic_batch: bool = False
    for idx in range(int(network.num_inputs)):
        tensor: Any = network.get_input(idx)
        shape: tuple[int, ...] = tuple(int(dim) for dim in tensor.shape)
        if any(dim < 0 for dim in shape[1:]):
            raise RuntimeError(f"ONNX input {tensor.name!r} has dynamic non-batch dims {shape}; trtkit requires static per-sample shapes.")
        if shape[0] < 0:
            has_dynamic_batch = True
            per_sample: tuple[int, ...] = shape[1:]
            profile.set_shape(tensor.name, (1, *per_sample), (config.opt_batch_size, *per_sample), (config.max_batch_size, *per_sample))
        elif shape[0] != config.max_batch_size:
            raise RuntimeError(
                f"ONNX input {tensor.name!r} has static batch {shape[0]} but the build requests {config.max_batch_size}; re-export or match batch sizes."
            )
    if has_dynamic_batch:
        builder_config.add_optimization_profile(profile)
    serialized: Any = builder.build_serialized_network(network, builder_config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path}.")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    # Publish atomically: ensure_engine trusts the cache path by existence, so a
    # killed build must never leave a truncated engine there.
    tmp_path: Path = engine_path.with_name(f"{engine_path.name}.tmp-{os.getpid()}")
    tmp_path.write_bytes(bytes(serialized))
    manifest: dict[str, Any] = {
        "onnx_path": str(onnx_path),
        "onnx_sha256": _onnx_content_hash(onnx_path),
        "engine_path": str(engine_path),
        "portable_engine": False,
        "rebuild_from_onnx_on_target_machine": True,
        "max_batch_size": config.max_batch_size,
        "opt_batch_size": config.opt_batch_size,
        "strongly_typed": True,
        "allow_tf32": config.allow_tf32,
        "workspace_gib": config.workspace_gib,
        "builder_optimization_level": config.builder_optimization_level,
        "tensorrt_version": str(trt.__version__),
        "cuda_device_name": torch.cuda.get_device_name(),
        "cuda_compute_capability": list(torch.cuda.get_device_capability()),
    }
    engine_path.with_suffix(engine_path.suffix + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(tmp_path, engine_path)


def ensure_engine(onnx_path: Path, config: TrtBuildConfig, *, cache_dir: Path = DEFAULT_TRT_CACHE_DIR) -> Path:
    """Return a cached engine for this ONNX file, building it on first use.

    Args:
        onnx_path: ONNX interchange file the engine is built from.
        config: Precision/batch/workspace build options.
        cache_dir: Engine cache root.

    Returns:
        Path to a ready-to-load engine matching this machine and config.
    """
    engine_path: Path = cached_engine_path(onnx_path, config, cache_dir=cache_dir)
    if not engine_path.exists():
        print(f"[trtkit] building TensorRT engine (one-time, may take minutes): {engine_path.name}")
        build_engine(onnx_path, engine_path, config)
    return engine_path


def _onnx_content_hash(onnx_path: Path) -> str:
    """Hex SHA-256 of an ONNX model, covering its external weight file if present.

    Args:
        onnx_path: ONNX model file (dynamo exports of large models keep weights
            in a sibling ``<name>.onnx.data`` file).

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
