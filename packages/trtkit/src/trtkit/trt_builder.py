"""TensorRT engine building and machine-local caching (the hub layer).

Engines are never committed or downloaded: they are sm-/version-specific
artifacts built once from a model's ONNX interchange file into a local cache
directory, with a JSON manifest recording how each engine was produced.

When TensorRT miscompiles a graph whose fusion spans an input-derived
reduction all the way to an output (garbage/NaN at every precision), mark the
reduction results as extra ONNX outputs to force materialization and split
the fusion — see prompt-da's ``export_promptda_onnx`` for a worked example.
"""

import hashlib
import json
import os
import threading
import time
from _thread import LockType
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn

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
    workspace_gib: float = 24.0
    """Workspace memory pool limit handed to the builder — a cap on tactic
    memory, not an upfront allocation."""
    builder_optimization_level: int = 3
    """TensorRT builder optimization level (0-5)."""
    extra_output_patterns: tuple[str, ...] = ()
    """Mark every intermediate tensor whose name contains one of these substrings as
    an additional engine output. Materializing a tensor bars TensorRT from fusing it
    with its consumers — the escape hatch for miscompiled fusions (e.g. plane-sweep
    cost-volume GridSample). Each pattern must match at least one tensor. Part of
    the engine cache key."""


def cached_engine_path(onnx_path: Path, config: TrtBuildConfig, *, cache_dir: Path = DEFAULT_TRT_CACHE_DIR, onnx_sha256: str | None = None) -> Path:
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
        onnx_sha256: Precomputed :func:`onnx_content_hash` digest, to avoid
            re-hashing large files; computed here when omitted.

    Returns:
        Deterministic engine path inside ``cache_dir``.
    """
    if not 1 <= config.opt_batch_size <= config.max_batch_size:
        raise ValueError(f"opt_batch_size must be within [1, max_batch_size], got opt={config.opt_batch_size} max={config.max_batch_size}.")
    import tensorrt as trt
    capability: tuple[int, int] = torch.cuda.get_device_capability()
    onnx_hash: str = (onnx_sha256 or onnx_content_hash(onnx_path))[:12]
    precision_key: str = "strong" if config.allow_tf32 else "strong-notf32"
    extra_outputs_key: str = (
        f"_x{hashlib.sha256('|'.join(config.extra_output_patterns).encode()).hexdigest()[:8]}"
        if config.extra_output_patterns
        else ""
    )
    name: str = (
        f"{onnx_path.stem}_b1-{config.opt_batch_size}-{config.max_batch_size}_{precision_key}"
        f"_w{config.workspace_gib:g}o{config.builder_optimization_level}{extra_outputs_key}"
        f"_trt{trt.__version__}_sm{capability[0]}{capability[1]}_{onnx_hash}.engine"
    )
    return cache_dir / name


def build_engine(onnx_path: Path, engine_path: Path, config: TrtBuildConfig, *, onnx_sha256: str | None = None) -> None:
    """Build a TensorRT engine from ONNX and write it plus a manifest.

    Args:
        onnx_path: ONNX interchange file. Its batch dimension may be dynamic;
            a dynamic profile spans ``1..config.max_batch_size`` (opt at
            ``config.opt_batch_size``) so callers run true batch sizes.
        engine_path: Output engine path (a ``.json`` manifest is written beside it).
        config: Batch/TF32/workspace build options.
        onnx_sha256: Precomputed :func:`onnx_content_hash` digest for the
            manifest; computed here when omitted.

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
    for pattern in config.extra_output_patterns:
        matched: bool = False
        for layer_index in range(int(network.num_layers)):
            layer: Any = network.get_layer(layer_index)
            for output_index in range(int(layer.num_outputs)):
                candidate: Any = layer.get_output(output_index)
                if pattern in candidate.name and not candidate.is_network_output:
                    network.mark_output(candidate)
                    matched = True
        if not matched:
            raise RuntimeError(f"extra_output_patterns entry {pattern!r} matched no network tensor in {onnx_path}.")
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

    duration_prior: float | None = None
    prior_mtime: float = float("-inf")
    for manifest_path in engine_path.parent.glob("*.engine.json"):
        try:
            prior_manifest: Any = json.loads(manifest_path.read_text())
            manifest_mtime: float = manifest_path.stat().st_mtime
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(prior_manifest, dict):
            continue
        prior_onnx_path: Any = prior_manifest.get("onnx_path")
        prior_build_seconds: Any = prior_manifest.get("build_seconds")
        if (
            isinstance(prior_onnx_path, str)
            and Path(prior_onnx_path).stem == onnx_path.stem
            # Same optimization level only: an o0 duration is no estimate for an o3 build.
            and prior_manifest.get("builder_optimization_level") == config.builder_optimization_level
            and isinstance(prior_build_seconds, (int, float))
            and not isinstance(prior_build_seconds, bool)
            and prior_build_seconds > 0.0
            and manifest_mtime > prior_mtime
        ):
            duration_prior = float(prior_build_seconds)
            prior_mtime = manifest_mtime

    class BuildProgress(trt.IProgressMonitor):
        """Spinner + elapsed clock + current builder phase — deliberately not a bar.

        A measured o3 build emits no progress callbacks for 99% of its duration
        (steps burst at start and end), and TensorRT's log stream is equally
        silent during the expensive tactic timing, so there is no data to drive
        a fraction-complete display. A prior ``build_seconds`` from an earlier
        manifest (same ONNX, same optimization level) appears as a static
        "typically ~2m" hint; the phase callbacks feed the trailing context
        text. TensorRT may invoke callbacks from multiple builder threads, so
        one lock guards the phase state; rich's own refresh thread animates the
        spinner and the clock.
        """

        def __init__(self, typical_seconds: float | None) -> None:
            super().__init__()
            self._lock: LockType = threading.Lock()
            self._totals: dict[str, int] = {}
            self._steps: dict[str, int] = {}
            self._stack: list[str] = []
            hint: str = ""
            if typical_seconds is not None:
                hint_label: str = f"~{typical_seconds / 60.0:.0f}m" if typical_seconds >= 60.0 else f"~{typical_seconds:.0f}s"
                hint = f" (typically {hint_label} on this machine)"
            self._progress: Progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                TextColumn("[dim]{task.fields[phase]}[/dim]"),
                transient=True,
                disable=not Console().is_terminal,
            )
            self._task: TaskID = self._progress.add_task(f"TensorRT build{hint}", total=None, phase="starting")
            self._progress.start()

        def close(self) -> None:
            self._progress.stop()

        def phase_start(self, phase_name: str, parent_phase: str | None, num_steps: int) -> None:
            with self._lock:
                self._totals[phase_name] = num_steps
                self._steps[phase_name] = 0
                self._stack.append(phase_name)
                self._show_phase()

        def step_complete(self, phase_name: str, step: int) -> bool:
            with self._lock:
                self._steps[phase_name] = step + 1  # step is the zero-based index of the completed step
                self._show_phase()
            return True

        def phase_finish(self, phase_name: str) -> None:
            with self._lock:
                if phase_name in self._stack:
                    self._stack.remove(phase_name)
                self._totals.pop(phase_name, None)
                self._steps.pop(phase_name, None)
                if self._stack:
                    self._show_phase()
                else:
                    self._progress.update(self._task, phase="finalizing")

        def _show_phase(self) -> None:
            # Show the deepest phase with real steps; micro-phases obscure more useful parent context.
            target: str = next((name for name in reversed(self._stack) if self._totals[name] > 1), self._stack[-1])
            self._progress.update(self._task, phase=f"{target} {self._steps[target]}/{self._totals[target]}")

    monitor: Any = BuildProgress(duration_prior)
    builder_config.progress_monitor = monitor
    build_started: float = time.perf_counter()
    try:
        serialized: Any = builder.build_serialized_network(network, builder_config)
    finally:
        monitor.close()
    build_seconds: float = time.perf_counter() - build_started
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path}.")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    # Publish atomically: ensure_engine trusts the cache path by existence, so a
    # killed build must never leave a truncated engine there.
    tmp_path: Path = engine_path.with_name(f"{engine_path.name}.tmp-{os.getpid()}")
    tmp_path.write_bytes(bytes(serialized))
    manifest: dict[str, Any] = {
        "onnx_path": str(onnx_path),
        "onnx_sha256": onnx_sha256 or onnx_content_hash(onnx_path),
        "engine_path": str(engine_path),
        "portable_engine": False,
        "rebuild_from_onnx_on_target_machine": True,
        "max_batch_size": config.max_batch_size,
        "opt_batch_size": config.opt_batch_size,
        "strongly_typed": True,
        "allow_tf32": config.allow_tf32,
        "workspace_gib": config.workspace_gib,
        "builder_optimization_level": config.builder_optimization_level,
        "build_seconds": build_seconds,
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
        config: Batch/TF32/workspace build options.
        cache_dir: Engine cache root.

    Returns:
        Path to a ready-to-load engine matching this machine and config.
    """
    onnx_sha256: str = onnx_content_hash(onnx_path)
    engine_path: Path = cached_engine_path(onnx_path, config, cache_dir=cache_dir, onnx_sha256=onnx_sha256)
    if not engine_path.exists():
        print(f"[trtkit] building TensorRT engine (one-time, may take minutes): {engine_path.name}")
        build_engine(onnx_path, engine_path, config, onnx_sha256=onnx_sha256)
    return engine_path


def onnx_content_hash(onnx_path: Path) -> str:
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
