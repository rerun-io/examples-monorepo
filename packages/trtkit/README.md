# trtkit — one PyTorch → ONNX → TensorRT home

The shared acceleration layer for the monorepo's model packages. It replaces
the TensorRT runner/builder copies that used to live in posekit, wilor-nano,
sapiens2-pose / sapiens-coco133-pose, prompt-da, and mamma with one
implementation, split in two layers:

- **runtime** — a backend-neutral tensor-function contract
  (`trtkit.base.TensorRuntime`: CUDA tensors in, CUDA tensors out, keyed by
  binding name) with three implementations: `TorchRuntime` (eager module,
  parity reference and fallback), `OnnxCudaRuntime` (ONNX Runtime CUDA EP with
  IOBinding onto torch memory — no host copies), and `TensorRtRuntime`
  (persistent buffers via `set_tensor_address`, `execute_async_v3`,
  static-batch zero-padding, optional CUDA-graph capture/replay).
- **hub** — artifact identity and machine-local caching:
  `TrtBuildConfig` / `build_engine` / `ensure_engine` plus generic ONNX
  introspection in `trtkit.onnx_graph` (`onnx_static_batch_size`).
  Model-family graph surgery (e.g. detector NMS stripping) stays model-side.

ONNX files are the portable artifacts. Engines are **machine-locked**
(TensorRT version + GPU compute capability), never committed, and rebuilt from
ONNX on each machine. Model-specific concerns — export wrappers, checkpoint
resolution, pre/postprocessing — stay in the model packages.

## Core usage

```python
from pathlib import Path
from trtkit import TensorRtRuntime, TrtBuildConfig, ensure_engine

# Build-or-reuse a cached engine from an ONNX file (first call builds, minutes).
engine_path: Path = ensure_engine(onnx_path, TrtBuildConfig(max_batch_size=32, opt_batch_size=8, precision="fp16"))

runtime = TensorRtRuntime(engine_path)                 # use_cuda_graph=True for tight latency loops
outputs = runtime({"images": frames_f32_cuda})         # dict[str, Tensor] in -> dict[str, Tensor] out
boxes = outputs["output0"]                             # sliced to your submitted batch size
```

Everything the runtime knows comes from the engine itself: `runtime.spec`
carries input/output names, per-sample shapes, dtypes, and `max_batch_size`.
**The engine is the batch authority** — don't mirror batch sizes into config.

Two contract rules:

- **Outputs are views into runtime-owned buffers that the next call
  overwrites.** Clone anything that must survive. For batches larger than the
  engine's max, use `trtkit.run_chunked(runtime, inputs)` — it chunks, clones,
  and concatenates so callers never touch the buffer contract.
- Dynamic-batch engines execute at your true batch size; static-batch engines
  zero-pad up to their baked batch (outputs still come back sliced).

## How model packages consume it

No wrapper classes. The model package declares its contract as *data* —
binding-name constants and a TypedDict — made load-bearing by one pure
boundary function (see `wilor_nano/api/tensorrt_runtime.py`):

```python
FULL_WILOR_INPUT_NAME = "img_patches"

class WiLorOutput(TypedDict):
    global_orient: Float[Tensor, "batch 1 3"]
    ...

def run_full_wilor(runtime: TensorRuntime, crops: Float[Tensor, "batch 256 256 3"]) -> WiLorOutput:
    return cast(WiLorOutput, runtime({FULL_WILOR_INPUT_NAME: crops}))
```

trtkit is transport; the model package owns meaning. Pipelines hold the bare
runtime (which also makes test fakes trivial: any `Callable[[Path],
TensorRuntime]` works as a factory).

For CLI-selectable backends, `trtkit.backends` provides tyro subcommand
unions (`BackendConfig`, `OnnxOrTrtBackendConfig`) and
`create_runtime_from_onnx(onnx_path, backend)`, which routes one ONNX artifact
to either the ORT session or a cached TensorRT engine — posekit's model zoo is
the reference consumer.

## Engine cache and identity

`ensure_engine` caches under `~/.cache/trtkit/trt` (override with
`TRTKIT_TRT_CACHE`). The filename encodes everything that invalidates an
engine: ONNX content hash (covering sibling `.onnx.data` external weights),
batch profile, precision, workspace, optimization level, TensorRT version, and
SM. A JSON manifest beside each engine records how it was produced. Builds
publish atomically, so a killed build never leaves a truncated engine.

`TrtBuildConfig.precision` is `fp32 | fp16 | bf16 | strong` (`strong` builds a
strongly-typed network that takes dtypes from the graph — the escape hatch for
fp16-overflow-prone ViTs). `allow_tf32=False` disables TF32 for strict-fp32
models and is part of the cache key (`-notf32`).

## Known TensorRT limitations

**Dynamic batch + Myelin `CHECK(is_const())`.** A graph whose reshape depends on
the batch symbol can fail to build with `dynamic_batch_max` set:

```
Could not find any implementation for node {ForeignNode[ONNXTRT_ShapeShuffle_…]}
MyelinCheckException: value.h:872: CHECK(is_const()) failed
```

Seen with ZipDepth-base, whose `F.unfold` upsampling head dynamo lowers to a
gather indexed off `Shape(...)`; head rewrites (`view(-1, …)`, static slices, the
NPU head) only move the failure earlier, and it reproduces identically on
TensorRT 10.16.1, 11.0.0, 11.1.0 and 11.2.1 on sm_120 (2026-08-30). It is a
Myelin limitation on a non-constant leading dim, not a graph bug, and it is not
in NVIDIA's release notes or tracker. Every other trtkit consumer here (PromptDA,
MoGe, MammaNet, RTMW, Sapiens) builds fine with a dynamic batch profile.

Workaround: export with `dynamic_batch_max=None` (static batch) and build one
engine per batch size (`TrtBuildConfig(max_batch_size=b, opt_batch_size=b)`).
Zero model change; upstream ZipDepth's own `export.py` hardcodes batch 1 for the
same reason. Related: fp16 via `model.half()`, not autocast — an autocast export
can leave a float conv kernel beside a half input, which a strongly-typed build
rejects at parse time.

## Dev

```bash
pixi run -e trtkit-dev --frozen lint       # also: typecheck, deadcode, tests
```

trtkit's own tests cover what it adds over the consolidated runners; the
moved behavior (three-backend parity, CUDA-graph replay, padding) is covered
end-to-end by posekit's `test_runtime_parity.py`, which runs a hermetic
TinyNet through all three backends.
