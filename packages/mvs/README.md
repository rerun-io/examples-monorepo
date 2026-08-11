# mvs

Multi-view stereo depth over ARKitScenes segments served by the Rerun cloud
catalog (CUDA-only: linux-64 + linux-aarch64).

One `RerunIterableDataset` carries every model input as a `Field`: the AV1
video, decoded straight to CUDA tensors through NVDEC, plus the six pose and
calibration columns — all riding a single fetch query. A stateful collate
composes per-frame poses, keeps a causal keyframe buffer, and assembles the
depth queries. The depth model is a plain ONNX artifact; `trtkit` runs it on
ONNX Runtime CUDA or a cached machine-local TensorRT engine (dynamic batch
1–32).

## Run

```bash
pixi run -e mvs --frozen mvs-catalog-depth   # depth demo (downloads the model on first use)
pixi run -e mvs --frozen mvs-live-mesh       # catalog video streaming only, no model
```

The model artifact lives in the private HF repo `pablovela5620/mvs-depth`;
`mvs-download-depth-model` fetches it with `hf download`.

The demo defaults to the ONNX Runtime backend, which is portable but slow:
the CUDA provider falls back to CPU kernels for parts of this graph and runs
roughly 6x slower than TensorRT. For real speed pass the `tensorrt`
subcommand (first use builds an engine, which takes a few minutes):

```bash
cd packages/mvs
python tools/demos/catalog_depth.py --data.segments 42899799 --rr-config.headless tensorrt
```

Tyro applies flags to the preceding subcommand, so parent flags
(`--data.segments`, `--rr-config.*`) must come before `tensorrt`/`onnx`.

## Tests

`pixi run -e mvs-dev --frozen tests` — golden parity vs CPU ONNX Runtime,
batch-vs-single-sample consistency, and TensorRT-vs-ONNX parity (the model-
and CUDA-dependent tests skip when either is unavailable).
