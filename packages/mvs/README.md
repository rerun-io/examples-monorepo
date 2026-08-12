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

The model artifact lives in the public HF repo `pablovela5620/mvs-depth`;
`mvs-download-depth-model` fetches it with `hf download` (no login needed).

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

## Depth model I/O

Dot-tier ONNX, fp32 I/O with the fp16-autocast split baked into the graph,
opset 17, dynamic batch `b` in 1–32 (the streaming demo runs `b=1`; exactly 7
source views per query). `DepthInputs` field names are exactly the ONNX input
names, so the feed dict cannot drift from the graph.

**Inputs** (`DepthInputs`):

| Name | Shape | What it is | Source in pipeline |
|---|---|---|---|
| `cur_image_b3hw` | b×3×384×512 | Current frame: RGB /255, ImageNet-normalized, bicubic-resized | NVDEC frame → `preprocess_image` (collate) |
| `src_image_bm3hw` | b×7×3×384×512 | 7 source views, same preprocessing | Keyframe buffer (cached GPU tensors) |
| `src_K_bm44` | b×7×4×4 | Source intrinsics at matching scale s1 (128×96) | Geometry fields → `s1_intrinsics` |
| `cur_invK_b44` | b×4×4 | Inverse current K at s1 | Geometry fields → `s1_intrinsics` |
| `src_cam_T_world_bm44` | b×7×4×4 | Source world→camera transforms | Pose fields, composed in the collate |
| `cur_world_T_cam_b44` | b×4×4 | Current camera→world transform | Pose fields, composed in the collate |

**Outputs** (`DepthOutputs`):

| Name | Shape | What it is |
|---|---|---|
| `depth_pred_s0_b1hw` | b×1×192×256 | Metric depth (meters) |
| `lowest_cost_bhw` | b×96×128 | Lowest plane-sweep matching cost (confidence proxy) |

s1 intrinsics: rescale native K to the depth resolution (`K[0]·256/W`,
`K[1]·192/H`), then halve once. TensorRT engines are built per machine and
cached by `trtkit` (keyed on ONNX hash, batch profile, TRT version, GPU arch).

## Tests

`pixi run -e mvs-dev --frozen tests` — golden parity vs CPU ONNX Runtime,
batch-vs-single-sample consistency, and TensorRT-vs-ONNX parity (the model-
and CUDA-dependent tests skip when either is unavailable).
