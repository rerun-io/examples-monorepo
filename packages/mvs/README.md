# mvs

Multi-view stereo depth over ARKitScenes segments served by the Rerun cloud
catalog (CUDA-only: linux-64 + linux-aarch64).

One `RerunIterableDataset` carries every model input as a `Field`: the AV1
video, decoded straight to CUDA tensors through NVDEC, plus the six pose and
calibration columns — all riding a single fetch query. A stateful collate
composes per-frame poses, keeps a causal keyframe buffer, and assembles the
depth queries. The depth model is a plain ONNX artifact; `trtkit` runs it as a
cached machine-local TensorRT engine (the default) or on ONNX Runtime CUDA (the
`onnx` fallback).

## Run

```bash
pixi run -e mvs --frozen mvs-catalog-depth   # depth demo (first use downloads the model and builds the TensorRT engine)
pixi run -e mvs --frozen mvs-live-mesh       # depth fused into a TSDF mesh, logged as it grows
```

The live mesh fuses each depth frame into a sparse colour TSDF (`packages/tsdf-rs`) at the catalog pose and logs
it with `simplecv.rerun_live_mesh.LiveMeshLog`: the changed blocks on the timeline while the segment plays (scrub
back to watch it grow), one mesh from the segment's last depth frame. The source video is relayed once as the
catalog's own AV1 VideoStream. Every depth frame is also logged as `mvs-catalog-depth` logs it: a 256×192
`pinhole_lowres` Pinhole and its depth as a millimetre PNG. Both tools send one default layout
(`mvs.catalog.blueprint`): the 3D scene with the moving camera beside the video and the predicted depth. A run
meshes exactly one segment (`--data.segments <id>`): segments restart their frame index at 0, so two in one
recording would mix their mesh histories.

The model artifact lives in the public HF repo `pablovela5620/mvs-depth`;
`mvs-download-depth-model` fetches it with `hf download` (no login needed).

### Backend

Both tools run the model as a TensorRT engine by default. The first run builds the engine and caches it per machine
(`~/.cache/trtkit/trt`, or the folder in `TRTKIT_TRT_CACHE`); later runs load it. On an RTX 5090 the first run of
`mvs-live-mesh` spends 112 s building the engine and takes 142 s in all; a later run of the same 30 s capture
(42897542, 1,766 depth frames) takes 27 s and peaks at about 5.4 GiB (5,578 MiB) of GPU memory.

The default engine has a batch-1 profile (`mvs.depth_engine.SMALL_TRT_BACKEND`): both tools feed the model one
frame at a time, and an engine keeps its buffers at its largest batch. trtkit's own default profile (max batch 32)
holds about 11 GiB. For a larger profile, name the `tensorrt` subcommand and set both batch flags; each profile builds
its own cached engine once:

```bash
pixi run -e mvs --frozen mvs-catalog-depth --data.segments 42897542 --rr-config.headless \
  tensorrt --backend.max-batch-size 8 --backend.opt-batch-size 8
```

`onnx` runs the model on ONNX Runtime CUDA. It needs no build, but it is about 12x slower on this model: ONNX
Runtime has no CUDA kernel for its 114 BatchNormalization nodes, so they and the casts around them run on the CPU. On the same RTX 5090 and
capture the live mesh takes 333 s instead of 27 s:

```bash
pixi run -e mvs --frozen mvs-live-mesh --data.segments 42897542 --rr-config.headless onnx
```

Tyro applies flags to the preceding subcommand, so parent flags
(`--data.segments`, `--rr-config.*`) must come before `tensorrt`/`onnx`.

## Depth model I/O

Dot-tier ONNX, fp32 I/O with the fp16-autocast split baked into the graph,
opset 17, dynamic batch `b` (the tools run `b=1`; exactly 7
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
