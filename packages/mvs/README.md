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

The live mesh fuses each depth frame into a sparse colour TSDF (`packages/tsdf-rs`) at the catalog pose, out to
3 m as the original demo does, and logs it with `simplecv.rerun_live_mesh.LiveMeshLog`: the changed blocks every
`--fusion.log-every` depth frames (default 10) while the segment plays, one mesh from the segment's last depth frame.
A run meshes exactly one segment (`--data.segments <id>`): segments restart their frame index at 0, so two in one
recording would mix their mesh histories.

### Layout

The recording carries the layout of the original SimpleRecon live-meshing demo by Pablo Vela
(pablovela5620/simplerecon-inference, `blueprint.py` and `live_meshing.py`), ported in `mvs.blueprint` and
`mvs.frame_views`; a worker thread logs each frame's views, as there:

- **Reconstruction** (left): the growing mesh, the current camera as a green frustum, its 7 source cameras as grey ones.
- **Current view**: the catalog's own AV1 VideoStream, relayed once, never re-encoded.
- **src 0 … src 6**: the frames the depth model matched against. Each is a `VideoFrameReference` into that same
  stream, at the time of the frame the decoder gave the model: no pixels are copied.
- **Depth** (16-bit millimetre PNG), **Normals** (JPEG 75, display only) and **Lowest cost**: the depth of the cost
  volume's best plane at 128×96, a millimetre PNG as well, because it takes only 64 values and stays exact.
- **Stage timings (ms)**: data wait, model forward, TSDF fuse, mesh extract and the log hand-off.

Both tools log the whole video first and then send the blueprint, whose time panel plays `video_time` from 0 at 1×.
The source views resolve on `video_time`, not on `frame`. `--log.no-source-cams` leaves out the filmstrip and the
source frusta. `mvs-catalog-depth` logs the same layout without the filmstrip, mesh and timings.

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
| `lowest_cost_bhw` | b×96×128 | Depth (meters) of the cost volume's best plane, at the matching size |

s1 intrinsics: rescale native K to the depth resolution (`K[0]·256/W`,
`K[1]·192/H`), then halve once. TensorRT engines are built per machine and
cached by `trtkit` (keyed on ONNX hash, batch profile, TRT version, GPU arch).

## Tests

`pixi run -e mvs-dev --frozen tests` — golden parity vs CPU ONNX Runtime,
batch-vs-single-sample consistency, and TensorRT-vs-ONNX parity (the model-
and CUDA-dependent tests skip when either is unavailable).
