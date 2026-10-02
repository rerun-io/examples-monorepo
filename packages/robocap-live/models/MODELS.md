# robocap-live hand-net models

2026-10-01. Weights: DetNet-F `detnet-demo-0929.weights.pt` (sha256 `edda4177…`), KeyNet-F `keynet-strong-pinch/best.weights.pt`
(sha256 `067cf7f6…`, with the pinch head, no visibility head). These are the weights the Python s66 runs used.

## Which files the runtime loads

| Backend | Call | DetNet | KeyNet |
|---|---|---|---|
| RK3588 NPU (cap) | `RknnNets::open(models_dir)` | `detnet_b1_fp16.rknn`, 3 contexts (cores 0, 1, 2) | `keynet_b1_int8_mmse.rknn`, 2 contexts (cores 1, 2) |
| ONNX Runtime (hosts) | `OrtNets::new(models_dir, &OrtConfig)` | `detnet_full.onnx` | `keynet.onnx` |

The NPU pair is **FP16 DetNet + INT8 KeyNet**:

- DetNet INT8 moves the detected centres 11.8 px on average on real s66 frames (14.9 px in the simulator). FP16 moves them 0.27 px.
- KeyNet INT8 (mmse calibration) moves the keypoints by 0.44 net px. This is inside the 2026-09-29 study's margin (0.47 net px for
  mmse INT8 on the device). It costs 2.8 ms for 4 crops, against 4.6 ms for FP16.
- `keynet_b1_fp16.rknn` is the drop-in fallback if INT8 noise matters: about 0.01 px from PyTorch. To use it, pass it to
  `RknnNets::with_files`.

## Files

`*.onnx` come from `pixi run -e handtrack python packages/handtrack/tools/export_nets_onnx.py --detnet-weights <pt> --keynet-weights <pt>
--work-dir <dir> --detnet-cache <dir> --keynet-cache <dir>` (opset 17, TorchScript exporter).
`*.rknn` come from `pixi run --manifest-path <rknn-toolkit2 project>/pixi.toml python packages/robocap-live/tools/rknn_convert.py
--work-dir <dir> --keynet-cache <dir> --calibration-dir <dir>` (rknn-toolkit2 2.3.2, target rk3588; that env is its own pixi
project, not an env of the root pixi.toml: the tool's docstring lists what it holds). The runtime on both caps is librknnrt 2.3.2 (429f97ae6b) with RKNPU driver 0.9.8.

| File | sha256 | Bytes | What |
|---|---|---|---|
| `detnet_full.onnx` | `f3bc8c78cda4e1aab7c2fd05e1c340e7b1fe29f18c0863324dbb9747b613eea9` | 5,781,667 | DetNet, full frame with the 4x4 pool inside, dynamic batch (ort) |
| `detnet_pooled_b1.onnx` | `6ed900049d198f9cffda30d9887c16613ca99e6ffb32fdfdbc6cf10c3d03b532` | 5,775,633 | DetNet, pooled input, batch 1 (RKNN source) |
| `keynet.onnx` | `71dc07cde470f5aa8547328d7995eb74ee6a8b18cf6cceb47a1d83b0019df1c4` | 7,773,501 | KeyNet + pinch head, dynamic batch (ort) |
| `keynet_b1.onnx` | `4974b4403f46cd9ec7898f78c51a592100521751db5549774f2efbc3b4d56bb5` | 7,773,495 | KeyNet, batch 1 (RKNN source) |
| `keynet_b4.onnx` | `aeb1c4baa852718d34b186dd1ae5abe9be6f24ccbb4aa28b5e046d259c02db6e` | 7,773,495 | KeyNet, batch 4 (RKNN source) |
| **`detnet_b1_fp16.rknn`** | `c1a0ffe41d57975f71ef3bb0597802f367620da5699bb8843daced1d2dbf19a9` | 3,371,706 | DetNet FP16 (**loaded**) |
| `detnet_b1_int8.rknn` | `306f037be04266252c35b91c87a09bd2e928521e21d8f1e2cd493943645cc6b5` | 2,017,454 | DetNet INT8 (rejected: see accuracy) |
| **`keynet_b1_int8_mmse.rknn`** | `60df340a6ecaa23c685377dafac085a83d1583ad3b303dab1ff86d5a4263058d` | 2,651,547 | KeyNet INT8, mmse calibration (**loaded**) |
| `keynet_b1_int8.rknn` | `39b625698f7f4967724cd3a90e50dfcd7581708078c842bae49c4725f108e4d6` | 2,651,547 | KeyNet INT8, normal calibration |
| `keynet_b1_fp16.rknn` | `5af64984d2db7d33a420f07a2005c516e558ade02464fc84e936f3cc8b965357` | 4,487,501 | KeyNet FP16 (fallback) |
| `keynet_b4_int8.rknn` | `f10be983bcadd7204ac8e7cc58c39fa733dc95e397442d8b14e242d40eb0a4ba` | 3,741,083 | KeyNet INT8, 4 crops per call (slower than 2 cores x batch 1; unused) |
| `keynet_b4_fp16.rknn` | `09c9b54dea948af48aec5b944aaa8fc8b3740d8fb6cded6bc43580fc58f0bad2` | 5,574,413 | KeyNet FP16, batch 4 (unused) |

`export_report.json` lists the ONNX checks. `rknn_report.json` holds the build records, the simulator scores and the device scores.

INT8 calibration uses three sets:

- the 2026-09-29 study's sets: 256 pooled DetNet frames and 256 KeyNet crops with their priors, from the UmeTrack and SHOW3D
  training caches;
- 87 DetNet frames from s66;
- 463 KeyNet crops from s66.

The s66 inputs are the odd-numbered samples of what handtrack's Python tracker (all 6 cameras) actually fed the
nets on the s66 clip. The even-numbered samples are the s66 evaluation set.

## Inputs and outputs

All shapes are row-major. Slot 0 is the left hand.

**DetNet**

- ONNX input `image`: f32 `[b,1,480,640]` (full) or `[1,1,120,160]` (pooled). Values are the 640x480 BarLetterbox frame / 255.
  The pooled input is the 4x4 mean of that frame.
- RKNN input: the same pooled `[1,1,120,160]`, **on the u8 scale**. The `/255` is inside the model (mean 0, std 255).
  - FP16 models get **f32, unrounded**: `RknnNets` computes the 4x4 mean with `pool4_mean_f32`.
  - INT8 models get u8: `pool4_u8`, rounded half to even like the training cache.
  - Why the f32 path matters: rounding the pooled input to u8 alone moves DetNet's centres by 2.9 px mean (p90 8.8 px) on s66
    frames.
- Outputs, f32:
  - `center` `[b,2,2]`: (cx / 640, cy / 480) in the net frame;
  - `radius` `[b,2]`: radius / 640;
  - `presence_logit` `[b,2]`.
- Decode with handtrack's `decode_detections`.

**KeyNet**

- Inputs:
  - `crop` `[b,1,96,96]`: [0, 1] in ONNX. RKNN takes it on the u8 scale with the `/255` inside the model. The FP16 model gets
    `crop * 255` as f32. The INT8 model gets `round(crop * 255)` as u8.
  - `keypoints` `[b,63]`: f32, the 21 x (u, v, d) prior, or zeros when untracked. Never normalised.
- Outputs, f32:
  - `heatmaps` `[b,21,18,18]`;
  - `distance` `[b,21,18]`;
  - `presence_logit` `[b,1]`;
  - `pinch_logit` `[b,1]`.
- Decode with handtrack's `decode_heatmaps` (log-quadratic peak) and `decode_distance`. A Rust port is in `nets::golden::decode_heatmaps`.
- In the ONNX graph, the pooled feature vector is `ReduceMean(keepdims=1)` + `Flatten`, so that RKNN can convert it. The math is the
  same as `KeyNetF.forward`.

## Latency

All times are milliseconds, p50 (p90 in brackets), wall clock per call through the `HandNets` trait.

- The process is pinned to the A76 cores (`taskset -c 4-7`).
- DetNet times include the CPU 4x4 pool.
- The governors were the cap defaults: NPU `rknpu_ondemand`, DDR `dmc_ondemand` (534 MHz when idle). Cap A uses the same defaults.
  The 2026-09-29 study measured the small nets 1.5-2x faster with the governors at `performance`.

**Cap B** (`robocap_fe62fa`, idle, 2026-10-01). Cap A dropped off the network at 00:39 (a power loss), so the
full matrix was measured on Cap B. The FP16 rows use the deployed input path: fp16 pass-through in the native NHWC layout,
commit `c41090e7`. The rows in brackets marked "f32 in" are the earlier path, which converted the input in the runtime.

| | FP16 | INT8 |
|---|---|---|
| DetNet, 1 frame (core 0) | **2.75 (3.54)**; f32 in: 3.3 (4.1) | 1.6-1.7 (1.8) |
| DetNet, 6 frames over cores 0-2 | **4.44 (4.58)**; f32 in: 4.8; one core, f32 in: 20.6 | - |
| DetNet phases, 1 frame | inputs_set 0.02 (pass-through; f32 in: 0.3-1.1), NPU 1.5-1.9, outputs_get 0.13 | NPU 1.39 on Cap A |
| KeyNet, 1 crop | 3.0 (3.9); f32 in: 3.2 | **1.8 (2.2)** |
| KeyNet, 2 crops (cores 1 + 2) | 2.35 (3.3) | **1.51 (1.60)** |
| KeyNet, 4 crops (cores 1 + 2) | 4.6 (6.5) | **2.81 (2.97)** |
| KeyNet, 8 crops (cores 1 + 2) | 9.3 (11.2) | 5.45 (7.07) |
| KeyNet phases, 1 crop | inputs_set 0.03 (pass-through; f32 in: 0.16-0.57), NPU 1.6-2.4, outputs_get 0.37 | inputs_set 0.09, NPU 1.65, outputs_get 0.23 |
| KeyNet, batch-4 model, 4 crops in one call on core 1 | 10.5 (f32 in) | 5.2-5.5 |
| KeyNet throughput, 2 contexts in parallel | 613 crops/s (f32 in) | 1,042-1,076 crops/s |
| CPU 4x4 pool, 640x480 to 120x160 | `pool4_mean_f32` 0.044 | `pool4_u8` 0.05-0.12 |

`rknn_native_probe` checks the pass-through path for an FP16 model. It feeds the model the same image through the f32 path and
through the native-layout path, at two value scales. On Cap B, `u8 / 255` gives a max output difference of 0.00000 against the f32
path. The u8 scale is wrong: the normalisation is not applied on the pass-through path.

**Cap A** (`robocap_f403b0`, 00:38, before it lost power). INT8 models with the study calibration; the FP16 runs had not started:

| | p50 |
|---|---|
| DetNet INT8, 1 call | 1.59 (NPU 1.39) |
| KeyNet INT8, 1 crop | 1.79 (NPU 1.47) |
| KeyNet INT8 batch 4 | 5.19 |
| 2 KeyNet contexts | 1,139 crops/s |
| CPU pool | 0.12 |

Cap A and Cap B agree within about 10 %.

**An RTX 3060 host**: ort 2.0.0-rc.11 driving onnxruntime-gpu 1.29.0, CUDA EP, TF32 off.

| | p50 |
|---|---|
| DetNet, 1 frame | 2.0 |
| DetNet, 6 frames | 7.1 |
| KeyNet, 1 crop | 1.96 |
| KeyNet, 2 crops | 2.00 |
| KeyNet, 4 crops | 2.08 |
| KeyNet, 8 crops | 3.15 |

A tracked frame (4 crops) costs about 2.8 ms of NPU on the cap. An acquisition frame adds about 2.75 ms of DetNet for one camera, or
4.4 ms for six cameras spread over the three cores. The FP16 pass-through outputs are bit-identical to the f32 path on the device:
the scores below are the same with either path.

## Accuracy against PyTorch FP32

Every comparison is against the same weights in PyTorch FP32 on the CPU, on the same inputs. The scores are decoded the way
handtrack decodes them.

- **held-out**: the training caches' held-out samples, 1,680 DetNet frames and 2,577 KeyNet crops (2,159 positive), UmeTrack + SHOW3D.
- **s66**: the real Cap A inputs that the Python tracker fed the nets on the s66 clip, 88 frames and 463 crops (426 with a hand).
  DetNet gets the real 640x480 frames, pooled the way the runtime pools them. KeyNet gets the unrounded perspective crops.
- DetNet:
  - centre and radius errors are over the slots that PyTorch reports present, in 640x480 net px;
  - presence agreement is over all slots (logit > 0).
- KeyNet:
  - keypoint shift is |decode(NPU) - decode(PyTorch)| over positive crops, in 96 px crop px;
  - "net px" maps the same shift through the crop's linear part (the study's rule);
  - GT is the mean ground-truth error in net px, PyTorch → NPU (held-out only);
  - the pinch columns compare sigmoid probabilities.

**ONNX Runtime**:

- Python ORT CPU on the full held-out sets:
  - DetNet max abs diff: center 2e-6, radius 3e-7, presence 6e-5;
  - KeyNet max abs diff: heatmaps 5e-6, distance 6e-6, presence 5e-5, pinch 3e-5.
- Rust `OrtNets` on the golden files:
  - CPU on an AMD Ryzen 9 9950X3D host: centre max 2e-4 px, keypoints max 1.1e-5 px;
  - CUDA on the RTX 3060 host: centre max 1.4e-4 px, keypoints max 1.4e-5 px, pinch probability max 1e-7.

**DetNet on RKNN** (sim = RKNN-Toolkit2 simulator, dev = Cap B through the Rust backend):

| Model | Set | Centre mean / p90 / max px, sim | Centre mean / p90 / max px, dev | Radius mean px, dev | Presence agreement, dev |
|---|---|---|---|---|---|
| **FP16** | held-out | 0.12 / 0.21 / 2.45 | 0.14 / 0.24 / 2.45 | 0.05 | 100 % |
| **FP16** | s66 | 0.24 / 0.49 / 1.02 | 0.27 / 0.52 / 1.20 | 0.05 | 100 % |
| INT8 | held-out | 3.75 / 6.73 / 50.5 | 4.01 / 7.24 / 51.9 | 1.41 | 99.7 % |
| INT8 | s66 | 14.9 / 44.3 / 55.9 | 11.8 / 35.2 / 56.4 | 2.20 | 97.7 % |

**KeyNet on RKNN**:

| Model | Set | Keypoint crop px mean / p90, dev | Net px, dev | GT net px, dev | d_rel mm, dev | Presence agreement, dev | Pinch prob error mean / p90, dev | Pinch agreement, dev |
|---|---|---|---|---|---|---|---|---|
| **INT8 mmse** | held-out | 0.285 / 0.537 | 0.437 | 4.326 → 4.358 | 0.42 | 99.96 % | 0.0105 / 0.032 | 99.2 % |
| **INT8 mmse** | s66 | 0.290 / 0.551 | - | - | 0.48 | 100 % | 0.0017 / 0.0027 | 100 % |
| INT8 normal | held-out | 0.374 / 0.722 | 0.577 | 4.326 → 4.373 | 0.44 | 99.96 % | 0.0154 / 0.048 | 98.5 % |
| INT8 normal | s66 | 0.373 / 0.707 | - | - | 0.50 | 99.8 % | 0.0025 / 0.0026 | 100 % |
| FP16 | held-out | 0.011 / 0.019 | 0.017 | 4.326 → 4.326 | 0.01 | 100 % | 0.0005 / 0.0014 | 100 % |
| FP16 | s66 | 0.010 / 0.019 | - | - | 0.02 | 100 % | 0.0001 / 0.0001 | 100 % |

The simulator matches the device to within 0.01-0.02 px for KeyNet. For example, mmse held-out is 0.286 crop px and 0.433 net px
in the simulator.

The 2026-09-29 study's margins were measured on the device for the older KeyNet `3d41b6d3`:

- INT8 vs FP32: 0.740 net px (normal) and 0.467 (mmse);
- INT8 GT cost: UmeTrack +0.022 px, SHOW3D +0.14 px (about +0.10 combined);
- DetNet INT8: 3.69 px vs FP32 on the held-out frames.

KeyNet INT8-mmse is inside those margins: 0.437 net px and a GT cost of +0.032 px. DetNet INT8 is not: 11.8 px on s66.

## Running

- **Cap.** `RknnNets::open("/root/robocap-live/models")` dlopens `/usr/lib/librknnrt.so`. The bench:
  `nets_bench rknn /root/robocap-live/models --golden <tests/data/nets> [--heldout <dir> --out <dir>]`.
  The golden test: `ROBOCAP_LIVE_MODELS=… ROBOCAP_LIVE_NETS_GOLDEN=… nets_rknn-<hash>`.
- **A CUDA host (ort CUDA).** Build with `--features ort`. ONNX Runtime and its CUDA libraries come from the monorepo's
  `handtrack` pixi env (no pip):
  ```
  ENV=<checkout>/.pixi/envs/handtrack
  ORT_DYLIB_PATH=$ENV/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.29.0 LD_LIBRARY_PATH=$ENV/lib \
    nets_bench ort <models dir> --device cuda
  ```
  `<checkout>` is a monorepo checkout with the `handtrack` env installed. `--device cpu` works without a GPU.
- **Scoring device outputs.** `rknn_convert.py --work-dir <dir> --keynet-cache <dir> --calibration-dir <dir> --score-device <out dir> --device-set heldout|s66 --device-label <label>`.
