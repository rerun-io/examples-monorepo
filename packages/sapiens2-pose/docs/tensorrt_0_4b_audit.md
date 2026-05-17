# Sapiens2 Pose 0.4B TensorRT Path

Updated: 2026-05-16

## Retained Path

The codebase now keeps the fastest strict-accuracy floating-point path from the 0.4B sweep:

- Precision: BF16 TensorRT
- Engine profile: static batch 1
- Runtime mode: CUDA Graph replay
- ONNX source: portable static batch-1 Sapiens2 pose graph
- TensorRT engine: machine-local, rebuilt per target GPU/driver/TensorRT version
- Video app default engine path: `${XDG_CACHE_HOME:-~/.cache}/sapiens2-pose/tensorrt/sapiens2_0_4b_pose_static_b1_bf16_current_static_graph.trt`

The prior FP8, INT8, INT4, NVFP4, per-layer precision forcing, dynamic batch, and layer-profiling experiment code was removed because it did not contribute to the current deployable path.

## Evidence

The retained BF16 engine passed the 5% output tolerance:

- Strict artifact comparison: max scalar relative `0.026866`
- Visible keypoint RRD comparison: max bbox fraction `0.036130`
- Best measured raw TensorRT heatmap median: about `26.8-27.3 ms`
- Best measured full video pose path with DETR boxes and TensorRT pose: about `91 ms/frame`

Reference artifacts from the optimization run remain under `/tmp/sapiens2_pose_trt_goal/`:

- ONNX: `/tmp/sapiens2_pose_trt_goal/onnx/sapiens2_0_4b_pose_dynamo_static_b1.onnx`
- Engine: `/tmp/sapiens2_pose_trt_goal/trt/sapiens2_0_4b_pose_static_b1_bf16_current_static_graph.trt`
- Manifest: `/tmp/sapiens2_pose_trt_goal/trt/sapiens2_0_4b_pose_static_b1_bf16_current_static_graph.trt.json`
- Baseline artifacts: `/tmp/sapiens2_pose_trt_goal/baseline/`

## Commands

Export ONNX:

```bash
pixi run -e sapiens2-pose-dev --frozen sapiens2-pose-export-onnx /tmp/sapiens2_pose_trt_goal/onnx/sapiens2_0_4b_pose_static_b1.onnx
```

Build BF16 TensorRT:

```bash
pixi run -e sapiens2-pose-dev --frozen sapiens2-pose-build-trt /tmp/sapiens2_pose_trt_goal/onnx/sapiens2_0_4b_pose_static_b1.onnx /tmp/sapiens2_pose_trt_goal/trt/sapiens2_0_4b_pose_static_b1_bf16.trt
```

Run TensorRT image inference:

```bash
pixi run -e sapiens2-pose-dev --frozen sapiens2-pose-trt-image --image-path <image> --engine-path <engine.trt> --rrd-path <out.rrd> --artifact-path <out.npz>
```

Benchmark retained path:

```bash
pixi run -e sapiens2-pose-dev --frozen sapiens2-pose-benchmark-trt --image-path <image> --baseline-artifact-path <baseline.npz> --engine-path <engine.trt>
```

## Validation

Required checks for code changes:

```bash
pixi run -e sapiens2-pose-dev --frozen ruff check packages/sapiens2-pose
pixi run -e sapiens2-pose-dev --frozen pytest -q packages/sapiens2-pose/tests
```
