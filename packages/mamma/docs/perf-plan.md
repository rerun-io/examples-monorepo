# Mamma realtime campaign — measurement protocol + ranked plan

Synthesized 2026-06-09 from NVIDIA TensorRT 11 benchmarking + optimization docs
(docs.nvidia.com/deeplearning/tensorrt/11.0.0/performance/{benchmarking,optimization}.html)
by the trt-perf-research workflow, targeted at this pipeline (RTX 5090 / CUDA 12.9).

Baseline: 65.9 s wall for 363 ticks = **181 ms/tick**; stage timers (async-bleed
caveat) attribute ~175 ms: track 74, fit 68 (stride-amortized), landmarks 17.6,
log_video 13. Goal: 33 ms/tick (12.1 s).

## Step 0 — Measurement protocol (before any optimization)

1. Lock clocks for every measurement AND every TRT engine build:
   `sudo nvidia-smi -lgc <max_sm> && sudo nvidia-smi -lmc <max_mem>` (find via
   `nvidia-smi -q -d SUPPORTED_CLOCKS`); watch throttle with `nvidia-smi dmon -s pcu`.
2. Per-stage CUDA events with ONE `torch.cuda.synchronize()` per tick — wall
   timers around async launches measure enqueue, not GPU time. Report p50/p95
   over 50 ticks after 10 warmup. Attribution rule: `sum(stage GPU ms)` vs wall;
   the difference is CPU/H2D/sync and is Step 1's target.
3. Trace capture (torch.profiler chrome trace via `tools/profile_pipeline.py`;
   nsys with `--cuda-graph-trace=node --gpu-metrics-device all` if installed).
   Read for: everything-on-stream-0 (zero overlap), inter-kernel gaps
   (CPU-bound enqueue), long memcpy bars (H2D), Tensor-Core active % in ViT.
4. Standalone `trtexec` baselines per network (`--fp16 --builderOptimizationLevel=4
   --noDataTransfers --timingCacheFile=*_sm120.timing.cache --dumpProfile
   --dumpLayerInfo`); unfused MatMul+Softmax+MatMul triples = fusion failed.

## Ranked steps (expected ms/tick saved)

1. **Kill hidden CPU/H2D/sync time (est. 50–90 ms)** — pinned host buffers
   allocated once + `non_blocking=True` copies; remove every mid-tick
   `.item()`/`.cpu()`/`synchronize()`; move stages off the default stream.
   Diagnostic-driven: read the timeline, don't guess.
2. **TRT FP16 EfficientTAM-ti encoder+propagation (40 → ~8–12 ms)** — compile
   the per-frame step only; memory bank stays as Python tensors fed as engine
   inputs. Fixed 512x512 → min=opt=max. Prefer
   `torch_tensorrt.compile(ir="dynamo", enabled_precisions={torch.float16},
   use_explicit_typing=True)`. Also: run 4 cameras on 4 CUDA streams.
3. **NVENC on a dedicated stream (14 → ~0 effective)** — encode tick N-1 while
   tick N infers; NVENC is separate hardware; sync only on the frame-write event.
4. **TRT FP16 MammaNet ViT-B/16+DETR (18 → ~4–6 ms)** — dims are Tensor-Core
   aligned; TRT 11 IAttention covers head size 64. Export with dynamo/opset>=17
   so SDPA stays fusable. One profile minShapes=1x3x512x384 opt/max=4x...
   Validate landmarks vs torch within mm tolerance before deploying.
5. **CUDA-graph the SMPL-X fit step (~22 → ~15 ms amortized)** — capture the
   full fixed-iter fwd+bwd+step loop; no early-exit; static buffers via
   `.copy_()`; `zero_grad(set_to_none=False)`. Smallest win, do last; the
   bigger residual levers are fewer iters or fitting async on its own stream.

Projected: ~28–38 ms/tick — the 33 ms target is achievable but tight.

## Do NOT bother

INT8/FP4 (FP16 projects to target; mm-scale calibration risk), max_aux_streams
(ViT layers strictly sequential), dynamic spatial shapes (crops are fixed),
preemptive BF16, TRT for the fit loop (training-style fwd+bwd — graphs are the
tool), detailed profilingVerbosity in production engines, TorchScript IR /
enqueueV2 / kFP16 (removed in TRT 11), reusing engines across GPUs (name
`*_trt11_sm120.plan`, keep in gitignored `.trt_cache/`, rebuild via pixi task).
