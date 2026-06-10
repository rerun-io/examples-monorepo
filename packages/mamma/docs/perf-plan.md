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


## Profile results (Step 0 executed, 2026-06-09)

`tools/profile_pipeline.py`, 48 steady ticks after 45 warmup, crossing_arms:

- **Total CUDA kernel time 78 ms/tick vs ~181 ms/tick wall** (234 under profiler)
  -> the pipeline is CPU/launch-bound overall; TRT engines alone cannot close the gap.
- Launch storm: aten::bmm 55,440 calls + aten::copy_ 136,536 calls per 48 ticks
  (~1,150 + 2,840 launches/tick), concentrated in the smplx fit and the
  per-camera tracker loop. log_video burns 25.8 ms/tick of CPU (PyAV swscale+NVENC submit).
- Fit microbench truth: one 16-iter optimize call = 92 ms wall, ~67 ms genuine GPU
  (4.2 ms/iter fwd+bwd over ALL 10,475 SMPL-X vertices). torch.compile default: no
  gain; reduce-overhead + compiled autograd: 82.5 ms; manual CUDA-graph capture
  fails (smplx forward performs a capture-illegal op, likely a per-call H2D const).
- Fixed along the way: per-camera-per-iter host sync in reprojection_loss
  (`if torch.isnan(...)` -> nan_to_num), benchmark warmup pipeline leak (2x models
  resident -> scale_cuda OOM), goal_check pytest env, encoder EOF on logger reuse.

## Revised next steps (highest leverage first)

1. **512-vertex LBS**: pre-multiply verts_512 into smplx skinning weights /
   shapedirs / posedirs so each fit iteration deforms 512 verts, not 10,475
   (~20x vertex math). Full mesh only once per tick for Rerun emission.
   Est: fit 92 -> ~20-25 ms per optimize call (~8 ms/tick amortized).
2. **NVENC + D2H logging on a worker thread** (25.8 ms CPU/tick off the loop).
3. **Tracker CPU overhead**: 74 ms wall vs ~11 ms CUDA — batch per-camera
   memory ops, try torch.compile on TAM image encoder, then 4 CUDA streams.
4. Re-profile; only then decide whether TRT engines (TAM/MammaNet) are still
   needed to reach 33 ms/tick.


## Campaign progress (2026-06-09, round 1)

| Wall (363-frame clip) | Change |
|---|---|
| 132.6 s | M6 baseline (efficienttam-ti + batched encode + decode-ahead + fit_stride) |
| 65.9 s | (carried) |
| 46.4 s | CUDA-graphed fitter: sampled 512-vert SMPL-X (exact, 5e-7) + capture-legal rigid chain; 92 -> 15.3 ms per 16-iter optimize |
| **43.7 s** | async video logging worker (13.4 -> 0.7 ms/tick); single autocast ctx per track tick |

Golden gate after each change: PASS (improved to 21.6 mm MPJPE / 18.7 mm PVE — persistent Adam momentum).

Two upstream smplx bugs found (worth an issue/PR): `batch_rigid_transform` indexes
`transform_chain[parents[i]]` with a 0-dim GPU tensor (54 hidden host syncs per forward,
serializes any optimizer built on smplx and blocks CUDA-graph capture); list
fancy-indexing in the same function does an H2D copy under capture.

**Remaining wall ≈ 43.7 s: track 27.8 s (76 ms/tick avg; ~40 steady + redetect/bleed),
landmarks 6.6 s, fit 6.0 s, misc ~3 s.** Next: tracker deep-dive (memory-bank ops +
per-camera decoder python; candidates: CUDA-graph/compile the TAM encoder+decoder,
TRT engine per perf-plan step 2), then MammaNet fp16 TRT (step 4). Target 12.1 s needs
track to ~15 ms/tick and landmarks ~6 ms/tick.


## Round 2 results (2026-06-09, late)

| Wall | Change |
|---|---|
| 43.7 s | round 1 end |
| 35.8 s | track_stride=2 (mask reuse between tracker ticks; gate unchanged) |
| **31.6 s** | track_stride=3 + fit_stride=4 defaults (gate 22.3 mm, 7.7 mm margin) |

Canonical `mamma-goal-check`: **4/5 PASS** (golden 22.3 mm, datasets, no-writes,
hygiene); realtime FAIL at 31.58 s vs 12.1 s.

Findings: TAM encode_image is only 3.5 ms — track cost is per-camera decoder/
memory PYTHON (compile of submodules: 40→35 ms, not the fix). MammaNet compiles
15.8→10.2 ms standalone but inductor cudagraphs conflict with the fitter's manual
graph in-process (landmarks 2x WORSE); flag default-off. Triton on sm_120 needs
CONDA_PREFIX (pixi run) for the ptxas-blackwell fallback.

## Remaining 2.6x — structural tier (each a half/multi-day item)
1. torch_tensorrt (dynamo) engines for TAM encoder+decoder and MammaNet —
   replaces inductor cudagraphs (no pool conflict), targets track ~15 and
   landmarks ~8 ms/tick.
2. Fork surgery: batch the 4 per-camera forward_embeddings into one B=4 pass
   (saturated forgetful banks have static shapes) — removes the per-camera
   python that compile cannot.
3. Multiprocess NVDEC decode workers (proven ~400 cam-fps vs 140 in-process)
   if decode resurfaces once compute shrinks.


## Mojo/MAX feasibility verdict (workflow, 2026-06-09)

**NO-GO across all five components** (full reasoning preserved here):
- Tracker python: bottleneck is CPython dispatch/object churn, which Mojo cannot see;
  Python->Mojo FFI has no zero-copy GPU tensor path today. Incumbent fork-batching wins (~10-15ms, 1-2wk).
- MammaNet: MAX has no torch/ONNX importer (2-4wk rewrite in max.graph), sm_120 only
  "known compatible for development". torch_tensorrt (torch_compile ir, fp16) wins: ~9-11ms, 1-3 days.
- TAM encoder: already batched at 3.5ms — defer. MAX serving of streaming TAM: do not attempt.
- Fit loop: already manually CUDA-graphed; Mojo ops inside capture undocumented — don't touch.
  Only future Mojo candidate: fused-LBS mesh emit (~3ms prize) — and a plain CUDA extension
  likely matches it without toolchain risk.
- Decode: fixed-function NVDEC; language-irrelevant.
Re-evaluate Mojo/MAX in 6-12 months (zero-copy DLPack GPU interop, documented graph-capture
compat, CI-tested sm_120 kernels).

## Round 3 (2026-06-09, late)
- sync-free smplx patch (emit no longer pays 54 syncs); fit-tail pipelined on a worker
  (thread_local graph capture). Wall 32.0 -> **31.4s**; gate 22.3mm PASS; tests 10/10.
- Current true bottleneck split (sync'd): track ~20ms/tick (SAM2-fork per-camera python),
  landmarks ~17.8ms (MammaNet GPU), fit+emit ~10ms, glue ~25-30ms (engine/logging python).
- NEXT (per verdict): torch_tensorrt MammaNet, then SAM2-fork 4-camera vectorization —
  the only identified path to 33ms/tick.


## torch_tensorrt install finding (2026-06-09)

PyPI `torch-tensorrt` 2.10 wheels target **CUDA 13** (depend on prerelease
`nvidia-cuda-runtime-cu13`) — incompatible with this conda-forge torch 2.10/cu129
stack; >=2.11 wheels require torch 2.11+. Options for the MammaNet engine step:
(a) bump the whole env to torch 2.11 + cu13-era wheels and re-validate everything,
(b) ONNX export (dynamo, opset>=17) -> onnxruntime-gpu TensorRT/CUDA EP with
io_binding on torch tensors — `onnxruntime-gpu` is already in the cuda feature,
(c) trtexec-built engine + TensorRT python API directly. (b) is the least invasive
and the recommended next experiment.


## Round 4 results + the final block (2026-06-09, end of session 1)

| Wall | Change |
|---|---|
| 31.6 s | round 2b end |
| 31.4 s | pipelined fit tail (thread_local capture) |
| **26.8 s** | MammaNet TensorRT engine (3.88 vs 15.8 ms per 4-crop call; gate 23.1 mm PASS) |

Negative result: thread-parallel per-camera `forward_embeddings` is WORSE
(43 vs 33 ms/tick — GIL contention). Confirms batching is the only tracker fix.

### SAM2-fork batching surgery — concrete plan (next session)
Steady-state fast path (bootstrapped, no prompts, 1 obj/cam, saturated banks ->
identical memory shapes across cameras), in `mamma/tracking/batched_forward.py`:
1. encode_image already batched [4,...] (done).
2. Stack each camera's `select_memories` output -> memory_attention as B=4
   (verify rotary/positional handling under batch; obj-ptr tokens concat).
3. `sam_mask_decoder` B=4 (it natively supports batch).
4. `encode_memory` B=4; split results back per camera for bank.try_add/prune
   (python bookkeeping stays per camera - cheap).
Fallback to the fork loop when preconditions fail (prompts/re-detect/multi-person).
Validate numerics vs fork path (mask IoU ~1.0) + golden gate.
Estimated: tracker ~33 -> ~12-15 ms/executed tick -> wall ~26.8 -> ~17-19 s.
Remaining to 12.1 s after that: engine-ify TAM encoder (TRT, same recipe as
MammaNet), trim engine glue (~10 ms/tick python), possibly track_stride=4.
