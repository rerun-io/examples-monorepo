"""4DAnyone inference: the run-path copy used by fdanyone_rerun.

Copied from ant-research/4DAnyone 6dcf626 (Apache-2.0, LICENSE beside this file) with the speed levers of the
4DAnyone-5090 port (branch upstream-fast @ 9c02b1e): ultrafast lossless working clip and spot-check verifies,
GPU-batched conditioning resize, veryfast skeleton x264, publish overlap, TAEW2.2 target decode, DiT/VAE prefetch,
FP32 RoPE, row-wise FP8 W8A8 (default), opt-in 3-step Turbo, the STEP_HOOKS seam, GVHMR REVISION and inline workers.
Every module reachable from the pipeline and its workers is kept (all view layouts, RCP, TCR, multi-GPU); upstream's
GUI (fdanyone/space) and nerfstudio export are not copied.
"""
