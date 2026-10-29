# hexels-bench

Hexels (arXiv 2610.37668) beside Brush 3DGS: upstream checkpoint export to hexel PLY, protocol training,
published-checkpoint renders, the shared NVS scorer, the benchmark queue and its tables, the 2D Image-GS
benchmark and the surface (TSDF mesh) evaluation. Every tool runs in the `hexels` env; the renderer and the
Brush trainer are the `gsplat-rust-renderer` package's binaries. Run directories follow one layout
(`hexels_bench.layout`, methods in `hexels_bench.methods`); published numbers are `hexels_bench/paper-numbers.json`.

## Brush on the Hexels NVS protocol

`hexels-train-brush` trains Brush with the inputs of the Hexels paper's NVS benchmark (Mip-NeRF 360, DTU,
Tanks and Temples). The published camera contract (`jhu-ccvl/Hexels`, `cameras/nvs/<dataset>/<scene>/contract.json`)
fixes the cameras, the every-8th held-out split, and the target size. Preparation decodes and resizes the images like
the upstream 2DGS loader and checks them against the contract's target hashes. Training starts from the SfM points
and caps the splat count at that number; all other Brush settings keep their defaults. Each run writes `model.ply`,
16-bit test renders from Brush's renderer, `cameras.json`, and `train.json` (loop and total seconds, peak training
VRAM, splat count).

```bash
pixi run -e hexels --frozen hexels-train-brush --dataset dtu --scenes scan24
```

## Benchmark queue and table

`hexels-bench` fills the run layout for Brush 7k, Hexels V1/V3 7k (upstream trainer) and the published 30k Hexels
checkpoints (upstream CUDA and our WGSL renders), one `(dataset, scene, method)` directory at a time. Each item runs
its tool through `pixi run -e <env> --frozen` (`gsplat-rust-renderer` for `gsplat render --fps`, `hexels` for the Brush
runner, upstream code and the shared scorer) while holding `flock` on the GPU lock file, appends the output to the
run's `log.txt`, and records failures in `runs/FAILURES.jsonl`. A run whose `metrics.json` scores every held-out view
is skipped, so a stopped queue resumes where it left off. `hexels-bench-table` turns the runs into `TABLE.json` (per
scene and per dataset, the published numbers beside ours) and `table.rrd` (paper-style tables as Rerun data).

```bash
pixi run -e hexels --frozen hexels-bench --scenes scan24 garden Truck     # every method on three scenes
pixi run -e hexels --frozen hexels-bench --methods brush-7k               # one method on every protocol scene
pixi run -e hexels --frozen hexels-bench-table
```
