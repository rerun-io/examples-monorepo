# metrictrail

NVIDIA MetricTrail is a monocular SLAM method in ViPE (`pipeline=metrictrail`). It is DPVO with two frozen priors:
MoGe-2 metric depth starts each new patch at a depth in metres and holds the bundle adjustment to that scale, and
DINOv3 ViT-S/16 features improve the patch matching. The output is camera poses and intrinsics at metric scale, plus
per-frame metric depth.

This package holds our code around upstream: benchmark presets, a run CLI with a Rerun logger, the VSLAM-LAB / paper
evaluation and the catalog layer. ViPE itself comes from `recipes/vipe-src`.

## Packaging

| Part | Where it comes from | Why |
| --- | --- | --- |
| ViPE + MetricTrail | `recipes/vipe-src`: the GitHub archive of ViPE `ead2e00`, sha256-pinned | `ead2e00` is the first commit with `vipe/metrictrail`; no release has it (v1.2.0 predates it). |
| CUDA kernels | the recipe compiles `vipe_ext` (without `lietorch_ext`) and MetricTrail's DPVO `cuda_corr` + `cuda_ba` at package build time (Python 3.12, conda-forge torch 2.12 cuda130, sm_86 + sm_120 + PTX) | upstream's `python -m vipe.metrictrail.build` step becomes part of `pixi install`. The stereo kernels are not built. |
| lietorch | patch 0001: ViPE's three vendored lietorch copies import `lietorch_mojo`; `build.sh` deletes the vendored code and `lietorch_backends` is not built | one lietorch for the workspace (lietorch-mojo). ViPE's copies are code-identical to upstream lietorch. |
| DINOv3 | patch 0002: `recipes/dinov3-src` (a run dependency) | the same ViT-S/16; patch tokens are bit-identical. |
| MoGe-2 (`moge2-l`) | patch 0003: `packages/monoprior`'s `MoGeModel.from_checkpoint` on ViPE's checkpoint (`Ruicheng/moge-2-vitl` @ `39c4d5e`); `build.sh` deletes ViPE's copy | the same code: depth, mask and focal are bit-identical on 48 TUM frames in fp16 and fp32. |
| DPVO | the recipe (ViPE's `vipe/metrictrail/dpvo`, `csrc/dpvo`), not `packages/dpvo` | MetricTrail's DPVO is a functional fork: metric-depth local BA, a metric global-BA backend, about 35 new config keys and a `cuda_ba` with 4 more arguments and a new kernel. `packages/dpvo` is a typed rewrite on Python 3.11. MetricTrail does not use DBoW loop closure, so `packages/dpretrieval` is not used. |

Weights are on the NAS (`/mnt/nas/ssd-8tb/metrictrail/weights`, `MANIFEST.md` beside them). The `metrictrail` feature
sets ViPE's checkpoint variables to these files, so a run never downloads.

## Run

lietorch comes from `packages/lietorch-mojo`: both envs compose the `lietorch-mojo` feature, and `metrictrail-run`
depends on `lietorch-mojo-build`, which compiles its Mojo extension for the local GPU on the first run. Use the prod
env `metrictrail` for runs, not `metrictrail-dev`. Paths must be absolute: the tasks run in `packages/metrictrail`.

One end-to-end run on TUM fr1_xyz (798 frames, about 3 min on the RTX 5090) to a `.rrd`:

```bash
XFORMERS_DISABLED=1 pixi run -e metrictrail --frozen metrictrail-run \
    --input /mnt/nas/ssd-8tb/metrictrail/data/TUM/rgb/xyz --benchmark tum \
    --gt-sequence /mnt/nas/ssd-8tb/datasets/VSLAM-LAB-Benchmark/RGBDTUM/rgbd_dataset_freiburg1_xyz \
    --output /tmp/mt-xyz --seed 1234 --rr-config.headless --rr-config.save /tmp/mt-xyz/xyz.rrd
```

`--input` takes a video, a frame directory or a VSLAM-LAB sequence. `--benchmark` selects upstream's settings
(`tum`, `7scenes`, `kitti`, `kitti_calibrated`, `wild_slam`, `lamaria`, `oxford`, `vbr`, or `video` for the README
quick start). `--no-capture` turns off the live intermediates (faster). Without a display, always pass
`--rr-config.headless` or `--rr-config.save`.

Other tasks:

- `metrictrail-log-run --run-dir <vipe_out> ...`: log a finished output directory (ours or upstream's) without the GPU.
- `metrictrail-check-kernels`: run one small op of each compiled kernel on the GPU.

## Evaluate

```bash
pixi run -e metrictrail --frozen metrictrail-eval --runs <out seed1234> <out seed1235> <out seed1236> \
    --dataset rgbdtum --out-dir <new dir> --vslamlab-dir <VSLAM-LAB checkout>
```

The task scores each output directory two ways and writes `report.json`:

- the paper protocol: ATE RMSE after one Sim(3) alignment per sequence, unweighted mean;
- VSLAM-LAB's own `evaluate_sequence` (a subprocess in the VSLAM-LAB checkout's `vslamlab` pixi env, with a private
  evo HOME).

Datasets: `rgbdtum`, `7scenes`, `kitti`.

## Catalog layers

```bash
pixi run -e metrictrail --frozen metrictrail-catalog-layer --runs-root <out seed1234> --report <report.json> \
    --benchmark tum --source metrictrail --producer "examples-monorepo metrictrail @ <sha>" \
    --catalog-url rerun+http://<host>:<port>
```

The task writes one layer `.rrd` per sequence to a new directory, `<results_root>/catalog/<source>/<run_tag>/`, and
never overwrites a layer file. With `--catalog-url` it registers the layers on the `dataforge-vslam-lab-<dataset>`
dataset: `ERROR` where the segment has no such layer yet, `REPLACE` where it has an older run. The layer name is
`metrictrail` (our port) or `metrictrail_upstream` (unmodified upstream ViPE). The run metrics are recording
properties in the layer, not sidecar files.

## Results against the paper

Metric: ATE RMSE after one Sim(3) alignment per sequence, unweighted mean, as in the paper. We wrote the tolerance
before the first result: a dataset passes if |mean - paper| <= max(3 sigma, 10 % of the paper value, half a unit of
the last printed digit), and the port matches upstream if |port - upstream| <= 3 sqrt(sigma_port^2 + sigma_up^2)
(0.2 cm floor). "Provisional" means fewer than 3 seeds.

"Port" means this package. The port results below were made before lietorch-mojo was wired in, with a temporary
provider (upstream lietorch C++ re-exported as `lietorch_mojo`); see Known gaps.
"Upstream" means unmodified ViPE `ead2e00` with its own lietorch and MoGe-2. All runs are on an RTX 5090.

<!-- results: p2/evalq/tables.json, 2026-11-06 ~11:34 UTC -->

| Dataset (paper table) | Paper | Upstream ViPE | Port | Port vs upstream |
| --- | --- | --- | --- | --- |
| TUM RGB-D fr1, 9 seq (Table 9) | 4.4 cm | 4.478 ± 0.074 cm, 3 seeds: PASS | 4.497 ± 0.273 cm, 2 seeds: PASS (provisional) | +0.019 cm, band 0.850: PASS (provisional) |
| 7-Scenes, 18 seq (Table 8) | 6.3 cm | 6.378 cm, 1 seed: PASS (provisional) | not run | not run |
| KITTI 00-10 (Tables 1, 12) | 14.05 m | 13.482 m, 1 seed: PASS (provisional) | not run | not run |

KITTI per sequence: upstream 01 is outside its band (27.98 m against the paper's 35.94 m, 1 seed).

The run speed of the port is the same as upstream's: 17.4 fps against 17.2 fps on TUM.

## Build time

Clean `vipe-src` rebuild, `pixi install -e metrictrail-dev`, 16 jobs, niced, on a shared host with a warm package
cache (a cold download was not measured):

| What | Time |
| --- | --- |
| `pixi install -e metrictrail-dev`, recipe rebuilt | 127 s |
| rattler-build recipe step | about 2 min |
| `vipe_ext` + DPVO kernels | 50 s + 69 s |

## Known gaps

- **Port results predate lietorch-mojo.** The tables above used upstream lietorch C++ as a temporary `lietorch_mojo`
  provider. One check on lietorch-mojo: TUM fr1_xyz, seed 1234, 1.295 cm against 1.297 cm with the temporary provider
  (Sim(3) scale 1.0999 against 1.0988).
- **Seeds.** Only TUM has 3 upstream seeds. The port has 2 TUM seeds and no 7-Scenes or KITTI runs yet.
- **KITTI stereo** is not built. The stereo JIT path (`VIPE_METRICTRAIL_DPVO_JIT=1`) has no Eigen 3.4.0 at run time.
- **linux-64 only.** linux-aarch64 (DGX Spark) was not tried.
- **monopriors brings its full PyPI closure** (diffusers, pytorch-lightning, sam3, ...), although MoGe-2 needs only a
  few of those packages.
- **`metrictrail-eval`'s default `--vslamlab-dir`** is a scratch path (`/tmp/metrictrail-scratch/vslamlab/VSLAM-LAB`).
  Pass your VSLAM-LAB checkout.
- **Catalog.** The layers are registered only on a private catalog. Registration on the shared Spark catalog waits
  for approval.
