# metrictrail

NVIDIA MetricTrail (ViPE `pipeline=metrictrail`) in the monorepo. Upstream ViPE @ `ead2e00` and its CUDA kernels come
from `recipes/vipe-src`; this package is our code around it: benchmark presets, a Rerun logger, the VSLAM-LAB / paper
evaluation and the catalog layer.

| module | what |
| --- | --- |
| `benchmarks.py` | upstream's per-benchmark settings (`configs/slam/README_metrictrail.md`) as presets: `tum`, `7scenes`, `kitti`, `kitti_calibrated`, `wild_slam`, `lamaria`, `oxford`, `vbr`, plus `video` (README quick start) |
| `apis/run.py` | run the pipeline exactly as upstream's `run.py` (Hydra + typed config), capture the live intermediates (`capture.py`), log everything (`outputs.py`) with the default blueprint (`blueprint.py`) |
| `apis/log_run.py` | log a finished output directory (ours or upstream's) without running a model |
| `eval/`, `apis/vslamlab_eval.py` | the paper protocol and VSLAM-LAB's own `evaluate_sequence` (subprocess into VSLAM-LAB's pixi env, private evo HOME) |
| `catalog/`, `apis/catalog_layer.py` | the `metrictrail` / `metrictrail_upstream` layer on dataforge's VSLAM-LAB segments (`p1/catalog/SPEC.md`) |

Entity layout of a run recording: `metrictrail/entities.py`. Colours: our port magenta `#E040FB`, upstream amber
`#F5A623`, ground truth green `#3DDC97`.

```bash
# run (GPU + a lietorch provider): a frame dir, a VSLAM-LAB sequence or a video
pixi run -e metrictrail python tools/run.py --input <dir|seq|video> --benchmark tum --output <dir> --seed 1234 \
    --rr-config.save run.rrd --rr-config.headless
# log a finished run (no GPU)
pixi run -e metrictrail python tools/log_run.py --run-dir <vipe_out> --name desk --input <frame dir> --benchmark tum \
    --gt-sequence <VSLAM-LAB seq> --label upstream --rr-config.save desk.rrd --rr-config.headless
# evaluate (one output dir per seed)
pixi run -e metrictrail python tools/vslamlab_eval.py --runs <seed1234> <seed1235> <seed1236> --dataset rgbdtum --out-dir <new dir>
# catalog layer
pixi run -e metrictrail python tools/catalog_layer.py --runs-root <seed1234> --report <report.json> --benchmark tum \
    --source metrictrail --producer "examples-monorepo metrictrail @ <sha>"
```

Without a display, always pass `--rr-config.headless` (or `--rr-config.save`): the default spawns a viewer.
