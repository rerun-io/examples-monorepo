# zipdepth

Training / evaluation / export code for [ZipDepth](https://github.com/fabiotosi92/ZipDepth)
(ECCV 2026, MIT), copied from the `pablovela5620/ZipDepth` fork @ `5a80354`.
The **network lives in monopriors** (`monopriors.third_party.zipdepth`) — this package only
imports it; there is no `zipdepth/model/` here. Inference for end users is
`monopriors.models.relative_depth.ZipDepthPredictor`. Weights: https://huggingface.co/pablovela5620/zipdepth.

```bash
pixi run -e zipdepth --frozen zipdepth-infer-rerun      # example image in Rerun (add --checkpoint for a trained .pth)
pixi run -e zipdepth --frozen zipdepth-train-smoke      # data → train → resume → torchrun → checkpoints load in monopriors (~3 min on a 5090)
```

Layout:
- `zipdepth/apis/` — typed tyro entry points authored here (`smoke_data`, `infer_rerun`, `train_smoke`).
- `zipdepth/upstream_cli/` — the fork's `scripts/*.py` (argparse), unmodified apart from model imports.
- `zipdepth/{data,loss,training,evaluation}/` — upstream training code, unmodified.
- `tools/` — thin shims only: `python tools/train.py --config configs/default.json ...` (see `UPSTREAM_README.md`).
