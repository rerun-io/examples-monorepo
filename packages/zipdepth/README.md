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
- `zipdepth/upstream_cli/` — the fork's `scripts/*.py` (argparse), unmodified apart from model imports (`infer.py` dropped: `ZipDepthPredictor` covers it).
- `zipdepth/{data,loss,training,evaluation}/` — upstream training code, unmodified.
- `tools/` — thin shims only: `python tools/train.py --config configs/default.json ...` (see `UPSTREAM_README.md`).

## Catalog training lane (`zipdepth-catalog`)

`zipdepth-train-catalog` fine-tunes on PromptDA pseudo-labels streamed from the Rerun catalog
(`arkitscenes-v2`, data on the NAS). Two loaders exist, selected with `--dataloader`:

- `current` (default) — `zipdepth/catalog/dataset.py`: one projected datafusion query per segment
  (`filter_segments` + `filter_contents` before `reader`, `fill_latest_at` for the sparse prompt/confidence
  cells), zero-copy Arrow blob views, PNG un-filter and sample building on the GPU, one NVDEC decoder per
  producer thread. Measured on the RTX 5090: **~150 frames/s, 7.7 ms median data wait per step**, GPU 84–90%,
  i.e. training is compute-bound (2026-09-03, 8 producers, batch 8, 768×1024).
- `rerun` — `zipdepth/catalog/dataloader_dataset.py`: the same samples through the idiomatic
  `rerun.experimental.dataloader.RerunIterableDataset` with our decoders as `ColumnDecoder`s. Bit-identical
  output (live equivalence test) but **8.4 frames/s** on 0.37: there is no public exact-timestamp sample index
  (only `FixedRateSampling`), and the dataloader retains every field's decoded output for a whole fetch block
  before emitting. Kept as the executable reference until those two API gaps close upstream.

Recipe note: `--preset v4` is the production recipe (peak lr `config/100`); `--preset fast` differs only in the peak
learning rate (`config/10`, 1e-4), which reaches the same quick-holdout quality in 2.5–3× fewer steps.
