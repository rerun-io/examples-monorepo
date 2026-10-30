# d31bf1da optical calibration candidate

The candidate is published at:

`rerun+http://dgx-spark:51235/dataset/18E3351991176D38631fefb9b4f3a944?segment_id=selfcap__2025-12-03__d31bf1da__ep001`

Pipeline implementation: `1eb2cee3`. SOMA implementation: `2345a75c8` from
`selfcap-soma-fit`, archived before execution. No source dataset was changed.

## Reproduce

From the repository root:

```bash
pixi run --frozen -e selfcap-curate selfcap-dev-cycle \
  --segment selfcap__2025-12-03__d31bf1da__ep001 \
  --tag <new-tag> --dataset selfcap-post-dev-exofit \
  --landmarks selfcap_curate/landmarks/selfcap__2025-12-03__d31bf1da__ep001.json \
  --exo-refine --focal-source fit --oak-egohand-assets /tmp/egohand
```

The completed run uses tag `exofit-20261029-v1`. Its files are under
`/home/pablo/dataforge-dev/selfcap-post-dev-exofit/runs/exofit-20261029-v1/`
on the catalog host. Layers: `base`, `sapiens2`, `calib`, `egohand_calib`,
`exo_landmarks`, `soma`, plus the existing source `egohand` file.

The 232-label record hashes to
`c0699f485acd77729735c8b50e117ae34a9b1206795b077ac74bb65f082bd1c1`.
The original labelling agent did not record its model; that remains explicit.
The all-data fit gives focals 852.62, 858.93, 921.00 and 904.48 px in rig order,
and a head-frame optical translation of (-29.84, -50.38, -24.08) mm.

## Held-out gate

Values are median / p90 pixels. This is the exact replay of the original
source calibration through the repository CLI, with labels read from the
candidate catalog. Both folds and the final model converged.

| Metric | Source | Candidate |
|---|---:|---:|
| Headset apertures | 38.53 / 45.12 | 4.56 / 9.98 |
| Phone apertures | 101.95 / 202.40 | 0.31 / 0.76 |
| Sapiens hands | 8.37 / 33.66 | 8.98 / 33.09 |
| Dense RTMW hands | 9.59 / 26.82 | 10.50 / 27.50 |
| Body transfer | 24.67 / 45.81 | 16.55 / 46.96 |

**The no-regression gate fails on hands.** This is a candidate, not an
accepted calibration correction. The new post run re-estimates its baseline,
so its baseline medians differ slightly: 38.38, 102.89, 8.39, 9.59 and 25.28 px.
Its candidate scores match the replay above within numerical precision.

The run's `reports/` contains both the post gate
(`selfcap__2025-12-03__d31bf1da__ep001.exo-gate.json`) and the source replay
(`gate.json`). The latter can be regenerated with:

```bash
pixi run --frozen -e selfcap-curate selfcap-exo-refine \
  --segment selfcap__2025-12-03__d31bf1da__ep001 \
  --dataset selfcap-post-dev --landmark-dataset selfcap-post-dev-exofit \
  --output-dir /tmp/exofit-replay-new
```

## Pixel evidence and checks

Rerun 0.38.1, native headless viewer at 1920×1080, `video_time=39862079571`,
40-second decode settle, using `/tmp/selfcap-5090/video/shot_url.py`:

- `/tmp/exofit-validation/full-soma.png`
- `/tmp/exofit-validation/rig02.png`
- `/tmp/exofit-validation/rig03.png`

These files and `notes.md` are also stored in the run's `evidence/` directory.
All were inspected. Phone and headset centres are near the labelled apertures;
labels are visible and the torso/arms stay on the person. The occluded lower
leg in rig_02 remains uncertain and extends through the table. The final full
view includes the SOMA mesh. SOMA used 4,265 frames over 0.3–178 s; its original
standalone output was given the segment id and mapped to `/world/pred/soma_x`
for registration, without changing the fitter.

Verification: 28 targeted tests passed; package Ruff and canonical package
pyrefly passed. No full gate was run. The original scratch dev-cycle tool was
not changed. All commits are local.
