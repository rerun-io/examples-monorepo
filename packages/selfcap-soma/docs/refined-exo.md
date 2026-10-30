# SOMA refit with Sol-refined exo cameras

Published only to `selfcap-post-dev/soma` for `selfcap__2025-12-03__d31bf1da__ep001`.
Input run: `exofit-display-v1`. Source SHA-256: `fd290c2430acf9660dec12c0ae98f0b45f25732e92dd5e943884f1a62a2ddead`.

Both columns use the **new source and camera calibration**. Before is the last published upper-body fit, rescored without refitting; after is the new fit. These are fit residuals, not independent ground-truth accuracy.

| Metric | Previous publish, rescored | Refit |
|---|---:|---:|
| Body, median / p90 mm | 36.55 / 60.52 | 21.66 / 43.82 |
| Shoulders, median / p90 mm | 50.97 / 63.78 | 27.10 / 35.06 |
| Ears, median / p90 mm | 42.46 / 65.02 | 26.24 / 32.26 |
| Quest hands, median / p90 mm | 5.84 / 11.34 | 6.09 / 11.37 |
| OAK fingers, median / p90 px | 11.53 / 26.83 | 11.56 / 26.76 |
| Chest angular speed p90 deg/s | 45.21 | 43.95 |
| Pelvis-height std mm | 22.90 | 36.69 |
| Head pitch median / absolute p90 deg | -8.84 / 11.06 | -8.38 / 11.40 |

The previous report used the older triangulation: body 16.83, shoulders 15.15, ears 3.93 mm medians. Those values are not directly comparable across the calibration change. Ears remain a material limitation at 26.24 mm in the new fit. Pelvis-height variation also increased. No loss, shape or head-attachment tuning was added in this run.

## Exo image cross-check

Median pixels, same raw detections and current cameras in both columns:

| Detector / view | Previous publish | Refit |
|---|---:|---:|
| rtmw / rig_00 | 14.48 | 13.05 |
| rtmw / rig_01 | 8.96 | 8.53 |
| rtmw / rig_02 | 11.31 | 10.64 |
| rtmw / rig_03 | 11.17 | 10.73 |
| sapiens2 / rig_00 | 11.66 | 9.54 |
| sapiens2 / rig_01 | 7.12 | 6.69 |
| sapiens2 / rig_02 | 9.48 | 8.68 |
| sapiens2 / rig_03 | 7.38 | 7.35 |

Sapiens2 supplied the triangulation, so its image score is not independent. RTMW is a separate detector on the same images.

## Configuration and evidence

Fitting settings are unchanged from `4c9e4175`: fixed earlier shared shape, zero knee/ankle/foot observation weights, frozen leg rotations, hip weight 0.15, temporal weight 0.1, ear position plus soft Quest rotation 0.3, Quest hand position/orientation weights 3/30. All leg rotations and retained shape were verified unchanged.

Display polish is from `666930e1`: fixed rest-pose waist selection (25 mm below hip joints), flattened display boundary and recomputed normals; exact fitted COCO points for overlays, confidence >=0.7 plus residual-scaled pixel tolerance. There are 3,483 retained vertices and 6,942 faces. No new published entity paths.

At 39.862 s in rig_02, shoulder errors are 10.62 / 15.51 px versus tolerances 30.18 / 28.30 px, with raw confidence 0.953 / 0.871. Both therefore classify as observed.

Full segment: 4367 frames, 0.208–182.117 s. Pose refinement and mesh generation: 9.61 s (2.20 ms/frame); peak CUDA allocation 426.7 MiB. This excludes catalog reads, evaluation and export. Published layer: 163,490,879 bytes (155.9 MiB).

Artifacts: `/tmp/soma-fit/refined-exo/`. `upper-body-report.json` holds all metrics; `selfcap__2025-12-03__d31bf1da__ep001.rrd` is the layer. `world.png` and `exo02.png` are maximized catalog panes at 39.862079571 s; `cut-preview.png` shows the waist from front, side and back.

Validation: unchanged entity set and recording IDs, finite mesh/normals, complete time range, seven camera overlays, no lower-body overlay points, exact retained shape and zero leg-chain rotations. Targeted unit/GPU tests and static checks passed during preparation; no full gate. No other layer, dataset or catalog blueprint was changed.

Pixel inspection confirmed: both shoulders, the shoulder line and torso links
are cyan in exo rig_02 at the requested time. The World view shows the head,
torso, arms and hands without legs. The waist edge is clean in front, side and
back views. Existing hand/wrist mesh limitations remain; the display polish
does not claim to repair hand anatomy.
