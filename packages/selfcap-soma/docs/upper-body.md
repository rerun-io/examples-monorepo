# Upper-body SOMA fit

Published configuration for `selfcap__2025-12-03__d31bf1da__ep001` in
`selfcap-post-dev/soma`. Supersedes the foot experiment.

The strict Sapiens2 source has SHA-256
`2c8999b0d42c2060fddc3275f7f9379b1b997285defac82eed1ff901a4a389bc`.
The run covers 4,367 frames, video_time 0.208–182.117 s. Both fits below were
scored on this same source, using upper-body COCO 5–12 for the body metric.
These are fit residuals, not independent ground-truth accuracy.

| Metric | Previous full body | Upper body |
|---|---:|---:|
| Body median / p90 mm | 19.97 / 50.55 | 16.83 / 45.42 |
| Shoulders median / p90 mm | 23.61 / 69.72 | 15.15 / 33.98 |
| Ears median / p90 mm | 4.13 / 9.78 | 3.93 / 8.99 |
| Quest hands median / p90 mm | 5.73 / 11.57 | 5.84 / 11.34 |
| OAK 15 finger joints median / p90 px | 11.82 / 27.16 | 11.53 / 26.83 |
| Head pitch median / absolute p90 deg | -6.69 / 8.90 | -8.84 / 11.06 |
| Chest angular speed p90 deg/s | 71.44 | 45.21 |
| Pelvis height std mm | 7.18 | 22.90 |

The angular-speed measure includes real motion; it is a jitter proxy. Pelvis
height is less constrained. Neck-to-pelvis tilt stays similar (median 3.34 to
3.60 degrees; p90 7.43 to 7.71), and the neck remains at least 0.538 m above
the pelvis. No upright term was added. A 0.1 pelvis/chest angular-velocity-change
prior reduces the angular jitter without forcing an upright pose.

## Exo cross-check

Median reprojection px, before → after, confident raw observations:

| View | RTMW | Sapiens2 |
|---|---:|---:|
| rig_00 | 11.90 → 11.71 | 8.64 → 8.70 |
| rig_01 | 8.37 → 8.29 | 6.35 → 6.22 |
| rig_02 | 10.38 → 10.17 | 8.14 → 7.85 |
| rig_03 | 11.04 → 11.03 | 7.52 → 7.21 |

Sapiens2 supplied the fit targets, so its cross-check is not independent.
RTMW offers a different detector, but both detectors share the same images.
Wrist twist p90 rose 26.44 → 34.92 degrees; finger roll p90 is essentially
unchanged (77.93 → 77.88); flipped hand faces p90 rose 0.268% → 0.301%.
These hand plausibility measures did not all improve.

## Configuration and display

- COCO 13–22 have zero observation weight; all leg-chain rotations stay exactly
  at rest. Hip landmarks use weight 0.15. No floor term is used.
- Shared identity, scale and symmetric bone ratios are retained exactly from
  `/tmp/soma-fit/feet-fix/current/calibration.json`; no torso-only shape refit.
- Ears supervise head position; Quest rotation weight 0.3, no head translation.
  Quest hand position/orientation weights remain 3 / 30.
- Vertices with summed thigh-and-descendant skinning weight ≥0.5 are removed,
  along with all touching faces. The pose-invariant crop has 3,488 vertices and
  6,950 faces. Normals are recomputed. Its open pelvis hem follows the skinning
  boundary; front and side views show a continuous V-shaped cut.
- Seven camera overlays keep the existing paths. Knees, ankles and feet are
  absent. Observed points stay cyan; inferred points and links are smaller grey.
- Final warm-start refinement: 7.11 s, 1.63 ms/frame, peak CUDA allocation 430 MiB.
  This excludes the earlier shape and refinement passes.

## Evidence

Local artifacts: `/tmp/soma-fit/upper-body/published/`.

- `upper-body-report.json`: complete metrics and source hash.
- `fit.rrd`: fit parameters and internal full mesh for reproducibility.
- `selfcap__2025-12-03__d31bf1da__ep001.rrd`: published upper-body layer,
  163,910,204 bytes (156.3 MiB).
- `cut-preview.png`: front/side mesh and pelvis-cut close-ups.
- `world.png`, `exo02.png`: catalog screenshots at 39.862079571 s.

Only the segment's `soma` layer was replaced. The catalog blueprint was not
changed; screenshots use a viewer-local blueprint to expose the SOMA entities.

Pixel inspection: the World view contains the head, torso, pelvis, arms and
hands, with no thighs/legs/feet. In exo rig_02, cyan hand points follow the
visible fingers broadly; other links are dim grey where unsupported. Knees and
ankles no longer draw across the table. The crop has an open underside with
pointed front/back ends, not a flat capped surface. The close-up also retains
visible wrist mesh artifacts; this upper-body change does not repair them.

Validation: targeted unit tests (11 passed), frozen-leg GPU integration test
(1 passed), Ruff and Pyrefly passed. Export checks cover recording identity,
all seven overlays, full time coverage, finite cropped mesh/normals, unchanged
entity paths, exact retained shared shape and zero leg-chain rotations.
