# Feet, limb symmetry and observed/inferred overlays

Status: code and first current-layer run complete. **Not published.** Wait for
explicit confirmation that the stricter `keypoints_sapiens2` layer is registered,
then refit and replace only `selfcap-post-dev/soma`.

The configuration remains Sapiens2 exo observations, ear position landmarks,
and soft Quest head rotation (weight 0.3, no head translation). Quest hand
positions and orientations remain enabled (weights 3 and 30).

## Foot landmarks and constraints

Three low-LOD SOMA-X v0.3.0 surface vertices are averaged per COCO landmark.
The mapping was inspected from top and side views in
`/tmp/soma-fit/feet-fix/landmark-mapping.png`.

| COCO | Landmark | Vertex IDs |
|---|---|---|
| 17 | Left big toe | 2034, 2144, 2026 |
| 18 | Left small toe | 1968, 1970, 1965 |
| 19 | Left heel | 2155, 2152, 2039 |
| 20 | Right big toe | 4238, 4348, 4230 |
| 21 | Right small toe | 4172, 4174, 4169 |
| 22 | Right heel | 4359, 4356, 4243 |

Foot surface positions use confidence-weighted Huber residuals, 20 mm scale,
weight 1.0. Local ankle rotation-vector limits are flex -35/+50°, yaw ±35°,
and roll ±25°. Toe-base limits are flex -20/+60° and yaw/roll ±10°. These are
soft modeling bounds, not clinical joint measurements. Prior weight is 0.2.
The toe-base guard was added after inspecting the first run's isolated curls;
the final stricter-source run must include it.

Floor estimation uses confident (≥0.7), nearly stationary (≤0.08 m/s) foot
landmarks. A 15th-percentile low-height seed is refined by a median within
20 mm. Contact applies only to valid stationary points within 30 mm of that
height. No missing detection creates a contact. The floor loss has weight 0.3
and a 20 mm scale. Current-layer estimate: **Y = 0.586123 m**; at least one
contact point is active in **6.18% left / 99.86% right** frames. Right-foot
observations are stable but not bit-identical repeats.

A symmetry penalty (weight 10, differences in 10 mm units) is applied to the
actual neutral upper-arm, forearm, thigh and shin lengths during shape fitting.
It constrains anatomy rather than assuming that matching native scale controls
alone produce equal lengths. Native identity, scale and bone ratios are then
frozen for pose fitting.

## First run on the current source

Source SHA256 over 3D positions/confidences:
`53db6232cd64588020b107195a5aedc39908f98b5d3efdc122dcee74e650b2a3`.
4,367 frames, 0.208324653–182.117411775 s. Shape: 55.84 s; pose + mesh:
71.54 s; peak allocated GPU memory 430 MiB. This was run while the shared GPU
was also used by the stricter-source worker; it is not an isolated benchmark.

Values are median/p90. Both predictions are evaluated against the same current
source. They are fitting consistency metrics, not independent ground truth.

| Metric | Published B before | Current candidate |
|---|---:|---:|
| Body, mm | 15.2 / 41.4 | 15.4 / 41.2 |
| Shoulders, mm | 22.9 / 65.5 | 16.5 / 51.4 |
| Ears, mm | 3.8 / 8.7 | 3.8 / 8.8 |
| Foot surfaces, mm | 58.5 / 147.7 | 6.5 / 11.0 |
| Quest hands, mm | 5.7 / 11.6 | 6.1 / 11.7 |
| OAK fingers, px | 11.82 / 27.16 | 11.93 / 27.33 |
| Head pitch median / abs p90, degrees | -6.7 / 8.9 | -8.6 / 10.7 |
| Left sole-up tilt, degrees | 16.2 / 32.4 | 15.1 / 41.5 |
| Right sole-up tilt, degrees | 11.0 / 13.8 | 4.6 / 6.6 |
| Ankle rotation median, left/right degrees | 0.0 / 0.0 | 21.4 / 11.2 |
| Wrist twist p90, degrees | 26.4 | 33.4 |
| Finger roll p90, degrees | 77.93 | 77.88 |
| Flipped hand/wrist faces p90, % | 0.268 | 0.335 |
| Collapsed hand/wrist faces p90, % | 0.368 | 0.402 |

The right foot improves. The left-foot tilt tail worsens under current weak
observations, so this candidate is not a completed fix and is not published.
The first-run geometry was inspected in `current/feet-comparison.png`; the
left foot remains poorly determined. New strong-view triangulation is required
before assessing the final candidate.

| Neutral segment, mm | Before left/right | After left/right |
|---|---:|---:|
| Upper arm | 269.58 / 272.83 | 276.98 / 276.99 |
| Forearm | 268.37 / 265.50 | 267.50 / 267.52 |
| Thigh | 392.64 / 424.18 | 409.00 / 409.08 |
| Shin | 479.47 / 439.58 | 461.57 / 461.43 |

Cross-check median px; each cell is **RTMW / Sapiens2**. Includes the newly
mapped foot surfaces for both before and after, so the numbers differ slightly
from the earlier no-foot evaluation.

| View | Before | Current candidate |
|---|---:|---:|
| rig_00 | 12.23 / 8.73 | 12.24 / 8.91 |
| rig_01 | 9.25 / 7.23 | 8.75 / 6.62 |
| rig_02 | 10.85 / 8.29 | 10.86 / 8.18 |
| rig_03 | 11.44 / 7.92 | 11.22 / 7.34 |

## Overlay contract

Existing `{pinhole}/pred/soma_x/joints_uv` paths are retained. A phone joint is
observed only when its raw Sapiens2 score is ≥0.7 and its position lies within
15 px of the projection. Other projections are inferred, not removed.
Observed points are cyan/radius 2.5; inferred points are grey/alpha 110/radius
1.1. A link is cyan only if both endpoints are observed; otherwise it is grey,
alpha 90 and thinner. Points2D and LineStrips2D share the same entity, avoiding
new blueprint entity paths. The old annotation connections are removed to
prevent an extra unstyled skeleton.

RGB uses nearby EgoHand 2D detections. OAK mono uses valid projected Quest hand
points and requires RGB EgoHand support; this is tracking/image support, not
an independent mono occlusion measurement. Non-hand OAK points are inferred.
Missing image detections stay missing. No camera, clock, or mount is modified.

Pixel check of the actual writer: `/tmp/soma-fit/feet-fix/style-check.png`.
This shows cyan observed links and smaller grey inferred links. The current
candidate layer is staged locally at
`/tmp/soma-fit/feet-fix/current/selfcap__2025-12-03__d31bf1da__ep001.rrd`.
Its machine-readable results are `current/foot-report.json` under the same root.

## Reproduce

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 pixi run -e selfcap-soma --frozen python packages/selfcap-soma/tools/feet_fit.py --output /tmp/soma-fit/feet-fix/strict
```

Use the installed environment's Python directly when running from a package
working directory. Do not register the candidate before the source-ready
message. Validate full-range mesh/normals, joints and all seven overlays, then
replace only the one segment's `soma` layer. Keep the original dataset blueprint.
