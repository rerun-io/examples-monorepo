# Sapiens2 SOMA comparison: d31bf1da ep001

Published **B (ears + soft Quest rotation)** as only `selfcap-post-dev/soma`. A has an implausible downward head posture despite a low ear residual. Both modes exclude Quest head translation and store a zero attachment translation. The camera pose origin is not used as an anatomical position.

## Setup

Full 0.208324653–182.117411775 s, 4,367 frames. Source columns are selected from the catalog schema, including their confidence component:

- Sapiens2: `/world/gt/sapiens2/coco133_xyz`; exo `/world/rig_0X/cam_00/pinhole/sapiens2/coco133_uv`.
- RTMW remains the CLI default (`--exo-source rtmw`); `--exo-source sapiens2` selects the new source.
- `--options.head-mode quest` preserves legacy attachment. `ears` is the CLI default and removes all Quest head losses; `ears-rotation` adds orientation only.
- Sapiens2 missing hand slots 91,92,96,100,104,108,112,113,117,121,125,129 remain absent. No zero targets or invented detections.
- Shared native identity (128 coefficients), scale and bone ratios fitted on every eighth valid frame before 25 s, then frozen. Identity prior 0.05, scale prior 2, bone-length weight 2; ear weight 3.
- A/B use the same initialization and 400 final all-pose steps, after the same 400 pose + 400 hand-repair steps. Quest hand position weight 3 and bone orientation weight 30 are unchanged. Procedural twists remain enabled.
- B uses rotation weight 0.3. Its constant Quest-to-skull rotation is estimated from first-part ear axes and torso-up. Torso-up sets the unobservable constant pitch; it is an assumption, not measured skull pitch.
- The first B trial used A to estimate the attachment and inherited its bad pitch. That discarded trial remains under `b/`; final B is `b-upright/`. Final A is `a-control/`; `a/` is its earlier initialization.

## Whole-segment residuals

3D values below are median/p90 mm against each fit's own training source. OAK uses the 15 requested finger indices, full 14-coefficient lens, catalog camera pose, and EgoHand 2D only. They are consistency measurements, not independent physical ground truth.

| Metric | RTMW legacy | Sapiens A | Sapiens B published |
|---|---:|---:|---:|
| Body | 22.0 / 45.6 | 14.8 / 40.7 | 15.2 / 41.4 |
| Shoulders | 28.9 / 54.2 | 20.6 / 60.4 | 22.9 / 65.5 |
| Ears | 10.0 / 21.2 | 3.6 / 8.6 | 3.8 / 8.7 |
| Quest hands | 8.3 / 16.1 | 5.6 / 11.3 | 5.7 / 11.6 |
| OAK fingers px | 13.4 / 29.4 | 11.7 / 26.9 | 11.8 / 27.2 |

## Head pitch and hand plausibility

Positive pitch is anatomical forward pointing up. A still has its chin near the chest and rapid pitch changes; B is plausible in the inspected frames. A/B 2D errors are nearly equal, so they do not independently resolve the pitch ambiguity.

| Metric | RTMW legacy | A | B |
|---|---:|---:|---:|
| Pitch median / absolute p90 deg | -12.3 / 20.3 | -73.4 / 85.0 | -6.7 / 8.9 |
| Pitch range deg | -52.3 to 21.8 | -89.8 to -1.3 | -13.6 to -0.0 |
| Pitch speed p90 deg/s | 97.1 | 180.4 | 12.6 |
| Wrist twist p90 deg | 25.3 | 28.1 | 26.4 |
| Finger roll p90 deg | 78.1 | 77.9 | 77.9 |
| Flipped hand/wrist faces p90 % | 0.469 | 0.248 | 0.268 |
| Collapsed hand/wrist faces p90 % | 0.602 | 0.368 | 0.368 |
| Fit + mesh seconds, excluding shared shape | 41.67 | 19.17 | 19.37 |

The flip/collapse values are proxies, not collision tests. Finger roll remains about 78 degrees: changing the exo source does not solve the remaining anatomical-prior limitation. Shared shape cost was 6.49 s. B: 19.37 s for fitting/mesh plus 6.49 s shared shape, about 5.92 ms/frame total; peak PyTorch allocation 426 MiB. Excludes catalog I/O, RRD writing, evaluation and screenshots. Legacy timing came from its earlier run.

## Cross-detector image check

Each cell is median px **against RTMW / against Sapiens2**. Four calibrated rectified phones, confidence >=0.7, mapped ears/body/hands only; matched frames and identical valid counts for all three fits.

| View | RTMW fit | A | B |
|---|---:|---:|---:|
| rig_00 | 13.64 / 11.49 | 12.17 / 8.65 | 12.23 / 8.73 |
| rig_01 | 8.82 / 8.76 | 8.47 / 6.48 | 8.46 / 6.49 |
| rig_02 | 10.72 / 9.85 | 10.81 / 8.28 | 10.85 / 8.29 |
| rig_03 | 11.83 / 9.02 | 11.01 / 7.14 | 11.02 / 7.23 |

The Sapiens fits improve agreement with Sapiens in all views. Against RTMW they improve three views and change rig_02 slightly (10.72 to 10.85 px for B). This does not establish that either detector is anatomically correct.

## Landmark definition differences

High-confidence matched Sapiens minus RTMW observations. Signed components are world x/y/z mm; magnitude is median/p90 mm.

| COCO landmark | Signed xyz | Magnitude |
|---|---:|---:|
| 3 | 54.2, 13.4, 29.9 | 63.8 / 87.9 |
| 4 | 1.4, 8.2, 7.8 | 15.5 / 34.1 |
| 5 | -7.6, 13.8, 55.4 | 59.1 / 88.4 |
| 6 | 21.5, 18.5, -2.3 | 42.3 / 85.4 |
| 11 | -17.7, 18.7, -48.5 | 56.8 / 71.3 |
| 12 | -19.5, 26.8, -25.2 | 42.9 / 62.3 |

Left ear differs by 64 mm, right ear by 15 mm; shoulder/hip definitions differ by 42–59 mm. Native shape was refit (scale 1.0966 versus 1.1186), not reused from RTMW. No fixed per-landmark displacement is silently subtracted.

## Artifacts and validation

- Final machine-readable report: `/tmp/soma-fit/sapiens-comparison/comparison-final.json`.
- Registered B: `/tmp/soma-fit/sapiens-comparison/b-upright/selfcap__2025-12-03__d31bf1da__ep001.rrd`, 176,448,137 bytes (168.27 MiB).
- Validated original application/recording IDs, all 4,367 joint/overlay frames and 1,457 mesh+normal frames through the full interval. Seven camera overlays contain only mapped body/hand/ear joints.
- Shared blueprint was not modified. Viewer evidence uses a local-only override to enable SOMA joints. In exo rig_02 the cyan hands overlap the visible hands with some residual offset; the hips and legs are mostly occluded and are not visually validated.
- Catalog screenshots at 39.862079571 s: `world-catalog-soma.png`, `exo02-catalog-soma.png` under the same artifact root; `head-final-preview.png` shows the A/B head close-ups. The original shared-blueprint captures are retained as `world-39.862.png` and `exo02-39.862.png`.
- Targeted source/head tests: 2 passed on host CUDA. Source/loss/projection tests: 6 passed. Changed-file Ruff and Pyrefly passed (unrelated missing-env path warnings). No full gate.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 pixi run -e selfcap-soma --frozen python packages/selfcap-soma/tools/source_comparison.py
```

## Foot defect raised during review

The foot mesh is not a valid reconstruction. Both `LeftFoot` and `RightFoot`
axis-angle parameters are exactly zero throughout the segment: the last
observed lower-body points are the ankles, so these leaf rotations have no
positional data gradient. There are no toe/heel landmarks, sole orientation,
floor/contact term, or foot pose prior beyond a pull toward neutral.

In published B, ankle residual median/p90 is 3.7/18.2 mm left and 5.6/11.0 mm
right against Sapiens2. Those small residuals do not establish correct feet.
At 39.862 s the left knee observation is absent, the fitted ankle heights differ
by about 10 cm, and the left foot's transformed neutral-up axis is tilted 37.3°
from world up (right 10.7°). No foot correction was included in this source/head
comparison. Foot landmarks or justified contact constraints are needed next;
occluded lower-body detections also need independent checks.

A second confirmed issue is asymmetry from independently fitted leg ratios:
neutral thigh/shin lengths are 392.6/479.5 mm left versus 424.2/439.6 mm right.
The left knee has no triangulated observation in 37.7% of frames. A symmetry
prior is needed before treating these fitted proportions as anatomical.

The omission is in the fitter, not wholly in the source: Sapiens2's COCO foot
slots 17–22 have 24, 46, 3,071, 4,367, 4,367 and 4,367 finite observations,
respectively. `coco_joint_names()` deliberately omits all six. The right foot
therefore has unused source evidence throughout the segment; the left has
many heel samples but almost no toe samples. Adding validated mesh foot
landmarks should precede any assumption that both soles remain flat.

## Per-view occlusion versus inferred overlays

The cyan overlay projects all finite SOMA joints into every camera; it does
not use that camera's detector confidence or an object-occlusion test. At
39.831673680 s (the fitted sample shown at the 39.862 s cursor), rig_02's two
ankle detections are already missing/zero-confidence. Their cyan projections
remain visible because they are inferred by the model. Rig_02 right-knee
confidence is 0.703 despite the table occlusion. Rig_01/03 provide both ankle
positions with confidence 0.957–0.977. High 3D confidence alone cannot answer
whether a point is visible in a particular image. Overlay visibility needs a
separate per-view convention; this run has not changed that convention.

Pixel inspection of `rig_01-visibility.png` and `rig_03-visibility.png` confirms
that the feet/ankles are visible in both source views at 39.862 s. Their high
confidence is not evidence of hidden-joint hallucination at that instant.
The rig_02 knee behind the table is a separate overconfidence concern. No
confidence threshold alone provides a per-view occlusion model.

The source implementation at `f3894bfc0` already uses exhaustive camera-pair
RANSAC, an 8 px reprojection gate, two refits, and at least two supporting views.
At the example frame, right-knee residuals are 31.6, 2.6, 13.1, and 1.6 px in
rig_00–03 respectively: the occluded rig_02 detection is rejected, while rig_01
and rig_03 support the 3D point. The left ankle nevertheless disagrees by 57.3 px
with rig_01's detection and 1.1 px with rig_03's, so the accepted support for that
landmark needs further audit before the 3D target is treated as reliable.

A confidence-policy weakness remains: triangulation accepts per-view scores
>=0.3, whereas displayed 2D detections require >=0.7; 3D confidence is the mean
score of accepted views. It is not a visibility probability or geometric
uncertainty. A weak view can help produce a finite point that is subsequently
used as a fitting target. More pair sampling does not solve this: stronger
view support, retained per-view inlier provenance, and explicit inference versus
observation display are the relevant next changes.
