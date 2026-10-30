# SelfCap SOMA fitting prototype

Current published run: [upper body with refined exo cameras](docs/refined-exo.md).
The original prototype description below documents the earlier baseline.

Fit one fixed-calibration SelfCap recording to NVIDIA SOMA-X body and hands.
This is an observation fitter, not an image model. It does not use EgoHand,
change calibration or clocks, or register catalog layers.

## Run

Use a host-side terminal in this workspace for Pixi installation.

```bash
pixi install --locked -e selfcap-soma
hf download nvidia/SOMA-X --revision f81a01796f652260ff930d5318d6c6b845a505ed \
  SOMA_neutral.npz SOMA_template_rig.usda SOMA_procedural_transforms.json LICENSE \
  --local-dir /tmp/soma-fit/assets
pixi run -e selfcap-soma --frozen soma-fit --shape-iterations 80
# Full-rate, head-covered part of the segment:
pixi run -e selfcap-soma --frozen soma-fit --start 0.3 --end 50.0 --stride 1 \
  --shape-iterations 80 --output /tmp/soma-fit/segment-final
pixi run -e selfcap-soma-dev --frozen gate
```

The package and weights both use Apache-2.0. The package is the published
`py-soma-x==0.3.0`; source reference is NVIDIA SOMA-X commit `d29dbe5`.
No upstream code is vendored. The downloaded third-party NPZ is read by SOMA;
we write no NPZ or NPY files.
The feature explicitly selects the workspace's two named Linux CUDA-13 platforms,
as the shared CUDA feature does; plain Linux platforms cannot solve GPU PyTorch
without their CUDA virtual package.

Asset SHA-256:

- `SOMA_neutral.npz`: `5829af4a5333abdba5ad449360b868faacee99d395971e62f35c0e7169432b63`
- `SOMA_template_rig.usda`: `a952cb7bdf4ccd801e355d53122d395b985b69d0d9181439edad5081bb171856`

## Fit and observation contract

The default is `selfcap-post-dev`, segment
`selfcap__2025-12-03__d31bf1da__ep001`, at `rerun+http://dgx-spark:51235`.
There is one bounded materialization for triangulation, one for Quest hands
and head together, and one static time-sync query. No video is read.

COCO body centers 5–16 map to SOMA anatomical joints. Foot surface points,
face landmarks and headset-covered points are omitted. Each COCO hand uses
SOMA's wrist and four joints per digit; the four non-thumb metacarpals are
skipped. Mapping names resolve against the actual public SOMA rig.

Quest head time is `video_time - quest_pose_offset`; hand reports are sampled
48 ms later. Hand interpolation rejects nonfinite/zero samples, >4 m/s wrist
jumps and neighbors, consecutive frozen samples, and gaps over 40 ms. Head
rotations use SLERP. The head term matches SOMA's left eye to the catalog's
left-eye origin. Its fixed orientation attachment is inferred from neutral
SOMA eyes and cranial axis. **This is not a measured skull/headset attachment.**

Adam optimizes per-frame root translation and 77 axis-angle rotations. Pose
settles for up to 100 iterations before fitting 16 shared native identity
coefficients and a global scale. Identity then freezes. FK runs as one GPU
batch; meshes are evaluated in batches of 16 only after fitting. The objective
uses source-normalized Huber distances (20 mm exo, 10 mm Quest), strong Quest
hand/eye terms, a head rotation term, weak pose and rotation-magnitude priors,
and timestamp-scaled velocity changes. Exo body and hand blocks have separate
weights. Missing observations contribute zero weight. Interpolation of missing
torso anchors is used only for initialization, not to invent loss targets.

## Outputs

| Entity | Data | Clock |
|---|---|---|
| `/world/fit/mesh` | low-LOD fitted world mesh | video_time |
| `/world/fit/joints` | 77 fitted joints and rig connections | video_time |
| `/world/inputs/{exo,quest}` | inputs used by the optimizer | video_time |
| `/world/detail/{fit,exo,quest}` | matching 42-joint hands for close comparison | video_time |
| `/parameters/{poses,root}` | per-frame fit parameters | video_time |
| `/parameters/{identity,scale}` | shared fitted parameters | static |

`fit.rrd` includes geometry, parameters, and a two-view blueprint.
`report.json` records finite-point median/p90 residuals, head rotation error,
optimization-plus-mesh time, and peak PyTorch allocated GPU memory. Timing
excludes asset construction, catalog reads, serialization and screenshot work.
The screenshot cursor uses latest-at: at 4.144 s the last native frame is
4.124828132 s; no new observation is invented at the cursor time.

## Prototype evidence (RTX 5090)

Full-rate 0.3–50.0 s query: 1,193 frames, 400 iterations, 80 shape iterations.
Optimization and mesh evaluation: 3.41 s, 2.86 ms/frame, 277 MiB peak allocated.
Shared scale: 1.041. At the 4.144 s cursor, fitted hand-to-Quest median/p90 is
4.13/8.90 mm; triangulation-to-Quest is 31.92/70.50 mm (41 finite input joints;
the fit provides all 42). Across the fitted interval:

| Residual | Median | p90 |
|---|---:|---:|
| Body to triangulation | 35.31 mm | 85.16 mm |
| Hands to triangulation | 18.65 mm | 62.18 mm |
| Hands to Quest | 5.36 mm | 11.09 mm |
| Left eye to Quest | 1.51 mm | 5.29 mm |
| Head rotation | 2.61° | 3.80° |

These are **training residuals**, not proof of better physical accuracy. Quest
is a strong fitting input. A clean mesh is not evidence that hidden twist,
surface shape, or fingers are correct. There is no learned pose prior, hand PCA,
foot contact, surface-foot mapping, image evidence, or uncertainty estimate.
The low-LOD mesh disables procedural twists and pose correctives. The next
useful test holds out Quest joints and compares visible independent image
landmarks, then adds EgoHand if it supplies independent observations.

An initial fit shrank to scale 0.69 because hand residuals dominated before pose
settled. It is retained only under `/tmp/soma-fit/window`; the staged objective
above replaced it. The integration test covers repeat backward passes before,
during and after identity freezing, including SOMA's rebind cache behavior.

## Independent evaluation and registered image overlays

```bash
pixi run -e selfcap-soma --frozen soma-evaluate
pixi run -e selfcap-soma --frozen soma-overlay
```

The overlay writer reuses the fitted 77-joint geometry, preserves the original
`dataforge` / segment recording IDs, and writes `/world/pred/soma_x/{mesh,joints}`.
It projects every joint into the four exo pinholes and all three OAK cameras at
`{pinhole}/pred/soma_x/joints_uv`, with a cyan skeleton annotation on each entity.
It does not copy video or change calibration. The projection and triangulation
subset in `camera_geometry.py` is copied from `selfcap-curate/keypoints.py` with
an AST parity test; all seven projected streams were also checked numerically
against direct use of the original helpers (zero difference, identical masks).

The layer was copied to
`dgx-spark:/home/pablo/dataforge-dev/selfcap-post-dev/soma/selfcap__2025-12-03__d31bf1da__ep001.rrd`
and registered **only** in `selfcap-post-dev`, layer `soma`, with replacement of
that same segment/layer. `soma-overlay` only writes the local file; it does not
register automatically. The registered layer remains the full-observation
baseline, not an experimental holdout fit.

At video_time 4144403454, the maximized exo rig_00, exo rig_02 and OAK rgb panes
were inspected in the native headless Viewer. The screenshots under
`/tmp/soma-fit/layer/` show visible cyan hands, but shoulders and upper torso sit
below the image landmarks. The supplied screenshot helper originally selected
the rightmost pane on a row; its temporary selector was corrected to select the
nearest `Maximize view` button to the requested title before accepting evidence.

Evaluation stays on 0.3–50.0 s of the same segment (1,193 frames):

- Quest hands are removed on `[0,2), [4,6), ...` before fitting; 617 held-out frames.
  Shared identity can use the remaining frames. The withheld labels never enter
  a fitting term. At 4.144 s, the last fitted sample is 4.124828132 s.
- Each phone is removed before robust triangulation and fitting, then scored
  against its own RTMW observations with confidence >=0.7. The baseline is the
  matching leave-camera-out triangulation, not the stored all-camera result.
- Body offsets are learned before 25 s. After 25 s, each of 10 body landmarks is
  withheld in turn, including from initialization, while other landmarks remain
  available. Wrist landmarks are excluded because hand terms locate the same
  anatomical point. This test compares frozen body-frame offsets with centers.
- An extra stress test removes *all* body observations after 25 s. It drifts badly
  (181.1 mm center median, 170.6 mm with offsets); this is not the main body test.
- OAK scores use the corrected EgoHand/Sapiens 2–6 s RRD. Side is matched by
  independent Quest projection with a 20 px ambiguity margin, not by either
  candidate's error. Only Quest-held-out frames and common finite points count.
  The artifact does not retain detector confidence, so no OAK confidence cutoff
  is claimed. Neither detector is a fitting input.

Median / p90 on common finite points, lower is better:

| Held-out target | Fit | Triangulation | Unit |
|---|---:|---:|---|
| Quest hands, all held-out blocks | 26.75 / 63.75 | 18.21 / 60.88 | mm |
| Quest hands, target frame | 42.18 / 71.75 | 31.92 / 70.50 | mm |
| rig_00 body | 27.83 / 50.18 | 20.13 / 37.12 | px |
| rig_00 hands | 10.09 / 31.00 | 11.50 / 33.91 | px |
| rig_01 body | 19.76 / 41.73 | 21.17 / 45.81 | px |
| rig_01 hands | 7.03 / 14.97 | 9.53 / 21.48 | px |
| rig_02 body | 31.48 / 61.76 | 24.70 / 57.13 | px |
| rig_02 hands | 10.04 / 25.62 | 9.25 / 29.19 | px |
| rig_03 body | 21.77 / 34.96 | 22.16 / 42.02 | px |
| rig_03 hands | 9.84 / 22.29 | 11.15 / 24.97 | px |
| OAK EgoHand right | 24.03 / 40.53 | 17.99 / 36.23 | px |
| OAK EgoHand left | 59.46 / 97.81 | 54.84 / 97.39 | px |
| OAK Sapiens right | 19.46 / 37.66 | 16.62 / 31.45 | px |
| OAK Sapiens left | 57.61 / 102.89 | 55.20 / 103.90 | px |

The leave-one-body-landmark-out test improves from **79.78 / 159.81 mm** at
joint centers to **55.37 / 130.64 mm** with frozen offsets (5,377 points). This
is a harder independent test, not a reduction of the original 35 mm training
residual. Offsets correct observation landmarks; they do not change the mesh
surface or establish accurate anatomical shape.

Sixteen fits took 61.59 s in optimization/mesh evaluation, 77.87 s for the full
experiment including model construction, queries and output. Results and
sample counts: `/tmp/soma-fit/evaluation-final/evaluation.json`. The independent
Quest holdout recording and inspected screenshot are under
`/tmp/soma-fit/evaluation-final/quest-holdout/`.

Conclusion: no general independent accuracy gain yet. Hands improve in three
of four held-out phone views, but lose against withheld Quest and the available
OAK image evidence. Calibration and prior wearer association are fixed and
shared, not held out. Further work should improve the observation model and
priors before treating this as better hand recovery.

## Full-segment viewer layer

The current registered `soma` layer covers the full available triangulated track:
**0.208324653–182.117411775 s**, 4,367 frames. The first head observation becomes
available 17 ms later, so that frame keeps its body/hand data while the absent
head term is masked. Missing head endpoints and gaps are never extrapolated into
fitting targets. All weights, priors, iteration counts and shape fitting stages
are unchanged from the full-observation run above.

```bash
pixi run -e selfcap-soma --frozen soma-fit --start 0.0 --end 183.0 --stride 1 \
  --iterations 400 --shape-iterations 80 --mesh-stride 3 \
  --output /tmp/soma-fit/full-segment
pixi run -e selfcap-soma --frozen soma-overlay \
  --source /tmp/soma-fit/full-segment/fit.rrd \
  --output /tmp/soma-fit/full-segment/selfcap__2025-12-03__d31bf1da__ep001.rrd
```

The layer has 1,457 low-LOD float32 meshes with normals (about 8 Hz, including
both endpoints), with 3D joints and all seven 2D overlays at the full fitted
rate. This is an output-size choice, not a change to the fitted trajectory.
It is 175,938,910 bytes (167.79 MiB), atomically copied to the same Spark `soma`
path and registered with `REPLACE` only in `selfcap-post-dev`.

2D overlays now use **only the 54 mapped COCO body/hand entries**, with matching
skeleton connections. Internal spine/head/eye/tip rig joints no longer project
as stray points on the OAK image. The 3D fit retains all 77 SOMA joints.
The recording carries their names so this mapping is explicit and testable.

Full fitting plus mesh evaluation took 8.14 s, with 451 MiB peak allocated GPU
memory. World-view screenshots at 4.144403454 s and 120 s are under
`/tmp/soma-fit/full-segment/`; the camera overlays use the same calibration and
full lens projection helpers as before.

## Shoulder diagnosis and corrected full-range fit

The full-duration layer above was superseded by a calibrated fit. The original
mapping was correct, but the original shape was too narrow and too short in the
torso. Removing head position alone did not solve that mismatch. Learning all
128 shape coefficients helped, but unconstrained position fitting traded away
upper-arm length. The selected fix combines a proper first-part shape stage,
native body bone ratios, a calibration-only bone-length objective, and a shared
camera-to-anatomical-eye attachment. Pose fitting then freezes all of them.

Calibration uses 61 well-observed samples before 25 s, 600 iterations with 300
shape iterations, 128 coefficients, identity prior 0.05, log-scale prior 2.0,
bone prior 0.1, and length weight 2.0. Native body bone ratios are bounded within
0.7–1.3; the fitted range is 0.832–1.054. Maximum identity coefficient is 3.27,
with L2 norm 12.08 across 128 dimensions. Global scale is 1.116. The fitted
Quest-frame camera-to-eye offset is `[0.00057, -0.05147, -0.04107]` metres.
This attachment is inferred, not an independent physical measurement.

Median lengths on the same 0.3–50 s diagnostic window, in millimetres:

| Quantity | RTMW / Quest | Before | Corrected |
|---|---:|---:|---:|
| Shoulder width | 420.7 | 337.5 | 407.3 |
| Hip width | 255.4 | 230.8 | 257.6 |
| Shoulder-mid to hip-mid | 620.8 | 537.1 | 599.1 |
| Upper arm | 285.3 | 273.7 | 266.1 |
| Forearm | 254.7 | 253.5 | 255.4 |
| Thigh | 415.5 | 423.3 | 392.5 |
| Shin | 362.5 | 408.9 | 351.6 |
| Tracked Quest camera to shoulder-mid | 175.1 | 212.4 | 190.8 |
| Anatomical fitted eye to shoulder-mid | 175.1 (tracked reference) | 212.6 | 205.8 |

Signed residuals in the same window, millimetres. Vertical is positive world Y;
lateral is positive toward anatomical left along the horizontal shoulder axis.

| Joint | Vertical before / after | Lateral before / after | Distance before / after |
|---|---:|---:|---:|
| Left shoulder | -57.2 / -14.0 | -48.3 / -10.6 | 89.5 / 30.2 |
| Right shoulder | -56.3 / -12.3 | 37.3 / 5.4 | 81.8 / 27.8 |
| Left hip | 8.7 / 1.7 | -11.0 / 3.7 | 18.4 / 14.7 |
| Right hip | 27.4 / 4.8 | 8.9 / -1.8 | 29.3 / 10.2 |
| Left elbow | -23.9 / -8.5 | 18.7 / 8.1 | 45.7 / 27.3 |
| Right elbow | -23.2 / -3.6 | -8.1 / -6.0 | 40.6 / 23.8 |
| Left wrist | -34.7 / -28.0 | -20.7 / -22.3 | 48.1 / 42.3 |
| Right wrist | -29.6 / -28.6 | 5.0 / 2.6 | 32.0 / 30.0 |

Shoulder distance ablations (left / right median mm): baseline 89.5 / 81.8;
head-position off 91.6 / 116.4; head weight 0.1 instead of 3.0: 72.8 / 71.9;
shared head offset alone 75.6 / 67.5; staged 128 coefficients 68.9 / 64.5;
staged 128 plus attachment 47.6 / 41.3; native bone ratios plus attachment
47.1 / 46.0; adding the neutral-length calibration term 30.2 / 27.8.
A COCO surface-landmark offset can explain part of the remaining 1–2 cm bias;
it cannot explain the original 83 mm width or 84 mm torso-length mismatch.
The independent leave-one-landmark-out evaluator still tests fixed offsets,
without treating them as a change to mesh anatomy.

Reproduce the ablations and calibration, then the full pose fit:

```bash
pixi run -e selfcap-soma --frozen soma-shoulders
pixi run -e selfcap-soma --frozen soma-fit --start 0.0 --end 183.0 --stride 1 \
  --iterations 400 --shape-iterations 80 --mesh-stride 3 \
  --calibration /tmp/soma-fit/shoulders/shape128-lengths-offset-calibration.json \
  --output /tmp/soma-fit/shoulder-corrected-full
pixi run -e selfcap-soma --frozen soma-overlay \
  --source /tmp/soma-fit/shoulder-corrected-full/fit.rrd \
  --output /tmp/soma-fit/shoulder-corrected-full/selfcap__2025-12-03__d31bf1da__ep001.rrd
```

The corrected full pose/mesh pass took 5.47 s (calibration is separate), peaking
at 397 MiB allocated. Full-range body training residual improves from
34.50 / 83.98 mm to 20.34 / 39.89 mm median/p90. This is a training residual,
not an independent accuracy result. Coverage, mesh sampling and the mapped-only
seven-camera overlays stay as described above. The corrected layer replaces
only `selfcap-post-dev/soma` for this segment. Registered-view screenshots at
4.144403454 s (World and exo rig_02) and 120 s (World) were inspected under
`/tmp/soma-fit/shoulder-corrected-full/`.

The independent evaluator supports the same calibration procedure while
excluding each evaluation source *before calibration as well as pose fitting*:

```bash
pixi run -e selfcap-soma --frozen soma-evaluate --calibrated \
  --options.shape-components 128 --options.identity-prior 0.05 \
  --options.scale-prior 2.0 --options.learn-head-offset \
  --options.learn-bone-scales --options.bone-length-weight 2.0 \
  --output /tmp/soma-fit/shoulder-heldout
```

The calibrated holdout rerun is at `/tmp/soma-fit/shoulder-heldout/evaluation.json`.
Its median/p90 checks remain separate from training residuals:

| Held-out check | Original fit | Corrected fit | Triangulation |
|---|---:|---:|---:|
| Quest hands (mm) | 26.75 / 63.75 | 22.88 / 62.13 | 18.21 / 60.88 |
| Body centers, leave-one-landmark-out (mm) | 79.78 / 159.81 | 53.02 / 132.11 | — |
| Body with fixed landmark offsets (mm) | 55.37 / 130.64 | 46.85 / 116.47 | — |
| rig_00 body (px) | 27.83 / 50.18 | 23.09 / 39.58 | 20.13 / 37.12 |
| rig_01 body (px) | 19.76 / 41.73 | 18.08 / 39.71 | 21.17 / 45.81 |
| rig_02 body (px) | 31.48 / 61.76 | 24.09 / 55.95 | 24.70 / 57.13 |
| rig_03 body (px) | 21.77 / 34.96 | 20.32 / 39.88 | 22.16 / 42.02 |

Thus body proportions improve, including three of four omitted-camera body
checks. The method still does not beat triangulation for withheld Quest hands
or OAK detector agreement. These remain same-segment development checks with
fixed shared calibration, not an untouched external accuracy benchmark.
The updated evaluation (16 pose fits plus 6 cached calibration fits) took
91.88 s of optimization/mesh time and 112.14 s total.

Final corrected layer:
`/tmp/soma-fit/shoulder-corrected-full/selfcap__2025-12-03__d31bf1da__ep001.rrd`,
175,921,774 bytes (167.77 MiB). It preserves all 4,367 joint frames and 1,457 mesh
frames from 0.208324653 to 182.117411775 s. All seven 2D streams contain mapped
body/hand landmarks only. The corrected mesh and overlays were verified in the
registered World and exo rig_02 views at 4.144403454 s, and World at 120 s.

## Ear-anchored head attachment (supersedes the eye-offset fit)

This revision removes the LeftEye position pin. It fits the COCO 3/4 ears to surface landmarks and learns one headset-to-skull rotation and translation from 61 well-observed frames before 25 s. Identity (128 coefficients), scale, bone ratios, and attachment are frozen for the subsequent pose fit. The earlier shape-stage bone-length constraints remain unchanged.

Low-LOD vertex averages: left **676, 649, 675**; right **2880, 2853, 2879**. These lie at the anterior ear-canal/tragus region. The selected vertices were checked in `/tmp/soma-fit/ear-landmarks-detail.png`. A GPU integration test compares sparse skinning with the full mesh, including gradients and frozen calibration.

The translation is camera-to-SOMA-Head, expressed in the local SOMA Head frame: **[+122.83, −101.10, −62.21] mm**. With the fitted rotation it is **[+6.51, −29.59, −168.11] mm** in Quest camera RDF coordinates. The sign is from tracked camera position toward the anatomical Head joint; this is not an eyeball estimate. The rotation matrix is stored in `ears/ears-offset-rotation-calibration.json` and the recording parameters.

The stored convention is `R_world_head = R_world_quest @ A` and `p_world_head ≈ p_world_quest + R_world_head @ t`; `A` is the fitted attachment matrix and `t` the reported local offset.

At full-segment scope, the attachment position residual is **3.07 / 5.77 mm** and the rotation residual is **10.98 / 23.53 degrees** (median / p90). Thus a rigid attachment does not explain every frame exactly. Ear landmark choice and head rotation remain prototype limitations.

### Controlled 0.3–50 s comparison

All rows below are fitting residuals, not independent accuracy. The ear candidate freezes shared calibration after the first 25 s, but still observes per-frame body and ears throughout.

| Landmark | Eye pin median mm | Ears + attachment median mm | Before/after vertical mm | Before/after lateral mm |
|---|---:|---:|---:|---:|
| left ear | 140.9 | 14.5 | -135.8 / 0.5 | -19.9 / 1.5 |
| right ear | 165.8 | 8.0 | -145.8 / -0.2 | -23.1 / -2.6 |
| left shoulder | 89.6 | 29.9 | -57.2 / -13.0 | -48.4 / -7.4 |
| right shoulder | 81.5 | 29.3 | -56.2 / -10.6 | 36.9 / 4.3 |
| left hip | 18.2 | 14.3 | 8.8 / 4.6 | -11.0 / 3.2 |
| right hip | 29.4 | 11.5 | 27.4 / 8.7 | 8.8 / -0.6 |
| left elbow | 45.4 | 27.3 | -23.9 / -9.5 | 18.8 / 7.6 |
| right elbow | 40.6 | 30.4 | -23.0 / -2.8 | -8.0 / -6.4 |
| left wrist | 47.9 | 41.8 | -34.0 / -28.1 | -20.4 / -23.0 |
| right wrist | 31.8 | 32.5 | -29.7 / -31.2 | 5.0 / 4.7 |

Lateral is positive toward the wearer’s left shoulder; vertical is world +Y. Per-frame shoulder and hip midpoints define torso length.

| Dimension | RTMW mm | Eye pin mm | Ear fit mm |
|---|---:|---:|---:|
| shoulder width | 420.7 | 337.5 | 410.9 |
| hip width | 255.4 | 230.8 | 256.4 |
| upper arm | 285.3 | 273.7 | 268.0 |
| forearm | 254.7 | 253.5 | 255.4 |
| thigh | 415.5 | 423.3 | 393.3 |
| shin | 362.5 | 408.9 | 351.7 |
| shoulder to hip midpoint | 620.8 | 537.1 | 595.4 |
| Quest tracked pose to shoulder midpoint | 175.1 | 212.5 | 193.4 |

The previous shape/length fit with a learned eye offset gave shoulder medians 30.2/27.8 mm. Ear anchoring gives 29.9/29.3 mm: it preserves that shoulder improvement while removing the invalid camera-to-eye assumption. Shape calibration, not merely switching the head constraint, explains most of the shoulder improvement. Residual joint-centre versus skin-landmark mismatch remains.

### Full segment and viewer evidence

Only `selfcap-post-dev`, segment `selfcap__2025-12-03__d31bf1da__ep001`, layer `soma`, was replaced. Coverage is 0.208324653–182.117411775 s: 4,367 joint/ear/overlay frames in all seven cameras and 1,457 low-LOD meshes with float32 normals. Layer size: **176,453,457 bytes (168.28 MiB)**. No other dataset was changed.

Full-segment body fitting residual: **21.93 / 42.52 mm**, compared with **34.50 / 83.98 mm** for the original eye pin. Ear residual: **10.11 / 21.79 mm**. Optimization plus mesh evaluation: **9.49 s, 2.17 ms/frame**, peak PyTorch allocation **420.72 MiB**. File encoding, catalog reads, and viewer time are excluded.

Artifacts:

- `/tmp/soma-fit/ears-full/fit.rrd` — complete local fit and parameter streams.
- `/tmp/soma-fit/ears-full/selfcap__2025-12-03__d31bf1da__ep001.rrd` — registered layer.
- `/tmp/soma-fit/ears-full/layer-validation.json` — identities, counts and endpoints.
- `/tmp/soma-fit/ears-full/world-4.144.png`, `exo-rig-02.png`, `world-120.png` — catalog screenshots.
- `/tmp/soma-fit/ears/diagnosis.json` — controlled residual and proportion tables.
- `/tmp/soma-fit/ears-heldout/evaluation.json` — independent evaluation.

### Independent checks

Each omitted phone is excluded before triangulation and before fitting the shared shape/attachment. Quest hand targets are removed in alternating two-second blocks before calibration. Ear evaluation removes both ears after 25 s. Body landmark tests remove the scored landmark after 25 s, then fit its observation offset only on earlier frames.

| Held-out measurement | Fit median / p90 | Comparison median / p90 | Unit | Points |
|---|---:|---:|---|---:|
| Quest hands: alternating 2 s held-out blocks | 22.54 / 62.59 | 18.21 / 60.88 | mm | 25763 |
| Quest hands: held-out target frame | 35.49 / 72.35 | 31.92 / 70.50 | mm | 41 |
| Ears [25,50]: withheld ears vs original eye pin | 21.57 / 48.68 | 148.17 / 171.51 | mm | 1199 |
| Body on [25,50]: fixed offsets vs joint centers | 94.28 / 173.76 | 97.37 / 176.74 | mm | 6574 |
| Body [25,50]: leave-one-landmark-out offsets vs centers | 45.39 / 114.36 | 53.16 / 133.20 | mm | 5377 |
| rig_00 leave-camera-out body | 22.84 / 39.30 | 20.13 / 37.12 | px | 7652 |
| rig_00 leave-camera-out hands | 10.24 / 31.51 | 11.50 / 33.91 | px | 43268 |
| rig_00 leave-camera-out ears | 35.44 / 47.53 | 34.22 / 47.82 | px | 2047 |
| rig_01 leave-camera-out body | 18.82 / 40.52 | 21.17 / 45.81 | px | 7906 |
| rig_01 leave-camera-out hands | 7.57 / 15.60 | 9.53 / 21.48 | px | 45941 |
| rig_01 leave-camera-out ears | 37.75 / 54.38 | 40.94 / 57.63 | px | 1380 |
| rig_02 leave-camera-out body | 24.18 / 57.46 | 24.70 / 57.13 | px | 9388 |
| rig_02 leave-camera-out hands | 10.03 / 26.03 | 9.25 / 29.19 | px | 44390 |
| rig_02 leave-camera-out ears | 30.26 / 42.98 | 31.89 / 49.17 | px | 1212 |
| rig_03 leave-camera-out body | 19.80 / 38.20 | 22.16 / 42.02 | px | 8730 |
| rig_03 leave-camera-out hands | 10.03 / 22.44 | 11.15 / 24.97 | px | 45554 |
| rig_03 leave-camera-out ears | 22.96 / 33.53 | 24.60 / 34.51 | px | 1391 |
| OAK /image/egohand/right: Quest-held-out frames | 20.78 / 34.73 | 17.99 / 36.23 | px | 1007 |
| OAK /image/sapiens/right: Quest-held-out frames | 18.11 / 30.25 | 16.62 / 31.45 | px | 719 |
| OAK /image/sapiens/left: Quest-held-out frames | 55.04 / 102.35 | 55.20 / 103.90 | px | 719 |
| OAK /image/egohand/left: Quest-held-out frames | 58.35 / 97.59 | 54.84 / 97.39 | px | 944 |

Comparison is triangulation unless the row explicitly names the original eye pin or joint centres. Phone comparison triangulation uses only the other three phones. The body offset rows compare corrected landmarks against uncorrected centres from the same held-out fit; the offsets do not move the displayed mesh.

The original eye-pin body leave-one-landmark-out centre error was 79.78 / 159.81 mm; it is now 53.16 / 133.20 mm. With observation offsets, it changes from 55.37 / 130.64 to 45.39 / 114.36 mm. With every body target absent after 25 s, error remains about 97 mm: this is not a strong unconstrained motion model.

The fit still loses to triangulation in the aggregate independent Quest hand test (22.54 vs 18.21 mm median). Body, hand and ear phone medians improve in three of four cameras, but OAK hand evidence remains mixed. No claim of general hand reconstruction improvement is supported.

Independent evaluation took 150.12 s total, including 129.63 s of optimization across 24 calibration/pose fits. Hyperparameters were selected during this prototype; these are held-out data terms, not a locked external benchmark.

Pixel review: the full mesh is visible in both World screenshots. The cyan shoulders closely follow the image shoulder line in exo rig_02 but remain slightly below it; hands still show disagreement. No stray unmapped head/spine point appears at the microwave. All screenshots show the requested maximized pane and time cursor.

Validation: `selfcap-soma-dev gate` passed (14 unit tests, 2 GPU integration tests); `ci ci` passed (383 structure tests and the lock check). Run logs are under `/tmp/soma-fit/ears-final-gate.log` and `/tmp/soma-fit/ears-ci.log`.

Reproduce the selected calibration with `pixi run --frozen -e selfcap-soma soma-shoulders`. Then:

```bash
pixi run --frozen -e selfcap-soma soma-fit --start 0.0 --end 183.0 --stride 1 \
  --iterations 400 --shape-iterations 80 --mesh-stride 3 \
  --calibration /tmp/soma-fit/ears/ears-offset-rotation-calibration.json \
  --output /tmp/soma-fit/ears-full
pixi run --frozen -e selfcap-soma soma-overlay \
  --source /tmp/soma-fit/ears-full/fit.rrd \
  --output /tmp/soma-fit/ears-full/selfcap__2025-12-03__d31bf1da__ep001.rrd
pixi run --frozen -e selfcap-soma soma-evaluate --output /tmp/soma-fit/ears-heldout \
  --calibrated --options.shape-components 128 --options.identity-prior 0.05 \
  --options.scale-prior 2.0 --options.learn-head-offset --options.learn-head-rotation \
  --options.use-ears --options.learn-bone-scales --options.bone-length-weight 2.0
```

## Hand mesh repair with raw Quest orientations

Confirmed cause: fitting only joint positions admitted severely twisted bone rotations. The 26-joint raw CSVs for this exact episode contain 183 columns (`ts_ns` + 26 × XYZ/XYZW), but ingest previously retained only the 21 position landmarks. The new typed `dataforge.quest_joint_poses` reader preserves all 26 positions and optional XYZW rotations; the existing ingestion API still exports its original positions, and no source dataset was re-ingested.

Raw source: `/mnt/nas/datasets/exoego-self-collected/main/cut/2025-12-03/d31bf1da-c23c-4e4b-b88b-eb5c5ad20e3f/episodes/episode-001/quest/{left,right}_hand_poses.csv`. Reading this source directly is an explicit exception for orientations absent from the catalog. Timing retains the 225.7 ms Quest offset and 48 ms hand latency.

Fix: estimate one fixed Quest-to-SOMA rotation per hand bone from bone direction and palm normal over the first 25 s (median alignment spread 1.01°). Initialize hand rotations from those frames, then fit a world-rotation term at weight 30 and matrix-error scale 0.03, while retaining the original Quest position weight 3 at 10 mm. Body/root, head and shoulder pose remain frozen during hand/forearm refinement. Enable SOMA procedural twists for final low-LOD skinning; public joint-centre parity is tested. Pose-corrective weights are absent from the available assets, so correctives remain off. Increasing mesh resolution alone cannot determine the missing bone rotations.

Twist display alone barely helped: flipped-face p90 was 11.04% without procedural twists and 11.01% with them. Rotational constraints were necessary. Wrist-location hypothesis was not confirmed: Quest minus triangulated wrist along the forearm direction was +15.76 mm left / +5.49 mm right (positive toward the hand), not 20 mm down the forearm. No wrist offset was introduced.

### Whole-segment before / after

| Measure | Before | After |
|---|---:|---:|
| Body median/p90 mm | 22.03 / 42.84 | 22.02 / 45.57 |
| Shoulders median/p90 mm | 29.38 / 55.03 | 28.93 / 54.23 |
| Ears median/p90 mm | 10.34 / 21.98 | 10.01 / 21.24 |
| Quest hands median/p90 mm | 4.73 / 11.43 | 8.29 / 16.06 |
| OAK RGB 15-finger median/p90 px | 11.47 / 26.89 | 13.40 / 29.36 |
| Wrist twist p90 degrees | 28.42 | 25.34 |
| Wrist twist max degrees | 107.23 | 75.74 |
| Finger roll p90 degrees | 158.75 | 78.14 |
| Outside broad finger bounds % | 85.02 | 18.42 |
| Flipped-face p90 % | 11.08 | 0.47 |
| Collapsed-face p90 % | 1.51 | 0.60 |

Position/image scores are fitting disagreements, not independent accuracy. Before is the previously registered Quest-trusting fit; after uses current catalog observations. The 15 OAK indices are 2,3,4,6,7,8,10,11,12,14,15,16,18,19,20, all finite samples without calibration trimming.

Roll uses swing–twist decomposition, not Euler angles. Broad diagnostic bounds are roll >45°, side swing >35°, or flex-swing magnitude >120°. These are screening proxies, not validated clinical anatomical limits. The remaining 18.4% flags and 78° finger-roll p90 are explicitly unresolved. A flipped face has a normal opposite the rest normal transported by blended bone rotations; collapsed means area below 20% of rest. These proxies do not certify absence of self-intersection.

Baseline worst frames by combined angle/collapse score: 50.622890713 and 24.415649348 s. The evaluation set also includes 39.862079571 and 4.144403454 s. Per-frame results are in `selected-metrics.json`; full diagnostic scalars are in `diagnosis-swing-twist.json`.

Visual review: at both worst frames and 39.862 s, the before mesh has a pinched wrist and folded/rolled palm. The repaired wrists are continuous and the palms/fingers have recognizable hand shapes. Some creases and curl remain. Each close-up contains only the cropped hand/forearm meshes, before on the left and after on the right; the open cut at the upper forearm is the deliberate crop boundary.

Artifacts in `/tmp/soma-fit/hand-repair-final/`: `hands-50.623.png`, `hands-24.416.png`, `hands-39.862.png`, `oak-snapshot.png`, `fit.rrd`, `summary.json`, `selected-metrics.json`, `diagnosis-swing-twist.json`, and the registered segment RRD.

Only the `soma` layer for this segment in `selfcap-post-dev` was replaced. Coverage remains 0.208324653–182.117411775 s: 4,367 joint/overlay frames, 1,457 meshes with normals. Layer size 176,458,543 bytes (168.28 MiB). Final fit/mesh evaluation took 41.67 s, 9.54 ms/frame, with 562.77 MiB peak allocated GPU memory; catalog reads, orientation calibration and diagnostics are excluded. The GPU was shared.

Validation: five targeted reader/frame/skin tests passed; the existing dataforge clock/mapping compatibility test passed; targeted typechecking and Ruff passed. No full gate. The lockfile only adds the existing editable dataforge package to the SOMA environments; other solves are unchanged.

Reproduce with `pixi run --frozen -e selfcap-soma soma-repair-hands`, then export with `soma-overlay --source /tmp/soma-fit/hand-repair-final/fit.rrd --output <layer.rrd>`. Registration remains a separate authorized action.

OAK pixel evidence uses `oak-snapshot.png`, an isolated native Rerun rendering of the catalog-decoded RGB frame at 39.835666282 s and the latest registered SOMA projection at 39.831673680 s (cursor 39.862079571 s). The current catalog blueprint did not display the 2D SOMA layer even though its samples were queryable. Explicit-frame and draw-order experiments did not resolve this and were removed; catalog 2D visibility remains an open viewer issue. The isolated image shows residual finger/wrist misalignment, consistent with the 13.4 px score.

## Sapiens2 source and ears-only head modes

`--exo-source sapiens2` selects the schema-discovered Sapiens2 triangulation;
`rtmw` remains the default. Missing Sapiens hand slots have zero influence.
`--options.head-mode ears` (the CLI default) removes Quest head position and rotation losses;
`ears-rotation` adds only its rotation. Both disable attachment translation.
The legacy `quest` mode remains available.

The full-segment [Sapiens2 comparison](docs/sapiens2-comparison.md) documents
A/B settings, cross-detector pixel checks and the published result. A's head
pitch was implausible; B was published with a soft orientation constraint.
Quest hand positions and orientations remain the hand targets. The camera
pose origin is never an anatomical head-position target in either new mode.
