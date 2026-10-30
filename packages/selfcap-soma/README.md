# SelfCap SOMA fitting prototype

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
