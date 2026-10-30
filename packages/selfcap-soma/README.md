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
