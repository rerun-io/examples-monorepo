# SelfCap alignment and calibration

Catalog-based alignment and calibration for explicit episodes of one SelfCap session.
Inputs are read only. Outputs are Sapiens2 detection layers, JSON records and RRD copies; registration
and layer swaps remain separate operator actions.

| Stage | Module | Output |
|---|---|---|
| Read one named segment | `catalog_io` | Rig layout, encoded video, gyro and Quest tracks |
| Rotation/repeat signals and correlations | `links`, `episode_links` | Coarse links, gyro-refined lags and repeat clocks |
| GPU phone hands | `hands_gpu`, `detection_io` | Property-free `sapiens2` layer RRDs |
| Shared camera/time fit | `quest_fit`, `session_fit`, `sync_policy` | Strict `TimeSync` JSON |
| Copy and retime base | `patch_base` | New aligned RRD with provenance |
| Inspect both clocks | `apis.sync_curves` | Rerun curves and residuals |

## Commands and guards

```bash
pixi run -e selfcap-curate --frozen selfcap-align \
  --base-dir /local/base --output-dir /local/aligned \
  --catalog-url rerun+http://127.0.0.1:51235 \
  --segment-ids selfcap__2025-12-30__65d00748__ep001 selfcap__2025-12-30__65d00748__ep002
```

`--key-file` accepts one explicit segment id or unprefixed episode key per line.
Selection must belong to one session; no catalog enumeration is performed.
Alignment reads presented-time bases containing Quest hands, factory metadata,
the session UUID and ego `first_frame_idx`. Older incomplete bases need conversion.
Models use Sapiens2 0.4B (batch four) and YOLOX (batch eight) through posekit/trtkit TensorRT.
The CLIs use per-machine model caches; callers can set cache environment variables for isolation. Existing targets, NAS output
and output beneath the input base directory are refused.

The legacy served-base migration remains supported:

```bash
pixi run -e selfcap-curate --frozen selfcap-patch-base \
  --keys 2025-11-24__027d8215__ep002 --sync-dir /local/sync --sync-name '{key}.json' \
  --served-dir /local/served --raw-root /readonly/cut --output-dir /local/migrated
```

Only this migration reads raw MP4/hand/factory metadata. It restores video
container starts, adds missing Quest hands and records provenance. Fresh-base
patching reads no raw files. A patched base cannot be patched again.
`--mode migrate-keep-clock` (default) restores the served clocks; `--mode migrate-realign` rebuilds alignment from the supplied record. Align uses the explicit `fresh` patch mode.
`--quest-video-delta-ms` changes only Quest video alignment. Neither CLI registers
or swaps layers; use `dataforge-swap-layer --layer` separately.

For evidence, run `selfcap-sync-curves --segment-id <id>` in the same environment.
Its shared `--rr-config.save <path>` and `--rr-config.headless` options support
saved recordings without a display. The tool reads each clock separately.

## Clocks

`capture_time` preserves presented video PTS, including the container start,
and raw CSV timestamps for IMU, Quest pose and hands. `video_time` aligns them
to OAK left. Offsets are milliseconds to **add** to capture time. RGB uses
`a_ms + b_ms_per_s * (t_start_session_s + capture_seconds)`; the session origin
is the ingested OAK left frame counter divided by 24. Phone alignment uses
`tau - 48 ms + quest_pose`. Missing measurements remain null with flags.

Frozen scratch links used container PTS. Catalog bases that dropped the start
therefore need per-stream corrections only in reference comparisons; production
measurements always use the catalog clock consistently. The pilot frozen coarse
links predate a scratch revision: recorded drift is 2.05 ms (left→pose) and
3.39 ms (left→gyro). Golden comparisons retain the final scratch references.

## Validation and provenance

Run `pixi run -e selfcap-curate-dev --frozen gate` for lint, typecheck, deadcode,
unit and integration tests. Run `tests-golden` separately with GPU access.
Units cover synthetic timing, DLT, fill/flag rules and patch invariants.
Integration checks explicit catalog links and migration against fresh conversion.
Goldens cover three CPU-fit sessions, GPU hand parity, and raw conversion through
the actual alignment CLI. Tests keep their reference tolerances and use temporary
bases and test-scoped servers; they never register on the shared catalog.

The read-only source is
`/home/pablo/exoego-data/dataforge-selfcap-aligned/v2/code/`.
Frozen comparison fixtures live in `tests/fixtures/`; other reference adapters
are test-only in `tests/frozen_v2.py`. Program records use strict pyserde JSON;
arrays remain typed streams rather than JSON payloads.

## Session calibration

| Stage | Module | Output |
|---|---|---|
| Exo fixed-focal fit and profiles | `exo_calibration` | Cameras, episode overrides, BA cost curves |
| OAK hands and mount | `hands_gpu`, `oak_mount` | Typed hands and Quest-rig RGB transform |
| Factory/Quest borrowing | `factory`, `catalog_io`, `device_tables` | Stereo transforms and pinholes |
| Write calibration | `calib_layer`, `apis.calibrate` | Session JSON and calibration-only RRDs |

```bash
pixi run -e selfcap-curate --frozen selfcap-calibrate --output-dir /local/calib --key-file /local/keys.txt
pixi run -e selfcap-curate --frozen selfcap-focal-profile --key-file /local/keys.txt --output /local/profile.json
pixi run -e selfcap-curate --frozen selfcap-pool --profiles /local/f_profile/*.json --output /local/device_intrinsics.json
```

Calibration reads the catalog's `sapiens2` layer, with GPU fallback for missing phone data. OAK hands are computed from catalog video. Tracks and factory values come from explicit segments. Outputs stay local; NAS writes, existing targets, registration and swaps are refused. `device_intrinsics.json` is rebuilt by the profile/pool commands. Rebuild `oak_device_rgb_R_imu.json` with `selfcap-pool-oak-rotation --free-paths /local/oak_free/*.json --gyro-dir /local/gyro --output /local/rotations.json`.
`borrowed_camera50.json` is curated by hand from verified device, healthy-segment and lever-arm facts.
Goldens cover three exo/OAK sessions, focal profiles and both pooled tables, GPU hands, served layer values including camera-50 borrowing, and raw alignment through calibration with cached detections.

## Keypoints

```bash
pixi run -e selfcap-curate --frozen selfcap-keypoints --calib-dir /local/calib --output-dir /local/complete \
  --catalog-url rerun+http://127.0.0.1:51235 --key-file /local/keys.txt --rate 2.0
```

Register base before calib in every batch: the catalog resolves duplicated static values by the newest write.
If frame planning, missing hands or decoding fails after model load, rerun with the unwritten keys only.
Order: `selfcap-calibrate` → `selfcap-keypoints` → one `dataforge-swap-layer --layer calib`.

`selfcap-post` runs align, calibrate, and keypoints in one process. It reads each episode's required video packets and raw Quest hands/head poses once and keeps them in memory for the later stages. Aligned phone timestamps use the same integer offsets as the base writer. Standalone stage commands still read from the catalog.

For `selfcap-post`, omit `--rate` to use the first episode's median native phone rate (rounded to 0.001 Hz), or pass `--rate` to select a frequency. This avoids a separate video read to measure the rate. The standalone `selfcap-keypoints` command still requires `--rate`.
Cameras come from local `--calib-dir` files; the catalog supplies patched videos, clocks and Quest anchors.
Explicit ids or a key file select one session. Preflight reports unprocessable episodes with `KEYPOINTS_SKIP`; good episodes run.
Existing targets, NAS outputs and outputs within the input directory are refused; no external registration or swap occurs.
Specify `--rate`: served rates were 2 Hz (3,979 episodes), 3 Hz (85), and 10 Hz (40). YOLOX-m HumanArt and RTMW-x use TensorRT batches of eight,
keeping three people per frame on a half-open `video_time` grid anchored at cut zero; nearest-frame ties go left.
Quest anchors use capture_time: left-eye head + quest_pose (100 ms gap), wrists −48 ms + quest_pose (40 ms gap).
Wearer selection uses head weight 2, person confidence ≥0.3 and box cost <0.5. Triangulation accepts joint detector scores
≥0.3, with radial undistortion, weighted DLT, pair RANSAC, 0.5° ray separation, 8 px inliers and two refits.
Confidence is the mean inlier score, multiplied by 0.9 for two-view support. It is not a visibility probability.
The 3D entity also stores `support_view_count`, `support_max_reprojection_px`, and `support_inlier_mask` per joint.
Each mask lists cameras in the static `support_camera_paths` order. A measured point needs at least two supporting cameras; temporal estimates carry `temporal_bridged` and retain the actual support count.
Body joints use a temporally consistent camera-pair path, a five-frame median, and bidirectional One Euro smoothing (1 Hz, beta 0.3). Smoothing stays within the 8 px geometric support gate. Temporal jumps above 30 mm second difference at 24 Hz and missing runs up to 0.5 seconds are interpolated between supported neighbours, with 75% of the smaller endpoint confidence. Longer gaps stay missing. Other triangulated joints retain beta 2.0; Quest hand tracking is unchanged.

The default `/world/skeleton` and `<cam>/pinhole/skeleton` retain ears, shoulders, elbows, the shoulder–hip torso quad,
and both 21-joint hands. Knees, ankles and feet remain in the hidden full COCO-133 layers.
Every view uses the same confidence colours: red at 0.7, yellow at 0.85, green at 1.0, clamped outside that range.
Low-confidence and bridged points remain visible. Blue annotation links use the shared typed keypoint archetypes and their class/keypoint IDs; custom confidence components carry per-joint and average confidence.
Quest hands have no per-joint confidence, so tracked Quest joints use 1.0. The support masks remain available for inspection.
AV1 reference/control/sample packets are retained; selected decoded frames are checked bit-for-bit in goldens.
Each output preserves every calib chunk and adds `/world/pred/rtmw_triangulated/coco133_xyz`, per-camera
`pinhole/pred/rtmw/coco133_uv` and `kp3d_*` fields in the same `calib` group; no external quality metrics are stored.
Frozen reference comparisons live in `tests/test_keypoints_*golden.py`. Those references use the earlier
acceptance and smoothing policy; temporal estimates now preserve short missing runs.

Calibration and focal profiling read alignment records from each selected catalog base. Phone detections come from the selected segment’s `sapiens2` layer; local sync JSON cannot override the base clocks. Focal profiling refuses missing detection data.

`selfcap-align` and `selfcap-calibrate` require a new, absent output directory. They build the session in a sibling staging directory and publish it with one rename after all outputs succeed. A failed run leaves no output directory; the same command can be retried.

Sync outputs use `selfcap__<key>.json` throughout; the migration command accepts an explicit `--sync-name` template for legacy input directories.

## Sapiens2 detection layer

Align writes `sapiens2/selfcap__<key>.rrd` beside the aligned bases. Each layer has
application `dataforge`, the segment’s recording ID, and **no properties**.
Phone identities come from the segment layout. All 133 points are stored at
`<cam>/pinhole/pred/sapiens2/coco133_uv`, with keypoint IDs and `confidence`;
the parent `pred/sapiens2` holds the COCO-133 annotation context. Missing values
and low scores are retained. Float32 xy and source float16 scores round-trip exactly.
`capture_time` is presented frame PTS in integer nanoseconds. `video_time` adds
`round(phone_offset_ms * 1e6)`, as in base patching. No container start is rebased.

Operator order: align → `dataforge-swap-layer --layer base` →
`dataforge-swap-layer --layer sapiens2` → calibrate → keypoints →
`dataforge-swap-layer --layer calib`. The existing swap command supports adding
an absent layer; each command uses explicit files and its normal idle gate.
Nothing in align, calibrate or the converter registers or swaps catalog data.

Phone goldens read the served `sapiens2` layer through private catalogs. Frozen OAK and
multi-person RTMW detector streams live under the local reference root’s
`golden-refs/oak-sapiens2/` and `golden-refs/rtmw-kp2d/` directories as viewer-readable RRDs.
Their typed components retain exact float64 seconds beside the integer-ns `video_time`
timeline. The tests read these fixtures directly; they do not import external research code.

## Candidate optical calibration

`selfcap-post --exo-refine` is the legacy optical diagnostic path. The batch
runner uses `selfcap-exo-label` with fixed device VP focals and no head offset.
`--exo-reference-dataset` selects a read-only source for labels and independent
RTMW checks. An episode without an `exo_landmarks` layer keeps the normal fit.
Only exo camera poses may change. Quest camera extrinsics, SLAM poses, hands,
and the OAK mount are preserved. The calibration file then owns triangulation,
display skeleton and camera-centre projections.

`exo_landmarks` stores labelled `Points2D` at
`/world/rig_XX/cam_00/pinhole/exo_landmarks` on `video_time`. Its recording
property holds a strict `Landmarks` record (view, seconds, target, pixels,
confidence, labeller, method and date). `quest0` and `quest1` are factory optical
centres; `phoneN` is exo rig N. The committed d31bf1da labels retain all 232
marks. Their original agent model was not recorded; provenance states that gap.

```bash
pixi run --frozen -e selfcap-curate selfcap-dev-cycle \
  --segment selfcap__2025-12-03__d31bf1da__ep001 \
  --tag exofit-example --dataset selfcap-post-dev-exofit \
  --landmarks selfcap_curate/landmarks/selfcap__2025-12-03__d31bf1da__ep001.json \
  --exo-refine --focal-source fit --oak-egohand-assets /tmp/egohand
```

This is an explicit publishing tool. It refuses production datasets, uses a
new run directory, and registers the default blueprint. With `--dataset selfcap-post-dev`,
it replaces only base, calib, egohand_calib and exo_landmarks; it preserves
SOMA, EgoHand and existing Sapiens2 layers. Separate candidates also copy the
source EgoHand file and alignment detections. The scratch `/tmp/selfcap-5090/dev_cycle.py`
remains independent and unchanged. Run files and reports live beneath
`<remote-root>/<dataset>/runs/<tag>/`.

The focal source is `table`, `fit`, or `fixed`. A `fixed` source requires
`--focal-file` with this strict JSON schema (use full phone IDs):

```json
{"focals":{"Dragon-413A7CC1":850.0,"Tiger-FE57138F":850.0,"Titanium-B00700F9":900.0,"Wolf-387E89E1":900.0},"method":"independent measurement"}
```

The values above illustrate the file format; they are not measured intrinsics.
The solver records the input hash, source, version, fitted cameras and optical
translation under `/__properties/calib`. Camera poses remain at
`/world/rig_XX/cam_00`; pinholes remain at their existing child paths.

For a read-only fit or an exact source-calibration comparison:

```bash
pixi run --frozen -e selfcap-curate selfcap-exo-refine \
  --dataset selfcap-post-dev --landmark-dataset selfcap-post-dev-exofit \
  --segment selfcap__2025-12-03__d31bf1da__ep001 --output-dir /tmp/exofit-check
```

This writes `refinement.json` and `gate.json`. `--calib-record` also writes a
calibration layer through the existing writer. The report pools two held-out
folds: alternate label frames and alternate 10-second hand blocks with a 200 ms
guard. RTMW hand checks use every third held-out frame; body transfer uses every
sixth frame and triangulates from the other cameras. The published model fits
all labels and Sapiens pairs after these checks. No random draws or wall-clock
values enter the solver.

The d31bf1da candidate reproduces the offline medians (pixels): headset
38.53 → 4.56, phones 101.95 → 0.31, Sapiens hands 8.37 → 8.98, independent
RTMW hands 9.59 → 10.50, body transfer 24.67 → 16.55. The hand regressions
fail the no-regression gate. The main dev promotion is explicit and retains
that gate result; it does not imply that the hand metrics improved.

### Bounded optical labeling

`selfcap-exo-label` reads an initial calibration and the segment's base,
Sapiens2 and independent RTMW observations. It writes local files only:

```bash
pixi run --frozen -e selfcap-curate selfcap-exo-label \
  --segment selfcap__2025-12-03__d31bf1da__ep001 \
  --calib-record /path/to/initial-session.json --output-dir /path/to/labels \
  --dataset selfcap-post-ep1-v2 --catalog-url rerun+http://dgx-spark:51235
```

The runner can call `selfcap_curate.exo_label.label_session` with the same
arguments. Its isolated worker has a 360-second deadline, including its Sol
children. Existing output directories are supported. Completed decisions resume;
failed or timed-out attempts reuse their completed packet and label files.
Use a new directory when source calibration or annotations change.

The tool samples four facing headset frames per view, and one frame for static
phones. Every other phone whose predicted centre falls inside the image receives a
540-pixel crop, including grazing-angle candidates. Crops carry a source
pixel grid; Sol runs read-only with medium reasoning and the fast tier on the
default personal account. Headsets and phones use separate calls; missing
front-facing phone targets get one automatic retry with the other identified
phones as context. Phone assignments must agree with a front-facing optical
axis and lie within 240 source pixels of the initial prediction. Coordinates
must fall inside the corresponding crop;
timestamps are canonicalized within one microsecond. A local dark-component snap
moves headset labels at most five source pixels.

Both fit folds share the static phone anchors. Their phone residual is therefore
an **in-sample consistency check**, not independent validation. Headset frames
and 10-second hand blocks are held out; body transfer uses independent RTMW
observations. The robust objective uses fixed relative block shares for headset anchors,
phone anchors, hands, and body (1:0.25:1.5:0.5), each divided equally among
active views and normalized by its observation count. Confidence weights are
normalized within each view. Weighting is applied to the robust cost, preserving
the six-pixel outlier scale. Fold fits and the all-frame fit use the same shares.
Acceptance requires optical medians to improve by at least 2 pixels and 10%,
body transfer median to improve, independent RTMW hand median to regress by at most 1.5 pixels,
and the original fit's Sapiens hand median to regress by at most 2 pixels as a secondary guard,
and all solvers to converge. The deployed model's per-view label medians must
also stay within two pixels of the corresponding full-pipeline fold scores.
Missing metrics reject the candidate.

`result.json` always records `segment`, `decision`, `reason`, `labels`,
`elapsed_s`, `thresholds`, and `gate`. The thresholds are configurable under `--thresholds.*` and saved in every result. Decisions are `accepted`, `rejected`, `timeout`, or
`failed`. Only `accepted` writes `refinement.json` and `calib.json`; all other
outcomes retain the initial calibration. `landmarks.json`, `packet.json`, PNGs,
and per-view logs provide provenance. Nothing is registered by this command.

The default `--focal-source device-vp` fixes each known device to its measured
image-only focal from `selfcap_curate/device_vp.json`. Unknown devices retain
the initial calibration focal. `--focal-source table` retains every initial
focal. Neither option fits focal, principal point, or distortion. Results record
`focals` and `focal_sources` per camera.

The estimator changes only exo camera poses. Quest SLAM, Quest cameras, OAK
mount/poses, and Quest hands are immutable truth. The old `head_translation_m`
compatibility field is always zero. An anchored hands/body bundle adjustment
polishes the optical solution before the final held-out gate; it retains the Sol
landmark terms and exact fixed intrinsics. Offset-enabled cached results are
invalidated by the new solver contract.
