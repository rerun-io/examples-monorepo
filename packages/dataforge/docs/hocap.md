# HO-Cap

## Source

Original archive mirror: `pablovela5620/hocap-original` at
`d9a562638bc4eda48451ce1211385371bbd17be0` (upstream IRVLUTD/HO-Cap).
Upstream project: https://irvlutd.github.io/HOCap/ .

`HocapConfig.root` defaults to `paths.raw_root() / "hocap"`; the host sets
`DATAFORGE_RAW_ROOT=/mnt/nas/datasets`. `--root` overrides it. Only zip
members are read. Extracted directories beside the zips are ignored. `download`
only verifies archive central directories, not every member's CRC. Discovery
sorts the sequence keys in `poses.zip`, skipping subjects without their image
zip. An explicit `--sequences` selection whose subject zip is absent
raises an error naming that zip. This supports the partial local test root.

Write conversions to local disk, e.g.
`DATAFORGE_OUTPUT_ROOT=/home/pablo/exoego-data/hocap/rrd`. Temporary AV1 files live
under that output root and are removed even on failure. A target beneath the
resolved raw root is rejected. Never extract or bulk-write on the NAS; only the
orchestrator copies finished files there, once, and makes them readable before
registration. No raw file is modified or removed.

All nine streams share the shipped frame index. No timestamps or frame rate are
shipped. `video_time = frame_index / FRAME_RATE`, rounded to the nearest
nanosecond; `FRAME_RATE = 30` is the one named source-rate constant in
`hocap_source.py`. The capture property
`clock_source` is exactly:

> no timestamps shipped; 30 Hz inferred from HO-Cap paper §5.1 (10 FPS subsample = mean frame step 3.0 in the released hpe benchmark GT)

`frame_index` is the second timeline on every temporal row. Container PTS does
not define the clock. Capture properties retain `source_revision`, per-camera
`source_resolution`, full `source_num_frames`, and converted `num_frames`.
JPEG sequences become color AV1 NVENC, CQ 36, GOP 60, no B-frames, through the
shared bounded-parallel encoder (3 workers). Clips are logged in camera order
while later encodes run. Both its sample-count check and the Rerun reader's count check
remain enabled. There is no CPU fallback. Full image-index and pose counts must
match metadata before the optional test-only `frame_limit` selects a prefix.
Limited outputs go under `<output_root>/preview-first<N>/`, separate from full
recordings. Archives and member indexes are opened once per dataset instance.

## Raw inventory

Paths below are zip members, not extracted inputs.

| Shipped file or stream | Rate / clock | Layer | Entity or not ingested: reason |
| --- | --- | --- | --- |
| `subject_N.zip: subject_N/seq/meta.yaml` | static | base | Capture/episode properties: subject, object IDs, task integer, sides, counts; selects extrinsics and cameras |
| `subject_N.zip: .../<serial>/color_*.jpg` (8 RealSense, 640×480) | inferred 30 Hz | base | `/world/rig_00..07/cam_00/pinhole/video`, serial-sorted |
| `subject_N.zip: .../hololens_kv5h72/color_*.jpg` (1280×720) | same index | base | `/world/rig_08/cam_00/pinhole/video` |
| `subject_N.zip: .../<serial>/depth_*.png` | same index | — | Not ingested: depth deferred; source uint16 mm, zero invalid, color-registered |
| `calibration.zip: calibration/intrinsics/<serial>.yaml: color` | static | base | Color pinholes, source resolution; distortion not applied because shipped pixels match the undistorted pinhole |
| Intrinsics `coeffs`, `depth`, `depth2color` | static | — | Not ingested: distortion would worsen label fit; depth calibration deferred with depth |
| `calibration.zip: calibration/extrinsics/<meta.extrinsics>` | static | base | `inv(tag_1) @ master_T_cam` on each RealSense camera; tag_0 unused |
| `calibration.zip: calibration/mano/subject_N.yaml` | static | hand_pose | Betas at `/world/gt/hands/<side>/mano`, shared by both hands |
| `poses.zip: .../poses_pv.npy` | same index | base | `/world/rig_08` temporal world-from-PV transform; repeated rows retained; invalid rows NaN, no carry-forward |
| `poses.zip: .../poses_m.npy` | same index | hand_pose | `/world/gt/hands/<side>/mano`: global orientation, 45 PCA coefficients, translation; −1 sentinel rows become NaN |
| Same `poses_m` plus betas and local MANO assets | same index | hand_mesh | `/world/gt/hands/<side>/mesh`, metres; absent sides omitted, invalid frame vertices empty |
| `poses.zip: .../poses_o.npy` | same index | object_pose | `/world/gt/objects/<id>` transforms and `/confidence`; xyzw + metres, metadata slot order |
| `labels.zip: .../<serial>/label_*.npz: hand_joints_3d` | same index | hand_pose | `/world/gt/coco133_xyz`; camera-to-world transform from an available camera per index |
| Same label: `hand_joints_2d` | same index | hand_pose | Eight `<pinhole>/coco133_uv` entities; no HoloLens 2D labels are invented |
| Same label: `cam_K`, `obj_poses` | same index | — | Not ingested separately: redundant copies of calibration and object poses, in camera frame |
| Same label: `seg_mask` | same index | — | Not ingested: segmentation deferred |
| Same label: `obj_class_inds`, `obj_class_names` | same index | — | Not ingested: segmentation visibility class lists deferred with masks; object identities come from metadata |
| `models.zip: models/<id>/textured_mesh.obj` + `.mtl` + `textured_mesh_0.png` | static | object_mesh | `/world/gt/objects/<id>/mesh`: Asset3D GLB built in memory from OBJ/MTL/PNG through the shared writer; confidence Scalars on the mesh entity |
| Same model: `cleaned_mesh_10000.obj`, `cleaned_mesh_2000.obj` | static | — | Not ingested: alternative untextured geometry; full textured mesh selected |

## Layers and entities

Layers are `base`, `hand_pose`, `hand_mesh`, `object_pose`, `object_mesh`.
They share `hocap__<subject>__<sequence>` recording IDs. Annotation layers suppress
automatic recording properties. Base owns the root COCO AnnotationContext and
right-handed Z-up world coordinates. Eight world-anchored exo rigs have one
camera each; the ninth rig follows the shipped HoloLens PV pose with an identity
rig-to-camera transform. Camera axes are OpenCV RDF.

HOCap hand order is already wrist, thumb, index, middle, ring, little. Right
joints copy directly to COCO slots 112–132, left to 91–111. **No Assembly mapping,
thumb midpoint or body wrist copy is used.** All other slots are NaN/0.0.
Present joints have confidence 1.0 because none is shipped; missing joints and
out-of-image pixels have NaN positions and confidence 0.0. The shared hand
writers enforce this rule. Missing label files produce missing 2D rows. 3D uses
the first available serial-sorted camera for each frame, so the known 369-file
master-camera gap in subject_2/20231022_200657 does not erase valid world joints.

Objects use the shared pose and mesh writers. Finite poses with nonzero
quaternions are trusted (confidence 1.0); invalid poses have confidence 0.0.
Mesh albedo alpha is 0 on invalid rows and 1 on valid rows. Meshes do not
duplicate object transforms. The adapter builds an Asset3D GLB in memory from
the shipped OBJ/MTL/PNG, with unique (position, UV, normal) corners to preserve
seams, normalized normals, flipped vertical UVs, and unchanged embedded PNG
bytes. The material has metallic factor 0.0 and roughness 1.0. The shared
`objects.log_object_mesh` writer adds confidence Scalars on the mesh entity.
No model file is written to disk. The textures render in the 0.38.1 viewer: see the pixel evidence under "Parity with simplecv".

The default blueprint has 3D, eight exo panes, and HoloLens. Each exo pane shows
its shipped 2D points. The table has 3D without video plus one HoloLens stream,
and subject/object/frame columns.

## Differences from simplecv

The source explorer's facts and audit identify these corrections:

- Use original zip paths and per-sequence metadata, not the hand-repacked sample.
- Use all shared frame indices, not the shortest video length (the sample audit
  dropped 511 of 741 labels). No stream-derived truncation or frame resampling.
- State the inferred clock instead of recovering a fabricated clock from an
  encode. Keep `frame_index`. Original PV has one distinct JPEG per index.
- Select the extrinsics filename from metadata, sort cameras by serial, and
  validate full pose/image counts before logging.
- Select available 3D label coverage; preserve per-camera 2D and its source gaps.
- Preserve raw MANO parameters and subject shape as data, plus object poses and
  textured meshes. Missing confidence is 0.0, never NaN.
- Keep the verified world transform and undistorted projection math.

Golden checks compare projected world joints to shipped 2D, median Euclidean
error ≤1 px per RealSense camera, and right MANO joints to shipped world joints
within 1e-4 m. HO-Cap labels use the official MANO v1.2 `MANO_LEFT.pkl`
through manopth (`hocap_toolkit/layers/mano_layer.py`, `config/mano_models`).
We evaluate simplecv's tracked `MANO_LEFT.pkl` from wilor-nano `mano_clean`.
The right pkls agree to 1.2e-7 m. The left difference is mean 0.22 mm,
max 4.7 mm; the shapedirs sign fix was tested and excluded (15 mm).
The 5 mm band applies to left `hand_mesh` only; shipped keypoints are unaffected.
Open item: re-check with the official v1.2 pkl before sign-off.
HoloLens calibration is rounded and source poses jitter; source misalignment
of tens of pixels is not corrected.

## Parity with simplecv

Measured 2026-09-25 on pablo-dl-server. simplecv ran `tools/batch_raw_to_rrd.py hocap` from `main`
@ 34ee7f4c in its prod env. Its input was the two sequences extracted from local zip copies into its
flat layout: color, labels, poses, meta; no depth, which it reads only with `--log-depths`. dataforge
ran `dataforge-convert hocap` @ 027b71b5 in its prod env on the same zips. simplecv row i is matched
with dataforge `frame_index` i. The full report, with every difference mapped to an audit defect, is
`/tmp/fleet-artifacts/exoego-migration/runs/hocap-parity.md`.

| sequence | coco133_xyz rows sc / df | video_time diff | joints present sc / df | max joint distance | missing-joint confidence sc / df | per-serial 2D (sc reprojected vs df shipped) | HoloLens pose | MANO mesh vertices |
|---|---|---|---|---|---|---|---|---|
| subject_5/20231027_112303 | 702 / 702 | 0 ns | 14,742 / 14,742 | 1.9e-7 m | NaN / 0.0 | median 0.79–0.80 px, max 1.41 px | 0 m, 3e-8 rad | right 1.2e-7 m |
| subject_5/20231027_113202 | 791 / 791 | 0 ns | 33,222 / 33,222 | 1.9e-7 m | NaN / 0.0 | median 0.79–0.80 px, max 1.41 px | 0 m, 3e-8 rad | right 1.2e-7 m, left 1.0e-7 m |

The 3D keypoints agree within float32 precision. The two sides use the same missing pattern and the
same timestamps: simplecv's h264 PTS at 30 fps equals `frame_index / 30` here. Differences, mapped to
the audit defects:
- Missing-joint confidence and `average_confidence`: simplecv writes NaN, the schema says 0.0.
- simplecv logs no `frame_index`.
- simplecv logs reprojected 2D instead of the shipped `hand_joints_2d`. The ≤ √2 px residual is the
  shipped integer truncation. simplecv also puts 2D on the HoloLens, which ships none.
- Rig order: simplecv follows the order of an intrinsics glob; dataforge sorts serials. Per-serial
  extrinsics agree to 1.2e-7 m.
- Only dataforge keeps the MANO parameters and the objects.

**New simplecv defect (not in the audit):** on right-only sequences simplecv also meshes the absent
left hand. It evaluates the all −1 `poses_m` row and draws the mesh about 1.7 m below the table on
every frame. dataforge omits hands that are not in `mano_sides`.
The audit's 230-of-741 truncation does not occur on the original release: every stream has
`num_frames` frames. dataforge writes all 741 rows of `subject_1/20231025_170650` (sample below).

Pixel evidence (dataforge sample files, headless 0.38.1 viewer, default blueprint, all five layers):
[subject_1/20231025_170650 f270](https://pablos-4800gt.ilish-ruler.ts.net:8768/exoego-migration/evidence/hocap-sample-s1-170650-f270-default-blueprint.png)
([3D zoom](https://pablos-4800gt.ilish-ruler.ts.net:8768/exoego-migration/evidence/hocap-sample-s1-170650-f270-3d-zoom.png)),
[subject_8/20231024_180111 f392](https://pablos-4800gt.ilish-ruler.ts.net:8768/exoego-migration/evidence/hocap-sample-s8-180111-f392-default-blueprint.png)
([exo zoom](https://pablos-4800gt.ilish-ruler.ts.net:8768/exoego-migration/evidence/hocap-sample-s8-180111-f392-exo-zoom.png)).

## Timing

Read `<output_root>/timing/convert.jsonl` with `dataforge.timing.load_records(path, ConvertRecord)`
and registration times with `load_records(path, RegisterRecord)`. Stages are `fetch` (zip
directory and source-table reads; no filesystem extraction), `transcode`, and `write:<layer>`.
Transcode is nested in base write time; do not sum overlapping stages. Capture length is
selected frames / `FRAME_RATE`.

Earlier runs of this branch measured 30.6 and 27.0 s per capture-minute with one encode at a time
(`1+c3f603cb2209`). Three parallel encodes (the review fixes) brought the rate down to the figures below.

**Baseline vs dataforge**, 2026-09-25, pablo-dl-server (RTX 5090 NVENC), prod envs, one process per
sequence, inputs on local NVMe. simplecv time = its batch timer ("Total time taken"). That covers the
JPEG→h264_nvenc preprocessing of all nine cameras (the first run on the extracted tree, so every
`output.mp4` was encoded) and the conversion. dataforge time = `convert.jsonl` `total_s`, all five
layers. Process start-up is excluded on both sides; wall times including `pixi run` were 15.8/16.1 s
(sc) and 9.0/9.2 s (df).

| sequence | capture | simplecv | simplecv s / capture-min | dataforge fetch | transcode (9 streams, 3 at once) | hand_pose | hand_mesh | object_mesh | dataforge total | dataforge s / capture-min | dataforge bytes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| subject_5/20231027_112303 | 23.4 s | 13.26 s | 34.0 | 2.21 s | 3.86 s | 0.48 s | 0.04 s | 0.09 s | 6.70 s | 17.2 | 28.5 MB |
| subject_5/20231027_113202 | 26.4 s | 13.67 s | 31.1 | 2.21 s | 3.85 s | 0.59 s | 0.08 s | 0.18 s | 6.95 s | 15.8 | 42.1 MB |

Soft gate: **passed**. dataforge takes about half of simplecv's time per capture-minute. Both
converters are bound by the NVENC encode. simplecv encodes the nine streams one at a time. dataforge
runs three encodes at once, and its transcode overlaps with base writing.

**hocap-sample** (8 sequences), same host and env, one process (`1+027b71b5fa67`). Here the zips
were read **over NFS from the NAS** (`--root /mnt/nas/datasets/hocap`) and the files were written
to local disk. Wall time for all 8 was 91 s, including start-up and the first central-directory reads.

| sequence | capture | fetch | transcode | hand_pose | hand_mesh | object_mesh | total | s / capture-min | bytes |
|---|---|---|---|---|---|---|---|---|---|
| subject_1/20231025_170650 | 24.7 s | 3.32 s | 7.32 s | 0.71 s | 0.04 s | 0.26 s | 11.68 s | 28.4 | 30.7 MB |
| subject_3/20231024_162327 | 29.3 s | 1.35 s | 6.31 s | 0.76 s | 0.04 s | 0.53 s | 9.02 s | 18.5 | 32.5 MB |
| subject_4/20231026_164131 | 39.1 s | 0.51 s | 8.98 s | 1.16 s | 0.12 s | 0.15 s | 10.96 s | 16.8 | 58.1 MB |
| subject_5/20231027_113535 | 28.5 s | 0.48 s | 6.34 s | 0.71 s | 0.08 s | 0.14 s | 7.79 s | 16.4 | 43.6 MB |
| subject_6/20231025_112229 | 38.8 s | 0.97 s | 10.21 s | 1.13 s | 0.12 s | 0.56 s | 13.04 s | 20.1 | 58.0 MB |
| subject_7/20231022_193506 | 37.9 s | 0.66 s | 9.34 s | 1.20 s | 0.06 s | 0.26 s | 11.55 s | 18.3 | 46.2 MB |
| subject_8/20231024_180111 | 38.5 s | 0.39 s | 9.94 s | 1.12 s | 0.11 s | 0.70 s | 12.29 s | 19.1 | 59.8 MB |
| subject_9/20231027_123814 | 44.2 s | 0.74 s | 9.85 s | 1.19 s | 0.13 s | 0.09 s | 12.03 s | 16.3 | 64.8 MB |

The NFS reads added little: 16–20 s per capture-minute, the same as the local copy. The exception
is the first sequence, 28.4, which pays for the first `labels.zip` directory read.
Registration time: pending (added by the orchestrator after `hocap-sample` is registered).
Full-corpus conversion waits for Pablo's approval.
