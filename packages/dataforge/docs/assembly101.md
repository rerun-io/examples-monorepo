# Assembly101

## Source

The video source is `pablovela5620/assembly101-720p` at
`001839131530cee9b2deb9ca66c025998d10cba4`. Official actions come from `cvml-nus/assembly101` at `bfc15ea5`,
`annotations/` only. The separate AssemblyHands annotation product is not ingested.

The read-only `root` defaults to `paths.raw_root() / "assembly101"`. It contains
`videos/av1-720-new/`, `assembly101_camera_and_hand_poses/`,
`assemblyhands-toolkit/calib/nimble_json_calib/`, and `manifests/`.
Set `DATAFORGE_RAW_ROOT=/mnt/nas/datasets` for NAS inputs, or pass `--root`
and `--annotations-root`. The environment variable sets the parent of `assembly101`
(default `data/raw`); `--root` sets the exact dataset root. The annotations default
is `paths.raw_root() / "assembly101/official/annotations"`.

For this lane, use `--root /home/pablo/exoego-data/assembly101/raw` and
`DATAFORGE_OUTPUT_ROOT=/home/pablo/exoego-data/assembly101/iter`. Conversion
refuses destinations below either input root or `/mnt/nas`. Only the orchestrator
copies finished recordings to the NAS. `--frame-limit N` writes under
`preview-firstN`, keeping previews separate from full recordings.

`download()` verifies the local tree without network access or writes. The mirror
manifest must contain an explicit `video_only=True` row to accept a capture without
pose members. Partial pose trees always fail verification.
`dataforge.datasets.assembly101_download.fetch_pose_members(destination, sequences,
members)` is an explicit driver-only transport. It opens the remote
`AssemblyPoses.zip` through `HfFileSystem` with 16 MiB range blocks and copies only
selected members. It skips files whose sizes match the zip directory, verifies
sizes and CRC during extraction, then replaces a temporary file. Never download
or extract the entire zip. Fetch calibration donors' fixed-extrinsic members too;
all 337 small fixed-extrinsic files and all 20 nimble files are staged in this lane.

Recordings use `assembly101__<sequence>` identities. The catalog name is
`assembly101`; registration supports `--catalog-name assembly101-sample` without
changing those identities. Discovery includes every directory with videos, in
sorted order. Conversion requires all 12 source views.

Each camera stores `source_resolution`, `stored_resolution`, source and written
frame counts, its lens source and AV1 metadata. Capture properties include
`source_revision`, `source_resolution`, `timestamp_t0`, `clock_source`, calibration
coverage and action/hand availability. Exo source images are 1920×1080, stored at
1280×720; ego source and stored images are 636×480. The mirror reduces source
10-bit mono to 8-bit yuv420p. No further color or bit-depth conversion is done.

All videos and pose rows use `video_time = frame_index / 60`, rounded to the
nearest nanosecond, plus `frame_index` as a sequence timeline. Source timestamps
are `t0 + k/60` rounded to milliseconds; `t0` is a device origin, not wall clock.
Mirror PTS are synthetic CFR. `property:capture:clock_source` records this evidence.
Streams are start-aligned and retain their own lengths; neither video nor labels
are trimmed to a common shortest stream. Official action indices are at 30 Hz:
an action frame `f` maps to video frame `2f` and time `f/30`.

## Raw inventory

| Shipped file or stream | Rate / clock | Layer | Entity or not ingested: reason |
| --- | --- | --- | --- |
| `videos/av1-720-new/<seq>/C*_rgb_low.mp4` | 60 Hz, start-aligned | base | `/world/rig_00..07/cam_00/pinhole/video`, AV1 VideoStream |
| `videos/av1-720-new/<seq>/HMC_*_mono10bit_low.mp4` | 60 Hz, start-aligned | base | `/world/rig_08/cam_00..03/pinhole/video` |
| `camera_extrinsics_fixed/<seq>.json` | static | base | Eight camera transforms, mm → m; also calibration-session identity |
| `camera_extrinsics_ego/<seq>.json` | 60 Hz, real frame keys | base | Temporal `/world/rig_08`, four static camera offsets |
| `nimble_json_calib/<seq>.json` | static | base | Per-camera Pinhole K and CameraDistortion model/coefficients |
| `timestamp/<seq>.json` | rounded device seconds | base | Capture `timestamp_t0` and clock evidence; not used as noisy video timing |
| `landmarks3D/<seq>.json` | 60 Hz, real keys | hand_pose | `/world/gt/coco133_xyz`, Points3DWithConfidence, mm → m |
| `landmarks2D/<seq>.json` | 60 Hz, real keys | hand_pose | All 12 `<pinhole>/coco133_uv`, Points2DWithConfidence; exo ×2/3, ego ×1 |
| `hand_confidences/<seq>.json` | per hand per frame | hand_pose | Confidence of every present joint of that hand, including derived slots |
| `fine-grained-annotations/{train,validation,test}.csv` | 30 Hz segment boundaries | actions | `/task/actions/fine`, all active labels, union across views |
| `coarse-annotations/coarse_labels/{assembly,disassembly}_<seq>.txt` | 30 Hz segment boundaries | actions | `/task/actions/coarse`, labels retain their assembly/disassembly part |
| `manifests/sequences.csv` | sequence metadata | — | Verification of pose coverage; not a separate stream |
| Other manifests, README | metadata | — | Source inventory/provenance; not sensor data |
| Action vocabularies, `head_actions.txt`, `tail_actions.txt`, coarse splits and view lists | metadata | — | Not ingested: segment rows already contain labels; benchmark splits are not streams |
| `camera_position_fixed`, `camera_position_ego` | static / 60 Hz | — | Not ingested: exact duplicate of extrinsic translation |
| `hand_bboxes` | 60 Hz | — | Not ingested: derived from shipped 2D landmarks |
| `xf_transf` | 60 Hz | — | Not ingested: wrist frame derived from 3D; axes are undocumented |
| `assembly-hands/annotations/{train,val}/*` | separate benchmark | — | Not ingested: separate AssemblyHands annotation product, pending decision |
| Official `poses@60fps`, TSM/DINOv2 features | alternate representations | — | Not ingested: outside the agreed source scope |
| Official full-resolution recordings | 60 Hz | — | Not ingested: this port uses the pinned AV1 mirror |
| `AssemblyPoses.zip` | archive transport | — | Selected members only; duplicate loose pose copies are not additional streams |

## Layers and entities

**Base.** Exo rigs follow the fixed serial order C10095, C10115, C10118, C10119,
C10379, C10390, C10395, C10404. Ego cameras are sorted numerically. The smallest
headset serial defines the moving rig frame. Static `rig_T_cam` is
`inv(world_T_reference[0]) @ world_T_cam[0]`; only the reference pose is temporal.
The measured source facts establish rigidity. All translations are converted from
mm to m. World up is +Y, using `world_up.WORLD_UP_VIEW_COORDINATES`; Assembly101
has no ingested accelerometer for the gravity estimator. The source measurement
(headset above wrists in +Y) supports this choice; pixel validation remains required.

Exo lenses resolve only through matching `camera_extrinsics_fixed` sets (absolute
tolerance 1e-3 in source mm, zero relative tolerance). Each camera uses its own
nimble record. The 20 nimble files cover 11 sessions / 249 pose sequences; 88 pose
sequences have no exo intrinsics. They retain transforms and video, with no exo
Pinhole and `calibration_source="none"`. Ego lenses resolve by serial across all
nimble files. The 17 video-only sequences have no transforms or Pinholes.

K is scaled once from nimble source dimensions to the stored video dimensions.
Calibration goes through simplecv camera types and `log_camera_node`, with distortion
stored as `CameraDistortion` on the pinhole entity. Nimble coefficients enter
simplecv's Brown-Conrady and Kannala-Brandt models unchanged; the golden test uses
those projection functions. Rerun 0.38.1 does not apply the distortion in its Viewer.
Pinhole-only residuals measured
by the driver are roughly 0.6–4.9 px median exo and 4–105 px ego in source pixels.
The blueprint therefore shows shipped 2D in camera panes without overlaying
undistorted 3D projections. AV1 packets are remuxed, never transcoded, including
prefix clips. Native AV1 pixel decoding still needs driver evidence.

**Hand pose.** The shared `hands.coco133_from_hands` adapter keeps the Assembly-21
mapping. Source hand 0 is left and 1 is right. Body wrists 9/10 copy hand wrists
91/112; thumb bases 92/113 are wrist/CMC midpoints; source palm 20 is dropped.
Uncovered or missing slots have NaN coordinates and confidence 0.0. Present slots
keep the shipped per-hand confidence, including exact zero. No 2D is reprojected,
clipped to the image, or removed for being behind a camera. Numeric JSON keys
are sorted; their values, not positions in a dictionary, define the timeline.
Members are read once each, with only one large decoded hand member held at a
time; consumed frames are released and output is sent in 256-frame column batches.
The 2D JSON itself can be large, so peak memory still scales with one decoded member.

**Actions.** Fine rows are deduplicated across all views by start, end and label.
At every boundary, the text contains all active segments sorted by start, with
end treated as exclusive. An empty document clears the track when none remain.
Coarse labels include their assembly/disassembly part. No actions file is written
when both sets are absent. A prefix before the first segment gets an empty row.
There are no MANO parameters, hand meshes or object layers in this source scope.

The default `exoego_blueprint` contains eight exo panes, a four-camera ego column,
a 3D scene with orbital controls and a fine-action strip. The table preview decodes
one exo video, with calibration/hand/coarse/fine availability fields.

## Differences from simplecv

| Old behavior / audit defect | This port and raw-source evidence |
| --- | --- |
| First AssemblyHands capture K reused globally | Match fixed-extrinsic sessions, use each nimble camera record; never borrow between days |
| First nimble file / last ego record used for every camera | Resolve the constant lens by each headset serial |
| Exo K scaled twice | One source-to-stored scale, exactly 2/3; ego factor 1 |
| Lens model absent | Preserve all coefficients; golden tests use both measured projection models; Viewer limitation is explicit |
| JSON dictionary position used as frame number; failures swallowed | Sort numeric keys and log the actual indices; errors propagate |
| Fabricated or missing confidence | Shipped per-hand values; NaN/0 only for missing joints |
| 2D produced by reprojection | Read shipped `landmarks2D` for all 12 cameras |
| Labels trimmed to the shortest stream | Each video and pose track retains its own start-aligned length |
| Ego cameras treated separately | One moving rigid headset, four static camera offsets |
| Actions not preserved as overlapping intervals | Union views and display every active coarse/fine segment |
| Fixed video subdirectory | The pinned mirror layout is explicit beneath an overridable root; no speculative layout search |

The golden test checks median ≤0.1 stored pixel on all 12 cameras over every 97th frame of
9011-c03f, using its 9012-c07c calibration session. It excludes derived thumb-base midpoints because
perspective projection and midpoint construction do not commute.

## Parity with simplecv

Measured 2026-09-25 on pablo-dl-server. The reference is simplecv's own rrds
(`/mnt/nas/datasets/exoego-forge-catalog-rig/assembly101/all/<seq>.rrd`). A fresh run of
`tools/batch_raw_to_rrd.py assembly101` from simplecv `main` @ 34ee7f4c (prod env, local layout) gives the
same values. dataforge ran `dataforge-convert assembly101` @ bef78372 in its prod env. Rows match by
frame k = round(video_time · 60): simplecv's `video_time` comes from the mirror's CFR PTS, dataforge's from
`frame_index / 60`, and both are k/60. The full report, with every difference mapped to an audit defect,
is `/tmp/fleet-artifacts/exoego-migration/runs/assembly101-parity.md`.

| sequence | coco133_xyz rows sc / df | max joint distance | present conf sc / df | missing conf sc / df | exo cameras | ego cameras (world, per serial) | exo 2D sc reprojected vs df shipped | ego 2D, per serial |
|---|---|---|---|---|---|---|---|---|
| 9011-c03f (843 headset) | 13,818 / 13,820 | **0.0 m** | 1.0 / shipped | NaN / 0.0 | ≤ 1e-7 m, 0° | ≤ 8.1e-8 m, 4.2e-6° | median 1.96–6.40 px | 3e-5 px (84355350) … 1.48 px |
| 9013-a28 (211 headset) | 13,936 / 13,936 | **0.0 m** | 1.0 / shipped | NaN / 0.0 | ≤ 1e-7 m, 0° | ≤ 3.5e-7 m, 1.1e-5° | median 3.09–4.60 px | 6.1–17.2 px |

The 3D keypoints are bit-identical. The differences, each mapped in the report: simplecv's fabricated
confidence (1.0 present, NaN missing); its trim to the shortest video (2 rows on 9011-c03f); one exo K for
every sequence and no exo lens model; one ego lens record for all four cameras (it matches only serial
84355350, where the reprojection agrees with the shipped 2D to 3e-5 px); its reprojected 2D, which also
turns off-image points into NaN with confidence 1.0; no `frame_index`; no actions. One ordering
difference is not in the audit: on the 211 headset simplecv names the cameras through a fixed alias table
(e1 = 21176875 … e4 = 21179183) and uses e1 as the rig reference, while dataforge sorts serials and uses the
smallest. The `rig_08` track and camera slots differ by that permutation; every world-from-camera pose agrees.

## Timing

Read `<output_root>/timing/convert.jsonl` with `dataforge.timing.load_records(path, ConvertRecord)` and
`timing/register.jsonl` with `load_records(path, RegisterRecord)`. Conversion records `fetch`, `remux`,
`write:base`, `write:hand_pose` and `write:actions`; `remux` is nested in base. No transcode stage runs.
Capture length is the longest stream / 60.

**Baseline vs dataforge**, 2026-09-25, pablo-dl-server, prod envs, one process per sequence (dataforge: one
process for both), inputs on local NVMe. simplecv has no preprocessing step for Assembly101: its loaders
read the 720p mirror directly (the mirror was built outside the repo), so the baseline is its conversion
alone. simplecv time = its batch timer ("Total time taken"), dataforge time = `total_s`; neither includes
Pixi or interpreter start-up (about 2.7 s for simplecv).

| sequence | capture | simplecv | dataforge | simplecv s / capture-min | dataforge s / capture-min |
|---|---|---|---|---|---|
| 9011-c03f | 230.4 s | 6.30 s | 7.58 s | 1.64 | **1.97** |
| 9013-a28 | 232.3 s | 6.31 s | 6.25 s | 1.63 | **1.61** |

Soft gate: dataforge is 20 % slower on 9011-c03f and equal on 9013-a28. Most of the gap on 9011-c03f is
`fetch` (1.68 s against 0.27 s on 9013-a28): the first sequence of each process pays a one-time load, the
cause of which was not measured further. The rest of the cost is the shipped 2D: `write:hand_pose` (5.3 s of
the total) decodes the 283 MB `landmarks2D` member and writes 12 shipped 2D tracks, which simplecv never
opens. It reprojects the 3D instead, which is cheaper and wrong by 2–17 px.

**assembly101-sample** (the 8 sample sequences, one process, same host, local raw root; 96 s wall for all
eight including start-up):

| sequence | capture | fetch | remux | write:base | write:hand_pose | write:actions | total | MB base / hand_pose / KB actions | s / capture-min |
|---|---|---|---|---|---|---|---|---|---|
| 9012-a17 | 358.9 s | 2.02 | 0.67 | 1.03 | 8.64 | 0.02 | 11.71 | 964 / 123 / 20 | **1.96** |
| 9031-c12d | 453.2 s | 0.89 | 0.67 | 0.97 | 10.39 | 0.00 | 12.25 | 1118 / 150 / 18 | **1.62** |
| 9033-b04d | 570.5 s | 0.81 | 0.92 | 1.25 | 13.47 | 0.00 | 15.53 | 1233 / 193 / 18 | **1.63** |
| 9042-a02 | 318.2 s | 0.39 | 0.50 | 0.73 | 7.43 | 0.00 | 8.55 | 829 / 108 / 12 | **1.61** |
| 9045-b05d | 630.5 s | 1.14 | 0.89 | 1.32 | 14.63 | 0.00 | 17.09 | 1534 / 216 / 22 | **1.63** |
| 9053-c08b | 360.4 s | 0.66 | 0.53 | 0.77 | 8.37 | 0.00 | 9.80 | 914 / 122 / 16 | **1.63** |
| 9064-a20 | 260.1 s | 0.31 | 0.31 | 0.54 | 6.01 | 0.00 | 6.86 | 595 / 89 / 15 | **1.58** |
| 9085-c01c | 436.0 s | 0.78 | 0.61 | 0.90 | 10.18 | 0.00 | 11.86 | 1057 / 148 / 21 | **1.63** |

Registration time is added when `assembly101-sample` is registered.
