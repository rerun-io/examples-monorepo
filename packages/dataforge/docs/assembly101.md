# Assembly101

## Source

The video source is `pablovela5620/assembly101-720p` at
`001839131530cee9b2deb9ca66c025998d10cba4`. The local mirror README declares
CC BY-NC 4.0. Official actions come from `cvml-nus/assembly101` at `bfc15ea5`,
`annotations/` only. The separate AssemblyHands annotation product is not ingested.

The read-only `root` defaults to `/mnt/nas/datasets/assembly101`. It contains
`videos/av1-720-new/`, `assembly101_camera_and_hand_poses/`,
`assemblyhands-toolkit/calib/nimble_json_calib/`, and `manifests/`.
`DATAFORGE_RAW_ROOT` changes the parent of `assembly101`; `--root` specifies the
exact dataset root. `--annotations-root` is independent and defaults to
`/mnt/nas/datasets/assembly101/official/annotations`.

For this lane, use `--root /home/pablo/exoego-data/assembly101/raw` and
`DATAFORGE_OUTPUT_ROOT=/home/pablo/exoego-data/assembly101/iter`. Conversion
refuses destinations below either input root or `/mnt/nas`. Only the orchestrator
copies finished recordings to the NAS. `--frame-limit N` writes under
`preview-firstN`, keeping previews separate from full recordings.

`download()` verifies the local tree without network access or writes. The mirror
manifest, when present, distinguishes missing pose assets from video-only captures.
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
| `nimble_json_calib/<seq>.json` | static | base | Per-camera Pinhole K and lens coefficients as camera AnyValues |
| `timestamp/<seq>.json` | rounded device seconds | base | Capture `timestamp_t0` and clock evidence; not used as noisy video timing |
| `landmarks3D/<seq>.json` | 60 Hz, real keys | hand_pose | `/world/gt/coco133_xyz`, Points3DWithConfidence, mm → m |
| `landmarks2D/<seq>.json` | 60 Hz, real keys | hand_pose | All 12 `<pinhole>/coco133_uv`, Points2DWithConfidence; exo ×2/3, ego ×1 |
| `hand_confidences/<seq>.json` | per hand per frame | hand_pose | Confidence of every present joint of that hand, including derived slots |
| `fine-grained-annotations/{train,validation,test}.csv` | 30 Hz segment boundaries | actions | `/task/actions/fine`, all active labels, union across views |
| `coarse-annotations/coarse_labels/{assembly,disassembly}_<seq>.txt` | 30 Hz segment boundaries | actions | `/task/actions/coarse`, labels retain their assembly/disassembly part |
| `manifests/sequences.csv` | sequence metadata | — | Verification of pose coverage; not a separate stream |
| Other manifests, README, LICENSE | metadata | — | Source inventory/provenance; not sensor data |
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
Rerun 0.38.1 cannot apply these distortion models. The camera node retains the
model name, coefficients in `k1..k6,p1..p4` order and `distortion_applied=False`.
The projection module implements OpenCV Brown-Conrady and OVFishEye62 (KB6 with
p1/p2 swapped) for consumers and the golden test. Pinhole-only residuals measured
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

Driver parity captures are `9011-c03f` (2021-02-01 16:02:39) and `9013-a28`
(2021-02-02 13:49:23), at matching `k/60` times. The required 1e-4 m parity
comparison and Viewer screenshots are pending driver execution. The local unit
suite is not evidence of pixel parity. The golden test checks median ≤0.1 stored
pixel on all 12 cameras over every 97th frame of 9011-c03f, using its 9012-c07c
calibration session. It excludes derived thumb-base midpoints because perspective
projection and midpoint construction do not commute.

## Timing

Left for the driver. Read `timing/convert.jsonl` with
`dataforge.timing.load_records(path, ConvertRecord)` and `timing/register.jsonl`
with `load_records(path, RegisterRecord)`. Record host, commit, sequence IDs,
output bytes, capture seconds and seconds per capture-minute. Conversion records
`fetch`, `write:base`, `write:hand_pose`, `write:actions`, and nested `remux` stages;
no transcode stage runs. Compare prod conversion with simplecv preprocessing plus
conversion on the same two captures, separating skipped runs. Pixel evidence,
the benchmark report and the soft timing gate must precede full-corpus approval.
