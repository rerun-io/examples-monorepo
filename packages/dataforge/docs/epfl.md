# EPFL-Smart-Kitchen-30

## Source

- Canonical release: https://zenodo.org/records/15535461 (videos) and https://zenodo.org/records/15551913 (poses and annotations).
- Mirror: https://huggingface.co/datasets/pablovela5620/epfl-smart-kitchen-av1 at `b91f2df8027e4d7e529939e78f856463f64a58cd`, CC BY-NC 4.0. This is an AV1 NVENC CQ 38 re-encode with the source frame counts preserved.
- `root` contains `Public_release_pose`; its default is `paths.raw_root() / "epfl-smart-kitchen"`. For this lane use `/mnt/nas/datasets/epfl-smart-kitchen`, read-only. `video_root` defaults to `root`; use `/home/pablo/exoego-data/epfl/raw` for the ten staged sessions.
- Local output: `DATAFORGE_OUTPUT_ROOT=/home/pablo/exoego-data/epfl/iter`. Conversion checks every RRD destination and the resolved `<output_root>/work` scratch directory, refusing `/mnt/nas` and either raw root, including symlink targets. No download or extraction occurs. `download()` verifies availability. Discovery reports incomplete sessions and missing staged video/meta files.
- One catalog dataset, `epfl`; sample registration is the driver's `epfl-sample`. Identity: `epfl__<split>__<subject>__<session>`. The two byte-identical train/test copies remain distinct recordings.
- `property:capture:source_revision`, `source_resolution`, and `video_source` identify the mirror and its provenance. Exo resolution is 1280×720; HoloLens is 896×504. Camera order is `output0`, `Aoutput0`–`Aoutput3`, `Boutput0`–`Boutput3`, then `hololens`; rigs 00–09 each have camera 00.
- All ten videos run at 30 Hz. `video_time` preserves `timestamps.txt` microseconds converted to nanoseconds, without rebasing. The clock is the Azure Kinect device clock, not Unix time. `frame_index` is the zero-based release row. HoloLens is already resampled by the release to this row census. MP4 PTS are replaced with the shipped timestamps, preserving the occasional one-tick gap. `property:capture:clock_source` records this rule.

## Raw inventory

Paths below are relative to the session in their release tree.

| Shipped file or stream | Rate / clock | Layer | Entity or not ingested: reason |
| --- | --- | --- | --- |
| `Public_release_videos/…/videos/{output0,Aoutput0..3,Boutput0..3}.mp4` | 30 Hz / device clock | base | `/world/rig_00..08/cam_00/pinhole/video`, AV1 samples remuxed without encoding |
| `videos/hololens.mp4` | release-resampled 30 Hz / same device clock | base | `/world/rig_09/cam_00/pinhole/video` |
| `meta_data/camera_matrix.json`: `K`, `dist`, `word2cam` | static, per session | base | Camera pinholes, transforms and full 8-coefficient distortion metadata; all nine exo cameras retained |
| Same JSON: `world2depth`, `depth_K`, `depth_dist` | static | — | not ingested: depth is outside this port |
| `meta_data/timestamps.txt` | one microsecond timestamp per RGB row | all | `video_time` timeline; unshifted |
| `meta_data/holo_data_wpose.csv`: `world2holo` | one row per RGB frame | base | `/world/rig_09` world transform, inverse of shipped `cam_T_world`; `[]` becomes a NaN transform |
| Same CSV: `eyes` | release row clock | — | not ingested: gaze deferred |
| Same CSV: `hololefts`, `holorights` | release row clock | — | not ingested: raw 26-joint HoloLens tracking deferred by facts §16 |
| `Public_release_pose/…/pose_3d/pose3d_mano.csv`: `kp3ds`, `kp3ds_conf` | release row clock | hand_pose | `/world/gt/coco133_xyz`, slots 91–132, shipped joints and confidence |
| Same CSV: `left/right_poses`, `Rh`, `Th`, `shapes` | release row clock | hand_pose | `/world/gt/hands/<side>/mano`, temporal `poses`, `Rh`, `Th`, `shapes` |
| Same CSV: `l2_dist_left/right` | release row clock | hand_pose | `/world/gt/hands/<side>/mano/l2_dist`, shipped Scalars, empty cells are NaN |
| Both pose CSVs: `rgb_frameid` | release row clock | hand_pose | `/world/gt/rgb_frameid`, checked for equality and consecutiveness |
| `pose_3d/pose3d_smpl.csv`: `kp3ds`, `kp3ds_conf` | release row clock | hand_pose | `/world/gt/coco133_xyz`, body slots 0–16 |
| Same CSV: `poses`, `Rh`, `Th`, `shapes`, `l2_dist` | release row clock | body_pose | `/world/gt/body/smpl` parameters and `/world/gt/body/smpl/l2_dist` Scalars |
| `annotations/actions_annotations.xlsx`: Start, End, Verbs, Nouns, Confusion | seconds from first frame | actions | `/task/actions/fine` TextDocuments; labels retain confusion flag and empty fields |
| `annotations/activity_annotations.json`: annotations | seconds from first frame | actions | `/task/actions/coarse` TextDocuments; distinct activities also in episode property |
| Same JSON: `datetime`, `video_file` | metadata | — | not ingested: annotation edit time and redundant source basename are not capture clocks |
| Optional `videos_depth/*.mp4`, `timestamps_depth/*.txt` | source depth clock | — | not ingested: depth deferred; not staged locally |
| Optional `IMUs/*.csv` | source IMU clock | — | not ingested: EPFL IMU is outside this lane's agreed scope; not staged locally |
| Audio, if present in video containers | container clock | — | not ingested: audio deferred |
| Mirror `README.md`, `LICENSE`, `manifests/{sequences.csv,encode_report.csv,encode_report.json,upload_payload.txt}` | corpus metadata | — | not ingested as streams; revision, resolution and re-encode provenance recorded in capture properties |

## Layers and entities

Seven files per recording: `base`, `hand_pose`, `body_pose`, `hand_mesh`, `body_mesh`, `projections`, `actions`. Only base sends the default blueprint and RecordingInfo. Base and actions publish in separate atomic recordings before the pose layers open; a later pose failure preserves those completed files. Every file shares the recording identity. Existing files are skipped unless forced. Prefix conversions use `preview-first<N>` and cannot replace full recordings.

`hand_pose` is the sole owner of the combined `Points3DWithConfidence` entity. Body and hands are separate fits joined on their shipped row key, following upstream's merge. Uncovered slots 17–90 are NaN/0. No derived wrist or thumb slots are introduced. Shipped confidence is retained, including values above one and zero-confidence positions. NaN confidence becomes zero. Body residual ≥0.09 or hand residual ≥0.06, and empty/nonfinite residuals, reject that fit: its positions become NaN and confidence zero. Residuals remain independent raw Scalars; they are never used as confidence magnitudes.

MANO meshes use simplecv `MANOLayerNP(use_pca=False)` with the shipped 45 finger angles, root replaced by `Rh`, and translation `Th + (R(Rh)-I) @ root_joint_template`. SMPL meshes use simplecv `SmplxLayerTorch(model_type="smpl", gender="neutral")` with the same origin-to-root pivot correction. Neutral gender follows facts §16 and EasyMocap's default. Shapes remain temporal data; FK rebuilds the model when shapes change. Meshes live at `/world/gt/hands/<side>/mesh` and `/world/gt/body/mesh`; rejected rows have empty vertex arrays. Both large pose CSVs are streamed together once per pending layer set, in batches of 64; no full-session mesh or pose table is materialized.

Each parameter entity has one static `AnyValues` with `use_pca=False`, `root="Rh"`, `translation_pivot="origin"`, and `source="pose3d_mano.csv"` or `"pose3d_smpl.csv"`. SMPL also records `gender="neutral (assumed)"`.

The SMPL root defaults to `paths.raw_root() / "body_models"`; pass `--smpl-model-root` to use another location. It must contain the official neutral SMPL model. Missing models fail clearly; no substitute or download is attempted. Mesh computation currently uses CPU simplecv layers.

**`body_mesh` is 10 Hz** (every third frame, `BODY_MESH_STRIDE`): a display layer, by decision (2026-09-25). Full-rate SMPL vertices cost 82 KB per frame, four times the session's videos, and Rerun 0.38 has no mesh skinning to pose one logged mesh from joint transforms. `body_pose` (SMPL parameters) and the keypoints stay at 30 Hz; `hand_mesh` stays at full rate.

`projections` applies OpenCV's complete rational Brown–Conrady model to all nine exo cameras. It writes only `<pinhole>/coco133_uv_projected`, with derived-source properties. Missing, behind-camera and offscreen points are NaN/0; visible points keep the 3D confidence. These cameras qualify for the agreed fisheye exception because Rerun's Pinhole cannot express their distortion. No shipped `coco133_uv` is invented. Exo panes include video plus derived pixels and exclude 3D geometry. HoloLens is a zero-distortion pinhole and includes world geometry. All nine exo panes are accessible through tabs; the table card decodes one stream and exposes subject, split, activity, frames.

Fine and coarse actions use half-open intervals. Seconds map to `round(s*30)`, then the source timestamp at that frame. Boundaries at or beyond the selected frame count are omitted; an interval ending at frame N remains active through frame N−1. Documents list every active segment; empty documents clear ended segments. Preview truncation does not pull future action boundaries into the prefix.

## Differences from simplecv

1. Empty residuals reject fits, as upstream `convert2d2a.py` specifies; simplecv treats them as accepted (audit defect 1).
2. Meshes follow the same rejection gate, clearing rejected frames instead of drawing opaque diverged fits.
3. Every stream uses unshifted device timestamps. This fixes the PTS/rebased-label clock split and preserves skipped device ticks.
4. `rgb_frameid` is checked and logged rather than ignored.
5. Every row's betas are logged and used; they are not collapsed to frame zero.
6. HoloLens missing transforms become NaN instead of carry-forward. Source drift remains visible and is recorded in provenance.
7. All nine exo panes are available, rather than a maximum of eight.
8. Accepted joints with shipped confidence zero retain their shipped position, as facts §8 requires; simplecv masks these positions.

Asset tests accept `DATAFORGE_EPFL_POSE_ROOT`, `DATAFORGE_EPFL_VIDEO_ROOT`, and `DATAFORGE_EPFL_PARITY_ROOT`; their defaults are `/mnt/nas/datasets/epfl-smart-kitchen`, `/home/pablo/exoego-data/epfl/raw`, and `/home/pablo/exoego-data/epfl/simplecv_root`. `DATAFORGE_SMPL_MODEL_ROOT` sets the test model root, defaulting to `/home/pablo/0Dev/work/rerun-projects/examples-monorepo/packages/lamp/data/body_models`. Tests skip when required assets are absent. These test defaults do not set library paths.

The XLSX reader supports only the release's single-sheet layout with inline strings and numeric cells. It rejects shared strings, formulas, other cell types, and additional worksheets.

Golden parity reads the simplecv reference RRD for `train/YH2007/2023_10_30_10_05_27` and streams both source pose CSVs through dataforge's parser, without importing simplecv loaders. Set `DATAFORGE_EPFL_REFERENCE_ROOT` to override `/mnt/nas/datasets/exoego-forge-catalog-rig`; absent or unreadable reference assets skip with their path. All **67,890** temporal rows match exactly after adding the first device timestamp to reference `video_time`; the extra reference row is static keypoint metadata. Across **3,962,854** common finite joint slots, maximum position and confidence errors are both **0** (limits: 1e-4 m and 1e-6). There are **2,592** dataforge-only finite slots, all explained by shipped zero confidence, **0** reference-only finite slots due to empty residuals, and **0** unexplained differences. The test checks each one-sided finite slot against the raw residual or confidence cell, consumes every CSV row, and retains only the current Arrow chunk and decoded CSV pair.

Both EPFL golden tests passed on 2026-09-25. The unchanged MANO FK test checks the first 60 source rows against shipped joints with ≤1 cm maximum and ≤3 mm median error. These are numerical gates; Viewer pixel evidence remains a separate driver check. Inspect HoloLens overlay drift as a source limitation, and confirm nonblank changing AV1 frames at two or more times.

Parity of the written rrds against the simplecv reference on both iteration sessions (2026-09-25, `train/YH2007/2023_10_30_10_05_27` and `train/YH2007/2023_10_18_15_25_04`): keypoints 67,890 / 68,610 rows, 3,962,854 / 3,911,913 common finite joints with 0 m position and 0 confidence error; 2,592 / 13,906 dataforge-only finite joints, all shipped confidence 0 (difference 8); no reference-only joints, because every empty-residual row also ships confidence 0 (so difference 1 does not change keypoints, only meshes). The nine exo cameras' `world_from_cam`, K and resolution are identical. HoloLens per-frame poses agree within 3.0e-5 (float32 quaternion vs float32 matrix) on the 63,833 / 64,929 finite rows once matched by frame index: the reference logs them on video PTS `i/30 s` (difference 3). The 4,057 / 3,681 rows with an empty `world2holo` are NaN here and carried forward in the reference (difference 6). No difference is unexplained.

## Timing

Iteration sessions, prod environments, pablo-dl-server, 2026-09-25, both converters run under the GPU lock on the same local inputs (pose CSVs copied to local disk, the local AV1 mirror). Times are each converter's own in-process total (`timing/convert.jsonl` for dataforge, simplecv's "Total time taken"); with process start-up the walls are 50.9/55.6 s (dataforge) and 34.3/33.5 s (simplecv).

| session | capture | dataforge total | dataforge s / capture-min | dataforge without body_mesh | simplecv total | simplecv s / capture-min |
|---|---|---|---|---|---|---|
| train/YH2007/2023_10_30_10_05_27 | 37.72 min | 48.4 s | 1.28 | 36.3 s (0.96) | 30.6 s | 0.81 |
| train/YH2007/2023_10_18_15_25_04 | 38.12 min | 53.1 s | 1.39 | 38.9 s (1.02) | 30.4 s | 0.80 |

dataforge stages (s, first / second session): `fetch:poses` 12.6 / 14.3, `write:body_mesh` 12.1 / 14.2, `write:projections` 11.4 / 12.2, `write:hand_mesh` 7.0 / 7.6, `write:base` 2.4 / 2.5 (`write:video` 0.5), `write:hand_pose` 1.3 / 1.5, `write:body_pose` 0.2 / 0.3. Output 8.5 GB per session: body_mesh 5.58, hand_mesh 1.26, base 1.07–1.12, projections 0.38, hand_pose 0.12, body_pose 0.03 GB. simplecv writes one 2.95–2.99 GB rrd per session (videos, keypoints, pinhole 2D, MANO meshes; no body mesh).

Sample conversion (8 sessions, prod env, poses read sequentially from the NAS, 2026-09-25). Only `test/YH2003` has `body_mesh`: at full rate the SMPL mesh costs 82 KB per frame (≈ 46 GB for the eight sessions), beyond the lane's 40 GB of local disk, so the other seven carry the six other layers until Pablo decides the mesh rate or storage.

| session | capture | total | s / capture-min | fetch:poses | write:projections | write:hand_mesh | write:body_mesh | output |
|---|---|---|---|---|---|---|---|---|
| test/YH2003/2023_06_02_09_20_42 | 28.95 min | 35.6 s | 1.23 | 10.0 s | 8.4 s | 5.2 s | 8.7 s | 6.67 GB |
| test/YH2004/2023_07_05_12_30_03 | 50.92 min | 44.6 s | 0.88 | 17.4 s | 13.5 s | 8.3 s | – | 3.87 GB |
| test/YH2011/2023_07_10_15_03_05 | 57.87 min | 52.0 s | 0.90 | 20.0 s | 15.6 s | 9.5 s | – | 4.32 GB |
| test/YH2029/2023_06_12_15_05_02 | 36.65 min | 33.0 s | 0.90 | 13.1 s | 9.7 s | 6.0 s | – | 2.88 GB |
| train/YH2018/2023_07_26_09_10_53 | 33.63 min | 29.8 s | 0.89 | 11.2 s | 9.1 s | 5.7 s | – | 2.63 GB |
| train/YH2021/2023_09_26_09_46_20 | 29.60 min | 26.0 s | 0.88 | 9.8 s | 7.9 s | 4.8 s | – | 2.38 GB |
| train/YH2025/2023_09_22_09_04_54 | 31.38 min | 28.8 s | 0.92 | 11.5 s | 8.6 s | 5.1 s | – | 2.48 GB |
| train/YH2040/2023_08_11_09_01_12 | 41.80 min | 36.9 s | 0.88 | 15.0 s | 11.0 s | 6.7 s | – | 3.27 GB |

The simplecv baseline is its conversion only (`simplecv.apis.batch_raw_to_rrd`, prod env `simplecv`). simplecv's preprocessing (`preprocess_epfl_smart_kitchen`) is the H.264 → AV1 transcode that produced the HF mirror; that mirror is also dataforge's declared source, so the step is shared by both paths and was not re-run (it would need the 192.75 GB `Public_release_videos.zip` from Zenodo 15535461). simplecv skipped its SMPL body mesh because its checkout has no SMPL model file, as in the reference rrds.

**Soft speed gate: not met.** dataforge is 1.6–1.7× slower per capture-minute than simplecv, and still 1.2–1.3× slower without the body mesh that simplecv does not write. Causes, by stage: `fetch:poses` parses every CSV row with `json.loads` and a pyserde `from_dict` per fit (12.6–14.3 s for 650 MB of CSV); `write:projections` runs the full 8-coefficient rational model for nine cameras over every joint and logs nine Points2D tracks (11–12 s); the SMPL mesh (12–14 s, 5.6 GB) is new work. Levers: parse the pose CSV columns in bulk (pose rows are a stream, not records), and time the projection stage's split between `cv2.projectPoints` and the Arrow logging before optimising it.

The canonical convert command writes `<output_root>/timing/convert.jsonl`, using `dataforge.timing`. Stages include metadata `fetch`, `fetch:poses`, preview `remux`, `write:video`, and each `write:<layer>`. No transcode stage runs for the staged AV1 videos. Output bytes, capture duration, host, version and failures use the shared conversion report.

Read with `dataforge.timing.load_records(path, ConvertRecord)`; registration uses `timing/register.jsonl` and `RegisterRecord`. Report elapsed seconds per capture-minute and separate skipped runs. Stages can overlap. Compare against simplecv preprocessing plus conversion in the production environment on the same two sessions; record the soft gate and benchmark artifact before any full-corpus run.
