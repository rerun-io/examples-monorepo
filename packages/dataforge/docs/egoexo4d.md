# Ego-Exo4D with Ego-Exo4D-HM

## Source

- Takes: Ego-Exo4D v2 (https://ego-exo4d-data.org), the public S3 release `s3://ego4d-consortium-sharing/egoexo-public/v2`
  that the official `egoexo` CLI reads. Access is gated: sign the Ego-Exo4D agreement (https://ego4ddataset.com); about two days
  later AWS keys arrive by email. **The keys expire 14 days after issue**; requesting access again renews them.
- Fits: Ego-Exo4D-HM (https://abhiram824.github.io/egoexo4d_human_meshes/, arXiv 2609.30187), the Hugging Face dataset
  `Ego-Exo4D-HM/npz-datasets` at `6e1cd862`, not gated. 2,649 takes, one npz per take, SMPL-H fits of the camera wearer
  made with SLAHMR from four exo GoPros (ViTPose body + HaMeR hand keypoints, triangulated). Only these takes are converted.
- Body models: `pablovela5620/mamma-streaming-data` at `03a1f4b5` (private): `body_models/smplh/SMPLH_MALE.npz` (the AMASS
  "extended SMPL+H" male model, 16 betas, the file SLAHMR ships) and `body_models/smplx/SMPLX_NEUTRAL.npz` (only its MANO hand
  PCA basis is read).
- Catalog `dataforge-egoexo4d`, sample `dataforge-egoexo4d-sample`. Identity: `egoexo4d__<take_name>`.
- Rig 0 is the wearer's Aria; rigs 1..N are the localized GoPros in `gopro_calibs.csv` order (HM's view order).
- `property:episode:{take_uid,activity,task,university,capture}` from `takes.json`; `property:capture:clock_source`,
  `clock_filled_frames` (frames whose timesync stamp was missing), `source_resolution`, `image_rotation_cw_deg` (90, the Aria
  MP4s) and `trajectory_coverage` (share of frames with a pose).
- Camera-source metadata: GoPros `source_width/height` 3840×2160 (2160×3840 portrait), `stored_width/height` 1920×1080;
  Aria streams carry `stream_id` (`214-1`, `1201-1`, `1201-2`, `211-1`). Everything is re-encoded (`cq` 36, `gop` 60).

## Get the raw data

Store the keys as AWS profile `egoexo4d` with `aws configure --profile egoexo4d` (machine-local `~/.aws/credentials`; or pass
`--aws-profile`), then:

```bash
pixi run -e dataforge dataforge-download egoexo4d --sequences cmu_bike02_4   # models, takes.json/captures.json, one fit
pixi run -e dataforge dataforge-download egoexo4d                            # every fit (~48.5 GB)
pixi run -e dataforge dataforge-download --list-remote egoexo4d              # JSON lines: take, bytes, take files
pixi run -e dataforge dataforge-convert egoexo4d --sequences cmu_bike02_4
```

- `download` fetches the shared files only: the two body models (pinned, sha256-verified), the release `metadata` part
  (`takes.json`, `captures.json`) and the selected fits into `<root>/hm/<take>/`.
- `convert` fetches each take's files from the release manifests, exactly what base reads: the frame-aligned videos of every
  GoPro and of the Aria's four streams (`takes`), `closed_loop_trajectory.csv` and `gopro_calibs.csv` (`take_trajectory`),
  the image-less VRS for the Aria calibration (`take_vrs_noimagestream`, path from its manifest), and the capture's `timesync.csv` (`captures`).
  `prefetch` fetches the next take while one converts. A file lands under `<root>/.dataforge-staging` and is renamed into place
  only at the manifest's size.
- Right after base is written the take files are deleted (`--keep-raw` keeps them). The fits, models, metadata and base's
  sidecars stay, so the derived layers rebuild without the release; `timesync.csv` stays because other takes of the capture share it.
- Default raw root `$DATAFORGE_RAW_ROOT/egoexo4d`; `--model-root` moves the body models.

## Raw inventory

| Shipped file or stream | Rate / clock | Layer | Entity or not ingested: reason |
| --- | --- | --- | --- |
| `takes/<t>/frame_aligned_videos/camNN.mp4` (`gpNN` at UPenn; the capture's non-ego cameras, quality-1 GoPros) | 30 Hz frame-aligned | base | `/world/rig_NN/cam_00/pinhole/video`, 1080p AV1 |
| same, GoPros with `quality != 1` | — | — | not ingested: Ego-Exo4D could not localize them (no pose) |
| `aria01_214-1.mp4` (RGB) | 30 Hz | base | `/world/rig_00/cam_00/pinhole/video` |
| `aria01_1201-1.mp4`, `aria01_1201-2.mp4` (SLAM) | 30 Hz | base | `/world/rig_00/cam_01`, `cam_02`; about one take in seven (IIITH, NUS, Uniandes, UPenn) ships them 640x480, the upright image resized to the readout's shape: base stores them 480x640 again (`stored_width/height`) |
| `aria01_211-1.mp4` (both eye cameras in one frame) | 10 Hz images in a 30 Hz video: two frames of three are black padding | base | `/world/rig_00/cam_03/pinhole/video`, the real images only, at their own times (a take whose padding breaks the one-in-three pattern is refused); video only: no single calibration describes the paired image |
| `trajectory/gopro_calibs.csv` | static | base | GoPro `world_T_cam` and KB4 lens |
| `trajectory/closed_loop_trajectory.csv` | ~1 kHz device clock | base | `/world/rig_00` `world_T_device` at each frame, interpolated inside ≤ 2 ms brackets by the shared MPS reader (`aria.read_trajectory`), else NaN |
| other `trajectory/` files, semidense points, eye gaze, audio, full VRS, annotations | — | — | not ingested: outside this port (the ego_pose GT layer was declined) |
| `<aria>_noimagestreams.vrs` tag `calib_json` | static | base | Aria camera calibrations (FISHEYE624), quarter-turned to the MP4 orientation |
| `captures/<c>/timesync.csv` `<aria>_214-1_capture_timestamp_ns` | per frame | all | `video_time` |
| HM `trans`, `root_orient`, `pose_body`, `hand_pose`, `betas_per_frame` | 30 Hz | body_pose | `/world/gt/body/smplh`, raw, every frame |
| HM `valid` | 30 Hz | body_pose | `/world/gt/body/smplh/valid` |
| HM `joints3d` (OpenPose 25 + 2×21 hands) | 30 Hz | body_pose | `/world/gt/coco133_xyz` |
| HM `chunk_ranges` | — | — | read: shape changes at chunk borders, checked to tile the take |
| HM `latent_pose` (VPoser) | — | — | not ingested: `pose_body` is its decoded form |
| HM `cam_R`, `cam_t`, `intrins`, `cam_dist`, `joints2d` | — | — | not ingested: SLAHMR's undistorted pinhole views (balance 0.8), not the shipped videos |

## Clocks

`video_time` is the Aria RGB capture time on the Aria device clock (ns), from `timesync.csv` rows
`timesync_start_idx .. timesync_end_idx - 1`; a missing stamp repeats the last one (counted in `clock_filled_frames`). Frame `i` of every frame-aligned video and of
the HM fit is row `timesync_start_idx + i`; `frame_index` is `i`. The trajectory's `tracking_timestamp_us` is the same device clock.

## Layers and entities

- **base**: the cameras and videos above, the Aria pose, capture and episode properties (`source_num_frames`: every source video must
  hold exactly the take's frame count, or base refuses before anything is published or deleted). Right after base is published,
  one sidecar `<output_root>/sidecars/<recording_id>/take.npz` (`times_ns`, `world_T_device` per frame, and the camera record as
  JSON: GoPro rows, the Aria `calib_json`, stored Aria stream sizes). Staleness goes by file age, so a plain retry finishes an
  interrupted rebuild: a base without a newer sidecar redoes the take (re-fetching it), a derived layer older than base is
  rebuilt, and a new base rebuilds every derived layer. A copy of the output tree must keep mtimes (`rsync -a`, `tar`).
- **body_pose**: raw SMPL-H parameters (`hand_pose` = 45 PCA coefficients per hand in SMPL-X's MANO basis, mean added), `valid`,
  and `coco133_xyz`: body 0–16 and feet 17–22 from BODY_25 (neck and mid-hip have no slot), hands 91–132, face empty. No shipped
  confidence, so 1.0; frames with `valid == 0` are NaN with confidence 0.
- **body_mesh**: SMPL-H male mesh every third frame (10 Hz display layer) plus every frame where `valid` changes (so an invalid
  stretch between stride frames still clears the mesh), from SLAHMR's model (shape basis padded to 300 columns so
  `smplx` keeps 16 betas). Its regressed joints match the shipped `joints3d` to 1e-6 m (`test_smplh_reproduces_shipped_joints`).
- **projections**: `coco133_xyz` through each GoPro's KB4 lens (OpenCV fisheye; checked against `cv2.fisheye.projectPoints`) and
  each calibrated Aria camera's FISHEYE624, at `<pinhole>/coco133_uv_projected`. No measured 2D.
- Fisheye rule: every camera pane shows the video and the projections only; the mesh is in the 3D view.
- Default blueprint: the scene, the four Aria panes in a column, GoPros 1–5 along the bottom. Table card: the scene without video
  beside GoPro 1. Both 3D views are rooted at `/world` (gravity-aligned, z up) with an orbital eye the viewer fits to the scene, so
  a kitchen and a soccer pitch both frame. GoPro frustums are 5 % of the layout's radius (at least 0.1 m), so they keep one
  on-screen size.

## Differences from simplecv

None: simplecv has no Ego-Exo4D adapter.

## Parity with simplecv

No simplecv reference. Parity is held against the release itself: the SMPL-H joints test above.

## Timing

Not measured on real takes yet. Synthetic 109-frame take (4 GoPros 4K, 4 Aria streams), 5090: 2.2 s for all four layers.

## Known gaps

Checked only against public format docs and synthetic takes until the first real take; verify on it:

- the timesync end bound (the sources disagree; the HM convention is used) and that every video has the take's frame count;
- that the Aria MP4s, SLAM and eye streams included, are quarter-turned from the sensor and the RGB MP4 is 1408×1408;
- that every MP4's frame count equals the timesync rows (base refuses otherwise) and the HM fit's (a fit mismatch is printed and
  the shorter one is converted).

Source faults kept as shipped: `unc_soccer_09-21-23_01_7`'s GoPro 1 is 7 m below the pitch, looking up (its pose in the HM fit is
the same).
