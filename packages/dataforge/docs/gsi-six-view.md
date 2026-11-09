# GSI Egocentric Six-View

## Source

- GSI's partner edition "Egocentric Six-View | 10h": edition `01a0f166-7994-7b43-a622-a325541c4356`, release
  `2026-09-v2`, 1,685 episodes, 10.3 h, 174 GB. Private: GSI shares it with a partner sign-in (username
  `gsi-ots-10h`, password handed over separately). dataforge reads both from `GSI_PARTNER_USERNAME` and
  `GSI_PARTNER_PASSWORD`, which live in `~/.secrets.env` and nowhere else.
- Catalog dataset `gsi-six-view`; one segment per episode, recording id `gsi-six-view__<episode_id>`.
- GSI ships each episode as Rerun recordings already (`gsi.ots.customer-rrd.v2`, written by rerun-sdk 0.37.1, application
  `gsi_off_the_shelf`, recording id = episode id). `gsi-six-view` repacks them into the exoego:v2 layout without
  decoding a video: lenses move GSI's cameras, pinholes, H.265 samples and SLAM pose; the hands are read column-wise,
  moved to the world frame and written through the shared hand helpers.
- `property:capture:*` on base: `category` (GSI's caption task family, from the manifest), `num_cameras` (6),
  `num_frames` (GSI's `n_frames`), `camera_names` (GSI's view of each `cam_NN`), `source_resolution`, `gsi_edition`,
  `gsi_source_release`, `gsi_rrd_contract`, `gsi_privacy_status` (`PRIVATE_INTERIM_EXISTING_BLUR`: faces are blurred)
  and the shipped files' `hands_rrd_sha256` / `slam_rrd_sha256`. GSI's own `property:episode:*` (caption, duration,
  device, source frame range, `start_ns`) is kept as shipped, and so is its `RecordingInfo` start time.
- `property:gsi_calibration:*` on base: the `/meta/calibration` fields no camera entity states: the eye and front
  stereo rectification (`*_P1`, `*_P2`, `*_R1`, `*_R2`, `*_Q`, `*_T_1_0`, `*_baseline`, `Q_frame`), the IMU
  extrinsics and time shifts (`*_T_cam_imu`, `*_timeshift_cam_imu`, `side_extrinsic_source`), `body_frame` and
  `frame_version`. `K`, `resolution`, `views`, `flu_from_opt`, `P_view_from_left_front`, `T_view_from_left_front` and
  `T_view_optical_from_left_front` are left out: the pinholes, `rig_T_cam` and `camera_names` carry them.

## Get the raw data

```bash
source ~/.secrets.env
export DATAFORGE_OUTPUT_ROOT=/mnt/nas/ssd-8tb/datasets/gsi-six-view
R=/mnt/nas/ssd-8tb/downloads/gsi/egocentric-six-view-10h
pixi run -e dataforge --frozen dataforge-download --list-remote gsi-six-view --root $R   # every episode, from the manifest
pixi run -e dataforge --frozen dataforge-download gsi-six-view --root $R --sequences <episode_id> ...
pixi run -e dataforge --frozen dataforge-download gsi-six-view --root $R                  # the whole edition
pixi run -e dataforge --frozen dataforge-convert gsi-six-view --root $R                   # --keep-download while the native mirror also converts
pixi run -e dataforge --frozen dataforge-register --catalog-url <url> gsi-six-view --root $R
```

GSI's API takes three calls, read from GSI's CLI (`gsi` 0.2.2), which dataforge does not install: partner sign-in for
a 24 h token, the signed manifest link, and pages of 25 episodes whose files carry a sha256 and a signed Cloudflare R2
link that lives an hour. The API has no per-episode query, so `--sequences` pages through the listing until every
selected episode has gone by. Every file is verified against the listing's size and sha256 before it takes its name;
a file already on disk and matching is skipped, and an episode whose layers are both published is not fetched again.
A failing link is fetched once more from a freshly signed page. R2 links answer only GET (HEAD is 403), so this
dataset streams them itself instead of using `transports.http_fetch`. Measured from pablo-dl-server: 19.8 MB/s, so
the whole edition takes about 2.4 h.

`convert` deletes an episode's two files once all of its layers are published, unless `--keep-download`; `manifest.json` stays. Point
`--root` at the NAS SSD tier: the download peaks at the whole edition's 174 GB.

## Raw inventory

| Shipped file or stream | Rate / clock | Layer | Entity or not ingested: reason |
| --- | --- | --- | --- |
| `hands.rrd` `/camera/<view>/image` `VideoStream` | H.265 (`hev1`), 30 fps | base | `/world/rig_00/cam_NN/pinhole/video`, samples unchanged |
| `hands.rrd` `/camera/<view>/image` `Pinhole` | static | base | `/world/rig_00/cam_NN/pinhole`: K, resolution and image-plane distance as shipped, `camera_xyz` FLU → RDF |
| `hands.rrd` `/camera/<view>` `Transform3D` | static | base | `/world/rig_00/cam_NN`: `rig_T_cam` between RDF frames (see Frames) |
| `hands.rrd` `/hand/<side>/keypoints3d` | per frame | pose | `/world/gt/coco133_xyz` slots 91-132, world frame; `projections`: `cam_NN/pinhole/coco133_uv_projected` |
| `hands.rrd` `/hand/<side>/vertices` | per frame | hand_mesh | `/world/gt/hands/<side>/mesh`, world frame, shared hand albedo; GSI's per-row normals and triangle copies dropped |
| `hands.rrd` `/hand/<side>/mano` | per frame | pose | `/world/gt/hands/<side>/mano` as shipped (rig frame); `is_usable` → `confidence`; wrist pose → `wrist` |
| `hands.rrd` `/meta/calibration` | static | base | `property:gsi_calibration:*`, minus what the cameras state |
| `hands.rrd` `/` `AnnotationContext` | static | — | replaced by the shared COCO-133 context |
| `hands.rrd` `/`, `/__properties` | static | base | `ViewCoordinates` FLU (GSI's world), episode properties and start time as shipped |
| `slam.rrd` `/slam/rig` | per frame | base | `/world/rig_00` `world_T_rig` (the rig frame turned to RDF); its two `capture_time`-only rows are dropped |
| `slam.rrd` `/slam/path`, `/slam/index`, `/slam/meta` | static | base | `/world/runs/slam/{trajectory,index,meta}` |
| `slam.rrd` `/camera`, `/hand` | per frame | — | not ingested: the same pose as `/slam/rig` (convert checks it for `/hand`) |
| `slam.rrd` `/slam/trail` | per frame | — | not ingested: a display trail; the follow view shows the path |

## Clocks

- `video_time` = GSI's `capture_time` minus the episode's `start_ns` (its first frame), so 0 at frame 0.
- `capture_time` (GSI's device duration, ns) stays on every base row; `frame` becomes `frame_index`.
- `source_frame` is dropped: it is `frame_index` plus `property:episode:source_frame_start`.
- The derived layers carry `video_time` and `frame_index`.

## Frames

- GSI's rig frame is left_front's FLU body frame (`/camera/left_front` is the identity), and its cameras are FLU too
  (`Pinhole:camera_xyz` = FLU). exoego:v2 asks for RDF cameras and an identity reference camera, so the rig frame
  becomes left_front's RDF frame: `world_T_rig` = GSI's pose · `FLU_R_RDF`, `rig_T_cam` = `FLU_R_RDF`ᵀ · GSI's ·
  `FLU_R_RDF`, `camera_xyz` = RDF. Every camera keeps its world pose (`test_every_camera_keeps_gsis_world_pose_in_an_rdf_frame`).
- `slam.rrd` gives `/camera` and `/hand` the same pose on every frame, so GSI's hands live in its rig frame; convert
  moves them to the world with that pose and refuses an episode where `/hand` and `/slam/rig` differ.
- Each pinhole is the calibration's `K` for its view, not the rectified `P1`/`P2`, and the shipped videos are
  undistorted (straight lines stay straight), so the panes project the meshes live through the pinholes and the
  `projections` layer uses the same pinholes.

## Layers and entities

- `base`: the rig node (`schema_version`, `reference` = `cam_00`, `num_cameras`), six cameras (`name` = GSI's view,
  `kind`, `camera_model` = `pinhole`, source keys), pinholes, H.265 samples, `world_T_rig`, the SLAM path.
- `pose`: `/world/gt/coco133_xyz` (GSI's 21 joints per hand in COCO-133 slots, nothing derived, confidence 1.0 for
  every finite joint: GSI ships no score), `/world/gt/hands/<side>/{confidence,wrist,mano}`. `confidence` is GSI's
  `is_usable`; `wrist` puts joint 0 at the MANO `global_orient`; `mano` keeps GSI's record (rig frame).
- `hand_mesh`: GSI's skinned MANO vertices in the world frame, on the frames GSI marks usable.
- `projections`: `coco133_uv_projected` of every view, through its pinhole, NaN off the image.
- The base is compacted with the object-store profile plus `fix_keyframe`. GSI logs `is_keyframe=false` on every
  non-key sample, which Rerun 0.42 rejects (the flag is a sparse marker), and the optimizer then leaves the video
  un-batched; `fix_keyframe` derives the flags from the H.265 samples.
- Default blueprint: the hands, frustums with their videos and the path in a view that rides the rig, beside the six
  views in eye, front and side pairs; each view shows its video, its `coco133_uv_projected` and both meshes. Table
  card: the same 3D view without videos beside left_front.
- GSI's H.265 makes ffmpeg skip SEI NALUs (type 39) and, after a seek, an IDR slice header (type 5). GSI's untouched
  files log the same warnings; every view still renders.

## gsi-six-view-native (temporary)

TODO(gsi-native): a comparison mirror for the GSI partnership; drop it, or keep it, once Pablo decides.
`gsi-six-view-native` reads the same download and repacks each shipped file unchanged under
`gsi-six-view-native__<episode_id>`: `base` = `hands.rrd`, `slam` = `slam.rrd`, compacted the same way, GSI's paths
and timelines (`capture_time`, `frame`, `source_frame`) untouched. Its blueprint shows GSI's world with the six
frustums (a `Pinhole`-only override, since GSI's video shares the pinhole entity) beside the six views with the
hands projected through each pinhole.

## Differences from simplecv

None: simplecv never read this dataset.

## Parity with simplecv

Not applicable.

## Timing

pablo-dl-server, 2026-11-08, 11 episodes (the smallest, largest, shortest, longest and densest plus one per five
categories), prod env, NAS SSD tier: `gsi-six-view` 0.6–7.6 s per episode (24.6 s for all 11, 1.72 GB);
`gsi-six-view-native` 0.1–1.9 s (6.0 s, 1.95 GB). The canonical layers are smaller because GSI repeats the triangle
list and normals on every mesh row. Download: 21.5 MB/s, about 2.3 h for the whole edition.
