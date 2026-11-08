# GSI Egocentric Six-View

## Source

- GSI's partner edition "Egocentric Six-View | 10h": edition `01a0f166-7994-7b43-a622-a325541c4356`, release
  `2026-09-v2`, 1,685 episodes, 10.3 h, 174 GB. Private: GSI shares it with a partner sign-in (username
  `gsi-ots-10h`, password handed over separately). dataforge reads both from `GSI_PARTNER_USERNAME` and
  `GSI_PARTNER_PASSWORD`, which live in `~/.secrets.env` and nowhere else.
- Catalog dataset `gsi-six-view`; one segment per episode, recording id `gsi-six-view__<episode_id>`.
- GSI ships each episode as Rerun recordings already (`gsi.ots.customer-rrd.v2`, written by rerun-sdk 0.37.1, application
  `gsi_off_the_shelf`, recording id = episode id), so dataforge repacks instead of converting: it decodes nothing and
  keeps GSI's entity paths, timelines and samples.
- `property:capture:*` on base: `category` (GSI's caption task family, from the manifest), `num_cameras` (6),
  `num_frames` (GSI's `n_frames`), `gsi_edition`, `gsi_source_release`, `gsi_rrd_contract`, `gsi_privacy_status`
  (`PRIVATE_INTERIM_EXISTING_BLUR`: faces are blurred) and the shipped files' `hands_rrd_sha256` / `slam_rrd_sha256`.
  GSI's own `property:episode:*` (caption, duration, device, source frame range) is kept as shipped.
- The camera-source contract (`log_camera_source`) does not apply: the camera nodes are GSI's, and each `Pinhole`
  carries its resolution.

## Get the raw data

```bash
source ~/.secrets.env
export DATAFORGE_OUTPUT_ROOT=/mnt/nas/ssd-8tb/datasets/gsi-six-view
R=/mnt/nas/ssd-8tb/downloads/gsi/egocentric-six-view-10h
pixi run -e dataforge --frozen dataforge-download --list-remote gsi-six-view --root $R   # every episode, from the manifest
pixi run -e dataforge --frozen dataforge-download gsi-six-view --root $R --sequences <episode_id> ...
pixi run -e dataforge --frozen dataforge-download gsi-six-view --root $R                  # the whole edition
pixi run -e dataforge --frozen dataforge-convert gsi-six-view --root $R
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

`convert` deletes an episode's two files once both of its layers are published; `manifest.json` stays. Point
`--root` at the NAS SSD tier: the download peaks at the whole edition's 174 GB.

## Raw inventory

| Shipped file or stream | Rate / clock | Layer | Entity or not ingested: reason |
| --- | --- | --- | --- |
| `hands.rrd` `/camera/{left,right}_{eye,front,side}/image` | H.265 (`hev1`), 30 fps, `capture_time` | base | as shipped; `VideoStream` + `Pinhole` |
| `hands.rrd` `/camera/<view>` | static | base | as shipped; `rig_T_cam` as `Transform3D` |
| `hands.rrd` `/hand/{left,right}/{vertices,keypoints3d,mano}` | per frame | base | as shipped; MANO `Mesh3D`, `Points3D`, parameters |
| `hands.rrd` `/meta/calibration` | static | base | as shipped; GSI's stereo and IMU calibration blob |
| `hands.rrd` `/`, `/__properties/episode` | static | base | as shipped; `ViewCoordinates`, `AnnotationContext`, episode properties |
| `hands.rrd` `/__properties` `RecordingInfo:name` | static | — | replaced by the dataforge recording name; `start_time` is kept |
| `slam.rrd` `/camera`, `/hand`, `/slam/{rig,path,trail,index,meta}` | per frame | slam | as shipped; world poses of the rig, cameras and hands |
| `slam.rrd` `/__properties` | static | — | not ingested: a derived layer restates no recording properties |

## Clocks

GSI's three timelines are kept: `capture_time` (device duration, ns), `frame` (episode frame index from 0) and
`source_frame` (frame index in the source capture). There is no `video_time`; the segment-table previews play on
`capture_time` (`GsiSixViewDataset.timeline`).

## Layers and entities

- `base`: GSI's `hands.rrd`. `slam`: GSI's `slam.rrd`, which poses `/camera` and `/hand` in the world.
- Both are compacted with the object-store profile plus `fix_keyframe`. GSI logs `is_keyframe=false` on every non-key
  sample, which Rerun 0.42 rejects (the flag is a sparse marker), and the optimizer then leaves the video un-batched;
  `fix_keyframe` derives the flags from the H.265 samples.
- Default blueprint: the world (hands, SLAM path, the six camera frustums) beside the six views in eye, front and side
  pairs. Each view projects both hands' MANO meshes and 3D keypoints through its pinhole. Table card: the hands and the
  path beside `left_front`, which projects the hands the same way, with category, caption and duration.
- GSI logs each `VideoStream` on its `Pinhole` entity (`/camera/<view>/image`), so an entity filter cannot keep the
  frustum and drop the video. The world overrides those entities with the `Pinhole` visualizer alone, so it draws the
  frustums and decodes no video.
- GSI's meshes are opaque. The blueprints override `Mesh3D:albedo_factor` with `hands.HAND_ALBEDO` (alpha 110), so the
  keypoints show through; the data stays as shipped.
- The projection needs no SLAM: `slam.rrd` gives `/camera` and `/hand` the same rig pose at every frame, so a view
  places the hands through GSI's `rig_T_cam` and pinhole alone. Each `Pinhole` is the calibration's `K` for its view,
  not the rectified `P1`/`P2`, and the shipped videos are undistorted, so a plain pinhole is correct.
- GSI's H.265 makes ffmpeg skip SEI NALUs (type 39) and, after a seek, an IDR slice header (type 5). GSI's untouched
  files log the same warnings; every view still renders.

## Differences from simplecv

None: simplecv never read this dataset.

## Parity with simplecv

Not applicable.

## Timing

pablo-dl-server, 2026-11-08, 11 episodes (the smallest, largest, shortest, longest and densest plus one per five
categories): 0.1–3.1 s per episode, 0.2 GB/s, peak RSS 2.1 GB on the largest (614 MB `hands.rrd`). The whole
edition converts in about 15 minutes.
