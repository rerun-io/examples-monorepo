# Gen-HumanEgo (GenRobot DAS-Ego, hands)

The hands release of GenRobot's DAS-Ego corpus. It is the same six-camera rig and MCAP topic family as
[Gen-HumanWholeBody](genrobot.md), without the body track and with metric depth. dataforge converts both
with one converter (`DasEgoDataset` in `datasets/genrobot.py`); this page lists only what differs. Read
[genrobot.md](genrobot.md) for everything shared: the double-sphere lens, `RIG_FROM_CALIB`, the clocks,
the IMU mounting, the H.264 pass-through and the MANO check.

## Source

- Hugging Face dataset `genrobot2025/Gen-HumanEgo` (gated), pinned at commit
  `3cca54aeba0d9af3ebf4995b035d23b41f9387da` (`gen_humanego.REVISION`, last modified 2026-09-25).
- 44,632 MCAPs, 62.0 TB, median 1.55 GB (0.09 to 4.07 GB), 1,848 hours. Every path has four domain levels:
  `data/<domain>/<scene>/<category>/<task>/<shard>/<uuid>.mcap`: 4 domains, 38 scenes, 91 categories,
  236 task folders. Two task folders have a space or capitals (`organize sewing supplies`,
  `Sewing_kit_assembly`); the shared path pattern accepts both.
- Besides `data/`: `README.md`, `assets/banner.jpg` and `egov4_urdf.zip` (the DAS-Ego link structure;
  not ingested: the shipped `T_b_c` already places every camera).
- Catalog name `gen-humanego`; the subset registers as `dataforge-gen-humanego-subset`. The recording id is
  `gen-humanego__<uuid>`. Properties are the whole-body ones (`property:capture:*`, `property:episode:*`).

## What differs from Gen-HumanWholeBody

| | Gen-HumanWholeBody (`genrobot`) | Gen-HumanEgo (`gen-humanego`) |
| --- | --- | --- |
| Size | 15,943 MCAPs, 8.8 TB | 44,632 MCAPs, 62.0 TB |
| Path depth | 2 or 4 domain levels | always 4 |
| Body | SMPL-X body22 skeleton (`robotx/motioncapture/...`) | none |
| Depth | none | camera2 Ego-Depth, 512x416 float16 (`depth` layer) |
| Hand topics | `handtracking/{left,right}` | the same, plus `handtracking_raw/{left,right}` and `handtracking/scores_debug` |
| Hand flags | `is_valid` False on every frame seen | `is_valid` True wherever MANO is shipped |
| Left MANO root | as shipped | as shipped on some episodes, mirrored on others (found per episode) |
| Layers | base, pose, actions, projections, hand_mesh | base, **depth**, pose, actions, projections, hand_mesh |

The README lists `camera[0-6]`; every episode seen has six cameras, `camera0` to `camera5`, as in the
whole-body release. The vendor page (docs.genrobot.ai, "DAS-Ego data introduction") documents the topics and
frames but not depth, the raw hands or the MANO conventions; those were solved from the data and from
GenRobot's `das-datakit` decoder.

## Raw inventory

Every episode seen ships these topics in addition to the whole-body ones (minus the body topic):

| Shipped topic | Rate | Layer | Entity, or not ingested: reason |
| --- | --- | --- | --- |
| `/robot0/sensor/camera2/depth` (`foxglove.CompressedImage`, `predz_i_zstd;dtype=float16`) | 30 Hz | depth | `rig_00/cam_06/pinhole/depth` (`DepthImage`, F16, metres) |
| `/robot0/sensor/camera2/depth_info` (`foxglove.DepthInfo`) | once | depth, sidecar | `rig_00/cam_06`: `kind = depth`, its own lens (`ds_coefficients`), the codec |
| `/robot0/sensor/camera2/conf_vis_compressed` (H.264) | 30 Hz | — | not ingested: a picture of the depth confidence over the image, not data |
| `/robot0/sensor/depth/vis_compressed` (H.264) | 30 Hz | — | not ingested: a colour-mapped rendering of the depth topic |
| `/robot0/handtracking/scores_debug` | 30 Hz | pose, sidecar | `world/gt/hands/<side>/scores`: `pred_score`, `pred_score_quality`, `pred_quality` (3) |
| `/robot0/handtracking_raw/{left,right}` | 30 Hz | sidecar | not in a layer: its MANO parameters land 0.26-0.29 m from its own bones under every convention tried |

Every hand, score and depth message is stamped on a reference (camera2) frame.

## Depth

- `depth_info` gives the size (512x416, camera2's 1600x1300 scaled by 0.32), intrinsics `K`, and its own
  double-sphere `xi` and `alpha`; `codec = float16_predz_i_frame_zstd`.
- The codec, read from `das-datakit` (`utils/topic_parser.py`): one zstd frame per image holding
  `u32 mode_bytes, u32 size` (`size` counts the mode bytes too), `mode_bytes` bytes the decoder skips, then one
  zigzag varint per pixel, row-major: the residual of the float16 bit pattern against `left + up - up_left`.
  That predictor inverts to a 2D prefix sum, so dataforge decodes a frame with two `cumsum` calls in about
  5 ms instead of the reference's per-pixel loop (`gen_humanego_depth.decode_predz`). It matches a scalar
  transcription of the reference bit for bit on real frames.
- The zstd frames carry their decompressed size, so pyarrow's zstd codec decodes them; no new dependency.
- Values are metres along the optical axis (z-depth). The fisheye border ships as NaN; the layer writes it as
  0, the viewer's "no depth". Checked on three smoke episodes: the shipped hand joints, projected through the
  depth lens at camera2's pose, lie a median 20-33 mm behind the depth surface (a hand's thickness); read as
  range instead, the offset spreads from 41 to 123 mm.
- The depth image gets its own camera node, `rig_00/cam_06` (`kind = "depth"`, name `camera2_depth`), at
  camera2's `rig_T_cam` with its own lens, so it never pretends to be camera2's 1600x1300 image.
- 3D views leave it out: a `Pinhole` back-projection would ignore the double-sphere lens.
- Size: about 306 KB per frame in the rrd (lz4 on raw float16), two thirds of an episode's output. Half
  resolution would be 82 KB per frame; a millimetre `EncodedDepthImage` (PNG or RVL) would be smaller still
  but no longer bit-exact. Kept at full resolution and bit-exact for now.

## Hands

- The pose layer is the whole-body one without the body (`with_body = False`): COCO-133 hand slots, wrists,
  MANO parameters, `quality` flags, per-hand confidence 1.0 where the hand message exists. It adds the
  shipped `scores` and the VIO path as one static strip at `/world/runs/vio/trajectory`.
- `is_valid` is True on every frame that ships MANO parameters (99.1-100 % of frames on the smoke episodes);
  `reproj_loss` is 0 and `bbox_touches_edge` False everywhere seen. `pred_score` is 1.0 on almost every
  frame; `pred_score_quality` varies (0.69 to 0.99), so it is kept as shipped and not used as confidence.
- **Left MANO root.** The fingers of the left hand are always in the right-hand convention (mirrored, as in
  the whole-body release). The root rotation is not consistent: on some episodes it is as shipped (the
  whole-body convention), on others it must be mirrored too (`M·R·M`, `M = diag(-1, 1, 1)`). Each convention
  lands at 0.00-0.16 mm median on its episodes and 37-120 mm on the others, so `check_mano` tries both for the
  left hand and keeps the closer; `property:hand_mesh:left_root_mirrored` records which. Seen mirrored on
  firmware 2.1.4 (V5) and 2.1.10 (V6), as shipped on 2.1.21 and 2.1.26 (V6); the subset run counts both.
- `handtracking_raw` is the unsmoothed track: its bones are within a few millimetres of the smoothed ones, but
  its MANO parameters fit nothing (0.26-0.29 m under all eight conventions tried), so it stays in the sidecar.

## Layers and entities

| Layer | Entities |
| --- | --- |
| `base` | As whole-body: root `ViewCoordinates`, COCO-133 context, `rig_00` with `world_T_rig`, six cameras, `imu_00`, `/frames`, `task/instruction` |
| `depth` | `rig_00/cam_06` (depth camera node, frustum) and `rig_00/cam_06/pinhole/depth` (`DepthImage`, F16, `meter = 1`, turbo) |
| `pose` | `world/gt/coco133_xyz`; per hand `wrist`, `confidence`, `mano`, `quality`, `scores`; `world/runs/vio/trajectory` |
| `actions` | `task/actions/coarse` and `task/actions/fine` as `StateChange` tracks |
| `projections` | per camera `pinhole/coco133_uv_projected` (hands only) |
| `hand_mesh` | per hand `world/gt/hands/<side>/mesh`, written when its MANO joints land within 2 mm of its bones |

Default blueprint: the hands in 3D from an upright eye that tracks the rig (`HANDS_EYE`), over the task and the
coarse and fine action lanes; on the right camera2 with its projected hands beside its depth, above the other
five cameras in yaw order. The table card is the whole-body one without the body.

## Subset

The release is 62 TB; the NAS SSD holds a subset. `--budget-tb 1.0` selects it (`genrobot.stratified_subset`):
every one of the 236 task folders in turn, the next episode in uuid order (random hex, so a random draw), task
order within a round fixed by a hash of the path, until the next episode would pass 1 TB. Every task gets an
episode before any task gets a second.

| | Subset | Release |
| --- | --- | --- |
| Episodes | 838 | 44,632 |
| MCAP size | 0.999 TB | 62.0 TB |
| Domains | domestic_services 633, business 166, industry 35, agriculture 4 | 31,708 / 8,279 / 4,636 / 9 |
| Scenes, categories, tasks covered | 38, 91, 236 (all) | 38, 91, 236 |
| Episodes per task | 1 to 4 (22 tasks have one) | 1 to 11,297 (`domestic_services/random/daily_work/daily_work`) |

The subset is task-balanced, not episode-balanced: industry has few task folders, so it holds 4 % of the
subset against 10 % of the release.

## Run

```bash
pixi run -e dataforge --frozen dataforge-download gen-humanego --root /mnt/nas/ssd-8tb/scratch/gen-humanego/raw
export DATAFORGE_OUTPUT_ROOT=/mnt/nas/ssd-8tb/datasets/gen-humanego
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0     # no hf_xet chunk cache under ~/.cache/huggingface
for i in 0 1 2 3 4 5 6 7; do              # one transient user unit per shard, each with MemoryMax=24G
  pixi run -e dataforge --frozen dataforge-convert gen-humanego --root /mnt/nas/ssd-8tb/scratch/gen-humanego/raw \
    --budget-tb 1.0 --shard-index $i --num-shards 8
done
pixi run -e dataforge --frozen dataforge-register --catalog-name dataforge-gen-humanego-subset gen-humanego --budget-tb 1.0
```

- Each worker fetches one MCAP (the next in the background) into the scratch root with `local_dir`, so the
  Hugging Face hub cache does not grow; checks its LFS sha256; writes the sidecar, the base and depth layers
  from it, then the derived layers from the sidecar; checks that every written layer file exists and is not
  empty; and only then deletes the MCAP. A failure after the fetch deletes the episode's layers and sidecar
  and keeps the MCAP for the retry, which is the same command.
- The SSD tier takes the output directly (`atomic_write` publishes each file 0644 by rename).

## Timing

Smoke, pablo-dl-server, prod environment: 0.25 s of conversion per second of capture; depth decoding is 60 %
of it. A 179 s episode (1.59 GB MCAP) writes base 559 MB, depth 1.70 GB, hand_mesh 102 MB, pose 15 MB,
projections 12 MB and a 0.1 GB sidecar in 45 s. The output is about 1.5 times the MCAP.

## Known gaps

- `conf_vis_compressed` and `depth/vis_compressed` are renderings and are not ingested; the depth confidence
  itself is not shipped as data.
- No depth point cloud: the 2D depth pane is the only view of the depth. A `depth_points` layer can be derived
  from the depth layer alone, without the MCAP.
- `handtracking_raw` stays in the sidecar.
- Audio stays in the sidecar, as in the whole-body release.
