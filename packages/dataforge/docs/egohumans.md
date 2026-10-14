# EgoHumans

## Source

- EgoHumans (Khirodkar, Bansal, Ma, Newcombe, Vo, Kitani; ICCV 2023): multi-person activities (tagging, lego assembly,
  fencing, basketball, volleyball, badminton, tennis) seen by up to four people's Aria glasses and ~15 static GoPros. Release:
  a Google Drive folder (`1JD963urzuzV_R_6FOVOtlx8UupwUuknR`, from `assets/DOWNLOAD.md` of `rawalkhirodkar/egohumans`), one
  tar.gz per sequence: 133 sequences, 597.6 GB, plus 9 COCO-format benchmark files (2.4 GB, derived, not converted). The code
  repo's license covers the code; the data has no separate license statement.
- The capture layout, the world, the rigs, the layers and the blueprint are Harmony4D's: see [harmony4d.md](harmony4d.md).
  This page lists only what differs.
- Catalog dataset `dataforge-egohumans`; identity `egohumans__<scene>__<sequence>` (`egohumans__01_tagging__004_tagging`).
- `property:capture:source_revision` names the Drive listing date and the config commit.

## Get the raw data

```bash
pixi run -e dataforge dataforge-download egohumans --list-remote                   # 133 sequences from the pinned listing
pixi run -e dataforge dataforge-download egohumans --sequences 02_lego/006_legoassemble
pixi run -e dataforge dataforge-convert egohumans --sequences 02_lego/006_legoassemble
```

- The folder listing (file ids and sizes; Drive publishes no checksums) is pinned in `dataforge/datasets/egohumans_drive.json`
  (2026-10-13), so a download needs no Google API key, no OAuth and no folder scraping. Each archive lands as `<file>.part`,
  is size-checked and renamed; a broken transfer resumes with a `Range` request.
- **Drive quota.** Drive refuses popular files for up to a day ("Quota exceeded: too many users have viewed or downloaded this
  file recently") and answers with an HTML page; `download` recognises it, keeps the part, reports the file and fails at the
  end so a rerun picks it up. It never slices a file into small ranges to get around the quota. On 2026-10-13 every file of the
  folder was refused; a user unit retried hourly for 48 h (Pablo's choice).
- Each sequence's upstream config is fetched from GitHub (`egohumans/configs/<activity>/<sequence>.yaml` at `d7875dd4`) into
  `<root>/configs/`. It lists the glasses (`INVALID_ARIAS`) and GoPros (`INVALID_EXOS`) the authors left out and the GoPros
  posed by hand (`CALIBRATION.MANUAL_EXO_CAMERAS`, pose in `colmap/workplace/camNN.npy`): 91 of the 133 configs pose some GoPros by hand, 29 drop glasses, 31 drop GoPros (all 133 parse).
- A tar.gz cannot be read in place: conversion extracts one sequence into the work directory (a file is placed by the part of
  its path after the `<sequence>` component, since the archives keep the authors' absolute paths; links and `..` are skipped),
  converts, and deletes the extraction. Peak local footprint: archive + extraction + rrds (up to ~45 GB for the largest).

## Raw inventory

As Harmony4D, with these differences:

| Shipped file or stream | Layer | Entity or not ingested: reason |
| --- | --- | --- |
| `processed_data/fit_poses3d/NNNNN.npy` | body_pose | COCO-17 keypoints (the release's benchmark is built from these); `refine_poses3d`, then `poses3d`, only when absent. `property` `keypoints_source` on each person names the directory used |
| `processed_data/{poses3d,refine_poses3d,init_smpl,poses2d}` | — | not ingested: earlier pipeline stages of the same people |
| `processed_data/bboxes/...` | projections | `<pinhole>/boxes/person_NN`, as shipped; read as upstream writes them (a list of `{bbox, human_name, human_id, color}` records per frame, `ego_exo_scene.get_bboxes`), where Harmony4D ships a dict keyed by person |
| `colmap/workplace/camNN.npy` | base | the pose of a hand-posed GoPro (camera-from-COLMAP, 3x4) |
| benchmark `coco/*.json`, `coco_track/*` | — | not ingested: COCO-format views of the same labels |

## Clocks

As Harmony4D: `video_time` = (frame − first frame) / 20 fps, `frame_index` the release frame number.

## Layers and entities

As Harmony4D. Up to four glasses (`rig_91`..`rig_94`) and people; the default blueprint shows the four rgb cameras and eight of
the GoPros (cam01, 03, …, 15).

## Differences from simplecv

None: simplecv never read EgoHumans.

## Parity with simplecv

No simplecv reader. The synthetic tests cover the config handling (invalid and hand-posed cameras), the extraction and the
Drive transport. **No real EgoHumans sequence has been converted yet** (Drive quota, 2026-10-13): the golden checks that
Harmony4D passes (shipped 2D, SMPL vertices) still have to run on one EgoHumans sequence.

## Timing

Pending the first real sequence.

## Known gaps

- Unverified on real data until Drive serves a sequence (above).
- The benchmark files are not converted.
